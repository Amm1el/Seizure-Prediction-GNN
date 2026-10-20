"""Training and evaluation.

Two stages, matching the poster's method:
    1. Self-supervised: train the GCN encoder with Deep Graph Infomax on every window
       graph from the training recordings. No labels are used.
    2. Supervised: train ``FusionClassifier`` (TCN + head, and optionally the encoder)
       with binary cross-entropy on the seizure labels.

Splitting by recording (or patient)
-----------------------------------
Windows overlap and neighboring windows look almost identical, and every recording has
its own "fingerprint" (electrode impedance, montage, the patient's background rhythm).
A random split over windows would put near-copies of test windows into training and
inflate every metric. ``split_by_recording`` assigns whole recordings to train, val or
test, so the test score measures generalization to unseen recordings. On real data the
id should be the patient id.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphLoader

from .data import RecordingGraphs, SequenceDataset, collate_sequences
from .models import BatchedDGI, GCNEncoder


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable on CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_recording(
    recordings: list[RecordingGraphs], val_fraction: float = 0.2, test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[RecordingGraphs], list[RecordingGraphs], list[RecordingGraphs]]:
    """Shuffle recording ids and assign whole recordings to train / val / test.

    Recordings that share an id (for example, several files from one patient) always
    land in the same split.
    """
    ids = sorted({r.recording_id for r in recordings})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, round(test_fraction * len(ids)))
    n_val = max(1, round(val_fraction * len(ids)))
    test_ids, val_ids = set(ids[:n_test]), set(ids[n_test : n_test + n_val])
    train = [r for r in recordings if r.recording_id not in test_ids | val_ids]
    val = [r for r in recordings if r.recording_id in val_ids]
    test = [r for r in recordings if r.recording_id in test_ids]
    return train, val, test


def make_loader(dataset: SequenceDataset, batch_size: int = 32, shuffle: bool = False, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sequences, generator=gen)


# --------------------------------------------------------------------------- stage 1: DGI


def pretrain_dgi(
    encoder: GCNEncoder,
    recordings: list[RecordingGraphs],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    verbose: bool = False,
) -> list[float]:
    """Self-supervised DGI pretraining of ``encoder`` on all window graphs (labels unused).

    Returns the mean loss per epoch. The encoder is trained in place.
    """
    dgi = BatchedDGI(encoder)
    opt = torch.optim.Adam(dgi.parameters(), lr=lr)
    graphs = [g for rec in recordings for g in rec.graphs]
    loader = GraphLoader(graphs, batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(epochs):
        dgi.train()
        total, count = 0.0, 0
        for b in loader:
            opt.zero_grad()
            pos_z, neg_z, summary = dgi(b.x, b.edge_index, b.edge_weight, batch=b.batch)
            loss = dgi.loss(pos_z, neg_z, summary)
            loss.backward()
            opt.step()
            total += loss.item() * b.num_graphs
            count += b.num_graphs
        history.append(total / count)
        if verbose:
            print(f"  DGI epoch {epoch + 1:2d}  loss {history[-1]:.4f}")
    return history


# --------------------------------------------------------------------------- stage 2: supervised


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, probabilities), both of shape (num_samples,)."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        ys.append(batch.y.numpy())
        ps.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, F1 (seizure = positive class) and ROC AUC."""
    pred = (prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": auc,
    }


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    return compute_metrics(*predict(model, loader))


def train_classifier(
    model: nn.Module,
    train_set: SequenceDataset,
    val_set: SequenceDataset,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> nn.Module:
    """Supervised training with class-weighted BCE; keeps the weights with the best val AUC.

    Seizure windows are the minority class, so the positive term of the loss is scaled
    by pos_weight = (#negatives / #positives) in the training set. Without it a model
    can score high accuracy by always predicting "no seizure".
    """
    labels = train_set.labels()
    pos_weight = torch.tensor((labels == 0).sum() / max((labels == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    train_loader = make_loader(train_set, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_set, batch_size)

    best_auc, best_state = -1.0, copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            opt.step()
        val = evaluate(model, val_loader)
        val_auc = val["auc"] if not np.isnan(val["auc"]) else val["f1"]
        if val_auc > best_auc:
            best_auc, best_state = val_auc, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch + 1:2d}  val acc {val['accuracy']:.3f}  f1 {val['f1']:.3f}  auc {val['auc']:.3f}")
    model.load_state_dict(best_state)
    return model
