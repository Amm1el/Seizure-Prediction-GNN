import numpy as np
import torch
from torch_geometric.loader import DataLoader as GraphLoader

from seizure_gnn.data import collate_sequences
from seizure_gnn.models import TCN, BatchedDGI, FusionClassifier, GCNEncoder, SpatialOnly, TemporalOnly
from seizure_gnn.train import (
    compute_metrics,
    evaluate,
    make_loader,
    pretrain_dgi,
    set_seed,
    split_by_recording,
    train_classifier,
)


def first_batch(dataset, size=5):
    return collate_sequences([dataset[i] for i in range(size)])


def test_sequence_dataset_and_collate(tiny_recordings, tiny_dataset):
    graphs, label, rec_id = tiny_dataset[0]
    assert len(graphs) == 4 and label in (0, 1)
    # Label is the window right after the sequence (horizon = 1).
    assert label == tiny_recordings[0].labels[4]
    batch = first_batch(tiny_dataset)
    assert len(batch.graphs) == 4 and batch.graphs[0].num_graphs == 5
    assert batch.feats.shape == (5, 4, 8 * 7)
    assert batch.y.shape == (5,)


def test_tcn_is_causal():
    set_seed(0)
    tcn = TCN(3, 8).eval()
    x = torch.randn(2, 6, 3)
    x2 = x.clone()
    x2[:, -1] += 10  # change only the last step
    h1 = tcn.net(x.transpose(1, 2))
    h2 = tcn.net(x2.transpose(1, 2))
    torch.testing.assert_close(h1[:, :, :-1], h2[:, :, :-1])  # earlier outputs unaffected
    assert tcn(x).shape == (2, 8)


def test_each_model_forward(tiny_dataset):
    batch = first_batch(tiny_dataset)
    for model in (
        FusionClassifier(GCNEncoder(7, 16), tcn_hidden=16),
        FusionClassifier(GCNEncoder(7, 16), tcn_hidden=16, freeze_encoder=True),
        SpatialOnly(7, 16),
        TemporalOnly(8 * 7, 16),
    ):
        out = model(batch)
        assert out.shape == (5,) and torch.isfinite(out).all()


def test_frozen_encoder_gets_no_gradient(tiny_dataset):
    model = FusionClassifier(GCNEncoder(7, 16), freeze_encoder=True)
    model(first_batch(tiny_dataset)).sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.tcn.parameters())


def test_dgi_loss_decreases(tiny_recordings):
    set_seed(0)
    encoder = GCNEncoder(7, 16)
    dgi = BatchedDGI(encoder)
    graphs = [g for rec in tiny_recordings[:2] for g in rec.graphs[:40]]
    b = next(iter(GraphLoader(graphs, batch_size=len(graphs))))
    pos_z, neg_z, s = dgi(b.x, b.edge_index, b.edge_weight, batch=b.batch)
    assert pos_z.shape == neg_z.shape == s.shape == (b.num_nodes, 16)

    opt = torch.optim.Adam(dgi.parameters(), lr=5e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = dgi.loss(*dgi(b.x, b.edge_index, b.edge_weight, batch=b.batch))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-5:]) < np.mean(losses[:5])

    history = pretrain_dgi(GCNEncoder(7, 16), tiny_recordings[:2], epochs=2)
    assert len(history) == 2


def test_metrics():
    m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.9]))
    assert m["accuracy"] == 0.75 and np.isclose(m["auc"], 1.0) and np.isclose(m["f1"], 0.8)


def test_split_has_no_recording_leakage(tiny_recordings):
    train, val, test = split_by_recording(tiny_recordings, seed=0)
    ids = [{r.recording_id for r in s} for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(s) for s in ids) == len(tiny_recordings)


def test_end_to_end_training_beats_chance(tiny_recordings):
    from seizure_gnn.data import SequenceDataset

    set_seed(0)
    train, val, test = split_by_recording(tiny_recordings, val_fraction=0.17, test_fraction=0.17, seed=0)
    train_set, val_set, test_set = (SequenceDataset(s, 4, 1) for s in (train, val, test))
    encoder = GCNEncoder(7, 16)
    pretrain_dgi(encoder, train, epochs=2)
    model = train_classifier(FusionClassifier(encoder, tcn_hidden=16), train_set, val_set, epochs=4, batch_size=32)
    metrics = evaluate(model, make_loader(test_set))
    majority = max(test_set.labels().mean(), 1 - test_set.labels().mean())
    assert metrics["auc"] > 0.7
    assert metrics["accuracy"] > majority and metrics["f1"] > 0  # better than always guessing one class
    print(f"tiny end-to-end: {metrics}, majority-class accuracy {majority:.2f}")
