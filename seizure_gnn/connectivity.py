"""Functional brain networks: one graph per EEG window.

Poster step: "transformed into functional brain networks, where nodes represent EEG
channels and edges were based on Pearson correlation and mutual information."

For a window of shape (C, L):
    * nodes  = the C channels
    * edges  = a weighted combination of |Pearson correlation| and mutual information,
               sparsified so each channel keeps only its strongest connections
    * node features = per-channel statistics and band powers of that window
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import welch
from torch_geometric.data import Data

# Classic clinical EEG bands in Hz.
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
FEATURE_NAMES = [f"log_{b}_power" for b in BANDS] + ["mean", "std", "line_length"]


# --------------------------------------------------------------------------- edges


def pearson_adjacency(window: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation between every pair of channels.

    Args:
        window: array of shape (C, L).

    Returns:
        Symmetric array of shape (C, C) with values in [0, 1] and a zero diagonal.
        Constant (flat) channels get zero correlation with everything.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(window)
    adj = np.abs(np.nan_to_num(corr, nan=0.0))
    np.fill_diagonal(adj, 0.0)
    return np.clip(adj, 0.0, 1.0)


def _discretize(window: np.ndarray, bins: int) -> np.ndarray:
    """Map each channel's samples to integer bin ids 0..bins-1 (equal-width bins per channel)."""
    lo = window.min(axis=-1, keepdims=True)
    hi = window.max(axis=-1, keepdims=True)
    scaled = (window - lo) / np.maximum(hi - lo, 1e-12)
    return np.minimum((scaled * bins).astype(np.int64), bins - 1)


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    """Histogram estimate of mutual information I(A; B) in nats between two signals.

    I(A; B) = sum_{i,j} p(i, j) * log( p(i, j) / (p(i) p(j)) )

    Args:
        a, b: arrays of shape (L,).
        bins: number of equal-width bins per signal.

    Returns:
        A non-negative float. Note that histogram estimates are biased upward for
        short windows, so independent signals give small positive values, not 0.
    """
    codes = _discretize(np.stack([a, b]), bins)
    joint = np.bincount(codes[0] * bins + codes[1], minlength=bins * bins).reshape(bins, bins)
    return float(_mi_from_joint_counts(joint[None])[0])


def _mi_from_joint_counts(joint: np.ndarray) -> np.ndarray:
    """MI for a stack of joint count tables of shape (P, bins, bins). Returns shape (P,)."""
    p_xy = joint / joint.sum(axis=(1, 2), keepdims=True)
    p_x = p_xy.sum(axis=2, keepdims=True)
    p_y = p_xy.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p_xy * np.log(p_xy / (p_x * p_y))
    return np.maximum(np.nansum(terms, axis=(1, 2)), 0.0)


def mi_adjacency(window: np.ndarray, bins: int = 16, normalize: bool = True) -> np.ndarray:
    """Pairwise mutual information between all channels of a window.

    All channel pairs are computed in one ``np.bincount`` call for speed.

    Args:
        window: array of shape (C, L).
        bins: histogram bins per channel.
        normalize: divide by log(bins), the largest MI possible with this many bins,
            so values fall in [0, 1] and can be averaged with Pearson weights.

    Returns:
        Symmetric non-negative array of shape (C, C) with a zero diagonal.
    """
    c = window.shape[0]
    codes = _discretize(window, bins)
    rows, cols = np.triu_indices(c, k=1)
    pair_ids = np.arange(len(rows))[:, None] * bins * bins
    flat = pair_ids + codes[rows] * bins + codes[cols]  # (P, L)
    joint = np.bincount(flat.ravel(), minlength=len(rows) * bins * bins).reshape(len(rows), bins, bins)
    mi = _mi_from_joint_counts(joint)
    if normalize:
        mi = mi / np.log(bins)
    adj = np.zeros((c, c))
    adj[rows, cols] = mi
    adj[cols, rows] = mi
    return adj


def combine_adjacency(pearson: np.ndarray, mi: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Weighted average of the two edge measures: alpha * pearson + (1 - alpha) * mi.

    Both inputs should already be on a [0, 1] scale (see ``mi_adjacency(normalize=True)``).
    """
    return alpha * pearson + (1.0 - alpha) * mi


def sparsify_topk(adj: np.ndarray, k: int) -> np.ndarray:
    """Keep each node's k strongest edges, then symmetrize.

    An edge survives if it is in the top k of either endpoint, so the result stays
    undirected. Every node keeps at least k neighbors (no isolated channels).

    Args:
        adj: array of shape (C, C) with a zero diagonal.

    Returns:
        Sparse symmetric array of shape (C, C).
    """
    c = adj.shape[0]
    k = min(k, c - 1)
    scores = adj.copy()
    np.fill_diagonal(scores, -np.inf)  # never pick a self loop
    keep = np.zeros_like(adj, dtype=bool)
    top = np.argsort(-scores, axis=1)[:, :k]
    keep[np.repeat(np.arange(c), k), top.ravel()] = True
    keep = keep | keep.T
    return np.where(keep, adj, 0.0)


def sparsify_threshold(adj: np.ndarray, threshold: float) -> np.ndarray:
    """Zero out every edge weaker than ``threshold``. Shape (C, C) in and out."""
    return np.where(adj >= threshold, adj, 0.0)


# --------------------------------------------------------------------------- node features


def node_features(window: np.ndarray, fs: float) -> np.ndarray:
    """Per-channel features for one window.

    Features (in order, see ``FEATURE_NAMES``):
        log band power for delta, theta, alpha and beta (Welch power spectral density),
        mean, standard deviation, and line length (mean absolute sample-to-sample change,
        a classic seizure marker).

    Args:
        window: array of shape (C, L).

    Returns:
        Array of shape (C, 7).
    """
    freqs, psd = welch(window, fs=fs, nperseg=min(window.shape[-1], int(fs)), axis=-1)
    df = freqs[1] - freqs[0]
    powers = []
    for lo, hi in BANDS.values():
        mask = (freqs >= lo) & (freqs < hi)
        powers.append(np.log(psd[:, mask].sum(axis=-1) * df + 1e-10))
    stats = [
        window.mean(axis=-1),
        window.std(axis=-1),
        np.abs(np.diff(window, axis=-1)).mean(axis=-1),
    ]
    return np.stack(powers + stats, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- graph


def window_adjacency(
    window: np.ndarray,
    alpha: float = 0.5,
    bins: int = 16,
    top_k: int | None = 4,
    threshold: float | None = None,
) -> np.ndarray:
    """Combined, sparsified adjacency for one window. Shape (C, L) -> (C, C)."""
    adj = combine_adjacency(pearson_adjacency(window), mi_adjacency(window, bins), alpha)
    if top_k is not None:
        adj = sparsify_topk(adj, top_k)
    if threshold is not None:
        adj = sparsify_threshold(adj, threshold)
    return adj


def adjacency_to_data(adj: np.ndarray, x: np.ndarray, y: int | None = None) -> Data:
    """Convert a dense adjacency and node features into a PyTorch Geometric ``Data``.

    Args:
        adj: symmetric array of shape (C, C); nonzero entries become edges.
        x: node features of shape (C, F).
        y: optional window label.

    Returns:
        ``Data`` with ``x`` (C, F), ``edge_index`` (2, E) holding both directions of
        every undirected edge, ``edge_weight`` (E,), and ``y`` (1,) if given.
    """
    src, dst = np.nonzero(adj)
    data = Data(
        x=torch.as_tensor(x, dtype=torch.float32),
        edge_index=torch.as_tensor(np.stack([src, dst]), dtype=torch.long),
        edge_weight=torch.as_tensor(adj[src, dst], dtype=torch.float32),
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.long)
    return data


def window_to_data(
    window: np.ndarray,
    fs: float,
    y: int | None = None,
    alpha: float = 0.5,
    bins: int = 16,
    top_k: int | None = 4,
    threshold: float | None = None,
) -> Data:
    """One EEG window (C, L) -> one functional brain network as a PyG ``Data`` object."""
    adj = window_adjacency(window, alpha=alpha, bins=bins, top_k=top_k, threshold=threshold)
    return adjacency_to_data(adj, node_features(window, fs), y)
