import numpy as np
import torch

from seizure_gnn.connectivity import (
    FEATURE_NAMES,
    combine_adjacency,
    mi_adjacency,
    mutual_information,
    node_features,
    pearson_adjacency,
    sparsify_threshold,
    sparsify_topk,
    window_to_data,
)

RNG = np.random.default_rng(0)
FS = 128.0


def random_window(c=8, length=256):
    shared = RNG.standard_normal(length)
    return RNG.standard_normal((c, length)) + np.linspace(0, 1, c)[:, None] * shared


def test_pearson_symmetric_zero_diag_in_unit_interval():
    adj = pearson_adjacency(random_window())
    assert adj.shape == (8, 8)
    np.testing.assert_allclose(adj, adj.T)
    np.testing.assert_array_equal(np.diag(adj), 0)
    assert adj.min() >= 0 and adj.max() <= 1


def test_pearson_handles_flat_channel():
    w = random_window()
    w[3] = 1.0
    adj = pearson_adjacency(w)
    assert np.isfinite(adj).all() and adj[3].sum() == 0


def test_mutual_information_nonnegative_and_detects_dependence():
    a = RNG.standard_normal(2000)
    independent = RNG.standard_normal(2000)
    dependent = a**2 + 0.1 * RNG.standard_normal(2000)  # nonlinear: Pearson ~ 0 but MI high
    mi_ind, mi_dep = mutual_information(a, independent), mutual_information(a, dependent)
    assert mi_ind >= 0 and mi_dep >= 0
    assert mi_dep > 3 * mi_ind
    assert abs(np.corrcoef(a, dependent)[0, 1]) < 0.1


def test_mi_adjacency_matches_pairwise_and_is_valid():
    w = random_window(5, 512)
    adj = mi_adjacency(w, bins=8, normalize=False)
    np.testing.assert_allclose(adj, adj.T)
    np.testing.assert_array_equal(np.diag(adj), 0)
    assert (adj >= 0).all()
    assert np.isclose(adj[1, 4], mutual_information(w[1], w[4], bins=8))
    assert mi_adjacency(w, bins=8).max() <= 1


def test_combine_and_sparsify():
    w = random_window()
    adj = combine_adjacency(pearson_adjacency(w), mi_adjacency(w), alpha=0.5)
    sparse = sparsify_topk(adj, k=2)
    np.testing.assert_allclose(sparse, sparse.T)
    assert ((sparse > 0).sum(axis=1) >= 2).all()  # every node keeps at least k neighbors
    assert (sparse > 0).sum() < (adj > 0).sum()
    assert (sparsify_threshold(adj, 0.3)[adj < 0.3] == 0).all()


def test_node_features_shape():
    feats = node_features(random_window(), FS)
    assert feats.shape == (8, len(FEATURE_NAMES)) and np.isfinite(feats).all()


def test_window_to_data():
    data = window_to_data(random_window(), FS, y=1, top_k=3)
    assert data.x.shape == (8, 7)
    assert data.edge_index.shape[0] == 2
    assert data.edge_weight.shape[0] == data.edge_index.shape[1]
    assert (data.edge_index[0] != data.edge_index[1]).all()  # no self loops
    # Undirected: every edge appears in both directions.
    edges = set(map(tuple, data.edge_index.t().tolist()))
    assert all((j, i) in edges for i, j in edges)
    assert data.y.item() == 1 and data.x.dtype == torch.float32
