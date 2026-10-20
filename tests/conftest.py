import os
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore", category=FutureWarning)

from seizure_gnn.data import SequenceDataset, SyntheticEEG, apply_feature_scaler, build_window_graphs, fit_feature_scaler  # noqa: E402


@pytest.fixture(scope="session")
def tiny_recordings():
    """Six short synthetic recordings turned into window graphs (shared across tests)."""
    gen = SyntheticEEG(n_channels=8, fs=64.0, seed=0)
    recs = [build_window_graphs(r, window_sec=2.0, stride_sec=1.0) for r in gen.generate_dataset(6, duration_sec=80)]
    mean, std = fit_feature_scaler(recs)
    return apply_feature_scaler(recs, mean, std)


@pytest.fixture(scope="session")
def tiny_dataset(tiny_recordings):
    return SequenceDataset(tiny_recordings, seq_len=4, horizon=1)
