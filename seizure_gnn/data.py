"""Recordings, window graphs and graph sequences.

Poster step: "Multiple graph representations were generated over time to capture
evolving brain activity."

Pipeline for one recording:
    raw (C, N) --preprocess--> windows (W, C, L) --connectivity--> W graphs
    W graphs --sliding sequence of length T--> training samples

Labeling choice (prediction framing)
------------------------------------
A sample is the sequence of T consecutive window graphs ending at window i, and its
label is the seizure label of window ``i + horizon``. With the default ``horizon=1``
the model sees only the past and must say whether the *next* window is a seizure,
which is a (very short-range) prediction task rather than detection of the current
window. ``horizon=0`` gives detection of the final window instead. Clinical seizure
prediction usually uses a pre-ictal horizon of minutes; a longer horizon is a one
argument change here, but it needs real data with long pre-ictal periods.

Real data loading is optional and documented in ``load_edf_recording`` and
``load_numpy_recording``. Everything else in the package is data-source agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .connectivity import window_to_data
from .preprocess import preprocess_recording, window_labels, window_signal


@dataclass
class Recording:
    """One continuous multichannel EEG recording.

    Attributes:
        recording_id: identifier used for leakage-free splitting (use the patient id
            if several recordings come from the same patient).
        signals: array of shape (C, N).
        labels: per-sample seizure labels of shape (N,), values 0 or 1.
        fs: sampling rate in Hz.
        channel_names: optional list of C channel names.
    """

    recording_id: str
    signals: np.ndarray
    labels: np.ndarray
    fs: float
    channel_names: list[str] = field(default_factory=list)


@dataclass
class RecordingGraphs:
    """The window graphs of one recording. ``graphs[w]`` is window w; ``labels`` has shape (W,)."""

    recording_id: str
    graphs: list[Data]
    labels: np.ndarray


# --------------------------------------------------------------------------- synthetic data


class SyntheticEEG:
    """Generates multichannel EEG-like signals with labeled synthetic "seizures".

    This exists so the whole pipeline can be run and tested without a data agreement.
    It is NOT a model of real epilepsy. The seizure signature is deliberately simple so
    the task is learnable:

    * background: per-channel low-pass filtered noise plus a ~10 Hz alpha rhythm and a
      weak shared component (mimicking volume conduction);
    * seizure: a shared rhythmic 3-8 Hz source (with a harmonic) is mixed into a random
      subset of "focal" channels and spreads to the others, with 2-3x amplitude. Because
      the channels share one source, cross-channel synchrony rises;
    * pre-ictal: for ``preictal_sec`` seconds before onset the same source fades in at low
      amplitude, so synchrony starts rising before the seizure. This is what makes the
      next-window prediction task learnable.

    Each recording draws its own seizure frequency, focal channels and gains, so
    recordings differ from each other (which is why we split by recording).
    """

    def __init__(
        self,
        n_channels: int = 16,
        fs: float = 128.0,
        preictal_sec: float = 8.0,
        effect_strength: float = 1.0,
        seed: int = 0,
    ):
        self.n_channels = n_channels
        self.fs = fs
        self.preictal_sec = preictal_sec
        self.effect_strength = effect_strength
        self.rng = np.random.default_rng(seed)

    def _background(self, n: int) -> np.ndarray:
        rng, c, fs = self.rng, self.n_channels, self.fs
        t = np.arange(n) / fs
        white = rng.standard_normal((c, n))
        kernel = np.exp(-np.arange(int(0.05 * fs)) / (0.01 * fs))  # smooth -> more low-frequency power
        noise = np.stack([np.convolve(w, kernel / kernel.sum(), mode="same") for w in white])
        noise /= noise.std(axis=1, keepdims=True)
        alpha_freq = rng.uniform(9, 11, size=(c, 1))
        alpha = rng.uniform(0.2, 0.6, size=(c, 1)) * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi, (c, 1)))
        shared = 0.3 * rng.standard_normal(n)
        return noise + alpha + shared[None, :]

    def generate_recording(self, recording_id: str, duration_sec: float = 120.0, n_seizures: int = 2) -> Recording:
        """Create one recording with ``n_seizures`` non-overlapping seizures of 12-24 s.

        Returns:
            ``Recording`` with signals of shape (C, N) and per-sample labels of shape (N,).
        """
        rng, c, fs = self.rng, self.n_channels, self.fs
        n = int(duration_sec * fs)
        t = np.arange(n) / fs
        x = self._background(n)
        labels = np.zeros(n, dtype=np.int64)

        # Seizure onsets spread evenly with jitter, leaving room for the pre-ictal ramp.
        slot = duration_sec / n_seizures
        for i in range(n_seizures):
            length = rng.uniform(12, 24)
            earliest = i * slot + self.preictal_sec + 2
            latest = (i + 1) * slot - length - 2
            onset = rng.uniform(earliest, max(earliest, latest))
            offset = min(onset + length, duration_sec - 1)

            freq = rng.uniform(3, 8)
            phase = 2 * np.pi * (freq * t + 0.3 * np.sin(2 * np.pi * 0.05 * t))  # slow frequency drift
            source = np.sin(phase) + 0.5 * np.sin(2 * phase)  # harmonic gives a spikier shape
            source /= source.std()

            # Focal channels get the full effect, the rest a weaker spread.
            focal = rng.choice(c, size=c // 3, replace=False)
            spread = np.full(c, rng.uniform(0.3, 0.6))
            spread[focal] = 1.0
            gain = rng.uniform(2.0, 3.0) * self.effect_strength

            # Mixing envelope: 0 -> 0.35 across the pre-ictal ramp, 1 during the seizure.
            env = np.zeros(n)
            pre = (t >= onset - self.preictal_sec) & (t < onset)
            env[pre] = 0.35 * (t[pre] - (onset - self.preictal_sec)) / self.preictal_sec
            ictal = (t >= onset) & (t < offset)
            env[ictal] = 1.0
            env *= self.effect_strength

            weight = np.clip(env[None, :] * spread[:, None], 0, 1)  # (C, N)
            x = (1 - 0.5 * weight) * x + weight * gain * source[None, :]
            labels[ictal] = 1

        x += 0.2 * rng.standard_normal(x.shape)  # sensor noise
        names = [f"CH{i:02d}" for i in range(c)]
        return Recording(recording_id, x.astype(np.float64), labels, fs, names)

    def generate_dataset(self, n_recordings: int = 12, duration_sec: float = 120.0, n_seizures: int = 2) -> list[Recording]:
        """Generate ``n_recordings`` independent recordings with ids ``rec00``, ``rec01``, ..."""
        return [self.generate_recording(f"rec{i:02d}", duration_sec, n_seizures) for i in range(n_recordings)]


# --------------------------------------------------------------------------- real data (optional)


def labels_from_intervals(n_samples: int, fs: float, intervals: list[tuple[float, float]]) -> np.ndarray:
    """Per-sample labels of shape (N,) from a list of (start_sec, end_sec) seizure intervals."""
    labels = np.zeros(n_samples, dtype=np.int64)
    for start, end in intervals:
        labels[int(start * fs) : int(end * fs)] = 1
    return labels


def load_numpy_recording(signals_path: str, fs: float, intervals: list[tuple[float, float]], recording_id: str) -> Recording:
    """Load a recording saved as a ``.npy`` array of shape (C, N).

    Seizure annotations come in as (start_sec, end_sec) pairs, which is how both
    CHB-MIT summary files and TUSZ annotation files express them after parsing.
    """
    signals = np.load(signals_path)
    return Recording(recording_id, signals, labels_from_intervals(signals.shape[1], fs, intervals), fs)


def load_edf_recording(
    edf_path: str,
    intervals: list[tuple[float, float]],
    recording_id: str,
    channels: list[str] | None = None,
) -> Recording:
    """Load an EDF file (the format of CHB-MIT and the TUH EEG Seizure Corpus) with MNE.

    ``mne`` is an optional dependency (``pip install mne``). Parsing each dataset's
    annotation files into ``intervals`` is left to the caller because the formats
    differ (CHB-MIT: ``chbXX-summary.txt``; TUSZ: per-file ``.csv``/``.tse`` files).

    Args:
        edf_path: path to the ``.edf`` file.
        intervals: seizure (start_sec, end_sec) pairs for this file.
        recording_id: use the patient id here to keep patients out of multiple splits.
        channels: channel names to keep, in order. Use the same list for every file so
            graphs have the same nodes across recordings.
    """
    try:
        import mne
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError("Reading EDF files needs MNE: pip install mne") from err

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="error")
    if channels is not None:
        raw.pick(channels)
    signals = raw.get_data()  # (C, N), volts
    fs = float(raw.info["sfreq"])
    return Recording(recording_id, signals, labels_from_intervals(signals.shape[1], fs, intervals), fs, list(raw.ch_names))


# --------------------------------------------------------------------------- graphs and sequences


def build_window_graphs(
    recording: Recording,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
    notch_hz: float | None = None,
    **graph_kwargs,
) -> RecordingGraphs:
    """Preprocess one recording and turn every window into a graph.

    ``graph_kwargs`` go to :func:`seizure_gnn.connectivity.window_to_data`
    (``alpha``, ``bins``, ``top_k``, ``threshold``).
    """
    clean = preprocess_recording(recording.signals, recording.fs, notch_hz=notch_hz)
    windows = window_signal(clean, recording.fs, window_sec, stride_sec)
    labels = window_labels(recording.labels, recording.fs, window_sec, stride_sec)
    graphs = [window_to_data(w, recording.fs, int(y), **graph_kwargs) for w, y in zip(windows, labels)]
    return RecordingGraphs(recording.recording_id, graphs, labels)


def fit_feature_scaler(recordings: list[RecordingGraphs]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of node features over every node of every graph (fit on training data only)."""
    x = torch.cat([g.x for rec in recordings for g in rec.graphs])
    return x.mean(dim=0), x.std(dim=0).clamp_min(1e-6)


def apply_feature_scaler(recordings: list[RecordingGraphs], mean: torch.Tensor, std: torch.Tensor) -> list[RecordingGraphs]:
    """Return copies of the recordings with standardized node features."""
    out = []
    for rec in recordings:
        graphs = []
        for g in rec.graphs:
            g = g.clone()
            g.x = (g.x - mean) / std
            graphs.append(g)
        out.append(RecordingGraphs(rec.recording_id, graphs, rec.labels))
    return out


class SequenceDataset(Dataset):
    """Sliding sequences of T window graphs with a prediction label.

    Item k is ``(graphs, label, recording_id)`` where ``graphs`` is a list of T ``Data``
    objects for windows i-T+1..i and ``label`` is the label of window ``i + horizon``.
    Sequences never cross recording boundaries.
    """

    def __init__(self, recordings: list[RecordingGraphs], seq_len: int = 6, horizon: int = 1):
        self.recordings = recordings
        self.seq_len = seq_len
        self.horizon = horizon
        self.index = []  # (recording position, start window)
        for r, rec in enumerate(recordings):
            n_windows = len(rec.graphs)
            for start in range(0, n_windows - seq_len - horizon + 1):
                self.index.append((r, start))

    def __len__(self) -> int:
        return len(self.index)

    def label(self, k: int) -> int:
        r, start = self.index[k]
        return int(self.recordings[r].labels[start + self.seq_len - 1 + self.horizon])

    def labels(self) -> np.ndarray:
        return np.array([self.label(k) for k in range(len(self))])

    def __getitem__(self, k: int):
        r, start = self.index[k]
        rec = self.recordings[r]
        return rec.graphs[start : start + self.seq_len], self.label(k), rec.recording_id


@dataclass
class SequenceBatch:
    """A mini-batch of B sequences.

    Attributes:
        graphs: list of T PyG ``Batch`` objects; ``graphs[t]`` holds window t of all B sequences.
        feats: handcrafted features without the graph, shape (B, T, C * F) (used by ``TemporalOnly``).
        y: float labels of shape (B,).
        recording_ids: list of B recording ids.
    """

    graphs: list[Batch]
    feats: torch.Tensor
    y: torch.Tensor
    recording_ids: list[str]


def collate_sequences(samples) -> SequenceBatch:
    """``collate_fn`` for a ``torch.utils.data.DataLoader`` over a ``SequenceDataset``."""
    seqs, labels, rec_ids = zip(*samples)
    seq_len = len(seqs[0])
    graphs = [Batch.from_data_list([seq[t] for seq in seqs]) for t in range(seq_len)]
    feats = torch.stack([torch.stack([g.x.flatten() for g in seq]) for seq in seqs])
    return SequenceBatch(graphs, feats, torch.tensor(labels, dtype=torch.float32), list(rec_ids))
