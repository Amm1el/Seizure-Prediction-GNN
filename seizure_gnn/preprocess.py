"""Signal preprocessing: filtering, normalization and windowing.

Poster step: "EEG data ... preprocessed to remove artifacts and noise" and
"EEG signals were segmented into fixed time windows".

Conventions used throughout the package:
    C = number of EEG channels
    N = number of samples in a recording
    W = number of windows
    L = samples per window (window_sec * fs)
    fs = sampling rate in Hz
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signals: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied to each channel.

    Removes slow drift (below ``low`` Hz) and high-frequency noise such as muscle
    artifact (above ``high`` Hz). ``filtfilt`` runs the filter forward and backward,
    so it does not shift the signal in time.

    Args:
        signals: array of shape (C, N).
        fs: sampling rate in Hz.
        low, high: pass band edges in Hz.
        order: filter order.

    Returns:
        Filtered array of shape (C, N).
    """
    nyquist = fs / 2.0
    high = min(high, nyquist * 0.99)  # the upper edge must sit below Nyquist
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signals, axis=-1)


def notch_filter(signals: np.ndarray, fs: float, freq: float = 60.0, quality: float = 30.0) -> np.ndarray:
    """Remove power-line interference at ``freq`` Hz (60 Hz in the US, 50 Hz in Europe).

    Args:
        signals: array of shape (C, N).

    Returns:
        Filtered array of shape (C, N). Returned unchanged if ``freq`` is at or
        above the Nyquist frequency.
    """
    if freq >= fs / 2.0:
        return signals
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signals, axis=-1)


def zscore(signals: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each channel to zero mean and unit variance.

    Args:
        signals: array of shape (C, N).

    Returns:
        Array of shape (C, N).
    """
    mean = signals.mean(axis=-1, keepdims=True)
    std = signals.std(axis=-1, keepdims=True)
    return (signals - mean) / (std + eps)


def window_signal(signals: np.ndarray, fs: float, window_sec: float, stride_sec: float) -> np.ndarray:
    """Cut a recording into fixed-length, possibly overlapping windows.

    Args:
        signals: array of shape (C, N).
        fs: sampling rate in Hz.
        window_sec: window length in seconds.
        stride_sec: step between window starts in seconds (stride < window gives overlap).

    Returns:
        Array of shape (W, C, L) with L = round(window_sec * fs).
        Trailing samples that do not fill a whole window are dropped.
    """
    win = int(round(window_sec * fs))
    step = int(round(stride_sec * fs))
    n = signals.shape[-1]
    if n < win:
        return np.empty((0, signals.shape[0], win), dtype=signals.dtype)
    starts = range(0, n - win + 1, step)
    return np.stack([signals[:, s : s + win] for s in starts])


def window_labels(
    sample_labels: np.ndarray, fs: float, window_sec: float, stride_sec: float, min_fraction: float = 0.5
) -> np.ndarray:
    """Turn per-sample seizure labels into one label per window.

    A window is labeled seizure (1) when at least ``min_fraction`` of its samples are
    seizure samples. The windows line up exactly with :func:`window_signal`.

    Args:
        sample_labels: array of shape (N,) with values 0 or 1.

    Returns:
        Integer array of shape (W,).
    """
    windows = window_signal(sample_labels[None, :].astype(float), fs, window_sec, stride_sec)
    return (windows[:, 0, :].mean(axis=-1) >= min_fraction).astype(np.int64)


def preprocess_recording(
    signals: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    notch_hz: float | None = None,
) -> np.ndarray:
    """Full preprocessing chain for one recording: band-pass, optional notch, z-score.

    Args:
        signals: array of shape (C, N).
        notch_hz: power-line frequency to remove, or None to skip the notch filter.

    Returns:
        Array of shape (C, N).
    """
    out = bandpass_filter(signals, fs, low, high)
    if notch_hz is not None:
        out = notch_filter(out, fs, notch_hz)
    return zscore(out)
