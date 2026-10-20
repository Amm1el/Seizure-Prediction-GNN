import numpy as np

from seizure_gnn.preprocess import bandpass_filter, notch_filter, preprocess_recording, window_labels, window_signal, zscore

FS = 128.0


def test_bandpass_keeps_shape_and_removes_out_of_band():
    t = np.arange(int(10 * FS)) / FS
    slow, mid, fast = np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 60 * t)
    x = np.stack([slow + mid + fast, mid])
    y = bandpass_filter(x, FS, 0.5, 40)
    assert y.shape == x.shape
    # What survives on channel 0 should be close to the 10 Hz component alone.
    core = slice(int(2 * FS), int(8 * FS))
    assert np.corrcoef(y[0, core], mid[core])[0, 1] > 0.98


def test_notch_and_zscore():
    x = np.random.default_rng(0).standard_normal((3, 1000)) * 5 + 2
    assert notch_filter(x, FS, 60).shape == x.shape
    z = zscore(x)
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-8)
    np.testing.assert_allclose(z.std(axis=1), 1, atol=1e-6)
    assert preprocess_recording(x, FS, notch_hz=60).shape == x.shape


def test_window_shapes_and_labels_align():
    c, seconds = 4, 10
    x = np.random.default_rng(0).standard_normal((c, int(seconds * FS)))
    w = window_signal(x, FS, window_sec=2.0, stride_sec=1.0)
    assert w.shape == (9, c, 256)  # starts at 0..8 s
    np.testing.assert_array_equal(w[1], x[:, 128:384])

    labels = np.zeros(x.shape[1], dtype=int)
    labels[int(5 * FS) :] = 1
    y = window_labels(labels, FS, 2.0, 1.0)
    assert y.shape == (9,)
    assert y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]  # window [4, 6) is half seizure -> 1


def test_too_short_recording_gives_no_windows():
    assert window_signal(np.zeros((2, 10)), FS, 2.0, 1.0).shape == (0, 2, 256)
