"""Seizure prediction from EEG with graph neural networks.

A 2026 reimplementation of the pipeline described in the 2025 UROP poster
"Predicting Seizures using Graph Neural Networks" (Ammiel Bowen, Dr. Yushun Dong,
Florida State University). The original experiment code is not public; this package
rebuilds the described method step by step:

    preprocess    -> filter, normalize and window raw EEG
    connectivity  -> one functional brain network (graph) per window
    data          -> recordings, sequences of window graphs, synthetic EEG
    models        -> GCN encoder, Deep Graph Infomax, TCN, fusion model, baselines
    train         -> DGI pretraining, supervised training, metrics
    ablation      -> Fusion vs single-model baselines
"""

__version__ = "0.1.0"
