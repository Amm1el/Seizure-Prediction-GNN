# Seizure Prediction with Graph Neural Networks

Undergraduate research with Dr. Yushun Dong, Department of Computer Science, Florida State University (September 2024 to July 2025). Presented at FSU's 2025 Undergraduate Research Opportunity Program (UROP) symposium.

**Poster:** [docs/UROP-2025-Seizure-Prediction-Poster.pdf](docs/UROP-2025-Seizure-Prediction-Poster.pdf)

## Question

Seizures are hard to predict, and existing machine learning models lack the speed and accuracy needed for real-time prediction. This project asked whether graph neural networks (GNNs), and in particular combinations of GNN models, can improve seizure prediction from EEG and shorten time to detection.

## Approach

1. **Data.** A publicly available EEG dataset labeled for seizure and non-seizure recordings, preprocessed to remove artifacts and noise.
2. **Graphs from EEG.** Signals are segmented into fixed time windows. Each window becomes a functional brain network: nodes are EEG channels, and edges are weighted by Pearson correlation and mutual information between channels. A sequence of graphs over time captures how brain activity evolves.
3. **Spatial features.** Deep Graph Infomax (DGI) learns node embeddings without labels (self-supervised), capturing the structure of each brain network.
4. **Temporal features.** A second GNN-based temporal model learns how those embeddings change across windows.
5. **Fusion.** Spatial and temporal features are combined to classify seizure versus non-seizure states.

The approach was developed through a literature review of existing GNN models, mathematical modeling of which combinations should work, and experiments testing different architectures.

## Preliminary results

- Combining DGI with temporal modeling reached accuracy in the **mid-70% range**, with F1 and AUC improvements over single-GNN models.
- Ablation studies across several candidate architectures suggest the combined model classifies seizure states better than either part alone.
- Results are preliminary. Planned next steps were hyperparameter optimization, cross-validation and statistical testing on larger datasets.

## Code

> **Honesty note.** This code is a 2026 reimplementation of the pipeline described in the poster; the original experiment code isn't public. The synthetic-data ablation is a smoke test of the pipeline, not a reproduction of the poster's results. No result in this repository comes from real EEG yet.

The `seizure_gnn` package rebuilds each step of the poster's method:

| Poster step | Module | What it does |
|---|---|---|
| Remove artifacts and noise | `seizure_gnn/preprocess.py` | 0.5-40 Hz Butterworth band-pass (`filtfilt`), optional 50/60 Hz notch, per-channel z-score |
| Segment into fixed time windows | `seizure_gnn/preprocess.py` | Overlapping windows (default 2 s, 1 s stride) and per-window seizure labels |
| Functional brain network per window | `seizure_gnn/connectivity.py` | Nodes = channels; edges = 0.5 · \|Pearson\| + 0.5 · histogram mutual information, top-k sparsified; node features = log delta/theta/alpha/beta power (Welch), mean, std, line length; output is a PyTorch Geometric `Data` |
| Sequence of graphs over time | `seizure_gnn/data.py` | Sliding sequences of T window graphs, labeled by the *next* window (prediction framing; `horizon=0` gives detection). Includes a `SyntheticEEG` generator and optional EDF/NumPy loaders |
| DGI self-supervised node embeddings | `seizure_gnn/models.py` | `GCNEncoder` trained with `torch_geometric.nn.DeepGraphInfomax` (feature-shuffling corruption, one summary per window graph) |
| GNN-based temporal model | `seizure_gnn/models.py` | Per-window graph embedding (mean-pooled GCN output) fed to a causal, dilated Temporal Convolutional Network (TCN) |
| Fuse spatial + temporal features | `seizure_gnn/models.py` | `FusionClassifier`: [final-window graph embedding ; TCN output] -> MLP -> seizure logit |
| Ablation vs single models | `seizure_gnn/models.py`, `seizure_gnn/ablation.py` | `SpatialOnly` (GCN on the last window), `TemporalOnly` (TCN on handcrafted features, no graph), and fusion without DGI pretraining |
| Training and evaluation | `seizure_gnn/train.py` | DGI pretraining, then supervised training with class-weighted BCE; accuracy, F1, ROC AUC; train/val/test split by recording so near-identical overlapping windows from one recording never sit on both sides of the split |

The original 2025 starter script is still at the root: `Main.py` loads the EEG-GNN-SSL electrode graph (adjacency matrix and electrode distances), builds it with NetworkX, converts it into a PyTorch Geometric `Data` object and loads the train/test seizure file markers; `graph_utils.py` draws the graph. The electrode graph and file markers follow the layout of the [EEG-GNN-SSL](https://github.com/tsy935/eeg-gnn-ssl) dataset preparation. Update the `path_to_data/...` paths at the top of `Main.py` before running it.

`docs/` holds the UROP poster.

### Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt

pytest -q                          # 19 tests, a few seconds on CPU
python scripts/run_ablation.py     # synthetic ablation, about 1.5 minutes on CPU
```

The ablation writes [`results/ablation_synthetic.md`](results/ablation_synthetic.md). Every model scores near the ceiling there because the synthetic seizures are easy to spot, so that table checks that the pipeline works; it says nothing about which model is better on real EEG.

### Next steps: running on real EEG

1. **Get data.** [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/1.0.0/) is open on PhysioNet. The [TUH EEG Seizure Corpus (TUSZ)](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) is larger but requires signing a data use agreement.
2. **Load it.** `seizure_gnn.data.load_edf_recording` reads EDF files with MNE (`pip install mne`). Parse each dataset's seizure annotations into `(start_sec, end_sec)` intervals and pick one fixed channel list so every graph has the same nodes. Use the patient id as `recording_id` so splits are patient-wise.
3. **Pick the task.** For true prediction, label sequences by a pre-ictal horizon of minutes rather than the next window, and handle the much stronger class imbalance.
4. **Evaluate properly.** Leave-one-patient-out cross-validation, a validation-tuned decision threshold, and statistical tests across folds, as the poster's planned next steps describe.

## Tech stack

Python, PyTorch, PyTorch Geometric, NumPy, SciPy, scikit-learn, NetworkX, pandas, Matplotlib, pytest

## References

- Tang et al., EEG-GNN-SSL: Self-supervised graph neural networks for improved EEG seizure analysis. [github.com/tsy935/eeg-gnn-ssl](https://github.com/tsy935/eeg-gnn-ssl)
- Veličković et al. (2019), Deep Graph Infomax. ICLR.

Thanks to Xihao Park (Osaka University) for guidance on GNNs and EEG analysis.

## Author

Ammiel Bowen · [ammielbowen.com](https://ammielbowen.com) · [LinkedIn](https://www.linkedin.com/in/ammielbowen/)
