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

## What is in this repository

This repository holds the starting data pipeline, not the full model:

- `Main.py` loads the electrode graph (adjacency matrix and electrode distances), builds it with NetworkX, converts it into a PyTorch Geometric `Data` object with both directions of every edge, and loads the train and test seizure file markers. Node features are placeholders here.
- `graph_utils.py` draws the electrode graph with Matplotlib.
- `docs/` holds the UROP poster.

The electrode graph and file markers follow the layout of the [EEG-GNN-SSL](https://github.com/tsy935/eeg-gnn-ssl) dataset preparation.

## Tech stack

Python, PyTorch, PyTorch Geometric, NetworkX, NumPy, pandas, Matplotlib

## Run

```bash
pip install torch torch_geometric networkx numpy pandas matplotlib
python Main.py
```

Update the `path_to_data/...` paths at the top of `Main.py` to point to your copy of the dataset.

## References

- Tang et al., EEG-GNN-SSL: Self-supervised graph neural networks for improved EEG seizure analysis. [github.com/tsy935/eeg-gnn-ssl](https://github.com/tsy935/eeg-gnn-ssl)
- Veličković et al. (2019), Deep Graph Infomax. ICLR.

Thanks to Xihao Park (Osaka University) for guidance on GNNs and EEG analysis.

## Author

Ammiel Bowen · [ammielbowen.com](https://ammielbowen.com) · [LinkedIn](https://www.linkedin.com/in/ammielbowen/)
