# Seizure Prediction with Graph Neural Networks

Early data-loading and graph-construction code from my undergraduate research with Dr. Yushun Dong in Florida State University's Department of Computer Science (2024 to 2025).

The project tested whether combining graph neural network models improves seizure prediction from EEG. EEG recordings become functional brain networks, with electrodes as nodes, and the models learn from both their structure and how they change over time. The work is summarized in my 2025 Undergraduate Research Opportunity Program (UROP) poster: [view the poster](https://ammielbowen.com/docs/UROP-2025-Seizure-Prediction-Poster.pdf).

## What is in this repository

This repository holds the starting pipeline, not the full model:

- `Main.py` loads the electrode adjacency matrix and distance data, builds the electrode graph with NetworkX, converts it into a PyTorch Geometric `Data` object, and loads the train and test seizure file markers. Node features are placeholders.
- `graph_utils.py` draws the electrode graph with Matplotlib.

## Tech stack

Python, PyTorch, PyTorch Geometric, NetworkX, pandas, Matplotlib

## Run

```bash
pip install torch torch_geometric networkx pandas matplotlib
python Main.py
```

Update the `path_to_data/...` paths in `Main.py` to point to your copy of the dataset.

## Author

Ammiel Bowen · [ammielbowen.com](https://ammielbowen.com) · [LinkedIn](https://www.linkedin.com/in/ammielbowen/)
