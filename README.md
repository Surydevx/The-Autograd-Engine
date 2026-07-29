# AutoGrad Engine

A lightweight, PyTorch-style scalar autograd engine and deep learning library built from scratch in pure Python.

This project implements a reverse-mode automatic differentiation engine over dynamically built directed acyclic graphs (DAGs). On top of this AutoGrad Engine core, it features a fully modular neural network API, adaptive optimizers, robust loss functions, anda robust diagnostic graph visualization tools using graphviz.

## Features

* **Core Autograd Engine:** Scalar `Value` objects that dynamically track operations and compute gradients via topological sort. Supports standard math operations, `exp`, `log`, `pow`, and non-linear activations (`relu`, `sigmoid`, `tanh`, `sin`, `cos`).
* **PyTorch-Style API:** Clean abstraction layers (`Module`, `Layer`, `Neuron`, `MLP`) that make defining architectures intuitive. Includes built-in support for model state saving/loading via JSON.
* **Modern Optimizers:** Includes implementations of SGD (with momentum), RMSprop, Adam, and AdamW (with decoupled weight decay).
* **Rich Loss Suite:** Built-in loss functions for regression and classification, including MSE, L1 (MAE), Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE), and Hinge Loss.
* **CLI Training Loop:** A full-featured training script (`main.py`) complete with early stopping, dynamic epoch shuffling, train/validation splitting, and a live terminal UI powered by `rich`.
* **Computational Graph Visualizer:** Export telemetry during the backward pass and render trimmed, highly-readable DAGs using Graphviz.

## Project Structure

``` bash
├── ARCHIVE
│   ├── generate_data.py
│   ├── main-2.py
│   ├── shuffle_data.py
│   ├── spiral.csv
│   └── xor.csv
├── AUTOGRAD
│   ├── data_procs_engine.py
│   ├── Engine.py
│   ├── Modules.py
│   ├── Optimizers.py
│   ├── __pycache__
│   │   ├── data_procs_engine.cpython-314.pyc
│   │   ├── Engine.cpython-314.pyc
│   │   ├── Modules.cpython-314.pyc
│   │   ├── Optimizers.cpython-314.pyc
│   │   ├── test_Engine.cpython-314-pytest-9.1.1.pyc
│   │   └── Utils.cpython-314.pyc
│   ├── test_Engine.py
│   └── Utils.py
├── dag_visualizer.py
├── main.py
├── Model
│   ├── computational_graph
│   ├── computational_graph.svg
│   ├── model_weights.json
│   ├── telemetry.json
│   └── training.svg
├── Pre-processed_data
├── __pycache__
│   ├── animate.cpython-314.pyc
│   ├── Engine.cpython-314.pyc
│   ├── shuffle_data.cpython-314.pyc
│   └── test_Engine.cpython-314-pytest-9.1.1.pyc
├── pyproject.toml
├── README.md
└── uv.lock

7 directories, 31 files
```

The core modules:

* `Engine.py` - The core mathematical engine, `Value` class, and topological graph traversal.
* `modules.py` - Neural network architectures (`MLP`, `LinearRegression`, `LogisticRegression`).
* `optimizers.py` - Optimization algorithms for gradient descent updates.
* `utils.py` - Loss functions and JSON telemetry exporters.
* `data_procs_engine.py` - Data ingestion, shuffling, and fallback toy datasets.
* `main.py` - The central CLI tool for configuring and running training loops.
* `dag_visualizer.py` - Renders `.svg` DAGs from exported graph telemetry.

## Quick Start

### Prerequisites

Make sure you have Python 3.8+ installed, along with `uv`:

```bash
uv sync
```
