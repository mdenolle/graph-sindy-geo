# PyG + PySINDy Graph Demo

A tiny end-to-end demo that:
- builds an irregular k-NN graph of sensors,
- simulates `x_dot = a * L x + b * x^2`,
- trains a small GNN to learn `x_dot`,
- uses **PySINDy** to recover a sparse, human-readable model.

## Quickstart (CPU)
```bash
make install
make test
make run
```

This installs CPU wheels for Torch/PyG. If you want CUDA, install the appropriate Torch build and matching PyG wheels.

## What to read

* `src/pyg_sindy_demo/demo.py` — main demo (functions + CLI entry).
* `tests/` — shape/sanity checks and SINDy coefficient recovery tests.

