from __future__ import annotations
import importlib.util
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

_TORCH_SPEC = importlib.util.find_spec("torch")
if _TORCH_SPEC:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_PYG_SPEC = importlib.util.find_spec("torch_geometric")
if _PYG_SPEC:
    from torch_geometric.data import Data  # type: ignore[assignment]
    from torch_geometric.nn import GCNConv  # type: ignore[assignment]
else:  # pragma: no cover - exercised when PyG is unavailable
    Data = None  # type: ignore[assignment]
    GCNConv = None  # type: ignore[assignment]

_PYSINDY_SPEC = importlib.util.find_spec("pysindy")
if _PYSINDY_SPEC:
    import pysindy as ps
else:  # pragma: no cover - exercised when PySINDy is unavailable
    ps = None  # type: ignore[assignment]


class _SimpleData:
    """Fallback container mimicking torch_geometric.data.Data."""

    def __init__(self, edge_index, pos, num_nodes):
        self.edge_index = edge_index
        self.pos = pos
        self.num_nodes = num_nodes

# -----------------------------
# Configs
# -----------------------------
@dataclass
class GraphConfig:
    N: int = 100
    k: int = 6
    noise_std: float = 0.01
    seed: int = 42

@dataclass
class SimConfig:
    T: int = 60
    dt: float = 0.05
    a_true: float = -0.8  # diffusion (smoothing)
    b_true: float = 0.6   # quadratic reaction

# -----------------------------
# Graph & Laplacian
# -----------------------------
def build_knn_graph(cfg: GraphConfig) -> Tuple[Data, np.ndarray, sp.csr_matrix]:
    rng = np.random.default_rng(cfg.seed)
    if torch is not None:
        torch.manual_seed(cfg.seed)
    pos = rng.random((cfg.N, 2), dtype=np.float64)

    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    knn_idx = np.argsort(dists, axis=1)[:, 1: cfg.k + 1]
    rows = np.repeat(np.arange(cfg.N), cfg.k)
    cols = knn_idx.reshape(-1)
    edge_index = np.stack([rows, cols], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    if torch is not None:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        pos_t = torch.tensor(pos, dtype=torch.float32)
    else:
        edge_index_t = edge_index
        pos_t = pos.astype(np.float32)

    if Data is not None and torch is not None:
        G = Data(edge_index=edge_index_t, pos=pos_t, num_nodes=cfg.N)
    else:
        G = _SimpleData(edge_index=edge_index_t, pos=pos_t, num_nodes=cfg.N)

    L = _normalized_laplacian(edge_index, cfg.N)

    return G, pos, L


def _normalized_laplacian(edge_index: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    row, col = edge_index
    weights = np.ones(row.shape[0], dtype=np.float64)
    adj = sp.coo_matrix((weights, (row, col)), shape=(num_nodes, num_nodes))
    # Ensure symmetry for undirected graph
    adj = adj.maximum(adj.T)
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    D_inv_sqrt = sp.diags(inv_sqrt_deg)
    identity = sp.eye(num_nodes, dtype=np.float64)
    laplacian = identity - D_inv_sqrt @ adj @ D_inv_sqrt
    return laplacian.tocsr()

# -----------------------------
# Dynamics & simulation
#   x_dot = a * L x + b * x^2
# -----------------------------
def f_graph(x: np.ndarray, L: sp.csr_matrix, a: float, b: float) -> np.ndarray:
    return a * (L @ x) + b * (x ** 2)

def rk4_step(x: np.ndarray, dt: float, L: sp.csr_matrix, a: float, b: float) -> np.ndarray:
    k1 = f_graph(x, L, a, b)
    k2 = f_graph(x + 0.5 * dt * k1, L, a, b)
    k3 = f_graph(x + 0.5 * dt * k2, L, a, b)
    k4 = f_graph(x + dt * k3, L, a, b)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_system(pos: np.ndarray, L: sp.csr_matrix, gcfg: GraphConfig, scfg: SimConfig):
    x0 = np.sin(2 * math.pi * pos[:, 0]) * np.cos(2 * math.pi * pos[:, 1])
    X = np.zeros((scfg.T, gcfg.N))
    Xdot = np.zeros_like(X)
    X[0] = x0
    Xdot[0] = f_graph(X[0], L, scfg.a_true, scfg.b_true)
    for t in range(1, scfg.T):
        X[t] = rk4_step(X[t-1], scfg.dt, L, scfg.a_true, scfg.b_true)
        Xdot[t] = f_graph(X[t], L, scfg.a_true, scfg.b_true)
    rng = np.random.default_rng(gcfg.seed)
    X_noisy = X + gcfg.noise_std * rng.standard_normal(X.shape)
    return X, Xdot, X_noisy

# -----------------------------
# Tiny GNN to learn x_dot
# -----------------------------
if torch is not None and GCNConv is not None:

    class GNNdxdt(nn.Module):
        def __init__(self, hidden=64):
            super().__init__()
            self.gcn1 = GCNConv(1, hidden)
            self.gcn2 = GCNConv(hidden, hidden)
            self.mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x_node, edge_index):
            h = F.silu(self.gcn1(x_node, edge_index))
            h = F.silu(self.gcn2(h, edge_index))
            return self.mlp(h)

    def train_gnn_dxdt(G: Data, X_noisy: np.ndarray, Xdot: np.ndarray, hidden=64, epochs=300, lr=5e-3, wd=1e-5):
        device = torch.device("cpu")
        model = GNNdxdt(hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        x_tensor = torch.tensor(X_noisy[..., None], dtype=torch.float32, device=device)  # (T,N,1)
        y_tensor = torch.tensor(Xdot[...,  None], dtype=torch.float32, device=device)   # (T,N,1)
        T = X_noisy.shape[0]
        T_train = int(0.7 * T)
        edge_index = G.edge_index.to(device)

        model.train()
        for _ in range(epochs):
            t_idx = np.random.randint(0, T_train)
            opt.zero_grad()
            pred = model(x_tensor[t_idx], edge_index)
            loss = F.mse_loss(pred, y_tensor[t_idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = [model(x_tensor[t], edge_index).squeeze(-1).cpu().numpy() for t in range(T)]
        Xdot_hat = np.stack(preds, axis=0)
        return model, Xdot_hat

else:  # pragma: no cover - exercised when torch/PyG is unavailable

    class GNNdxdt:  # type: ignore[override]
        def __init__(self, *_, **__):
            raise ImportError("train_gnn_dxdt requires torch and torch_geometric to be installed.")

    def train_gnn_dxdt(*_, **__):  # type: ignore[override]
        raise ImportError("train_gnn_dxdt requires torch and torch_geometric to be installed.")

# -----------------------------
# PySINDy custom library
# -----------------------------
if ps is not None:

    class GraphLibrary(ps.CustomLibrary):
        def __init__(self, L: sp.csr_matrix):
            self.L = L.tocsr()
            super().__init__(library_functions=[], function_names=[])

        def fit(self, x, y=None):
            return self

        def transform(self, x):
            raise NotImplementedError("Use transform_graph_time_series(T,N).")

        def transform_graph_time_series(self, X_TN: np.ndarray) -> np.ndarray:
            T, N = X_TN.shape
            x = X_TN.reshape(-1)
            x2 = x ** 2
            Lx = np.vstack([(self.L @ X_TN[t]) for t in range(T)]).reshape(-1)
            return np.stack([x, x2, Lx], axis=1)

        def get_feature_names(self, input_features=None):
            return ["x", "x^2", "Lx"]

    def run_sindy(Phi: np.ndarray, xdot: np.ndarray, dt: float):
        opt = ps.STLSQ(alpha=1e-3, threshold=0.02, max_iter=50)
        sindy = ps.SINDy(feature_library=ps.IdentityLibrary(), optimizer=opt)
        sindy.fit(Phi, t=dt, x_dot=xdot.reshape(-1, 1))
        return sindy

else:

    class GraphLibrary:  # type: ignore[override]
        def __init__(self, L: sp.csr_matrix):
            self.L = L.tocsr()

        def fit(self, x, y=None):
            return self

        def transform(self, x):
            raise NotImplementedError("Use transform_graph_time_series(T,N).")

        def transform_graph_time_series(self, X_TN: np.ndarray) -> np.ndarray:
            T, N = X_TN.shape
            x = X_TN.reshape(-1)
            x2 = x ** 2
            Lx = np.vstack([(self.L @ X_TN[t]) for t in range(T)]).reshape(-1)
            return np.stack([x, x2, Lx], axis=1)

        def get_feature_names(self, input_features=None):
            return ["x", "x^2", "Lx"]

    class _SimpleSINDy:
        def __init__(self, coefficients: np.ndarray):
            self._coef = coefficients.reshape(1, -1)

        def coefficients(self) -> np.ndarray:
            return self._coef

        def print(self, precision: int = 4):  # pragma: no cover - only used in CLI
            fmt = np.array2string(self._coef, precision=precision, suppress_small=True)
            print(f"SimpleSINDy coefficients: {fmt}")

    def _stlsq(Phi: np.ndarray, xdot: np.ndarray, threshold: float = 0.02, max_iter: int = 50) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(Phi, xdot, rcond=None)
        coef = coef.reshape(-1)
        for _ in range(max_iter):
            small = np.abs(coef) < threshold
            if np.all(~small):
                break
            coef[small] = 0.0
            large_mask = ~small
            if not np.any(large_mask):
                break
            Phi_large = Phi[:, large_mask]
            solved, *_ = np.linalg.lstsq(Phi_large, xdot, rcond=None)
            coef[large_mask] = solved
        return coef

    def run_sindy(Phi: np.ndarray, xdot: np.ndarray, dt: float):  # noqa: ARG001
        coef = _stlsq(Phi, xdot.reshape(-1))
        return _SimpleSINDy(coef)

# -----------------------------
# CLI entry
# -----------------------------
def main():
    gcfg = GraphConfig()
    scfg = SimConfig()
    G, pos, L = build_knn_graph(gcfg)
    X, Xdot, X_noisy = simulate_system(pos, L, gcfg, scfg)
    model, Xdot_hat = train_gnn_dxdt(G, X_noisy, Xdot)

    lib = GraphLibrary(L)
    T_train = int(0.7 * scfg.T)
    Phi_train = lib.transform_graph_time_series(X_noisy[:T_train])

    sindy_true = run_sindy(Phi_train, Xdot[:T_train], scfg.dt)
    sindy_hat  = run_sindy(Phi_train, Xdot_hat[:T_train], scfg.dt)

    gt = np.array([0.0, scfg.b_true, scfg.a_true])
    print("\n=== SINDy (TRUE derivatives) ===")
    sindy_true.print(precision=4)
    print("Coefficients [x, x^2, Lx]:", sindy_true.coefficients())
    print("\n=== SINDy (GNN-estimated derivatives) ===")
    sindy_hat.print(precision=4)
    print("Coefficients [x, x^2, Lx]:", sindy_hat.coefficients())
    print("\nGround truth [x, x^2, Lx]:", gt)

if __name__ == "__main__":
    main()
