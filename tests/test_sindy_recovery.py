import numpy as np
from pyg_sindy_demo.demo import (
    GraphConfig, SimConfig, build_knn_graph,
    simulate_system, GraphLibrary, run_sindy
)

def test_sindy_coefficients_close():
    # Smaller problem to keep CI fast/stable
    gcfg = GraphConfig(N=60, k=5, noise_std=0.005, seed=123)
    scfg = SimConfig(T=40, dt=0.05, a_true=-0.8, b_true=0.6)
    G, pos, L = build_knn_graph(gcfg)
    X, Xdot, X_noisy = simulate_system(pos, L, gcfg, scfg)

    lib = GraphLibrary(L)
    T_train = int(0.7 * scfg.T)
    Phi = lib.transform_graph_time_series(X_noisy[:T_train])
    sindy = run_sindy(Phi, Xdot[:T_train], scfg.dt)
    coef = sindy.coefficients().ravel()

    # Ground truth ordering: [x, x^2, Lx]
    gt = np.array([0.0, scfg.b_true, scfg.a_true])

    # Allow some tolerance due to noise and finite data
    assert abs(coef[0]) < 0.05
    assert abs(coef[1] - gt[1]) < 0.1
    assert abs(coef[2] - gt[2]) < 0.1
