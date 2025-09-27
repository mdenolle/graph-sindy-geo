import numpy as np
from pyg_sindy_demo.demo import GraphConfig, SimConfig, build_knn_graph, simulate_system

def test_graph_shapes():
    gcfg = GraphConfig(N=40, k=4)
    G, pos, L = build_knn_graph(gcfg)
    assert G.num_nodes == gcfg.N
    # kNN undirected ⇒ ~2*k edges per node (within factor for symmetry)
    assert G.edge_index.shape[1] >= gcfg.N * gcfg.k
    assert L.shape == (gcfg.N, gcfg.N)

def test_sim_outputs():
    gcfg = GraphConfig(N=30, k=4, noise_std=0.0)
    scfg = SimConfig(T=10, dt=0.05)
    G, pos, L = build_knn_graph(gcfg)
    X, Xdot, X_noisy = simulate_system(pos, L, gcfg, scfg)
    assert X.shape == (scfg.T, gcfg.N)
    assert np.allclose(X_noisy, X)  # no noise
    # basic sanity: nonzero derivatives
    assert np.linalg.norm(Xdot) > 0
