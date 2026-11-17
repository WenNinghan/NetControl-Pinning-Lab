"""
Canonical and domain-informed directed complex network generators.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def erdos_renyi_directed(n: int, mean_degree: float, seed: int = 42) -> nx.DiGraph:
    """Generate a directed Erdos-Renyi network with expected average degree <k>."""
    rng = np.random.default_rng(seed)
    p = np.clip(mean_degree / max(1, n - 1), 0.0, 1.0)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    adj = rng.random((n, n)) < p
    np.fill_diagonal(adj, False)
    rows, cols = np.where(adj)
    for u, v in zip(rows.tolist(), cols.tolist()):
        G.add_edge(u, v, weight=float(rng.uniform(0.4, 1.2)))
    return G


def scale_free_directed(n: int, mean_degree: float, gamma_exp: float = 2.4, seed: int = 42) -> nx.DiGraph:
    """
    Generate a directed scale-free network using the static Goh-Kahng-Kim model
    with tunable degree heterogeneity.
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 / max(1.05, gamma_exp - 1.0)
    ranks = np.arange(1, n + 1, dtype=float)
    weights = ranks ** (-alpha)
    # Permute in/out weights independently to mimic realistic directed hubs
    w_out = rng.permutation(weights)
    w_in = rng.permutation(weights)
    prob_matrix = np.outer(w_out, w_in)
    np.fill_diagonal(prob_matrix, 0.0)
    prob_matrix /= prob_matrix.sum()

    target_edges = int(round(n * mean_degree))
    flat_probs = prob_matrix.ravel()
    nonzero_mask = flat_probs > 0
    max_possible = int(nonzero_mask.sum())
    target_edges = min(target_edges, max_possible)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    if target_edges > 0:
        chosen = rng.choice(n * n, size=target_edges, replace=False, p=flat_probs)
        us, vs = np.divmod(chosen, n)
        for u, v in zip(us.tolist(), vs.tolist()):
            G.add_edge(u, v, weight=float(rng.uniform(0.4, 1.2)))
    return G


def small_world_directed(n: int, k_out: int = 3, p_rewire: float = 0.2, seed: int = 42) -> nx.DiGraph:
    """Generate a directed Watts-Strogatz small-world network."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for offset in range(1, k_out + 1):
            v = (u + offset) % n
            if rng.random() < p_rewire:
                candidates = [c for c in range(n) if c != u and not G.has_edge(u, c)]
                if candidates:
                    v = int(rng.choice(candidates))
            G.add_edge(u, v, weight=float(rng.uniform(0.5, 1.1)))
    return G


def reaction_cascade_network(n: int = 40, branch_factor: float = 2.6, seed: int = 42) -> nx.DiGraph:
    """
    Generate a hierarchical ozone-pollutant reaction & transport cascade network:
    - Primary precursors (top tier, ~25%)
    - Reactive intermediates & radical chain carriers (middle tier, ~50%)
    - Terminal transformation products & recycling feedback loops (bottom tier, ~25%)
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    n1 = max(2, int(0.25 * n))
    n2 = max(4, int(0.50 * n))
    n3 = n - n1 - n2

    for i in range(n):
        tier = "precursor" if i < n1 else ("intermediate" if i < n1 + n2 else "byproduct")
        G.add_node(i, tier=tier, decay_rate=float(rng.uniform(0.35, 0.95)))

    tier1 = list(range(0, n1))
    tier2 = list(range(n1, n1 + n2))
    tier3 = list(range(n1 + n2, n))

    # Forward pathway edges: Precursor -> Intermediate
    for u in tier1:
        k = max(1, int(rng.poisson(branch_factor)))
        targets = rng.choice(tier2, size=min(len(tier2), k), replace=False)
        for v in targets:
            G.add_edge(int(u), int(v), weight=float(rng.uniform(0.5, 1.4)))

    # Intermediate -> Intermediate cross-reactions & Intermediate -> Byproduct
    for u in tier2:
        k_mid = max(1, int(rng.poisson(branch_factor * 0.6)))
        mid_candidates = [x for x in tier2 if x != u]
        for v in rng.choice(mid_candidates, size=min(len(mid_candidates), k_mid), replace=False):
            G.add_edge(int(u), int(v), weight=float(rng.uniform(0.3, 1.0)))

        k_out = max(1, int(rng.poisson(branch_factor * 0.5)))
        for v in rng.choice(tier3, size=min(len(tier3), k_out), replace=False):
            G.add_edge(int(u), int(v), weight=float(rng.uniform(0.4, 1.1)))

    # Weak radical recycling feedback from intermediates/byproducts
    n_feedback = max(2, n // 8)
    for _ in range(n_feedback):
        u = int(rng.choice(tier2 + tier3))
        v = int(rng.choice(tier1 + tier2))
        if u != v:
            G.add_edge(u, v, weight=float(rng.uniform(0.15, 0.45)))

    return G


def build_continuous_system_matrix(G: nx.DiGraph, damping: float = 1.25, coupling: float = 0.35) -> np.ndarray:
    """
    Construct a continuous-time linear state dynamics matrix A in dx/dt = A x + B u
    from a directed weighted network G=(V, E).

    Here A_{ij} = coupling * w_{j->i} (incoming influence from j to i),
    and diagonal entries A_{ii} = -(damping + sum_j A_{ij}) ensure dissipative stability
    when unforced, matching physical diffusive/kinetic network transport.
    """
    n = G.number_of_nodes()
    nodes = list(range(n))
    W_adj = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    # W_adj[j, i] is weight of edge j -> i, so A = coupling * W_adj.T
    A = coupling * W_adj.T.copy()
    in_strength = A.sum(axis=1)
    np.fill_diagonal(A, -(damping + 0.5 * in_strength))
    return A
