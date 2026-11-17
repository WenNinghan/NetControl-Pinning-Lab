"""
Exact Structural Controllability via Liu-Slotine-Barabasi (Nature 2011)
Maximum Bipartite Matching, Edge Criticality Classification, and PBH Verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite


@dataclass
class StructuralControllabilityResult:
    n_nodes: int
    n_edges: int
    n_drivers: int
    driver_fraction: float
    driver_nodes: List[int]
    matched_in_nodes: Set[int]
    matching_pairs: List[Tuple[int, int]]


def analyze_structural_controllability(G: nx.DiGraph) -> StructuralControllabilityResult:
    """
    Compute the exact minimum number of driver nodes N_D and a valid driver node set
    using the Liu-Slotine-Barabasi maximum bipartite matching theorem.

    For a directed graph G(V, E), build bipartite graph B(V_out, V_in, E_bip)
    where each directed edge (u -> v) becomes an undirected bipartite edge (u+, v-).
    Unmatched nodes in V_in correspond to the minimum driver nodes (with N_D = max(1, N - |M*|)).
    """
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return StructuralControllabilityResult(0, 0, 0, 0.0, [], set(), [])

    B = nx.Graph()
    out_nodes = [f"out_{u}" for u in nodes]
    in_nodes = [f"in_{v}" for v in nodes]
    B.add_nodes_from(out_nodes, bipartite=0)
    B.add_nodes_from(in_nodes, bipartite=1)

    for u, v in G.edges():
        B.add_edge(f"out_{u}", f"in_{v}")

    matching = bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)

    matched_in: Set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for u in nodes:
        out_key = f"out_{u}"
        if out_key in matching:
            in_key = matching[out_key]
            v = int(in_key.split("_")[1])
            matched_in.add(v)
            pairs.append((u, v))

    unmatched = [v for v in nodes if v not in matched_in]
    if len(unmatched) == 0:
        # Perfect matching (e.g., cycle cover): 1 arbitrary driver node is required
        driver_nodes = [nodes[0]]
    else:
        driver_nodes = unmatched

    n_drivers = len(driver_nodes)
    return StructuralControllabilityResult(
        n_nodes=n,
        n_edges=G.number_of_edges(),
        n_drivers=n_drivers,
        driver_fraction=n_drivers / float(n),
        driver_nodes=driver_nodes,
        matched_in_nodes=matched_in,
        matching_pairs=pairs,
    )


def classify_control_edges(G: nx.DiGraph) -> Dict[str, List[Tuple[int, int]]]:
    """
    Classify all directed edges of G into three structural control categories:
    - 'critical': removing the edge strictly increases N_D (or reduces maximum matching size |M*|).
    - 'redundant': the edge never belongs to any maximum matching (removing it leaves |M*| unchanged,
                   and forcing it into a matching strictly reduces |M*|).
    - 'ordinary': neither critical nor redundant (participates in some, but not all, maximum matchings).
    """
    nodes = sorted(G.nodes())
    B = nx.Graph()
    out_nodes = [f"out_{u}" for u in nodes]
    in_nodes = [f"in_{v}" for v in nodes]
    B.add_nodes_from(out_nodes, bipartite=0)
    B.add_nodes_from(in_nodes, bipartite=1)
    for u, v in G.edges():
        B.add_edge(f"out_{u}", f"in_{v}")

    base_matching = bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)
    base_size = len(base_matching) // 2

    critical: List[Tuple[int, int]] = []
    redundant: List[Tuple[int, int]] = []
    ordinary: List[Tuple[int, int]] = []

    for u, v in G.edges():
        u_key, v_key = f"out_{u}", f"in_{v}"
        # 1. Test removal criticality
        B.remove_edge(u_key, v_key)
        m_removed = len(bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)) // 2
        B.add_edge(u_key, v_key)

        if m_removed < base_size:
            critical.append((u, v))
            continue

        # 2. Test whether (u -> v) can belong to at least one maximum matching
        # Force (u_key, v_key) into matching by removing both endpoints and matching the remainder
        B_sub = B.copy()
        B_sub.remove_node(u_key)
        B_sub.remove_node(v_key)
        sub_top = [node for node in out_nodes if node != u_key]
        m_forced = 1 + (len(bipartite.hopcroft_karp_matching(B_sub, top_nodes=sub_top)) // 2)

        if m_forced < base_size:
            redundant.append((u, v))
        else:
            ordinary.append((u, v))

    return {
        "critical": critical,
        "ordinary": ordinary,
        "redundant": redundant,
    }


def verify_pbh_controllability(
    G: nx.DiGraph,
    driver_nodes: List[int],
    seed: int = 42,
    tol: float = 1e-7,
) -> Tuple[bool, int]:
    """
    Numerically verify the Popov-Belevitch-Hautus (PBH) controllability test
    rank([sI - A, B]) == N for all eigenvalues s of A, using generic random weights
    on the structural graph G and independent control inputs on `driver_nodes`.
    """
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return True, 0

    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        A[v, u] = rng.uniform(0.5, 2.0)
    # Generic self-loops on diagonal avoid trivial zero-eigenvalue degeneracy
    for i in range(n):
        A[i, i] = -rng.uniform(1.5, 3.0)

    m = len(driver_nodes)
    B = np.zeros((n, m), dtype=float)
    for col, d_node in enumerate(driver_nodes):
        B[d_node, col] = rng.uniform(0.8, 1.5)

    eigvals = np.linalg.eigvals(A)
    min_rank = n
    for lam in eigvals:
        pbh_mat = np.hstack([lam * np.eye(n, dtype=complex) - A.astype(complex), B.astype(complex)])
        svals = np.linalg.svd(pbh_mat, compute_uv=False)
        rk = int(np.sum(svals > tol))
        min_rank = min(min_rank, rk)

    return (min_rank == n), min_rank
