"""
Exact Structural Controllability via Liu-Slotine-Barabasi (Nature 2011)
Maximum Bipartite Matching, Edge Criticality Classification, and PBH Verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite


@dataclass
class StructuralControllabilityResult:
    n_nodes: int
    n_edges: int
    n_drivers: int
    driver_fraction: float
    driver_nodes: List[Any]
    matched_in_nodes: Set[Any]
    matching_pairs: List[Tuple[Any, Any]]


def _sorted_nodes(G: nx.DiGraph) -> List[Any]:
    nodes = list(G.nodes())
    try:
        return sorted(nodes)
    except TypeError:
        return nodes


def analyze_structural_controllability(G: nx.DiGraph) -> StructuralControllabilityResult:
    """
    Compute the exact minimum number of driver nodes N_D and a valid driver node set
    using the Liu-Slotine-Barabasi (Nature 2011) maximum bipartite matching theorem.

    For a directed graph G(V, E), build bipartite graph B(V+, V-, E_bip)
    where each directed edge (u -> v) becomes an undirected bipartite edge (("out", u), ("in", v)).
    Unmatched nodes in V- correspond to the minimum driver nodes, with N_D = max(1, N - |M*|).
    """
    nodes = _sorted_nodes(G)
    n = len(nodes)
    if n == 0:
        return StructuralControllabilityResult(0, 0, 0, 0.0, [], set(), [])

    B = nx.Graph()
    out_nodes = [("out", u) for u in nodes]
    in_nodes = [("in", v) for v in nodes]
    B.add_nodes_from(out_nodes, bipartite=0)
    B.add_nodes_from(in_nodes, bipartite=1)

    for u, v in G.edges():
        B.add_edge(("out", u), ("in", v))

    matching = bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)

    matched_in: Set[Any] = set()
    pairs: List[Tuple[Any, Any]] = []
    for u in nodes:
        out_key = ("out", u)
        if out_key in matching:
            _, v = matching[out_key]
            matched_in.add(v)
            pairs.append((u, v))

    unmatched = [v for v in nodes if v not in matched_in]
    if len(unmatched) == 0:
        # Perfect matching (|M*| == N, e.g., cycle cover): N_D = max(1, N - |M*|) = 1
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


def classify_control_edges(G: nx.DiGraph) -> Dict[str, List[Tuple[Any, Any]]]:
    """
    Classify all directed edges of G into the three Liu-Slotine-Barabasi (Nature 2011)
    structural control link categories:
    - 'critical': removing the edge strictly increases the minimum driver count
                  N_D = max(1, N - |M*|).
    - 'redundant': the edge never belongs to any maximum bipartite matching
                   (forcing (u+, v-) into a matching strictly reduces |M*|).
    - 'ordinary': neither critical nor redundant (belongs to at least one maximum
                  matching, while removing it leaves N_D unchanged).
    """
    nodes = _sorted_nodes(G)
    n = len(nodes)
    if n == 0 or G.number_of_edges() == 0:
        return {"critical": [], "ordinary": [], "redundant": []}

    B = nx.Graph()
    out_nodes = [("out", u) for u in nodes]
    in_nodes = [("in", v) for v in nodes]
    B.add_nodes_from(out_nodes, bipartite=0)
    B.add_nodes_from(in_nodes, bipartite=1)
    for u, v in G.edges():
        B.add_edge(("out", u), ("in", v))

    base_matching = bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)
    base_size = len(base_matching) // 2
    base_nd = max(1, n - base_size)

    critical: List[Tuple[Any, Any]] = []
    redundant: List[Tuple[Any, Any]] = []
    ordinary: List[Tuple[Any, Any]] = []

    for u, v in G.edges():
        u_key, v_key = ("out", u), ("in", v)
        # 1. Test removal criticality on N_D = max(1, N - |M*|)
        B.remove_edge(u_key, v_key)
        m_removed = len(bipartite.hopcroft_karp_matching(B, top_nodes=out_nodes)) // 2
        B.add_edge(u_key, v_key)
        rem_nd = max(1, n - m_removed)

        if rem_nd > base_nd:
            critical.append((u, v))
            continue

        # 2. Test whether (u -> v) belongs to at least one maximum matching:
        # Force (u_key, v_key) into matching by removing both endpoints and matching the remainder.
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
    driver_nodes: List[Any],
    seed: int = 42,
    tol: float = 1e-7,
) -> Tuple[bool, int]:
    """
    Numerically verify the Popov-Belevitch-Hautus (PBH) controllability test:
        rank([s I_N - A, B]) == N   for all s in C (all eigenvalues of A plus s = 0),
    using generic random weights strictly on the edges of G (zero artificial self-loops,
    since independent nodal self-loops alter structural controllability per Cowan et al. 2012)
    and m = len(driver_nodes) independent control input columns in B.

    Per Liu-Slotine-Barabasi (Nature 2011, SI Sec. II.B, Cactus Theorem), if G contains
    root strongly connected components (SCCs with zero incoming links from outside) that
    contain no unmatched node (e.g., isolated cycle covers), the first external control
    signal u_0 also couples to one node in each uncovered root SCC without increasing the
    number of independent inputs m.
    """
    rng = np.random.default_rng(seed)
    nodes = _sorted_nodes(G)
    n = len(nodes)
    if n == 0:
        return True, 0
    m = len(driver_nodes)
    if m == 0:
        return False, 0

    node_to_idx = {u: idx for idx, u in enumerate(nodes)}

    # Strictly preserve G's exact zero/nonzero pattern (A[v, u] != 0 iff (u -> v) in E(G))
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        A[node_to_idx[v], node_to_idx[u]] = rng.uniform(0.5, 2.0)

    B = np.zeros((n, m), dtype=float)
    driver_set = set(driver_nodes)
    for col, d_node in enumerate(driver_nodes):
        B[node_to_idx[d_node], col] = rng.uniform(0.8, 1.5)

    # Cactus theorem root-SCC accessibility coupling (only for non-top-linked cycle SCCs):
    # Connect input column 0 to any root SCC of size >= 2 (or self-looped) that has no driver node,
    # ensuring accessibility of closed cycles without adding extra control columns.
    condensation = nx.condensation(G)
    for scc_id in condensation.nodes():
        if condensation.in_degree(scc_id) == 0:
            members = condensation.nodes[scc_id]["members"]
            if driver_set.isdisjoint(members):
                has_internal_edge = len(members) >= 2 or any(G.has_edge(w, w) for w in members)
                if has_internal_edge:
                    rep = sorted(members)[0] if all(isinstance(x, (int, str)) for x in members) else next(iter(members))
                    B[node_to_idx[rep], 0] = rng.uniform(0.8, 1.5)

    eigvals = list(np.linalg.eigvals(A)) + [0.0 + 0.0j]
    min_rank = n
    for lam in eigvals:
        pbh_mat = np.hstack([lam * np.eye(n, dtype=complex) - A.astype(complex), B.astype(complex)])
        svals = np.linalg.svd(pbh_mat, compute_uv=False)
        rk = int(np.sum(svals > tol))
        min_rank = min(min_rank, rk)

    return (min_rank == n), min_rank
