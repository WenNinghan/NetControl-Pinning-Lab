"""
Generic structural controllability for dedicated actuators and shared signals.
Matching-based edge classification and numerical PBH tests of explicit input layouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


@dataclass
class StructuralControllabilityResult:
    n_nodes: int
    n_edges: int
    n_drivers: int
    driver_fraction: float
    driver_nodes: List[Any]
    matched_in_nodes: Set[Any]
    matching_pairs: List[Tuple[Any, Any]]
    source_components: List[List[Any]] = field(default_factory=list)

    @property
    def minimum_input_channels(self) -> int:
        """Minimum independent signal count when one signal may actuate several sites."""
        return max(1, self.n_nodes - len(self.matching_pairs)) if self.n_nodes else 0

    @property
    def shared_input_sites(self) -> List[List[Any]]:
        """One explicit minimum-channel layout on these same physical sites."""
        channels = [[v] for v in self.driver_nodes if v not in self.matched_in_nodes]
        extra_sites = [v for v in self.driver_nodes if v in self.matched_in_nodes]
        if not channels:
            return [extra_sites] if extra_sites else []
        channels[0].extend(extra_sites)
        return channels


def _sorted_nodes(G: nx.DiGraph) -> List[Any]:
    nodes = list(G.nodes())
    try:
        return sorted(nodes)
    except TypeError:
        return sorted(nodes, key=lambda v: (type(v).__module__, type(v).__qualname__, repr(v)))


def analyze_structural_controllability(G: nx.DiGraph) -> StructuralControllabilityResult:
    """
    Return a minimum dedicated layout (one independent input per listed site).

    Every right-unmatched vertex of a maximum state matching needs a site, and
    every source strongly connected component (SCC) needs an input path. Among
    maximum matchings, maximize the number of source SCCs with an unmatched
    vertex, then add one site per still-uncovered source SCC. This realizes the
    dedicated structural input selection bound m + beta - alpha, where m is
    the matching deficiency, beta the number of source SCCs, and alpha the
    maximum number of those SCCs assignable to unmatched vertices.

    The shared-channel alternative is reported separately by shared_input_sites.
    Both statements concern generic, independently weighted graph edges.
    """
    nodes = _sorted_nodes(G)
    n = len(nodes)
    if n == 0:
        return StructuralControllabilityResult(0, 0, 0, 0.0, [], set(), [])

    index = {node: i for i, node in enumerate(nodes)}
    condensation = nx.condensation(G)
    sources = [
        sorted(condensation.nodes[c]["members"], key=index.__getitem__)
        for c, degree in condensation.in_degree() if degree == 0
    ]
    sources.sort(key=lambda members: index[members[0]])

    # Rows are target states. A column is a state predecessor, a single-use
    # representative of a source SCC, or that row's private slack. Relative to
    # slack, a state match is worth beta+1 and an SCC match is worth 1.
    # Thus no gain in SCC coverage can sacrifice one actual state match.
    n_sources = len(sources)
    rows = list(range(n))
    cols = [n + n_sources + i for i in range(n)]
    weights = [1] * n
    for source in nodes:
        for target in G.successors(source):
            rows.append(index[target])
            cols.append(index[source])
            weights.append(n_sources + 2)
    for i, members in enumerate(sources):
        rows.extend(index[v] for v in members)
        cols.extend([n + i] * len(members))
        weights.extend([2] * len(members))

    assignment = csr_matrix((weights, (rows, cols)), shape=(n, 2 * n + n_sources))
    target_rows, source_cols = min_weight_full_bipartite_matching(assignment, maximize=True)
    pairs = [(nodes[col], nodes[row]) for row, col in zip(target_rows, source_cols) if col < n]
    matched_in = {target for _, target in pairs}
    sites = set(nodes) - matched_in
    for members in sources:
        if sites.isdisjoint(members):
            sites.add(members[0])
    driver_nodes = [v for v in nodes if v in sites]
    n_drivers = len(driver_nodes)
    return StructuralControllabilityResult(
        n_nodes=n,
        n_edges=G.number_of_edges(),
        n_drivers=n_drivers,
        driver_fraction=n_drivers / float(n),
        driver_nodes=driver_nodes,
        matched_in_nodes=matched_in,
        matching_pairs=pairs,
        source_components=sources,
    )


def classify_control_edges(G: nx.DiGraph) -> Dict[str, List[Tuple[Any, Any]]]:
    """
    Classify all directed edges of G into the three Liu-Slotine-Barabasi (Nature 2011)
    structural control link categories:
    - 'critical': removing the edge strictly increases the minimum independent signal count
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
    """Check exactly the listed dedicated sites, using one input column per site."""
    return verify_pbh_input_placement(G, [[node] for node in driver_nodes], seed=seed, tol=tol)


def verify_pbh_input_placement(
    G: nx.DiGraph,
    input_sites: Sequence[Sequence[Any]],
    seed: int = 42,
    tol: float = 1e-7,
) -> Tuple[bool, int]:
    """Check an explicit input support with seeded generic nonzero A and B weights.

    Each group is one independent column of B. Every site in that group
    receives that signal. No other states are actuated. This is a numerical
    check of one realization; generic structural analysis is a separate step.
    """
    rng = np.random.default_rng(seed)
    nodes = _sorted_nodes(G)
    n = len(nodes)
    if n == 0:
        return True, 0

    node_to_idx = {u: idx for idx, u in enumerate(nodes)}
    A = np.zeros((n, n), dtype=float)
    for u in nodes:
        for v in sorted(G.successors(u), key=node_to_idx.__getitem__):
            A[node_to_idx[v], node_to_idx[u]] = rng.uniform(0.5, 2.0)

    B = np.zeros((n, len(input_sites)), dtype=float)
    for col, group in enumerate(input_sites):
        for node in group:
            B[node_to_idx[node], col] = rng.uniform(0.8, 1.5)

    eigvals = list(np.linalg.eigvals(A)) + [0.0 + 0.0j]
    min_rank = n
    for lam in eigvals:
        pbh_mat = np.hstack([lam * np.eye(n, dtype=complex) - A.astype(complex), B.astype(complex)])
        svals = np.linalg.svd(pbh_mat, compute_uv=False)
        rk = int(np.sum(svals > tol))
        min_rank = min(min_rank, rk)

    return (min_rank == n), min_rank
