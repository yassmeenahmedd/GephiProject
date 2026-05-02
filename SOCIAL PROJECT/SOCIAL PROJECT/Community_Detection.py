"""
Community Detection Module
===========================
Implements three graph community detection algorithms with full evaluation
metrics and side-by-side comparison visualisation.

Algorithms
----------
1. Louvain Communities    — modularity optimization method
2. Vertex Subgraph        — iterative subgraph extraction via node similarity
3. Girvan–Newman          — edge-betweenness hierarchical splitting

Evaluation Metrics
------------------
Internal (no ground truth needed):
  • Conductance           — average inter-community edge ratio (lower = better)
  • Intra-cluster density — average density inside clusters

External (compared to a reference partition / node attribute):
  • Normalised Mutual Information (NMI)

Comparison
  • Side-by-side metric table*/
  • Community graph visualisation per algorithm
  • Aggregate comparison dashboard
"""

import math
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ─── colour palette ───────────────────────────────────────────────────────────
_COMM_COLORS = [
    "#4a90d9", "#e87040", "#5cba6e", "#9b59b6", "#f0a500",
    "#2ec4b6", "#e6194b", "#3cb44b", "#f032e6", "#42d4f4",
    "#9A6324", "#aaffc3", "#ffd8b1", "#800000", "#808000",
]
_BG    = "#0b0b14"
_PANEL = "#11111f"
_TEXT  = "#dde0f0"
_GRID  = "#252540"
_CYAN  = "#00d4ff"
_GOLD  = "#f0c040"
_GREEN = "#50e090"
_RED   = "#ff5566"

CLASS_COLOR_MAP = {
    "1A": "#e6194b", "1B": "#f58231", "2A": "#3cb44b", "2B": "#bfef45",
    "3A": "#4363d8", "3B": "#42d4f4", "4A": "#911eb4", "4B": "#f032e6",
    "5A": "#9A6324", "5B": "#C8860A", "Teachers": "#2ec4b6",
}

# Algorithm display names & accent colours for the comparison dashboard
ALGO_META = {
    "Louvain":       {"color": "#00d4ff", "icon": "⬡"},
    "Vertex-Subgraph": {"color": "#f0c040", "icon": "◈"},
    "Girvan-Newman":  {"color": "#50e090", "icon": "⬢"},
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, alpha=0.6)
    if title:   ax.set_title(title,   color=_TEXT, fontsize=11, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, color=_TEXT, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, color=_TEXT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
#  1. DETECTION ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def detect_louvain(G: nx.Graph, resolution: float = 1.0) -> Dict[Any, int]:
    """
    Louvain Community Detection.

    A modularity-based algorithm that greedily optimizes modularity
    by iteratively merging communities.

    Parameters
    ----------
    G : nx.Graph
        Input graph (directed graphs are converted to undirected).
    resolution : float
        Resolution parameter (default 1.0). Higher values → more communities.
        Values < 1.0 → fewer, larger communities; > 1.0 → more, smaller ones.

    Returns
    -------
    Dict[Any, int]
        {node: community_id}
    """
    UG = G.to_undirected() if G.is_directed() else G
    try:
        communities = list(nx.community.louvain_communities(UG, resolution=resolution))
    except Exception as exc:
        warnings.warn(
            f"Louvain community detection failed ({exc}); "
            f"falling back to connected components.",
            UserWarning,
        )
        return detect_connected_components(G)

    partition: Dict[Any, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    return partition


def detect_vertex_subgraph(
    G: nx.Graph,
    n_communities: int = 5,
    similarity: str = "jaccard",
    seed: int = 42,
) -> Dict[Any, int]:
    """
    Vertex Subgraph Community Detection.

    Builds communities by iteratively extracting dense subgraphs based
    on structural node similarity (Jaccard or Adamic-Adar on neighbour sets).

    Algorithm:
      1. Compute pairwise similarity for all edges.
      2. Seed selection: pick the highest-degree node as the first seed.
      3. Grow a subgraph by greedily adding the most similar neighbour
        until density drops below a threshold or subgraph is too large.
      4. Assign community, remove those nodes, repeat for next seed.
      5. Remaining nodes form singleton / residual communities.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    n_communities : int
        Approximate target number of communities (actual count may differ).
    similarity : str
        'jaccard' or 'adamic_adar'. Jaccard is faster; Adamic-Adar rewards
        shared neighbours with low degree (hub-penalising).
    seed : int
        Random seed for tie-breaking during seed selection.

    Returns
    -------
    Dict[Any, int]
        {node: community_id}
    """
    UG = G.to_undirected() if G.is_directed() else G
    rng = np.random.default_rng(seed)

    # ── Similarity computation ───────────────────────────────────────────────
    def _jaccard(u, v):
        nu = set(UG.neighbors(u))
        nv = set(UG.neighbors(v))
        union = nu | nv
        return len(nu & nv) / len(union) if union else 0.0

    def _adamic_adar(u, v):
        common = set(UG.neighbors(u)) & set(UG.neighbors(v))
        score = 0.0
        for w in common:
            deg = UG.degree(w)
            if deg > 1:
                score += 1.0 / math.log(deg)
        return score

    sim_fn = _adamic_adar if similarity == "adamic_adar" else _jaccard

    # Pre-compute edge similarities
    edge_sim: Dict[Tuple, float] = {}
    for u, v in UG.edges():
        s = sim_fn(u, v)
        edge_sim[(u, v)] = s
        edge_sim[(v, u)] = s

    # ── Subgraph extraction ──────────────────────────────────────────────────
    remaining = set(UG.nodes())
    partition: Dict[Any, int] = {}
    cid = 0
    target_size = max(2, len(UG.nodes()) // max(n_communities, 1))

    while remaining and cid < n_communities:
        # Seed: highest-degree node among remaining
        seed_node = max(
            remaining,
            key=lambda n: (UG.degree(n), float(rng.random())),
        )
        subgraph_nodes = {seed_node}
        frontier = set(UG.neighbors(seed_node)) & remaining

        while frontier and len(subgraph_nodes) < target_size * 1.5:
            # Pick the frontier node most similar (avg) to current subgraph
            best_node, best_score = None, -1.0
            for candidate in frontier:
                # Average similarity to all nodes already in subgraph
                scores = [
                    edge_sim.get((candidate, n), 0.0)
                    for n in subgraph_nodes
                    if UG.has_edge(candidate, n)
                ]
                avg = float(np.mean(scores)) if scores else 0.0
                if avg > best_score:
                    best_score = avg
                    best_node  = candidate

            if best_node is None or best_score < 1e-9:
                break

            # Density check BEFORE committing: peek whether adding best_node
            # would drop density below the threshold. If so, skip this candidate
            # and try the next-best one rather than killing growth entirely.
            # BUG FIX: the original code added best_node, then broke out of the
            # inner loop on a density violation — discarding the node but also
            # stopping all further growth, leaving most nodes as residuals.
            trial_nodes = subgraph_nodes | {best_node}
            if len(trial_nodes) > 3:
                trial_density = nx.density(UG.subgraph(trial_nodes))
                if trial_density < 0.10:
                    # Remove this candidate from frontier so we don't retry it,
                    # but keep growing with other candidates
                    frontier.discard(best_node)
                    continue

            subgraph_nodes.add(best_node)
            frontier.discard(best_node)
            frontier |= (set(UG.neighbors(best_node)) & remaining) - subgraph_nodes

        for node in subgraph_nodes:
            partition[node] = cid
            remaining.discard(node)
        cid += 1

    # BUG FIX: the original code gave every remaining node its own unique cid
    # (cid += 1 per node), identical to the k-clique singleton inflation bug.
    # Fix: assign all residual nodes the SAME shared cid, then use neighbour-
    # majority vote to absorb them into the closest real community if possible.
    if remaining:
        residual_cid = cid  # one shared id for nodes with no community yet
        for node in remaining:
            partition[node] = residual_cid

        # Neighbour-majority vote: iteratively re-assign residuals whose
        # neighbours have already been placed in a real community.
        changed = True
        while changed:
            changed = False
            still_residual = [n for n in remaining if partition[n] == residual_cid]
            for node in still_residual:
                neighbour_cids = [
                    partition[nb]
                    for nb in UG.neighbors(node)
                    if partition.get(nb, residual_cid) != residual_cid
                ]
                if neighbour_cids:
                    from collections import Counter as _Counter
                    partition[node] = _Counter(neighbour_cids).most_common(1)[0][0]
                    changed = True

    return partition


def detect_girvan_newman(
    G: nx.Graph,
    n_communities: Optional[int] = None,
    max_communities: int = 20,
) -> Dict[Any, int]:
    """
    Girvan–Newman Community Detection.

    Iteratively removes the edge with the highest betweenness centrality.

    Stopping criterion (in priority order):
      1. If ``n_communities`` is given → stop when component count reaches it.
      2. Otherwise → track modularity at every split and stop at the peak
         (the partition with the highest modularity score). This avoids the
         "always returns exactly N" bug caused by a fixed hard target.
      3. Hard cap at ``max_communities`` components to prevent runaway splits
         on graphs with many small connected components.

    Parameters
    ----------
    G : nx.Graph
        Input graph (directed graphs are converted to undirected).
    n_communities : int or None
        If set, split until this many components exist (hard target).
        If None (default), stop at the modularity-optimal split.
    max_communities : int
        Upper bound on components when using modularity-optimal mode.

    Returns
    -------
    Dict[Any, int]
        {node: community_id}, sorted largest community first.

    Notes
    -----
    - O(m² n) per step — expensive on large graphs.
    - Edge weights are used in betweenness computation when present.
    - The modularity-optimal mode explores up to ``max_communities`` splits
      and returns whichever produced the highest modularity, so the result
      reflects the graph's actual structure rather than a user-supplied count.
    """
    UG = G.to_undirected() if G.is_directed() else G
    H  = UG.copy()

    def _make_partition(graph: nx.Graph) -> Dict[Any, int]:
        part: Dict[Any, int] = {}
        for cid, comp in enumerate(
            sorted(nx.connected_components(graph), key=len, reverse=True)
        ):
            for node in comp:
                part[node] = cid
        return part

    # ── Mode 1: hard target ──────────────────────────────────────────────────
    if n_communities is not None:
        n_communities = max(1, int(n_communities))
        comp = nx.number_connected_components(H)
        while comp < n_communities and H.number_of_edges() > 0:
            centrality = nx.edge_betweenness_centrality(H, weight="weight")
            if not centrality:
                break
            H.remove_edge(*max(centrality, key=centrality.get))
            comp = nx.number_connected_components(H)
        return _make_partition(H)

    # ── Mode 2: modularity-optimal (default) ────────────────────────────────
    # Explore splits one at a time; record modularity at each step.
    # Return the partition that achieved the maximum modularity.
    best_partition = _make_partition(H)

    # BUG FIX: nx.community.modularity raises ValueError when the partition
    # contains only one community (undefined Q for a single group).  Guard it.
    initial_sets = partition_to_sets(best_partition)
    if len(initial_sets) >= 2:
        try:
            best_modularity = nx.community.modularity(
                UG, initial_sets, weight="weight"
            )
        except Exception:
            best_modularity = 0.0
    else:
        # Single component: Q is conventionally 0 (no community structure yet)
        best_modularity = 0.0

    while (
        H.number_of_edges() > 0
        and nx.number_connected_components(H) < max_communities
    ):
        centrality = nx.edge_betweenness_centrality(H, weight="weight")
        if not centrality:
            break
        H.remove_edge(*max(centrality, key=centrality.get))

        current_partition  = _make_partition(H)
        current_components = partition_to_sets(current_partition)

        # Need ≥ 2 communities for modularity to be meaningful
        if len(current_components) < 2:
            continue

        try:
            current_modularity = nx.community.modularity(
                UG, current_components, weight="weight"
            )
        except Exception:
            continue

        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_partition  = current_partition

    return best_partition


def detect_connected_components(G: nx.Graph) -> Dict[Any, int]:
    """Baseline: connected-component decomposition (used as fallback)."""
    UG = G.to_undirected() if G.is_directed() else G
    partition: Dict[Any, int] = {}
    for cid, comp in enumerate(
        sorted(nx.connected_components(UG), key=len, reverse=True)
    ):
        for node in comp:
            partition[node] = cid
    return partition


# ── Algorithm registry ────────────────────────────────────────────────────────
ALGORITHMS = {
    "Louvain":        detect_louvain,
    "Vertex-Subgraph": detect_vertex_subgraph,
    "Girvan-Newman":   detect_girvan_newman,
}


def run_community_detection(
    G: nx.Graph,
    algorithm: str = "Girvan-Newman",
    **kwargs,
) -> Dict[Any, int]:
    """
    Run a named community detection algorithm and return {node: community_id}.

    Parameters
    ----------
    G         : nx.Graph — the input graph
    algorithm : str      — one of 'Louvain', 'Vertex-Subgraph', 'Girvan-Newman'
    **kwargs             — forwarded to the algorithm (e.g. k=4, n_communities=6)

    Returns
    -------
    Dict[Any, int]  {node: community_id}
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(ALGORITHMS)}"
        )
    fn = ALGORITHMS[algorithm]
    try:
        return fn(G, **kwargs)
    except TypeError:
        return fn(G)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def partition_to_sets(partition: Dict[Any, int]) -> List[set]:
    """Convert {node: cid} → list of sets, sorted largest-first."""
    groups: Dict[int, set] = defaultdict(set)
    for node, cid in partition.items():
        groups[cid].add(node)
    return sorted(groups.values(), key=len, reverse=True)


def partition_stats(partition: Dict[Any, int]) -> Dict[str, Any]:
    """Return basic size statistics of the partition."""
    sets  = partition_to_sets(partition)
    sizes = [len(s) for s in sets]
    return {
        "n_communities": len(sets),
        "sizes":         sizes,
        "min_size":      min(sizes) if sizes else 0,
        "max_size":      max(sizes) if sizes else 0,
        "mean_size":     round(float(np.mean(sizes)), 2) if sizes else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. INTERNAL EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def eval_conductance(G: nx.Graph, partition: Dict[Any, int]) -> float:
    """
    Average conductance across communities.
    conductance(S) = cut(S) / min(vol(S), vol(V\\S))
    Lower is better — dense, isolated communities have low conductance.
    """
    UG    = G.to_undirected() if G.is_directed() else G
    sets  = partition_to_sets(partition)
    if len(sets) < 2:
        return 0.0
    scores = []
    for comm in sets:
        cut     = nx.cut_size(UG, comm, weight="weight")
        vol     = sum(d for _, d in UG.degree(comm, weight="weight"))
        vol_bar = sum(d for _, d in UG.degree(
            [n for n in UG.nodes() if n not in comm], weight="weight"
        ))
        denom = min(vol, vol_bar)
        if denom > 0:
            scores.append(cut / denom)
    return round(float(np.mean(scores)), 4) if scores else float("nan")


def eval_intra_density(G: nx.Graph, partition: Dict[Any, int]) -> float:
    """
    Average internal edge density across communities.
    density(S) = edges_in_S / (|S|*(|S|-1)/2)
    Higher is better — captures compactness of each community.
    """
    UG        = G.to_undirected() if G.is_directed() else G
    sets      = partition_to_sets(partition)
    densities = []
    for comm in sets:
        if len(comm) < 2:
            continue
        sub = UG.subgraph(comm)
        densities.append(nx.density(sub))
    return round(float(np.mean(densities)), 4) if densities else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  3. EXTERNAL EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def _labels_from_attr(G: nx.Graph, attr: str = "Class") -> List[str]:
    return [str(G.nodes[n].get(attr, "N/A")) for n in G.nodes()]


def _labels_from_partition(G: nx.Graph, partition: Dict[Any, int]) -> List[int]:
    return [partition.get(n, -1) for n in G.nodes()]


def eval_nmi(
    G: nx.Graph, partition: Dict[Any, int], ground_truth_attr: str = "Class"
) -> float:
    """Normalised Mutual Information between partition and a node attribute."""
    if not SKLEARN_OK:
        return _nmi_manual(G, partition, ground_truth_attr)
    true = _labels_from_attr(G, ground_truth_attr)
    pred = _labels_from_partition(G, partition)
    return round(float(normalized_mutual_info_score(true, pred)), 4)


def _nmi_manual(G, partition, attr):
    true = _labels_from_attr(G, attr)
    pred = _labels_from_partition(G, partition)
    n    = len(true)
    if n == 0:
        return 0.0
    from collections import Counter
    tc = Counter(true); pc = Counter(pred)
    jc: Dict = defaultdict(int)
    for t, p in zip(true, pred):
        jc[(t, p)] += 1

    def H(counts):
        tot = sum(counts.values())
        return -sum((c / tot) * math.log(c / tot + 1e-12)
                    for c in counts.values() if c > 0)

    ht = H(tc); hp = H(pc)
    mi = sum(
        cnt / n * math.log((cnt / n) / ((tc[t] / n) * (pc[p] / n) + 1e-12))
        for (t, p), cnt in jc.items()
    )
    denom = (ht + hp) / 2
    return round(mi / denom, 4) if denom > 1e-12 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  4. FULL EVALUATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_partition(
    G: nx.Graph,
    partition: Dict[Any, int],
    ground_truth_attr: str = "Class",
) -> Dict[str, Any]:
    """
    Run selected internal + external metrics and return a flat dict.

    Parameters
    ----------
    G                 : input graph
    partition         : {node: community_id}
    ground_truth_attr : node attribute name used as ground-truth label

    Returns
    -------
    Dict with keys: n_communities, min_size, max_size, mean_size,conductance, intra_density, nmi.
    """
    stats   = partition_stats(partition)
    return {
        # community size stats
        "n_communities": stats["n_communities"],
        "min_size":      stats["min_size"],
        "max_size":      stats["max_size"],
        "mean_size":     stats["mean_size"],
        # internal metrics
        "conductance":   eval_conductance(G, partition),
        "intra_density": eval_intra_density(G, partition),
        # external metrics
        "nmi":eval_nmi(G, partition, ground_truth_attr),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  5. RUN ALL THREE ALGORITHMS + COLLECT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_algorithms(
    G: nx.Graph,
    resolution: float = 1.0,
    n_communities: int = 5,
    similarity: str = "jaccard",
    seed: int = 42,
    ground_truth_attr: str = "Class",
) -> List[Dict]:
    """
    Run Louvain, Vertex-Subgraph, and Girvan-Newman on G.

    Returns
    -------
    List of dicts, one per algorithm:
        {
        "algorithm": str,
        "partition": Dict[Any, int],
        "metrics":   Dict[str, Any],
        }
    """
    configs = [
        ("Louvain",        {"resolution": resolution}),
        ("Vertex-Subgraph", {"n_communities": n_communities,"similarity": similarity, "seed": seed}),
        ("Girvan-Newman",   {"n_communities": n_communities}),
    ]
    results = []
    for name, kwargs in configs:
        print(f"  Running {name}...", end=" ", flush=True)
        try:
            partition = run_community_detection(G, name, **kwargs)
            metrics   = evaluate_partition(G, partition, ground_truth_attr)
            results.append({
                "algorithm": name,
                "partition": partition,
                "metrics":   metrics,
            })
            print(f"✓  ({metrics['n_communities']} communities, "
                f"conductance={metrics['conductance']:.3f})")
        except Exception as exc:
            warnings.warn(f"{name} failed: {exc}")
            print(f"✗  ({exc})")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  6. VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_community_graph(
    G: nx.Graph,
    partition: Dict[Any, int],
    pos: Optional[Dict] = None,
    output_path: str = "_community.png",
    title: str = "Community Detection",
    algo_color: str = _CYAN,
    seed: int = 42,
    figsize: Tuple = (14, 11),
    dpi: int = 150,
) -> str:
    """Network coloured by community membership, node size ∝ degree."""
    if pos is None:
        k_layout = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
        pos = nx.spring_layout(G, k=k_layout, seed=seed, weight="weight")

    n_comm      = max(partition.values(), default=0) + 1
    node_colors = [
        _COMM_COLORS[partition.get(n, 0) % len(_COMM_COLORS)]
        for n in G.nodes()
    ]
    degs = [G.degree(n) for n in G.nodes()]
    mn, mx = min(degs), max(degs)
    sizes  = [20 + 200 * (d - mn) / max(mx - mn, 1) for d in degs]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.axis("off")

    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    wmin, wmax = (min(weights), max(weights)) if weights else (1, 1)
    ew = [0.15 + 1.2 * (w - wmin) / max(wmax - wmin, 1) for w in weights]
    nx.draw_networkx_edges(
        G, pos, width=ew, alpha=0.2, edge_color="#9999bb", ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=sizes,
        alpha=0.92, linewidths=0.5, edgecolors=algo_color, ax=ax
    )

    patches = [
        mpatches.Patch(
            color=_COMM_COLORS[c % len(_COMM_COLORS)],
            label=f"Community {c + 1}",
        )
        for c in range(n_comm)
    ]
    leg = ax.legend(
        handles=patches, loc="lower left", framealpha=0.75,
        facecolor="#1a1a2e", edgecolor="#555577",
        labelcolor="white", title_fontsize=9, fontsize=8,
    )
    leg.get_title().set_color("white")
    ax.set_title(title, color=algo_color, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path

def render_community_size_dist(
    partition: Dict[Any, int],
    output_path: str = "_comm_sizes.png",
    title: str = "Community Size Distribution",
    bar_color: str = _CYAN,
    figsize: Tuple = (10, 5),
    dpi: int = 150,
) -> str:
    """Bar chart of community sizes sorted descending."""
    sets   = partition_to_sets(partition)
    sizes  = sorted([len(s) for s in sets], reverse=True)
    labels = [f"C{i + 1}" for i in range(len(sizes))]
    colors = [_COMM_COLORS[i % len(_COMM_COLORS)] for i in range(len(sizes))]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.bar(labels, sizes, color=colors, edgecolor="#222244", linewidth=0.5)
    _style_ax(ax, title, "Community", "Size (# nodes)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  7. COMPARISON TABLE IMAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison_table_img(
    results: List[Dict],
    output_path: str = "_comm_compare.png",
    dpi: int = 150,
) -> str:
    """
    Render a styled comparison table image for all three algorithms.

    Columns displayed:
    Algorithm | # Communities | Conductance | Intra-Density | NMI
    """
    if not results:
        return output_path

    col_keys = [
        "n_communities", "conductance", "intra_density", "nmi",
    ]
    col_lbls = [
        "# Comms", "Conductance", "Intra-Density", "NMI",
    ]

    # ── Identify best value per numeric column (for highlighting) ────────────
    # Higher is better for all except conductance (lower is better)
    lower_better = {"conductance"}

    def _fmt(val):
        if isinstance(val, int):
            return str(val)
        try:
            f = float(val)
            return "—" if math.isnan(f) else f"{f:.3f}"
        except Exception:
            return str(val)

    # Collect column values to find best
    col_values: Dict[str, List] = {k: [] for k in col_keys}
    for r in results:
        for k in col_keys:
            col_values[k].append(r["metrics"].get(k, float("nan")))

    def _best_idx(vals, lower_better=False):
        valid = [(i, v) for i, v in enumerate(vals)
                if not (isinstance(v, float) and math.isnan(v))]
        if not valid:
            return -1
        fn = min if lower_better else max
        return fn(valid, key=lambda x: x[1])[0]

    best_idx = {
        k: _best_idx(col_values[k], lower_better=(k in lower_better))
        for k in col_keys
    }

    # Build cell data
    rows = []
    for r in results:
        row = [r["algorithm"]] + [_fmt(r["metrics"].get(k)) for k in col_keys]
        rows.append(row)

    fig_h = max(2.5, 0.45 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(16, fig_h), facecolor=_BG)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Algorithm"] + col_lbls,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)

    header_colors = {
        "Louvain":        ALGO_META["Louvain"]["color"],
        "Vertex-Subgraph": ALGO_META["Vertex-Subgraph"]["color"],
        "Girvan-Newman":   ALGO_META["Girvan-Newman"]["color"],
    }

    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor(_GRID)
        if ri == 0:
            # Header row
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color=_CYAN, weight="bold")
        else:
            algo_name = rows[ri - 1][0]
            # Alternate row shading
            base_bg = "#141428" if ri % 2 == 0 else "#0f0f1a"
            cell.set_facecolor(base_bg)

            if ci == 0:
                # Algorithm name cell — colour-coded
                accent = header_colors.get(algo_name, _TEXT)
                cell.set_text_props(color=accent, weight="bold")
            else:
                # Check if this is the best value in this column
                col_key_idx = ci - 1  # 0-indexed into col_keys
                if col_key_idx < len(col_keys):
                    col_key = col_keys[col_key_idx]
                    if best_idx[col_key] == ri - 1:
                        # Highlight winner
                        cell.set_facecolor("#1a2a1a")
                        cell.set_text_props(color=_GREEN, weight="bold")
                    else:
                        cell.set_text_props(color=_TEXT)
                else:
                    cell.set_text_props(color=_TEXT)

    ax.set_title(
        "Community Detection — Algorithm Comparison",
        color=_TEXT, fontsize=12, fontweight="bold", pad=14,
    )

    # Legend: green = best value in column
    fig.text(
        0.01, 0.02,
        "★ Green cells indicate the best value in each column",
        color=_GREEN, fontsize=8, alpha=0.8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  8. FULL COMPARISON DASHBOARD  (all algorithms, side-by-side)
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison_dashboard(
    G: nx.Graph,
    results: List[Dict],
    pos: Optional[Dict] = None,
    output_path: str = "_comparison_dashboard.png",
    seed: int = 42,
    dpi: int = 150,
) -> str:
    """
    Generate a full comparison dashboard with:
      Row 0 : algorithm name headers
      Row 1 : community graph per algorithm (side-by-side)
      Row 2 : community size bar chart per algorithm (side-by-side)
      Row 3 : combined metric bar chart (grouped by metric)

    Parameters
    ----------
    G        : the graph that was analysed
    results  : output of run_all_algorithms()
    pos      : fixed node layout (computed once for fair comparison)
    output_path : save location for the PNG
    seed     : for layout if pos is None

    Returns
    -------
    str  — output_path
    """
    if not results:
        warnings.warn("No results to render.")
        return output_path

    n_algos = len(results)

    # Compute a shared layout once for visual consistency
    if pos is None:
        k_layout = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
        pos = nx.spring_layout(G, k=k_layout, seed=seed, weight="weight")

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig_w = max(16, 6 * n_algos)
    fig_h = 52  # tall dashboard
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)

    # GridSpec: 5 rows — graphs | size charts | metric bars | table
    gs = GridSpec(
        5, n_algos,
        figure=fig,
        height_ratios=[5, 4, 3, 4, 2.5],
        hspace=0.45,
        wspace=0.30,
    )

    # ── Row 0 title banner ───────────────────────────────────────────────────
    # drawn as a super-title; individual col titles are axis titles below

    # ── Row 1: Community graphs ──────────────────────────────────────────────
    for col, res in enumerate(results):
        ax  = fig.add_subplot(gs[0, col])
        name = res["algorithm"]
        meta = ALGO_META.get(name, {"color": _CYAN, "icon": "●"})
        part = res["partition"]
        n_c  = res["metrics"]["n_communities"]

        ax.set_facecolor(_BG)
        ax.axis("off")

        node_colors = [
            _COMM_COLORS[part.get(n, 0) % len(_COMM_COLORS)]
            for n in G.nodes()
        ]
        degs = [G.degree(n) for n in G.nodes()]
        mn, mx = min(degs), max(degs)
        sizes  = [15 + 160 * (d - mn) / max(mx - mn, 1) for d in degs]

        weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
        wmin, wmax = (min(weights), max(weights)) if weights else (1, 1)
        ew = [0.1 + 0.8 * (w - wmin) / max(wmax - wmin, 1) for w in weights]

        nx.draw_networkx_edges(
            G, pos, width=ew, alpha=0.18, edge_color="#8888aa", ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=sizes,
            alpha=0.90, linewidths=0.5, edgecolors=meta["color"], ax=ax
        )

        patches = [
            mpatches.Patch(
                color=_COMM_COLORS[c % len(_COMM_COLORS)],
                label=f"C{c + 1}",
            )
            for c in range(min(n_c, 8))
        ]
        leg = ax.legend(
            handles=patches, loc="lower left", framealpha=0.6,
            facecolor="#0f0f1a", edgecolor="#333355",
            labelcolor="white", fontsize=7, ncol=2,
        )
        ax.set_title(
            f"{meta['icon']} {name}\n{n_c} communities  |  "
            f"Q={res['metrics']['modularity']:.3f}",
            color=meta["color"], fontsize=11, fontweight="bold", pad=8,
        )

    # ── Row 3: Community size distributions ──────────────────────────────────
    for col, res in enumerate(results):
        ax   = fig.add_subplot(gs[2, col])
        name = res["algorithm"]
        meta = ALGO_META.get(name, {"color": _CYAN})
        sets = partition_to_sets(res["partition"])
        sizes_sorted = sorted([len(s) for s in sets], reverse=True)
        lbls  = [f"C{i + 1}" for i in range(len(sizes_sorted))]
        clrs  = [_COMM_COLORS[i % len(_COMM_COLORS)] for i in range(len(sizes_sorted))]

        ax.bar(lbls, sizes_sorted, color=clrs, edgecolor=_BG, linewidth=0.4)
        _style_ax(ax, "Size Distribution", "Community", "# Nodes")
        ax.tick_params(axis="x", labelsize=7, rotation=45)

    # ── Row 4: Grouped metric bar chart across algorithms ────────────────────
    metric_cols = [
        ("conductance",   "Conductance",    True),   # lower is better
        ("intra_density", "Intra-Density",  False),
        ("nmi",           "NMI",            False),
    ]
    ax_bar = fig.add_subplot(gs[3, :])
    ax_bar.set_facecolor(_PANEL)

    x         = np.arange(len(metric_cols))
    bar_width = 0.22
    offsets   = np.linspace(-(n_algos - 1) / 2, (n_algos - 1) / 2, n_algos) * bar_width

    for i, res in enumerate(results):
        name  = res["algorithm"]
        meta  = ALGO_META.get(name, {"color": _CYAN})
        m     = res["metrics"]
        vals  = []
        for mkey, _, _ in metric_cols:
            v = m.get(mkey, 0)
            if v != v: v = 0
            vals.append(float(v))
        bars = ax_bar.bar(
            x + offsets[i], vals, bar_width,
            label=name, color=meta["color"], alpha=0.82,
            edgecolor=_BG, linewidth=0.4,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([lbl for _, lbl, _ in metric_cols],
                            color=_TEXT, fontsize=10)
    ax_bar.set_ylabel("Score", color=_TEXT, fontsize=10)
    ax_bar.tick_params(colors=_TEXT)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(_GRID)
    ax_bar.grid(axis="y", color=_GRID, linewidth=0.4, alpha=0.6)
    ax_bar.legend(
        facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT,
        fontsize=10, loc="upper right",
    )
    ax_bar.set_title(
        "Metric Comparison — All Algorithms",
        color=_TEXT, fontsize=12, fontweight="bold", pad=10,
    )

    # Annotate "lower is better" for conductance
    for col_i, (_, lbl, lb) in enumerate(metric_cols):
        if lb:
            ax_bar.annotate(
                "↓ lower", xy=(col_i, 0), xytext=(col_i, -0.08),
                color="#ff8844", fontsize=7, ha="center",
            )

    # ── Row 5: Summary text table ─────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[4, :])
    ax_tbl.axis("off")

    col_keys2 = ["n_communities", "modularity", "coverage", "performance",
"conductance", "intra_density", "nmi", "ari", "v_measure"]
    col_hdrs  = ["# Comms", "Modularity", "Coverage", "Performance",
                "Conductance", "Intra-Density", "NMI", "ARI", "V-Measure"]

    def _fmt2(val):
        if isinstance(val, int): return str(val)
        try:
            f = float(val)
            return "—" if math.isnan(f) else f"{f:.3f}"
        except Exception:
            return str(val)

    tbl_data = [
        [res["algorithm"]] + [_fmt2(res["metrics"].get(k)) for k in col_keys2]
        for res in results
    ]

    tbl = ax_tbl.table(
        cellText=tbl_data,
        colLabels=["Algorithm"] + col_hdrs,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    algo_color_map = {
        res["algorithm"]: ALGO_META.get(
            res["algorithm"], {"color": _CYAN}
        )["color"]
        for res in results
    }

    # Find best per column
    lower_better_set = {"conductance"}
    best_per_col: Dict[str, int] = {}
    for k in col_keys2:
        vals_col = []
        for res in results:
            v = res["metrics"].get(k, float("nan"))
            try:
                vals_col.append(float(v))
            except Exception:
                vals_col.append(float("nan"))
        valid = [(i, v) for i, v in enumerate(vals_col) if not math.isnan(v)]
        if valid:
            fn = min if k in lower_better_set else max
            best_per_col[k] = fn(valid, key=lambda x: x[1])[0]

    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor(_GRID)
        if ri == 0:
            cell.set_facecolor("#09091a")
            cell.set_text_props(color=_CYAN, weight="bold", fontsize=9)
        else:
            algo_name = results[ri - 1]["algorithm"]
            cell.set_facecolor("#0f0f1e" if ri % 2 else "#131328")
            if ci == 0:
                accent = algo_color_map.get(algo_name, _TEXT)
                cell.set_text_props(color=accent, weight="bold")
            else:
                col_key = col_keys2[ci - 1]
                if best_per_col.get(col_key) == ri - 1:
                    cell.set_facecolor("#0e1e0e")
                    cell.set_text_props(color=_GREEN, weight="bold")
                else:
                    cell.set_text_props(color=_TEXT)

    ax_tbl.set_title(
        "Full Metrics Summary",
        color=_TEXT, fontsize=11, fontweight="bold", pad=10,
    )

    # ── Super title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "Community Detection — Side-by-Side Comparison",
        color=_TEXT, fontsize=16, fontweight="bold", y=0.995,
    )

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  9. CONVENIENCE: run everything and save all artefacts
# ─────────────────────────────────────────────────────────────────────────────

def compare_community_detection(
    G: nx.Graph,
    resolution: float = 1.0,
    n_communities: int = 5,
    similarity: str = "jaccard",
    seed: int = 42,
    ground_truth_attr: str = "Class",
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    Full pipeline: run all three algorithms, evaluate, and save all plots.

    Parameters
    ----------
    G                 : input graph
    k                 : resolution parameter for Louvain algorithm
    n_communities     : target communities for Vertex-Subgraph & Girvan-Newman
    similarity        : 'jaccard' or 'adamic_adar' (Vertex-Subgraph)
    seed              : random seed
    ground_truth_attr : node attribute for external metrics
    output_dir        : folder to save PNG outputs

    Returns
    -------
    Dict with keys:
      'results'    — list of {algorithm, partition, metrics}
      'dashboard'  — path to comparison dashboard PNG
      'table'      — path to comparison table PNG
      'per_algo'   — {algo: {graph,sizes}} per-algorithm PNGs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Running community detection algorithms...")
    results = run_all_algorithms(
        G,
        resolution=resolution,
        n_communities=n_communities,
        similarity=similarity,
        seed=seed,
        ground_truth_attr=ground_truth_attr,
    )

    if not results:
        warnings.warn("No results produced. Check your graph.")
        return {"results": [], "dashboard": None, "table": None, "per_algo": {}}

    # Shared layout for all graphs
    k_layout = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, k=k_layout, seed=seed, weight="weight")

    per_algo: Dict[str, Dict[str, str]] = {}

    for res in results:
        name = res["algorithm"]
        meta = ALGO_META.get(name, {"color": _CYAN})
        safe = name.replace("-", "_").replace(" ", "_").lower()

        g_path = os.path.join(output_dir, f"_{safe}_graph.png")
        s_path = os.path.join(output_dir, f"_{safe}_sizes.png")

        render_community_graph(
            G, res["partition"], pos=pos,
            output_path=g_path,
            title=f"{name} — Community Graph",
            algo_color=meta["color"],
            seed=seed,
        )
        render_community_size_dist(
            res["partition"],
            output_path=s_path,
            title=f"{name} — Sizes",
            bar_color=meta["color"],
        )
        per_algo[name] = {
            "graph": g_path,
            "sizes": s_path,
        }

    # Comparison artefacts
    dash_path  = os.path.join(output_dir, "_comparison_dashboard.png")
    table_path = os.path.join(output_dir, "_comparison_table.png")

    print("Rendering comparison dashboard...")
    render_comparison_dashboard(
        G, results, pos=pos, output_path=dash_path, seed=seed
    )
    print("Rendering comparison table...")
    render_comparison_table_img(results, output_path=table_path)

    print(f"\n✓ Dashboard saved  → {dash_path}")
    print(f"✓ Table saved      → {table_path}")
    for name, paths in per_algo.items():
        print(f"  {name}: {list(paths.values())}")

    # Print a quick summary to stdout
    print("\n" + "=" * 68)
    print(f"  {'ALGORITHM':<20} {'# COMM':>7} {'MODULARITY':>11} {'COVERAGE':>9} {'NMI':>7}")
    print("  " + "-" * 64)
    for res in results:
        m = res["metrics"]
        print(
            f"  {res['algorithm']:<20} "
            f"{m['n_communities']:>7} "
            f"{m['modularity']:>11.4f} "
            f"{m['coverage']:>9.4f} "
            f"{m['nmi']:>7.4f}"
        )
    print("=" * 68 + "\n")

    return {
        "results":   results,
        "dashboard": dash_path,
        "table":     table_path,
        "per_algo":  per_algo,
    }
