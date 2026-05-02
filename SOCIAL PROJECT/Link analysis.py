"""
Link Analysis Module
=====================
PageRank, Betweenness Centrality, and other link-analysis techniques.
Also includes HITS (hubs & authorities), Katz, and Eigenvector centrality.

All compute_* functions return plain dicts.
All render_* functions save PNGs and return the path.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

_BG    = "#0f0f1a"
_PANEL = "#141428"
_TEXT  = "#ddddee"
_GRID  = "#2a2a44"
_CYAN  = "#00d4ff"
_CORAL = "#e87040"
_GREEN = "#5cba6e"
_AMBER = "#f0a500"
_PURPLE= "#9b59b6"

CLASS_COLOR_MAP = {
    "1A":"#e6194b","1B":"#f58231","2A":"#3cb44b","2B":"#bfef45",
    "3A":"#4363d8","3B":"#42d4f4","4A":"#911eb4","4B":"#f032e6",
    "5A":"#9A6324","5B":"#C8860A","Teachers":"#2ec4b6",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, alpha=0.6)
    if title:  ax.set_title(title,  color=_TEXT, fontsize=11, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel,color=_TEXT, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel,color=_TEXT, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
#  1. COMPUTE LINK-ANALYSIS SCORES
# ─────────────────────────────────────────────────────────────────────────────

def compute_pagerank(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 500,
    weight: str = "weight",
) -> Dict[Any, float]:
    """
    Compute PageRank scores for every node.
    For undirected graphs, edges are treated as bidirectional.
    Returns {node: pagerank_score} (scores sum to 1).
    """
    return nx.pagerank(G, alpha=alpha, max_iter=max_iter, weight=weight)


def compute_betweenness(
    G: nx.Graph,
    normalized: bool = True,
    weight: str = "weight",
) -> Dict[Any, float]:
    """
    Brandes betweenness centrality — fraction of shortest paths through node.
    Returns {node: betweenness} normalised to [0,1].
    """
    return nx.betweenness_centrality(G, normalized=normalized, weight=weight)


def compute_eigenvector(
    G: nx.Graph,
    max_iter: int = 1000,
    weight: str = "weight",
) -> Dict[Any, float]:
    """
    Eigenvector centrality — importance weighted by neighbors' importance.
    Falls back to degree centrality for disconnected graphs.
    """
    try:
        UG = G.to_undirected() if G.is_directed() else G
        return nx.eigenvector_centrality(UG, max_iter=max_iter, weight=weight)
    except nx.PowerIterationFailedConvergence:
        return dict(nx.degree_centrality(G))


def compute_hits(G: nx.Graph) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    """
    HITS algorithm — returns (hubs, authorities).
    For undirected graphs both scores are identical.
    """
    try:
        hubs, auth = nx.hits(G, max_iter=500)
        return hubs, auth
    except Exception:
        deg = dict(nx.degree_centrality(G))
        return deg, dict(deg)


def compute_katz(
    G: nx.Graph,
    alpha: float = 0.005,
    beta: float = 1.0,
    weight: str = "weight",
) -> Dict[Any, float]:
    """
    Katz centrality — counts all paths with exponential decay by length.
    alpha must be < 1/largest_eigenvalue.
    """
    try:
        return nx.katz_centrality(G, alpha=alpha, beta=beta,
                                   normalized=True, weight=weight)
    except Exception:
        return dict(nx.degree_centrality(G))


def compute_closeness(
    G: nx.Graph,
    weight: str = None,
) -> Dict[Any, float]:
    """Closeness centrality — how close a node is to all others."""
    return nx.closeness_centrality(G, distance=weight)


def compute_degree_centrality(G: nx.Graph) -> Dict[Any, float]:
    """Normalised degree centrality (degree / (N-1))."""
    return nx.degree_centrality(G)


def compute_all_link_metrics(G: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """
    Convenience function: compute all link-analysis metrics at once.
    Returns a dict of {metric_name: {node: score}}.
    """
    hubs, auth = compute_hits(G)
    return {
        "pagerank":           compute_pagerank(G),
        "betweenness":        compute_betweenness(G),
        "eigenvector":        compute_eigenvector(G),
        "katz":               compute_katz(G),
        "closeness":          compute_closeness(G),
        "degree_centrality":  compute_degree_centrality(G),
        "hubs":               hubs,
        "authorities":        auth,
    }


def get_top_nodes(
    scores: Dict[Any, float],
    n: int = 10,
    ascending: bool = False,
) -> List[Tuple[Any, float]]:
    """Return the top-n nodes by score."""
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return sorted_items[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  2. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def render_link_analysis_dashboard(
    G: nx.Graph,
    metrics: Dict[str, Dict[Any, float]],
    pos: Optional[Dict] = None,
    output_path: str = "_link_analysis.png",
    color_attr: str = "Class",
    seed: int = 42,
    top_n: int = 10,
    figsize: Tuple = (20, 16),
    dpi: int = 150,
) -> str:
    """
    4-panel dashboard:
      Top-left:  PageRank distribution + top nodes
      Top-right: Betweenness distribution + top nodes
      Bot-left:  Network coloured by PageRank
      Bot-right: Centrality scatter (PageRank vs Betweenness)
    """
    if pos is None:
        k = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
        pos = nx.spring_layout(G, k=k, seed=seed, weight="weight")

    pr = metrics.get("pagerank", {})
    bt = metrics.get("betweenness", {})
    ev = metrics.get("eigenvector", {})

    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

    # ── Panel 1: PageRank bar chart top-N ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    top_pr = get_top_nodes(pr, top_n)
    ids    = [str(n) for n, _ in top_pr]
    scores = [s for _, s in top_pr]
    bars = ax1.barh(ids[::-1], scores[::-1], color=_CYAN, alpha=0.85,
                    edgecolor="#222244", linewidth=0.5)
    _style_ax(ax1, f"PageRank — Top {top_n} Nodes", "PageRank Score", "Node")

    # ── Panel 2: Betweenness bar chart top-N ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    top_bt = get_top_nodes(bt, top_n)
    ids_b  = [str(n) for n, _ in top_bt]
    scores_b = [s for _, s in top_bt]
    ax2.barh(ids_b[::-1], scores_b[::-1], color=_CORAL, alpha=0.85,
             edgecolor="#222244", linewidth=0.5)
    _style_ax(ax2, f"Betweenness Centrality — Top {top_n}", "Score", "Node")

    # ── Panel 3: Network coloured by PageRank ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(_BG); ax3.axis("off")
    pr_vals = np.array([pr.get(n, 0) for n in G.nodes()])
    if pr_vals.max() > pr_vals.min():
        norm_pr = (pr_vals - pr_vals.min()) / (pr_vals.max() - pr_vals.min())
    else:
        norm_pr = np.full_like(pr_vals, 0.5)
    cmap_fn = cm.get_cmap("plasma")
    node_colors_pr = [mcolors.to_hex(cmap_fn(v)) for v in norm_pr]
    degs = [G.degree(n) for n in G.nodes()]
    mn, mx = min(degs), max(degs)
    sizes  = [20 + 200 * (d - mn) / max(mx - mn, 1) for d in degs]
    nx.draw_networkx_edges(G, pos, alpha=0.12, edge_color="#aaaacc",
                           width=0.5, ax=ax3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_pr,
                           node_size=sizes, alpha=0.9,
                           linewidths=0.4, edgecolors="#ffffff", ax=ax3)
    sm = cm.ScalarMappable(cmap=cmap_fn,
                            norm=mcolors.Normalize(vmin=pr_vals.min(),
                                                   vmax=pr_vals.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, fraction=0.03, pad=0.02,
                 label="PageRank").ax.yaxis.set_tick_params(color=_TEXT,
                                                             labelcolor=_TEXT)
    ax3.set_title("Network coloured by PageRank", color=_TEXT,
                  fontsize=11, fontweight="bold")

    # ── Panel 4: PR vs Betweenness scatter ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cmap_class = {str(d.get(color_attr, "N/A")):
                  CLASS_COLOR_MAP.get(str(d.get(color_attr, "N/A")), "#97c2fc")
                  for _, d in G.nodes(data=True)}
    node_colors_cl = [CLASS_COLOR_MAP.get(str(G.nodes[n].get(color_attr,"N/A")),
                                           "#97c2fc") for n in G.nodes()]
    x_pr = [pr.get(n, 0) for n in G.nodes()]
    x_bt = [bt.get(n, 0) for n in G.nodes()]
    ax4.scatter(x_pr, x_bt, c=node_colors_cl, s=sizes,
                alpha=0.8, edgecolors="#ffffff", linewidths=0.3, zorder=2)
    _style_ax(ax4, "PageRank vs Betweenness", "PageRank", "Betweenness")

    present = {str(G.nodes[n].get(color_attr,"N/A")) for n in G.nodes()}
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in CLASS_COLOR_MAP.items() if l in present]
    if patches:
        ax4.legend(handles=patches, title=color_attr, loc="upper right",
                   framealpha=0.7, facecolor="#1a1a2e", edgecolor="#555577",
                   labelcolor="white", fontsize=7, title_fontsize=8)

    fig.suptitle("Link Analysis Dashboard", color=_TEXT,
                 fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_centrality_comparison(
    G: nx.Graph,
    metrics: Dict[str, Dict[Any, float]],
    output_path: str = "_centrality_compare.png",
    top_n: int = 15,
    figsize: Tuple = (18, 8),
    dpi: int = 150,
) -> str:
    """
    Grouped bar chart comparing top nodes across multiple centrality measures.
    """
    measure_keys   = ["pagerank", "betweenness", "eigenvector", "closeness", "katz"]
    measure_labels = ["PageRank", "Betweenness", "Eigenvector", "Closeness", "Katz"]
    measure_colors = [_CYAN, _CORAL, _GREEN, _AMBER, _PURPLE]

    # Identify top-N nodes by PageRank (or degree if PR not available)
    ref_metric = metrics.get("pagerank") or metrics.get("degree_centrality", {})
    top_nodes  = [n for n, _ in get_top_nodes(ref_metric, top_n)]
    node_lbls  = [str(n) for n in top_nodes]

    # Normalise each metric to [0,1] for comparison
    def _norm(d): 
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn: return {k: 0.5 for k in d}
        return {k: (v-mn)/(mx-mn) for k,v in d.items()}

    x = np.arange(len(top_nodes))
    width = 0.15
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)

    for i, (key, lbl, col) in enumerate(zip(measure_keys, measure_labels, measure_colors)):
        if key not in metrics: continue
        normed = _norm(metrics[key])
        vals   = [normed.get(n, 0) for n in top_nodes]
        ax.bar(x + i * width, vals, width, label=lbl, color=col, alpha=0.82,
               edgecolor="#222244", linewidth=0.4)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(node_lbls, rotation=45, ha="right", color=_TEXT, fontsize=8)
    _style_ax(ax, f"Centrality Comparison — Top {top_n} Nodes by PageRank",
              "Node", "Normalised Score")
    ax.legend(loc="upper right", framealpha=0.7, facecolor="#1a1a2e",
              edgecolor="#555577", labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_hits_chart(
    hubs: Dict[Any, float],
    authorities: Dict[Any, float],
    output_path: str = "_hits.png",
    top_n: int = 15,
    figsize: Tuple = (14, 6),
    dpi: int = 150,
) -> str:
    """Side-by-side bar charts for HITS hubs and authorities."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor=_BG)

    for ax, scores, title, col in [
        (ax1, hubs,       f"HITS Hubs — Top {top_n}",        _AMBER),
        (ax2, authorities,f"HITS Authorities — Top {top_n}",  _GREEN),
    ]:
        top = get_top_nodes(scores, top_n)
        ids = [str(n) for n, _ in top]
        vs  = [s for _, s in top]
        ax.barh(ids[::-1], vs[::-1], color=col, alpha=0.85,
                edgecolor="#222244", linewidth=0.5)
        _style_ax(ax, title, "Score", "Node")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def build_metrics_table(
    G: nx.Graph,
    metrics: Dict[str, Dict[Any, float]],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Return a list of dicts (one per node) with all link-analysis scores,
    sorted by PageRank descending.
    """
    pr = metrics.get("pagerank", {})
    bt = metrics.get("betweenness", {})
    ev = metrics.get("eigenvector", {})
    kt = metrics.get("katz", {})
    cl = metrics.get("closeness", {})
    hb = metrics.get("hubs", {})
    au = metrics.get("authorities", {})
    dc = metrics.get("degree_centrality", {})

    rows = []
    for n in G.nodes():
        rows.append({
            "node":           str(n),
            "degree":         G.degree(n),
            "pagerank":       round(pr.get(n, 0), 6),
            "betweenness":    round(bt.get(n, 0), 6),
            "eigenvector":    round(ev.get(n, 0), 6),
            "katz":           round(kt.get(n, 0), 6),
            "closeness":      round(cl.get(n, 0), 6),
            "hubs":           round(hb.get(n, 0), 6),
            "authorities":    round(au.get(n, 0), 6),
        })
    rows.sort(key=lambda r: r["pagerank"], reverse=True)
    return rows[:top_n] if top_n else rows