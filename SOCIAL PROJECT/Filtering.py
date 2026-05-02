"""
Phase 5 — Enhanced Filtering Module
=====================================
Provides rich node-filtering capabilities for the Social Network Analysis Tool.

Filter strategies
-----------------
1. Centrality-based  — degree, betweenness, closeness, eigenvector, pagerank
                        (normalised range sliders for each)
2. Membership-based  — community id, Class attribute
3. Combined          — AND / OR combination of strategies 1 and 2

New in this version
-------------------
* Five centrality measures (degree, betweenness, closeness, eigenvector, pagerank)
* filter_by_centrality_ranges() — fine-grained per-measure range control
* filter_by_community_and_centrality() — keep nodes that are BOTH in selected
  communities AND within centrality thresholds
* render_five_centrality_distributions() — 5-panel histogram overview
* render_centrality_radar() — per-node radar chart for selected nodes
"""

import math
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection


# ─────────────────────────────────────────────────────────────────────────────
#  Shared palette
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLOR_MAP: Dict[str, str] = {
    "1A": "#e6194b", "1B": "#f58231",
    "2A": "#3cb44b", "2B": "#bfef45",
    "3A": "#4363d8", "3B": "#42d4f4",
    "4A": "#911eb4", "4B": "#f032e6",
    "5A": "#9A6324", "5B": "#C8860A",
    "Teachers": "#2ec4b6",
}

COMM_COLORS: List[str] = [
    "#4a90d9", "#e87040", "#5cba6e", "#9b59b6",
    "#f0a500", "#2ec4b6", "#e6194b", "#3cb44b",
    "#f032e6", "#42d4f4", "#fb923c", "#a3e635",
]

_BG    = "#0f0f1a"
_PANEL = "#141428"
_TEXT  = "#ddddee"
_GRID  = "#2a2a44"
_BLUE  = "#4a90d9"
_TEAL  = "#2ec4b6"
_CORAL = "#e87040"
_AMBER = "#f0a500"
_GREEN = "#5cba6e"
_PINK  = "#f472b6"
_PURPLE= "#9b59b6"

CENT_COLORS = {
    "degree":      _BLUE,
    "betweenness": _CORAL,
    "closeness":   _GREEN,
    "eigenvector": _AMBER,
    "pagerank":    _PINK,
}

CENT_LABELS = {
    "degree":      "Degree Centrality",
    "betweenness": "Betweenness Centrality",
    "closeness":   "Closeness Centrality",
    "eigenvector": "Eigenvector Centrality",
    "pagerank":    "PageRank",
}


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", color=_TEXT)


# ─────────────────────────────────────────────────────────────────────────────
#  1. CENTRALITY COMPUTATION  (5 measures)
# ─────────────────────────────────────────────────────────────────────────────

def compute_centralities(G: nx.Graph) -> Dict[str, Dict[Any, float]]:
    """
    Compute five centrality measures for every node.

    Returns
    -------
    dict with keys:
        "degree"           : {node: normalised_degree}
        "betweenness"      : {node: normalised_betweenness}
        "closeness"        : {node: normalised_closeness}
        "eigenvector"      : {node: normalised_eigenvector}
        "pagerank"         : {node: normalised_pagerank}
        "raw_degree"       : {node: raw_int_degree}
        "raw_betweenness"  : {node: raw_betweenness}
        "raw_closeness"    : {node: raw_closeness}
        "raw_eigenvector"  : {node: raw_eigenvector}
        "raw_pagerank"     : {node: raw_pagerank}
    """
    # ── Degree ───────────────────────────────────────────────────────────────
    raw_deg: Dict[Any, float] = dict(G.degree())

    # ── Betweenness ──────────────────────────────────────────────────────────
    raw_bet: Dict[Any, float] = nx.betweenness_centrality(G, normalized=True, weight="weight")

    # ── Closeness ────────────────────────────────────────────────────────────
    raw_clo: Dict[Any, float] = nx.closeness_centrality(G)

    # ── Eigenvector ──────────────────────────────────────────────────────────
    try:
        UG = G.to_undirected() if G.is_directed() else G
        raw_eig: Dict[Any, float] = nx.eigenvector_centrality_numpy(UG, weight="weight")
    except Exception:
        raw_eig = {n: 0.0 for n in G.nodes()}

    # ── PageRank ─────────────────────────────────────────────────────────────
    try:
        raw_pr: Dict[Any, float] = nx.pagerank(G, weight="weight", max_iter=200)
    except Exception:
        raw_pr = {n: 1.0 / max(G.number_of_nodes(), 1) for n in G.nodes()}

    def _norm(scores: Dict) -> Dict[Any, float]:
        vals = list(scores.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

    return {
        "degree":          _norm(raw_deg),
        "betweenness":     _norm(raw_bet),
        "closeness":       _norm(raw_clo),
        "eigenvector":     _norm(raw_eig),
        "pagerank":        _norm(raw_pr),
        "raw_degree":      {n: int(d) for n, d in raw_deg.items()},
        "raw_betweenness": raw_bet,
        "raw_closeness":   raw_clo,
        "raw_eigenvector": raw_eig,
        "raw_pagerank":    raw_pr,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. COMMUNITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> Dict[Any, int]:
    """
    Assign community ids using connected-component analysis.
    Returns {node: community_id}, sorted largest-first.
    """
    UG = G.to_undirected() if G.is_directed() else G
    comps = sorted(nx.connected_components(UG), key=len, reverse=True)
    membership: Dict[Any, int] = {}
    for cid, comp in enumerate(comps):
        for node in comp:
            membership[node] = cid
    return membership


# ─────────────────────────────────────────────────────────────────────────────
#  3. FILTER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_centrality(
    G: nx.Graph,
    centralities: Dict[str, Dict],
    degree_range: Tuple[float, float]      = (0.0, 1.0),
    betweenness_range: Tuple[float, float] = (0.0, 1.0),
    closeness_range: Tuple[float, float]   = (0.0, 1.0),
    eigenvector_range: Tuple[float, float] = (0.0, 1.0),
    pagerank_range: Tuple[float, float]    = (0.0, 1.0),
) -> nx.Graph:
    """
    Keep nodes whose normalised centrality scores all fall within the
    specified ranges.  Supports all five centrality measures.
    """
    deg = centralities.get("degree",      {})
    bet = centralities.get("betweenness", {})
    clo = centralities.get("closeness",   {})
    eig = centralities.get("eigenvector", {})
    pr  = centralities.get("pagerank",    {})

    keep: List[Any] = [
        n for n in G.nodes()
        if (degree_range[0]      <= deg.get(n, 0) <= degree_range[1]      and
            betweenness_range[0] <= bet.get(n, 0) <= betweenness_range[1] and
            closeness_range[0]   <= clo.get(n, 0) <= closeness_range[1]   and
            eigenvector_range[0] <= eig.get(n, 0) <= eigenvector_range[1] and
            pagerank_range[0]    <= pr.get(n, 0)  <= pagerank_range[1])
    ]
    return G.subgraph(keep).copy()


def filter_by_membership(
    G: nx.Graph,
    communities: Optional[Dict[Any, int]] = None,
    community_ids: Optional[Set[int]] = None,
    classes: Optional[Set[str]] = None,
    class_attr: str = "Class",
) -> nx.Graph:
    """
    Keep nodes that belong to selected communities AND/OR classes.
    """
    if communities is None:
        communities = detect_communities(G)

    keep: List[Any] = []
    for n, data in G.nodes(data=True):
        comm_ok  = (community_ids is None) or (communities.get(n) in community_ids)
        class_ok = (classes is None) or (str(data.get(class_attr, "N/A")) in classes)
        if comm_ok and class_ok:
            keep.append(n)
    return G.subgraph(keep).copy()


def filter_by_community_and_centrality(
    G: nx.Graph,
    centralities: Dict[str, Dict],
    communities: Dict[Any, int],
    community_ids: Optional[Set[int]] = None,
    degree_range: Tuple[float, float]      = (0.0, 1.0),
    betweenness_range: Tuple[float, float] = (0.0, 1.0),
    closeness_range: Tuple[float, float]   = (0.0, 1.0),
    eigenvector_range: Tuple[float, float] = (0.0, 1.0),
    pagerank_range: Tuple[float, float]    = (0.0, 1.0),
    operator: str = "AND",
) -> nx.Graph:
    """
    Filter by community membership AND/OR centrality ranges simultaneously.
    This is the key function for requirement 2:
      'filtering nodes based on their membership in specific communities
       OR their centrality scores falling within certain ranges.'

    operator="AND" → node must pass BOTH community and centrality tests
    operator="OR"  → node must pass at least ONE test
    """
    cent_sub = filter_by_centrality(
        G, centralities,
        degree_range, betweenness_range, closeness_range,
        eigenvector_range, pagerank_range,
    )
    memb_sub = filter_by_membership(G, communities, community_ids)

    cent_nodes = set(cent_sub.nodes())
    memb_nodes = set(memb_sub.nodes())

    if operator.upper() == "AND":
        keep = cent_nodes & memb_nodes
    else:
        keep = cent_nodes | memb_nodes
    return G.subgraph(keep).copy()


def filter_combined(
    G: nx.Graph,
    centralities: Dict[str, Dict],
    communities: Optional[Dict[Any, int]] = None,
    degree_range: Tuple[float, float]      = (0.0, 1.0),
    betweenness_range: Tuple[float, float] = (0.0, 1.0),
    closeness_range: Tuple[float, float]   = (0.0, 1.0),
    eigenvector_range: Tuple[float, float] = (0.0, 1.0),
    pagerank_range: Tuple[float, float]    = (0.0, 1.0),
    community_ids: Optional[Set[int]]   = None,
    classes: Optional[Set[str]]         = None,
    class_attr: str = "Class",
    operator: str = "AND",
) -> nx.Graph:
    """Combined centrality + membership filter with AND/OR logic."""
    if communities is None:
        communities = detect_communities(G)

    cent_sub = filter_by_centrality(
        G, centralities,
        degree_range, betweenness_range, closeness_range,
        eigenvector_range, pagerank_range,
    )
    memb_sub = filter_by_membership(G, communities, community_ids, classes, class_attr)

    cent_nodes = set(cent_sub.nodes())
    memb_nodes = set(memb_sub.nodes())

    keep = (cent_nodes & memb_nodes) if operator.upper() == "AND" else (cent_nodes | memb_nodes)
    return G.subgraph(keep).copy()


# ─────────────────────────────────────────────────────────────────────────────
#  4. SUMMARY / STATS
# ─────────────────────────────────────────────────────────────────────────────

def get_filter_stats(
    G_original: nx.Graph,
    G_filtered: nx.Graph,
    centralities: Dict[str, Dict],
    communities: Dict[Any, int],
) -> Dict[str, Any]:
    """Return a dict of filter statistics suitable for GUI display."""
    filt_nodes = list(G_filtered.nodes())
    deg = centralities.get("raw_degree",      {})
    bet = centralities.get("raw_betweenness", {})
    clo = centralities.get("raw_closeness",   {})
    eig = centralities.get("raw_eigenvector", {})
    pr  = centralities.get("raw_pagerank",    {})

    def _safe_mean(lst):
        return round(float(np.mean(lst)), 4) if lst else 0.0

    filt_deg = [deg.get(n, 0) for n in filt_nodes]
    filt_bet = [bet.get(n, 0) for n in filt_nodes]
    filt_clo = [clo.get(n, 0) for n in filt_nodes]
    filt_eig = [eig.get(n, 0) for n in filt_nodes]
    filt_pr  = [pr.get(n,  0) for n in filt_nodes]

    return {
        "n_original":          G_original.number_of_nodes(),
        "e_original":          G_original.number_of_edges(),
        "n_filtered":          G_filtered.number_of_nodes(),
        "e_filtered":          G_filtered.number_of_edges(),
        "pct_nodes":           round(G_filtered.number_of_nodes() / max(G_original.number_of_nodes(), 1) * 100, 1),
        "pct_edges":           round(G_filtered.number_of_edges() / max(G_original.number_of_edges(), 1) * 100, 1),
        "mean_degree":         _safe_mean(filt_deg),
        "mean_betweenness":    _safe_mean(filt_bet),
        "mean_closeness":      _safe_mean(filt_clo),
        "mean_eigenvector":    _safe_mean(filt_eig),
        "mean_pagerank":       _safe_mean(filt_pr),
        "communities_present": sorted({communities.get(n) for n in filt_nodes if n in communities}),
    }


def print_filter_summary(G_original, G_filtered, centralities=None, communities=None, label="Filtered"):
    n_o, e_o = G_original.number_of_nodes(), G_original.number_of_edges()
    n_f, e_f = G_filtered.number_of_nodes(), G_filtered.number_of_edges()
    print(f"\n{'═'*52}")
    print(f"  FILTER SUMMARY — {label}")
    print(f"{'═'*52}")
    print(f"  Nodes: {n_o} → {n_f}  ({n_f/max(n_o,1)*100:.1f}% kept)")
    print(f"  Edges: {e_o} → {e_f}  ({e_f/max(e_o,1)*100:.1f}% kept)")
    if communities:
        ids = sorted({communities.get(n) for n in G_filtered.nodes() if n in communities})
        print(f"  Communities present: {ids}")
    print(f"{'═'*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  5. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _node_colors_for(G, attr="Class"):
    return [CLASS_COLOR_MAP.get(str(d.get(attr, "N/A")), "#97c2fc")
            for _, d in G.nodes(data=True)]


def _degree_sizes(G, lo=20, hi=260):
    degs = [G.degree(n) for n in G.nodes()]
    if not degs: return []
    mn, mx = min(degs), max(degs)
    if mx == mn: return [(lo+hi)/2] * len(degs)
    return [lo + (hi-lo)*(d-mn)/(mx-mn) for d in degs]


def render_filter_comparison(
    G: nx.Graph,
    G_filtered: nx.Graph,
    output_path: str,
    centralities: Optional[Dict[str, Dict]] = None,
    communities: Optional[Dict[Any, int]] = None,
    color_attr: str = "Class",
    layout_seed: int = 42,
    figsize: Tuple[int, int] = (20, 10),
    dpi: int = 150,
    title: str = "Filtering Result",
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=_BG)
    k = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
    pos_full = nx.spring_layout(G, k=k, seed=layout_seed, weight="weight")
    pos_filt = {n: pos_full[n] for n in G_filtered.nodes() if n in pos_full}
    filt_set = set(G_filtered.nodes())

    for ax, (graph, pos, subtitle) in zip(axes, [
        (G,          pos_full, f"Original  ({G.number_of_nodes()}N  {G.number_of_edges()}E)"),
        (G_filtered, pos_filt, f"Filtered  ({G_filtered.number_of_nodes()}N  {G_filtered.number_of_edges()}E)"),
    ]):
        ax.set_facecolor(_BG); ax.axis("off")
        weights = [d.get("weight", 1) for _, _, d in graph.edges(data=True)]
        w_lo, w_hi = min(weights, default=1), max(weights, default=1)
        ew = [0.15 + 1.2*(w-w_lo)/(w_hi-w_lo) if w_hi > w_lo else 0.4 for w in weights]

        if graph is G:
            nc = [CLASS_COLOR_MAP.get(str(d.get(color_attr,"N/A")), "#97c2fc")
                  if n in filt_set else "#2a2a44"
                  for n, d in graph.nodes(data=True)]
            for (u, v), ew_val in zip(graph.edges(), ew):
                if u in pos and v in pos:
                    col = "#bbbbbb" if (u in filt_set and v in filt_set) else "#333355"
                    alp = 0.3 if (u in filt_set and v in filt_set) else 0.06
                    x0, y0 = pos[u]; x1, y1 = pos[v]
                    ax.plot([x0,x1],[y0,y1], color=col, lw=ew_val, alpha=alp, zorder=1)
        else:
            nc = _node_colors_for(graph, color_attr)
            nx.draw_networkx_edges(graph, pos, width=ew, alpha=0.3, edge_color="#bbbbbb", ax=ax)

        sz = _degree_sizes(graph)
        nx.draw_networkx_nodes(graph, pos, node_color=nc,
                               node_size=sz if sz else [60],
                               alpha=0.92, linewidths=0.5, edgecolors="#ffffff", ax=ax)
        ax.set_title(subtitle, color=_TEXT, fontsize=12, fontweight="bold", pad=8)

    present = {str(d.get(color_attr,"N/A")) for _,d in G.nodes(data=True)}
    patches = [mpatches.Patch(color=c, label=l)
               for l,c in CLASS_COLOR_MAP.items() if l in present]
    fig.legend(handles=patches, title=color_attr, loc="lower center",
               ncol=min(len(patches),6), framealpha=0.75,
               facecolor="#1a1a2e", edgecolor="#555577",
               labelcolor="white", fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, color=_TEXT, fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_centrality_distributions(
    G: nx.Graph,
    centralities: Dict[str, Dict],
    G_filtered: Optional[nx.Graph] = None,
    output_path: str = "centrality_dist.png",
    figsize: Tuple[int, int] = (22, 9),
    dpi: int = 150,
) -> str:
    """Five-panel histogram — one for each centrality measure."""
    measures = ["degree", "betweenness", "closeness", "eigenvector", "pagerank"]
    fig, axes = plt.subplots(1, 5, figsize=figsize, facecolor=_BG)
    filt_nodes = set(G_filtered.nodes()) if G_filtered else set()

    for ax, key in zip(axes, measures):
        scores = centralities.get(key, {})
        col    = CENT_COLORS[key]
        vals_all  = list(scores.values())
        vals_filt = [scores[n] for n in filt_nodes if n in scores]

        ax.hist(vals_all,  bins=20, color=col, alpha=0.3, label="All",      density=True)
        if vals_filt:
            ax.hist(vals_filt, bins=20, color=col, alpha=0.85, label="Filtered", density=True)
            lo, hi = min(vals_filt), max(vals_filt)
            ax.axvspan(lo, hi, alpha=0.12, color=col)
            ax.axvline(lo, color=col, lw=1.2, ls="--")
            ax.axvline(hi, color=col, lw=1.2, ls="--")
        ax.legend(fontsize=7, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
        _style_ax(ax, "Score", "Density", CENT_LABELS[key])

    fig.suptitle("Centrality Distributions — shaded = filtered range",
                 color=_TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_centrality_scatter(
    G: nx.Graph,
    centralities: Dict[str, Dict],
    G_filtered: Optional[nx.Graph] = None,
    output_path: str = "centrality_scatter.png",
    color_attr: str = "Class",
    figsize: Tuple[int, int] = (20, 6),
    dpi: int = 150,
) -> str:
    """Degree vs Betweenness / Closeness / Eigenvector / PageRank."""
    fig, axes = plt.subplots(1, 4, figsize=figsize, facecolor=_BG)
    filt_nodes = set(G_filtered.nodes()) if G_filtered else set()
    nd = centralities.get("degree",      {}); nb = centralities.get("betweenness", {})
    nc = centralities.get("closeness",   {}); ne = centralities.get("eigenvector", {})
    np_ = centralities.get("pagerank",   {})

    node_list = list(G.nodes(data=True))
    colors = [CLASS_COLOR_MAP.get(str(d.get(color_attr,"N/A")), "#97c2fc") for _,d in node_list]

    pairs = [
        ([nd.get(n,0) for n,_ in node_list], [nb.get(n,0) for n,_ in node_list],
         "Degree", "Betweenness", "Degree vs Betweenness"),
        ([nd.get(n,0) for n,_ in node_list], [nc.get(n,0) for n,_ in node_list],
         "Degree", "Closeness", "Degree vs Closeness"),
        ([ne.get(n,0) for n,_ in node_list], [np_.get(n,0) for n,_ in node_list],
         "Eigenvector", "PageRank", "Eigenvector vs PageRank"),
        ([nb.get(n,0) for n,_ in node_list], [nc.get(n,0) for n,_ in node_list],
         "Betweenness", "Closeness", "Betweenness vs Closeness"),
    ]

    for ax, (xs, ys, xl, yl, ttl) in zip(axes, pairs):
        bg_x = [x for x,(n,_) in zip(xs,node_list) if n not in filt_nodes]
        bg_y = [y for y,(n,_) in zip(ys,node_list) if n not in filt_nodes]
        bg_c = [c for c,(n,_) in zip(colors,node_list) if n not in filt_nodes]
        ax.scatter(bg_x, bg_y, c=bg_c, s=18, alpha=0.15, edgecolors="none")

        hl_x = [x for x,(n,_) in zip(xs,node_list) if n in filt_nodes]
        hl_y = [y for y,(n,_) in zip(ys,node_list) if n in filt_nodes]
        hl_c = [c for c,(n,_) in zip(colors,node_list) if n in filt_nodes]
        if hl_x:
            ax.scatter(hl_x, hl_y, c=hl_c, s=65, alpha=0.92,
                       edgecolors="#ffffff", linewidths=0.4, zorder=3, label="Filtered")
        ax.legend(fontsize=7, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
        _style_ax(ax, xl, yl, ttl)

    fig.suptitle("Centrality Scatter — highlighted = filtered nodes",
                 color=_TEXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_community_membership(
    G: nx.Graph,
    communities: Dict[Any, int],
    G_filtered: Optional[nx.Graph] = None,
    output_path: str = "community_membership.png",
    color_attr: str = "Class",
    layout_seed: int = 42,
    figsize: Tuple[int, int] = (14, 11),
    dpi: int = 150,
) -> str:
    filt_nodes = set(G_filtered.nodes()) if G_filtered else set(G.nodes())
    k   = 1.5 / math.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, k=k, seed=layout_seed, weight="weight")
    num_comm = max(communities.values(), default=0) + 1
    nc = [COMM_COLORS[communities.get(n, 0) % len(COMM_COLORS)] for n in G.nodes()]
    alphas = [0.92 if n in filt_nodes else 0.12 for n in G.nodes()]

    weights = [d.get("weight", 1) for _,_,d in G.edges(data=True)]
    w_lo, w_hi = min(weights, default=1), max(weights, default=1)
    ew = [0.15 + 1.2*(w-w_lo)/(w_hi-w_lo) if w_hi > w_lo else 0.4 for w in weights]

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_BG); ax.axis("off")

    for (u,v), ew_val in zip(G.edges(), ew):
        if u in pos and v in pos:
            col = "#bbbbbb" if (u in filt_nodes and v in filt_nodes) else "#222244"
            alp = 0.3 if (u in filt_nodes and v in filt_nodes) else 0.05
            x0,y0 = pos[u]; x1,y1 = pos[v]
            ax.plot([x0,x1],[y0,y1], color=col, lw=ew_val, alpha=alp, zorder=1)

    degs = [G.degree(n) for n in G.nodes()]
    mn_d, mx_d = min(degs), max(degs)
    sizes = [20 + 200*(G.degree(n)-mn_d)/max(mx_d-mn_d,1) for n in G.nodes()]

    for n, col, sz, alpha in zip(G.nodes(), nc, sizes, alphas):
        if n not in pos: continue
        x, y = pos[n]
        ec = "#ffffff" if n in filt_nodes else "#333355"
        lw = 1.0 if n in filt_nodes else 0.3
        ax.scatter(x, y, s=sz, color=col, alpha=alpha, edgecolors=ec, linewidths=lw, zorder=2)

    patches = [mpatches.Patch(color=COMM_COLORS[c % len(COMM_COLORS)], label=f"Community {c+1}")
               for c in range(num_comm)]
    leg = ax.legend(handles=patches, title="Communities", loc="lower left",
                    framealpha=0.75, facecolor="#1a1a2e", edgecolor="#555577",
                    labelcolor="white", fontsize=8)
    leg.get_title().set_color("white")
    ax.set_title(f"Community Membership  —  {len(filt_nodes)} of {G.number_of_nodes()} nodes highlighted",
                 color=_TEXT, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def run_filtering_pipeline(
    G: nx.Graph,
    output_dir: str = ".",
    degree_range: Tuple[float, float]      = (0.3, 1.0),
    betweenness_range: Tuple[float, float] = (0.0, 1.0),
    closeness_range: Tuple[float, float]   = (0.4, 1.0),
    eigenvector_range: Tuple[float, float] = (0.0, 1.0),
    pagerank_range: Tuple[float, float]    = (0.0, 1.0),
    community_ids: Optional[Set[int]]   = None,
    classes: Optional[Set[str]]         = None,
    class_attr: str = "Class",
    filter_mode: str = "centrality",
    combine_op: str = "AND",
    color_attr: str = "Class",
    layout_seed: int = 42,
    dpi: int = 150,
) -> Dict[str, Any]:
    import os
    os.makedirs(output_dir, exist_ok=True)
    centralities = compute_centralities(G)
    communities  = detect_communities(G)

    if filter_mode == "centrality":
        G_filtered = filter_by_centrality(
            G, centralities,
            degree_range, betweenness_range, closeness_range,
            eigenvector_range, pagerank_range,
        )
    elif filter_mode == "membership":
        G_filtered = filter_by_membership(G, communities, community_ids, classes, class_attr)
    elif filter_mode == "community_and_centrality":
        G_filtered = filter_by_community_and_centrality(
            G, centralities, communities, community_ids,
            degree_range, betweenness_range, closeness_range,
            eigenvector_range, pagerank_range, combine_op,
        )
    else:
        G_filtered = filter_combined(
            G, centralities, communities,
            degree_range, betweenness_range, closeness_range,
            eigenvector_range, pagerank_range,
            community_ids, classes, class_attr, combine_op,
        )

    paths = {}
    p = os.path.join(output_dir, "filter_comparison.png")
    render_filter_comparison(G, G_filtered, p, centralities, communities, color_attr, layout_seed, dpi=dpi)
    paths["comparison"] = p

    p = os.path.join(output_dir, "centrality_distributions.png")
    render_centrality_distributions(G, centralities, G_filtered, p, dpi=dpi)
    paths["centrality_dist"] = p

    p = os.path.join(output_dir, "centrality_scatter.png")
    render_centrality_scatter(G, centralities, G_filtered, p, color_attr=color_attr, dpi=dpi)
    paths["centrality_scatter"] = p

    p = os.path.join(output_dir, "community_membership.png")
    render_community_membership(G, communities, G_filtered, p, color_attr=color_attr,
                                 layout_seed=layout_seed, dpi=dpi)
    paths["community_membership"] = p

    stats = get_filter_stats(G, G_filtered, centralities, communities)
    return {
        "G_filtered": G_filtered,
        "centralities": centralities,
        "communities":  communities,
        "stats":        stats,
        "output_paths": paths,
    }