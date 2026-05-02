"""
Phase 4: Graph Metrics & Statistics Module
==========================================
Computes, stores, and visualises a comprehensive set of network metrics.

Metric groups
-------------
1. Global summary       — nodes, edges, density, components, diameter, radius,
                          APL, transitivity, assortativity, small-world index
2. Degree distribution  — raw degrees, PDF, CDF, log-log fit, power-law test
3. Clustering           — per-node (unweighted + weighted), avg, distribution
4. Path lengths         — all-pairs histogram, APL, eccentricity distribution
5. Rich-club            — rich-club coefficient curve
6. Per-node table       — degree, clustering, eccentricity, triangles (exportable)

All compute_* functions return plain Python dicts/lists → easy to pass to GUI.
All render_* functions produce PNG files and return their paths.
"""
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

def _to_simple_graph(G: nx.Graph) -> nx.Graph:
    """
    Convert MultiGraph/MultiDiGraph → Graph/DiGraph
    by collapsing parallel edges.
    """
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        if G.is_directed():
            H = nx.DiGraph()
        else:
            H = nx.Graph()

        for u, v, data in G.edges(data=True):
            if H.has_edge(u, v):
                # accumulate weight if exists
                if "weight" in data:
                    H[u][v]["weight"] = H[u][v].get("weight", 0) + data["weight"]
                else:
                    H[u][v]["weight"] = H[u][v].get("weight", 0) + 1
            else:
                H.add_edge(u, v, **data)

        H.add_nodes_from(G.nodes(data=True))
        return H

    return G
# ─────────────────────────────────────────────────────────────────────────────
#  1. GLOBAL SUMMARY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Return a dict of scalar graph-level metrics.

    All values are plain Python floats/ints so they serialise cleanly.
    Slow metrics (diameter, APL) are computed only when the graph is
    connected; otherwise we work on the largest connected component.
    """
    G = _to_simple_graph(G)  # 🔥 ADD THIS LINE
    directed = G.is_directed()
    UG = G.to_undirected() if directed else G          # undirected view

    # ── component analysis ────────────────────────────────────────────────
    if directed:
        n_comp   = nx.number_weakly_connected_components(G)
        connected = nx.is_weakly_connected(G)
        lcc       = max(nx.weakly_connected_components(G), key=len)
    else:
        n_comp   = nx.number_connected_components(G)
        connected = nx.is_connected(G)
        lcc       = max(nx.connected_components(G), key=len)

    G_lcc = UG.subgraph(lcc).copy()

    # ── degree stats ──────────────────────────────────────────────────────
    degrees  = [d for _, d in G.degree()]
    deg_mean = float(np.mean(degrees))
    deg_std  = float(np.std(degrees))
    deg_min  = int(min(degrees))
    deg_max  = int(max(degrees))

    # ── path / diameter (on LCC) ──────────────────────────────────────────
    t0 = time.time()
    apl      = float(nx.average_shortest_path_length(G_lcc))
    diameter = int(nx.diameter(G_lcc))
    radius   = int(nx.radius(G_lcc))
    path_ms  = round((time.time() - t0) * 1000)

    # ── clustering ────────────────────────────────────────────────────────
    avg_clustering  = float(nx.average_clustering(UG))
    transitivity    = float(nx.transitivity(UG))        # = global clustering coeff
    n  = G.number_of_nodes()


    return {
        # topology
        "nodes":            n,
        "edges":            G.number_of_edges(),
        "directed":         directed,
        "density":          round(float(nx.density(G)), 6),
        "self_loops":       nx.number_of_selfloops(G),
        # connectivity
        "is_connected":     connected,
        "n_components":     n_comp,
        "lcc_size":         len(lcc),
        # degree
        "degree_mean":      round(deg_mean, 3),
        "degree_std":       round(deg_std, 3),
        "degree_min":       deg_min,
        "degree_max":       deg_max,
        # paths
        "avg_path_length":  round(apl, 4),
        "diameter":         diameter,
        "radius":           radius,
        "path_compute_ms":  path_ms,
        # clustering
        "avg_clustering":   round(avg_clustering, 4),
        "transitivity":     round(transitivity, 4),
    }


def print_global_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print the global metrics dict."""
    sep = "═" * 48
    print(f"\n{sep}")
    print("  GRAPH METRICS SUMMARY")
    print(sep)
    rows = [
        ("Type",               "Directed" if metrics["directed"] else "Undirected"),
        ("Nodes",              metrics["nodes"]),
        ("Edges",              metrics["edges"]),
        ("Density",            metrics["density"]),
        ("Self-loops",         metrics["self_loops"]),
        ("──────────────────", "──────────────────"),
        ("Connected",          metrics["is_connected"]),
        ("Components",         metrics["n_components"]),
        ("LCC size",           metrics["lcc_size"]),
        ("──────────────────", "──────────────────"),
        ("Degree mean",        metrics["degree_mean"]),
        ("Degree std",         metrics["degree_std"]),
        ("Degree min / max",   f"{metrics['degree_min']} / {metrics['degree_max']}"),
        ("──────────────────", "──────────────────"),
        ("Avg path length",    metrics["avg_path_length"]),
        ("Diameter",           metrics["diameter"]),
        ("Radius",             metrics["radius"]),
        ("──────────────────", "──────────────────"),
        ("Avg clustering",     metrics["avg_clustering"]),
        ("Transitivity",       metrics["transitivity"]),
        ("Avg Degree",      metrics["degree_mean"]),
    ]
    for label, value in rows:
        print(f"  {label:<22} {value}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  2. DEGREE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_degree_distribution(G: nx.Graph) -> Dict[str, Any]:
    """
    Return degree sequence, PDF, CDF, and log-log regression stats.
    For directed graphs returns in-degree and out-degree separately.
    """
    G = _to_simple_graph(G)  # 🔥 ADD THIS
    directed = G.is_directed()

    if directed:
        in_deg  = sorted([d for _, d in G.in_degree()],  reverse=True)
        out_deg = sorted([d for _, d in G.out_degree()], reverse=True)
        degrees = sorted([d for _, d in G.degree()], reverse=True)
    else:
        degrees = sorted([d for _, d in G.degree()], reverse=True)
        in_deg = out_deg = []

    counts = Counter(degrees)
    k_vals = sorted(counts.keys())
    pdf    = np.array([counts[k] for k in k_vals], dtype=float)
    pdf   /= pdf.sum()
    cdf    = np.cumsum(pdf)

    # Log-log linear fit (power-law check) — filter zeros
    log_k  = np.log10(np.array(k_vals, dtype=float) + 1e-9)
    log_p  = np.log10(pdf + 1e-9)
    coeffs = np.polyfit(log_k, log_p, 1)   # slope = -γ for power law
    gamma  = -round(float(coeffs[0]), 3)
    r2     = float(np.corrcoef(log_k, log_p)[0, 1] ** 2)

    return {
        "degrees":  degrees,
        "in_deg":   in_deg,
        "out_deg":  out_deg,
        "k_vals":   k_vals,
        "pdf":      pdf.tolist(),
        "cdf":      cdf.tolist(),
        "gamma":    gamma,           # power-law exponent estimate
        "log_r2":   round(r2, 4),   # fit quality (1 = perfect power law)
        "directed": directed,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  3. CLUSTERING COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_clustering(G: nx.Graph) -> Dict[str, Any]:
    """
    Per-node and global clustering coefficients (weighted + unweighted).
    Also returns triangle counts and square clustering.
    """
    G = _to_simple_graph(G)  # 🔥 ADD THIS
    UG = G.to_undirected() if G.is_directed() else G
    cc      = nx.clustering(UG)                    # unweighted, per node
    cc_w    = nx.clustering(UG, weight="weight")   # weighted
    tris    = nx.triangles(UG)
    sq_cc   = nx.square_clustering(UG)             # 4-cycle based

    # Group by attribute if available
    per_class: Dict[str, List[float]] = defaultdict(list)
    for node, val in cc.items():
        cls = str(G.nodes[node].get("Class", "N/A"))
        per_class[cls].append(val)

    class_avg = {k: round(float(np.mean(v)), 4) for k, v in per_class.items()}

    return {
        "per_node":        {n: round(v, 4) for n, v in cc.items()},
        "per_node_weighted": {n: round(v, 4) for n, v in cc_w.items()},
        "triangles":       {n: int(v) for n, v in tris.items()},
        "square_cc":       {n: round(v, 4) for n, v in sq_cc.items()},
        "avg_clustering":  round(float(nx.average_clustering(UG)), 4),
        "avg_weighted":    round(float(nx.average_clustering(UG, weight="weight")), 4),
        "transitivity":    round(float(nx.transitivity(UG)), 4),
        "class_avg":       class_avg,   # mean clustering per class
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. PATH LENGTH DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_path_lengths(G: nx.Graph) -> Dict[str, Any]:
    """
    All-pairs shortest path lengths, eccentricity, APL, diameter, radius.
    Works on the largest connected component if graph is disconnected.
    """
    G = _to_simple_graph(G)  # 🔥 ADD THIS
    UG = G.to_undirected() if G.is_directed() else G

    # Use LCC
    lcc  = max(nx.connected_components(UG), key=len)
    G_lcc = UG.subgraph(lcc).copy()

    t0 = time.time()
    all_pairs = dict(nx.all_pairs_shortest_path_length(G_lcc))
    elapsed   = round(time.time() - t0, 3)

    all_lengths = [
        length
        for src, targets in all_pairs.items()
        for tgt, length in targets.items()
        if src != tgt
    ]

    dist_counter = Counter(all_lengths)
    k_vals  = sorted(dist_counter.keys())
    counts  = [dist_counter[k] for k in k_vals]
    total   = sum(counts)
    pdf     = [c / total for c in counts]
    cdf     = list(np.cumsum(pdf))

    ecc = nx.eccentricity(G_lcc, sp=all_pairs)

    return {
        "apl":         round(float(np.mean(all_lengths)), 4),
        "diameter":    int(max(all_lengths)),
        "radius":      int(min(ecc.values())),
        "k_vals":      k_vals,
        "counts":      counts,
        "pdf":         pdf,
        "cdf":         cdf,
        "eccentricity": {n: int(e) for n, e in ecc.items()},
        "lcc_size":    len(lcc),
        "compute_s":   elapsed,
    }
# ─────────────────────────────────────────────────────────────────────────────
#  5. PER-NODE METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compute_node_metrics_table(G: nx.Graph) -> List[Dict[str, Any]]:
    """
    Build a list-of-dicts (one row per node) with:
    degree, clustering, triangles, eccentricity,
    and all node attributes from the original CSV.
    Suitable for display in a GUI table or export to CSV.
    """
    G = _to_simple_graph(G)   # 🔥 ADD THIS LINE
    UG  = G.to_undirected() if G.is_directed() else G

    cc   = nx.clustering(UG)

    rows = []
    for node, data in G.nodes(data=True):
        row: Dict[str, Any] = {"id": node}
        row.update(data)                                 # CSV attributes
        row["degree"]              = G.degree(node)
        row["clustering"]          = round(cc.get(node, 0), 4)
        if G.is_directed():
            row["in_degree"]  = G.in_degree(node)
            row["out_degree"] = G.out_degree(node)
        rows.append(row)

    # Sort by degree descending
    rows.sort(key=lambda r: -r["degree"])
    return rows
# ─────────────────────────────────────────────────────────────────────────────
#  6. RICH-CLUB COEFFICIENT
# ─────────────────────────────────────────────────────────────────────────────
def compute_rich_club(G: nx.Graph) -> Dict[str, Any]:
    """Rich-club coefficient φ(k) for each degree threshold k."""
    UG = G.to_undirected() if G.is_directed() else G
    rc = nx.rich_club_coefficient(UG, normalized=False)
    return {
        "k_vals": list(rc.keys()),
        "phi":    [round(v, 4) for v in rc.values()],
    }

# ─────────────────────────────────────────────────────────────────────────────
#  RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
_BG    = "#0f0f1a"
_PANEL = "#141428"
_TEXT  = "#ddddee"
_GRID  = "#2a2a44"
_BLUE  = "#4a90d9"
_TEAL  = "#2ec4b6"
_CORAL = "#e87040"
_AMBER = "#f0a500"
def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color=_TEXT)


def render_degree_distribution(
    dd: Dict,
    output_path: str,
    figsize: Tuple[int,int] = (16, 10),
    dpi: int = 150,
) -> str:
    """
    4-panel degree distribution figure:
      [A] Histogram (PDF bar chart)
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(1, 1, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(1) for c in range(1)]

    k    = np.array(dd["k_vals"], dtype=float)
    pdf  = np.array(dd["pdf"])
    # ── A: PDF bar ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(k, pdf, width=max(1, (k[-1]-k[0])/len(k)*0.9),
           color=_BLUE, alpha=0.85, edgecolor=_PANEL, linewidth=0.3)
    _style_ax(ax, "Degree k", "P(k)", "Degree Distribution (PDF)")   
    fig.suptitle("Degree Distribution Analysis", color=_TEXT,
                 fontsize=15, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_clustering_analysis(
    G: nx.Graph,
    cc_data: Dict,
    output_path: str,
    figsize: Tuple[int,int] = (16, 10),
    dpi: int = 150,
) -> str:
    """
    3-panel clustering analysis:
      [A] Histogram of per-node clustering coefficients
      [C] Mean clustering per class (bar chart)
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[0, c]) for c in range(2)]

    cc_vals  = np.array(list(cc_data["per_node"].values()))
    cc_w_vals= np.array(list(cc_data["per_node_weighted"].values()))

    # ── A: Histogram ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(cc_vals,   bins=30, color=_BLUE,  alpha=0.7, label="Unweighted", density=True)
    ax.hist(cc_w_vals, bins=30, color=_CORAL, alpha=0.5, label="Weighted",   density=True)
    ax.axvline(cc_data["avg_clustering"], color=_AMBER, linewidth=1.8,
               linestyle="--", label=f"Avg={cc_data['avg_clustering']}")

    ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "Clustering Coefficient", "Density",
              "Clustering Coefficient Distribution")

    # ── C: Per-class bar chart ────────────────────────────────────────────
    ax = axes[1]
    class_data = cc_data.get("class_avg", {})
    if class_data:
        classes = sorted(class_data.keys())
        vals    = [class_data[c] for c in classes]
        colors  = [_BLUE if c != "Teachers" else _AMBER for c in classes]
        bars = ax.bar(classes, vals, color=colors, edgecolor=_PANEL,
                      linewidth=0.4, alpha=0.88)
        ax.tick_params(axis="x", rotation=45)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=_TEXT, fontsize=7)
        ax.axhline(cc_data["avg_clustering"], color=_CORAL,
                   linestyle="--", linewidth=1.2,
                   label=f"Global avg={cc_data['avg_clustering']}")
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "Class", "Mean Clustering", "Avg Clustering by Class")

    fig.suptitle("Clustering Coefficient Analysis", color=_TEXT,
                 fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path

def render_path_length_analysis(
    pl_data: Dict,
    output_path: str,
    figsize: Tuple[int,int] = (9, 6),
    dpi: int = 150,
) -> str:
    """
    Single-panel path length figure:
      Histogram of shortest path lengths
    """
    fig, ax1 = plt.subplots(figsize=figsize, facecolor=_BG)

    k_vals = pl_data["k_vals"]
    counts = pl_data["counts"]
    total  = sum(counts)
    pdf    = [c / total for c in counts]

    # ── Path length histogram ──────────────────────────────────────────────
    bars = ax1.bar(k_vals, pdf, color=_TEAL, alpha=0.85,
                   edgecolor=_PANEL, linewidth=0.3, width=0.7)
    for bar, p in zip(bars, pdf):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f"{p*100:.1f}%", ha="center", va="bottom",
                 color=_TEXT, fontsize=9)
    ax1.legend(fontsize=9, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax1, "Shortest Path Length", "Fraction of Pairs",
              "Shortest Path Length Distribution")

    # Annotation box
    info = (f"APL = {pl_data['apl']}\n"
            f"Diameter = {pl_data['diameter']}\n"
            )
    ax1.text(0.97, 0.97, info, transform=ax1.transAxes,
             ha="right", va="top", fontsize=8.5, color=_AMBER,
             bbox=dict(fc=_BG, ec=_GRID, alpha=0.85, boxstyle="round,pad=0.4"))

    fig.suptitle("Shortest Path Length Distribution", color=_TEXT,
                 fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path

# def render_rich_club(
#     rc_data: Dict,
#     output_path: str,
#     figsize: Tuple[int,int] = (9, 5),
#     dpi: int = 150,
# ) -> str:
#     """Rich-club coefficient φ(k) curve."""
#     fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
#     k   = rc_data["k_vals"]
#     phi = rc_data["phi"]
#     ax.plot(k, phi, color=_CORAL, linewidth=2)
#     ax.fill_between(k, phi, alpha=0.18, color=_CORAL)
#     _style_ax(ax, "Degree threshold k", "Rich-Club φ(k)",
#               "Rich-Club Coefficient")
#     ax.text(0.97, 0.97,
#             "φ(k): fraction of edges\namong top-k degree nodes",
#             transform=ax.transAxes, ha="right", va="top",
#             fontsize=8.5, color=_AMBER,
#             bbox=dict(fc=_BG, ec=_GRID, alpha=0.8, boxstyle="round,pad=0.3"))
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
#                 facecolor=fig.get_facecolor())
#     plt.close()
#     return output_path