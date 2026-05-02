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

import math
import time
import warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple


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

    # ── assortativity ─────────────────────────────────────────────────────
    try:
        assortativity = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assortativity = float("nan")

    # ── small-world estimate (σ proxy) ────────────────────────────────────
    # σ = (C/C_rand) / (L/L_rand) where C_rand ≈ k/n, L_rand ≈ ln(n)/ln(k)
    n  = G.number_of_nodes()
    k  = deg_mean
    if k > 1 and n > 1:
        c_rand   = k / n
        l_rand   = math.log(n) / math.log(max(k, 1.01))
        sw_sigma = round((avg_clustering / max(c_rand, 1e-9)) /
                          (apl / max(l_rand, 1e-9)), 4)
    else:
        sw_sigma = float("nan")

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
        # assortativity & small-world
        "assortativity":    round(assortativity, 4),
        "sw_sigma":         sw_sigma,
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
        ("Assortativity",      metrics["assortativity"]),
        ("Small-world σ",      metrics["sw_sigma"]),
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
    degree, clustering, weighted_clustering, triangles, eccentricity,
    and all node attributes from the original CSV.
    Suitable for display in a GUI table or export to CSV.
    """
    UG  = G.to_undirected() if G.is_directed() else G
    lcc = max(nx.connected_components(UG), key=len)
    ecc = nx.eccentricity(UG.subgraph(lcc))

    cc   = nx.clustering(UG)
    cc_w = nx.clustering(UG, weight="weight")
    tris = nx.triangles(UG)

    rows = []
    for node, data in G.nodes(data=True):
        row: Dict[str, Any] = {"id": node}
        row.update(data)                                 # CSV attributes
        row["degree"]              = G.degree(node)
        row["clustering"]          = round(cc.get(node, 0), 4)
        row["clustering_weighted"] = round(cc_w.get(node, 0), 4)
        row["triangles"]           = int(tris.get(node, 0))
        row["eccentricity"]        = int(ecc.get(node, -1))
        if G.is_directed():
            row["in_degree"]  = G.in_degree(node)
            row["out_degree"] = G.out_degree(node)
        rows.append(row)

    # Sort by degree descending
    rows.sort(key=lambda r: -r["degree"])
    return rows


def export_node_metrics_csv(rows: List[Dict], path: str) -> str:
    """Write the node metrics table to a CSV file."""
    import csv, os
    if not rows:
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


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
_GREEN = "#5cba6e"
_PURPLE= "#9b59b6"


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
      [B] CDF (cumulative)
      [C] Log-log scatter + power-law fit line
      [D] Degree rank plot (Zipf)
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    k    = np.array(dd["k_vals"], dtype=float)
    pdf  = np.array(dd["pdf"])
    cdf  = np.array(dd["cdf"])
    degs = np.array(dd["degrees"])

    # ── A: PDF bar ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.bar(k, pdf, width=max(1, (k[-1]-k[0])/len(k)*0.9),
           color=_BLUE, alpha=0.85, edgecolor=_PANEL, linewidth=0.3)
    _style_ax(ax, "Degree k", "P(k)", "Degree Distribution (PDF)")
    ax.text(0.97, 0.95,
            f"μ={np.mean(degs):.1f}\nσ={np.std(degs):.1f}",
            transform=ax.transAxes, ha="right", va="top",
            color=_AMBER, fontsize=9,
            bbox=dict(fc=_BG, ec=_GRID, alpha=0.8, boxstyle="round,pad=0.3"))

    # ── B: CDF ────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(k, cdf, color=_TEAL, linewidth=2)
    ax.fill_between(k, cdf, alpha=0.15, color=_TEAL)
    ax.axhline(0.5, color=_AMBER, linestyle="--", linewidth=1, alpha=0.7,
               label="50th percentile")
    # Mark median degree
    median_deg = float(np.median(degs))
    ax.axvline(median_deg, color=_CORAL, linestyle=":", linewidth=1,
               label=f"Median={median_deg:.0f}")
    ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "Degree k", "Cumulative P(k)", "Degree CDF")

    # ── C: Log-log + power-law fit ────────────────────────────────────────
    ax = axes[2]
    # Only plot non-zero PDF points
    mask = pdf > 0
    ax.scatter(np.log10(k[mask]+1), np.log10(pdf[mask]+1e-9),
               color=_CORAL, s=25, alpha=0.85, zorder=3, label="Empirical")
    # Fit line
    log_k_fit = np.log10(k[mask] + 1)
    log_p_fit = np.log10(pdf[mask] + 1e-9)
    coeffs    = np.polyfit(log_k_fit, log_p_fit, 1)
    fit_line  = np.polyval(coeffs, log_k_fit)
    ax.plot(log_k_fit, fit_line, color=_AMBER, linewidth=1.5,
            linestyle="--", label=f"Fit γ≈{dd['gamma']}  R²={dd['log_r2']}")
    ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "log₁₀(k+1)", "log₁₀ P(k)", "Log-Log (Power-Law Check)")

    # ── D: Degree rank (Zipf) ────────────────────────────────────────────
    ax = axes[3]
    ranks = np.arange(1, len(degs) + 1)
    ax.plot(ranks, degs, color=_GREEN, linewidth=1.8)
    ax.fill_between(ranks, degs, alpha=0.15, color=_GREEN)
    _style_ax(ax, "Rank", "Degree", "Degree Rank Plot")
    ax.text(0.97, 0.95,
            f"Max={int(degs[0])}  Min={int(degs[-1])}",
            transform=ax.transAxes, ha="right", va="top",
            color=_AMBER, fontsize=9,
            bbox=dict(fc=_BG, ec=_GRID, alpha=0.8, boxstyle="round,pad=0.3"))

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
      [B] Clustering vs Degree scatter
      [C] Mean clustering per class (bar chart)
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    cc_vals  = np.array(list(cc_data["per_node"].values()))
    cc_w_vals= np.array(list(cc_data["per_node_weighted"].values()))
    degs     = np.array([G.degree(n) for n in cc_data["per_node"]])

    # ── A: Histogram ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(cc_vals,   bins=30, color=_BLUE,  alpha=0.7, label="Unweighted", density=True)
    ax.hist(cc_w_vals, bins=30, color=_CORAL, alpha=0.5, label="Weighted",   density=True)
    ax.axvline(cc_data["avg_clustering"], color=_AMBER, linewidth=1.8,
               linestyle="--", label=f"Avg={cc_data['avg_clustering']}")
    ax.axvline(cc_data["transitivity"], color=_GREEN, linewidth=1.5,
               linestyle=":", label=f"Transitivity={cc_data['transitivity']}")
    ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "Clustering Coefficient", "Density",
              "Clustering Coefficient Distribution")

    # ── B: Clustering vs Degree ───────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(degs, cc_vals, c=cc_w_vals, cmap="plasma",
                    s=25, alpha=0.7, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Weighted CC", color=_TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=_TEXT, labelcolor=_TEXT)
    # Trend line
    if len(degs) > 2:
        z = np.polyfit(degs, cc_vals, 1)
        p = np.poly1d(z)
        xs = np.linspace(degs.min(), degs.max(), 100)
        ax.plot(xs, p(xs), color=_AMBER, linewidth=1.5,
                linestyle="--", alpha=0.8, label="Trend")
        ax.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax, "Degree", "Clustering Coefficient",
              "Clustering vs Degree")

    # ── C: Per-class bar chart ────────────────────────────────────────────
    ax = axes[2]
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
    figsize: Tuple[int,int] = (14, 6),
    dpi: int = 150,
) -> str:
    """
    2-panel path length figure:
      [A] Histogram of shortest path lengths
      [B] Eccentricity distribution
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    k_vals = pl_data["k_vals"]
    counts = pl_data["counts"]
    total  = sum(counts)
    pdf    = [c / total for c in counts]

    # ── A: Path length histogram ──────────────────────────────────────────
    bars = ax1.bar(k_vals, pdf, color=_TEAL, alpha=0.85,
                   edgecolor=_PANEL, linewidth=0.3, width=0.7)
    ax1.axvline(pl_data["apl"], color=_AMBER, linewidth=2,
                linestyle="--", label=f"APL = {pl_data['apl']}")
    ax1.axvline(pl_data["diameter"], color=_CORAL, linewidth=1.5,
                linestyle=":", label=f"Diameter = {pl_data['diameter']}")
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
            f"Radius = {pl_data['radius']}\n"
            f"LCC nodes = {pl_data['lcc_size']}")
    ax1.text(0.97, 0.97, info, transform=ax1.transAxes,
             ha="right", va="top", fontsize=8.5, color=_AMBER,
             bbox=dict(fc=_BG, ec=_GRID, alpha=0.85, boxstyle="round,pad=0.4"))

    # ── B: Eccentricity distribution ──────────────────────────────────────
    ecc_vals = list(pl_data["eccentricity"].values())
    ecc_counter = Counter(ecc_vals)
    ecc_k    = sorted(ecc_counter.keys())
    ecc_c    = [ecc_counter[k] for k in ecc_k]

    bars2 = ax2.bar(ecc_k, ecc_c, color=_PURPLE, alpha=0.85,
                    edgecolor=_PANEL, linewidth=0.3, width=0.6)
    ax2.axvline(pl_data["diameter"], color=_CORAL, linewidth=1.5,
                linestyle=":", label=f"Diameter={pl_data['diameter']}")
    ax2.axvline(pl_data["radius"],   color=_GREEN, linewidth=1.5,
                linestyle="--", label=f"Radius={pl_data['radius']}")
    for bar, c in zip(bars2, ecc_c):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 str(c), ha="center", va="bottom",
                 color=_TEXT, fontsize=10)
    ax2.legend(fontsize=9, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax2, "Eccentricity", "Number of Nodes",
              "Eccentricity Distribution")

    fig.suptitle("Path Length Analysis", color=_TEXT,
                 fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_rich_club(
    rc_data: Dict,
    output_path: str,
    figsize: Tuple[int,int] = (9, 5),
    dpi: int = 150,
) -> str:
    """Rich-club coefficient φ(k) curve."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    k   = rc_data["k_vals"]
    phi = rc_data["phi"]
    ax.plot(k, phi, color=_CORAL, linewidth=2)
    ax.fill_between(k, phi, alpha=0.18, color=_CORAL)
    _style_ax(ax, "Degree threshold k", "Rich-Club φ(k)",
              "Rich-Club Coefficient")
    ax.text(0.97, 0.97,
            "φ(k): fraction of edges\namong top-k degree nodes",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=_AMBER,
            bbox=dict(fc=_BG, ec=_GRID, alpha=0.8, boxstyle="round,pad=0.3"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def render_metrics_dashboard(
    G: nx.Graph,
    global_m: Dict,
    dd: Dict,
    cc_data: Dict,
    pl_data: Dict,
    output_path: str,
    figsize: Tuple[int,int] = (22, 16),
    dpi: int = 130,
) -> str:
    """
    Master dashboard: all metric plots in one figure (3×3 grid).
    Suitable for a single-page overview / report page.
    """
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # Row 0 — degree
    ax_pdf  = fig.add_subplot(gs[0, 0])
    ax_cdf  = fig.add_subplot(gs[0, 1])
    ax_rank = fig.add_subplot(gs[0, 2])

    # Row 1 — clustering
    ax_cc_hist  = fig.add_subplot(gs[1, 0])
    ax_cc_deg   = fig.add_subplot(gs[1, 1])
    ax_cc_class = fig.add_subplot(gs[1, 2])

    # Row 2 — paths + rich club + stats card
    ax_path = fig.add_subplot(gs[2, 0])
    ax_ecc  = fig.add_subplot(gs[2, 1])
    ax_card = fig.add_subplot(gs[2, 2])

    k    = np.array(dd["k_vals"], dtype=float)
    pdf  = np.array(dd["pdf"])
    cdf  = np.array(dd["cdf"])
    degs = np.array(dd["degrees"])
    cc_vals = np.array(list(cc_data["per_node"].values()))
    cc_w    = np.array(list(cc_data["per_node_weighted"].values()))
    deg_cc  = np.array([G.degree(n) for n in cc_data["per_node"]])

    # ── Degree PDF ────────────────────────────────────────────────────────
    ax_pdf.bar(k, pdf, width=max(1,(k[-1]-k[0])/len(k)*0.85),
               color=_BLUE, alpha=0.82, edgecolor=_PANEL, linewidth=0.2)
    _style_ax(ax_pdf, "Degree k", "P(k)", "Degree Distribution")

    # ── CDF ───────────────────────────────────────────────────────────────
    ax_cdf.plot(k, cdf, color=_TEAL, linewidth=2)
    ax_cdf.fill_between(k, cdf, alpha=0.15, color=_TEAL)
    ax_cdf.axhline(0.5, color=_AMBER, linewidth=1, linestyle="--", alpha=0.7)
    _style_ax(ax_cdf, "Degree k", "Cumulative P(k)", "Degree CDF")

    # ── Rank plot ─────────────────────────────────────────────────────────
    ranks = np.arange(1, len(degs)+1)
    ax_rank.plot(ranks, degs, color=_GREEN, linewidth=1.5)
    ax_rank.fill_between(ranks, degs, alpha=0.15, color=_GREEN)
    _style_ax(ax_rank, "Rank", "Degree", "Degree Rank")

    # ── CC Histogram ──────────────────────────────────────────────────────
    ax_cc_hist.hist(cc_vals, bins=25, color=_BLUE, alpha=0.75,
                    label="Unweighted", density=True)
    ax_cc_hist.hist(cc_w, bins=25, color=_CORAL, alpha=0.5,
                    label="Weighted", density=True)
    ax_cc_hist.axvline(cc_data["avg_clustering"], color=_AMBER,
                       linewidth=1.5, linestyle="--",
                       label=f"Avg={cc_data['avg_clustering']}")
    ax_cc_hist.legend(fontsize=7, facecolor=_BG, edgecolor=_GRID,
                      labelcolor=_TEXT)
    _style_ax(ax_cc_hist, "CC", "Density", "Clustering Coefficients")

    # ── CC vs Degree ──────────────────────────────────────────────────────
    ax_cc_deg.scatter(deg_cc, cc_vals, c=cc_w, cmap="plasma",
                      s=18, alpha=0.7, edgecolors="none")
    _style_ax(ax_cc_deg, "Degree", "CC", "CC vs Degree")

    # ── CC by Class ───────────────────────────────────────────────────────
    class_data = cc_data.get("class_avg", {})
    if class_data:
        cls = sorted(class_data.keys())
        v   = [class_data[c] for c in cls]
        ax_cc_class.bar(cls, v, color=_PURPLE, alpha=0.85,
                        edgecolor=_PANEL, linewidth=0.3)
        ax_cc_class.axhline(cc_data["avg_clustering"], color=_CORAL,
                            linestyle="--", linewidth=1)
        ax_cc_class.tick_params(axis="x", rotation=45, labelsize=7)
    _style_ax(ax_cc_class, "Class", "Mean CC", "CC by Class")

    # ── Path histogram ────────────────────────────────────────────────────
    pk = pl_data["k_vals"]
    pp = [c/sum(pl_data["counts"]) for c in pl_data["counts"]]
    ax_path.bar(pk, pp, color=_TEAL, alpha=0.85,
                edgecolor=_PANEL, linewidth=0.3, width=0.65)
    ax_path.axvline(pl_data["apl"], color=_AMBER, linewidth=1.5,
                    linestyle="--", label=f"APL={pl_data['apl']}")
    ax_path.legend(fontsize=8, facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax_path, "Path Length", "Fraction", "Path Length Distribution")

    # ── Eccentricity ──────────────────────────────────────────────────────
    ecc_v = list(pl_data["eccentricity"].values())
    ec    = Counter(ecc_v)
    ax_ecc.bar(sorted(ec.keys()), [ec[k] for k in sorted(ec.keys())],
               color=_PURPLE, alpha=0.85, edgecolor=_PANEL, linewidth=0.3,
               width=0.55)
    _style_ax(ax_ecc, "Eccentricity", "# Nodes", "Eccentricity Distribution")

    # ── Stats card ────────────────────────────────────────────────────────
    ax_card.set_facecolor(_PANEL)
    ax_card.axis("off")
    m = global_m
    lines = [
        ("NETWORK SUMMARY", None, 13, _TEXT, "bold"),
        ("─" * 28, None, 8, _GRID, "normal"),
        (f"Nodes",           f"{m['nodes']}",              10, _TEXT, "normal"),
        (f"Edges",           f"{m['edges']}",              10, _TEXT, "normal"),
        (f"Density",         f"{m['density']}",            10, _TEXT, "normal"),
        (f"Components",      f"{m['n_components']}",       10, _TEXT, "normal"),
        ("─" * 28, None, 8, _GRID, "normal"),
        (f"Avg Degree",      f"{m['degree_mean']}",        10, _BLUE, "normal"),
        (f"Degree Std",      f"{m['degree_std']}",         10, _BLUE, "normal"),
        (f"Degree Range",
         f"{m['degree_min']}–{m['degree_max']}",           10, _BLUE, "normal"),
        ("─" * 28, None, 8, _GRID, "normal"),
        (f"Avg Path Length", f"{m['avg_path_length']}",   10, _TEAL, "normal"),
        (f"Diameter",        f"{m['diameter']}",           10, _TEAL, "normal"),
        (f"Radius",          f"{m['radius']}",             10, _TEAL, "normal"),
        ("─" * 28, None, 8, _GRID, "normal"),
        (f"Avg Clustering",  f"{m['avg_clustering']}",    10, _CORAL, "normal"),
        (f"Transitivity",    f"{m['transitivity']}",      10, _CORAL, "normal"),
        (f"Assortativity",   f"{m['assortativity']}",     10, _CORAL, "normal"),
        (f"Small-world σ",   f"{m['sw_sigma']}",          10, _AMBER, "bold"),
    ]
    y = 0.97
    for item in lines:
        if len(item) == 5:
            label, value, size, color, weight = item
        else:
            continue
        if value is None:
            ax_card.text(0.05, y, label, transform=ax_card.transAxes,
                         fontsize=size, color=color, fontweight=weight,
                         va="top")
        else:
            ax_card.text(0.05, y, label, transform=ax_card.transAxes,
                         fontsize=size, color=color, fontweight=weight,
                         va="top")
            ax_card.text(0.95, y, value, transform=ax_card.transAxes,
                         fontsize=size, color=color, fontweight="bold",
                         va="top", ha="right")
        y -= 0.052

    fig.suptitle("Graph Metrics Dashboard — Primary School Contact Network",
                 color=_TEXT, fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path