"""
Phase 2: Visualization Module
Renders the social network using PyVis (interactive HTML) and Matplotlib (static).
Supports coloring/sizing nodes by any attribute, multiple layouts, and edge weight encoding.
"""

import os
import math
import colorsys
import itertools
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import Optional, Dict, Any, List, Tuple


# ─────────────────────────────────────────────
#  Color palettes
# ─────────────────────────────────────────────

# 11-color qualitative palette (distinct, colorblind-friendly where possible)
QUALITATIVE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#9A6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
]

CLASS_COLOR_MAP = {
    "1A": "#e6194b", "1B": "#f58231",
    "2A": "#3cb44b", "2B": "#bfef45",
    "3A": "#4363d8", "3B": "#42d4f4",
    "4A": "#911eb4", "4B": "#f032e6",
    "5A": "#9A6324", "5B": "#fabed4",
    "Teachers": "#aaffc3",
}

GENDER_COLOR_MAP = {
    "M": "#4363d8",
    "F": "#e6194b",
    "Unknown": "#888888",
}


def _build_color_map(G: nx.Graph, color_attr: str) -> Dict[Any, str]:
    """Map each unique value of `color_attr` to a hex color string."""
    # Use preset maps for known attributes
    if color_attr == "Class":
        return CLASS_COLOR_MAP
    if color_attr == "Gender":
        return GENDER_COLOR_MAP

    values = sorted(set(
        str(d.get(color_attr, "N/A"))
        for _, d in G.nodes(data=True)
    ))
    return {v: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] for i, v in enumerate(values)}


def _node_colors(G: nx.Graph, color_attr: Optional[str], color_map: Dict) -> List[str]:
    fallback = "#97c2fc"
    if not color_attr:
        return [fallback] * G.number_of_nodes()
    return [
        color_map.get(str(data.get(color_attr, "N/A")), fallback)
        for _, data in G.nodes(data=True)
    ]


def _node_sizes_by_attr(
    G: nx.Graph,
    size_attr: Optional[str],
    min_size: float = 5,
    max_size: float = 40,
) -> List[float]:
    """Scale node sizes linearly between min_size and max_size based on a numeric attribute."""
    if not size_attr:
        return [10.0] * G.number_of_nodes()

    raw = []
    for _, data in G.nodes(data=True):
        v = data.get(size_attr)
        try:
            raw.append(float(v))
        except (TypeError, ValueError):
            raw.append(0.0)

    lo, hi = min(raw), max(raw)
    if hi == lo:
        return [min_size + (max_size - min_size) / 2] * len(raw)
    return [
        min_size + (max_size - min_size) * (v - lo) / (hi - lo)
        for v in raw
    ]


def _node_sizes_by_degree(
    G: nx.Graph, min_size: float = 5, max_size: float = 40
) -> List[float]:
    degrees = [G.degree(n) for n in G.nodes()]
    lo, hi = min(degrees), max(degrees)
    if hi == lo:
        return [min_size] * len(degrees)
    return [
        min_size + (max_size - min_size) * (d - lo) / (hi - lo)
        for d in degrees
    ]


# ─────────────────────────────────────────────
#  PyVis interactive HTML
# ─────────────────────────────────────────────

def visualize_pyvis(
    G: nx.Graph,
    output_path: str = "network.html",
    color_attr: Optional[str] = "Class",
    size_by: str = "degree",          # "degree" | attribute name | "uniform"
    show_labels: bool = True,
    label_attr: Optional[str] = None, # node attribute to use as label
    edge_width_by_weight: bool = True,
    title: str = "Social Network",
    height: str = "750px",
    width: str = "100%",
    physics_enabled: bool = True,
) -> str:
    """
    Render the graph as an interactive HTML file using PyVis.

    Returns the path to the generated HTML file.
    """
    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        bgcolor="#1a1a2e",
        font_color="white",
        directed=G.is_directed(),
        notebook=False,
    )

    # Physics options for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.8
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 200 }
      },
      "edges": {
        "smooth": { "type": "continuous" }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # Build color map
    color_map = _build_color_map(G, color_attr) if color_attr else {}

    # Compute sizes
    if size_by == "degree":
        sizes = _node_sizes_by_degree(G, min_size=8, max_size=40)
    elif size_by == "uniform":
        sizes = [14.0] * G.number_of_nodes()
    else:
        sizes = _node_sizes_by_attr(G, size_by, min_size=8, max_size=40)

    nodes_list = list(G.nodes(data=True))

    # Add nodes
    for i, (node_id, data) in enumerate(nodes_list):
        # Label
        if label_attr and label_attr in data:
            label = str(data[label_attr])
        elif show_labels:
            label = str(data.get("label", node_id))
        else:
            label = ""

        # Color
        if color_attr:
            val = str(data.get(color_attr, "N/A"))
            color = color_map.get(val, "#97c2fc")
        else:
            color = "#97c2fc"

        # Tooltip: show all attributes
        tooltip_lines = [f"<b>ID: {node_id}</b>"]
        for k, v in data.items():
            tooltip_lines.append(f"{k}: {v}")
        tooltip = "<br>".join(tooltip_lines)

        net.add_node(
            int(node_id) if hasattr(node_id, 'item') else node_id,
            label=label,
            color=color,
            size=float(sizes[i]),
            title=tooltip,
            font={"size": 10, "color": "white"},
        )

    # Add edges
    weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
    w_max = max(weights) if weights else 1
    w_min = min(weights) if weights else 1

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        if edge_width_by_weight and w_max > w_min:
            width_val = 0.3 + 3.0 * (float(w) - float(w_min)) / (float(w_max) - float(w_min))
        else:
            width_val = 1.0

        edge_tooltip = f"Weight: {w}"
        for k, val in data.items():
            if k != "weight":
                edge_tooltip += f"<br>{k}: {val}"

        net.add_edge(
            int(u) if hasattr(u, 'item') else u,
            int(v) if hasattr(v, 'item') else v,
            width=width_val,
            title=edge_tooltip,
            color={"color": "rgba(200,200,200,0.25)", "highlight": "#ffffff"},
        )

    net.save_graph(output_path)

    # Inject legend into the HTML
    if color_attr and color_map:
        _inject_legend(output_path, color_map, color_attr, title)

    return output_path


def _inject_legend(html_path: str, color_map: Dict, attr_name: str, title: str):
    """Inject a floating legend + title bar into the PyVis HTML output."""
    legend_items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
        f'<div style="width:14px;height:14px;border-radius:50%;background:{color};flex-shrink:0"></div>'
        f'<span style="font-size:12px">{label}</span></div>'
        for label, color in color_map.items()
    )

    legend_html = f"""
<div style="position:fixed;top:16px;left:16px;background:rgba(20,20,40,0.88);
            color:white;padding:14px 18px;border-radius:10px;z-index:9999;
            border:1px solid rgba(255,255,255,0.15);min-width:150px;
            font-family:sans-serif;backdrop-filter:blur(4px)">
  <div style="font-weight:600;font-size:13px;margin-bottom:10px;
              border-bottom:1px solid rgba(255,255,255,0.2);padding-bottom:6px">
    {attr_name}
  </div>
  {legend_items}
</div>
<div style="position:fixed;top:16px;left:50%;transform:translateX(-50%);
            background:rgba(20,20,40,0.75);color:white;padding:8px 20px;
            border-radius:8px;z-index:9999;font-family:sans-serif;
            font-size:15px;font-weight:600;border:1px solid rgba(255,255,255,0.15)">
  {title}
</div>
"""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("<body>", f"<body>\n{legend_html}", 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────
#  Matplotlib static plots
# ─────────────────────────────────────────────

def _get_layout(G: nx.Graph, layout: str, seed: int = 42) -> Dict:
    """Return a position dictionary for the given layout name."""
    layout = layout.lower()
    if layout == "spring" or layout == "fruchterman_reingold":
        return nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(G.number_of_nodes()))
    elif layout == "kamada_kawai":
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=seed)
    elif layout == "circular":
        return nx.circular_layout(G)
    elif layout == "shell":
        # Group nodes by Class attribute for shell layout
        groups = {}
        for n, d in G.nodes(data=True):
            key = d.get("Class", d.get("community", "default"))
            groups.setdefault(key, []).append(n)
        shells = list(groups.values())
        return nx.shell_layout(G, nlist=shells if len(shells) > 1 else None)
    elif layout == "spectral":
        return nx.spectral_layout(G)
    elif layout == "random":
        return nx.random_layout(G, seed=seed)
    elif layout == "radial":
        # Use ego graph of highest-degree node as centre
        center = max(G.degree(), key=lambda x: x[1])[0]
        return nx.kamada_kawai_layout(G)
    else:
        return nx.spring_layout(G, seed=seed)


def visualize_matplotlib(
    G: nx.Graph,
    output_path: str = "network.png",
    layout: str = "spring",
    color_attr: Optional[str] = "Class",
    size_by: str = "degree",
    show_labels: bool = False,
    label_attr: Optional[str] = None,
    edge_alpha: float = 0.25,
    edge_width_by_weight: bool = True,
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150,
    title: str = "Social Network",
    seed: int = 42,
) -> str:
    """
    Render a static high-resolution PNG using Matplotlib + NetworkX.
    Returns the path to the saved image.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.axis("off")

    # Layout
    pos = _get_layout(G, layout, seed=seed)

    # Colors
    color_map = _build_color_map(G, color_attr) if color_attr else {}
    node_colors = _node_colors(G, color_attr, color_map)

    # Sizes
    if size_by == "degree":
        raw_sizes = _node_sizes_by_degree(G, min_size=20, max_size=300)
    elif size_by == "uniform":
        raw_sizes = [60.0] * G.number_of_nodes()
    else:
        raw_sizes = _node_sizes_by_attr(G, size_by, min_size=20, max_size=300)

    # Edge widths
    weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
    w_max = max(weights) if weights else 1
    w_min = min(weights) if weights else 1
    if edge_width_by_weight and w_max > w_min:
        edge_widths = [
            0.2 + 1.8 * (w - w_min) / (w_max - w_min) for w in weights
        ]
    else:
        edge_widths = [0.5] * len(weights)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=edge_alpha,
        edge_color="#aaaaaa",
        ax=ax,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=raw_sizes,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="#ffffff",
        ax=ax,
    )

    # Labels
    if show_labels:
        if label_attr:
            labels = {n: str(d.get(label_attr, n)) for n, d in G.nodes(data=True)}
        else:
            labels = {n: str(d.get("label", n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos, labels=labels,
            font_size=6, font_color="white", font_weight="bold", ax=ax
        )

    # Legend
    if color_attr and color_map:
        patches = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in color_map.items()
            if any(str(d.get(color_attr)) == lbl for _, d in G.nodes(data=True))
        ]
        legend = ax.legend(
            handles=patches,
            title=color_attr,
            loc="lower left",
            framealpha=0.75,
            facecolor="#1a1a2e",
            edgecolor="#555577",
            labelcolor="white",
            title_fontsize=10,
            fontsize=9,
        )
        legend.get_title().set_color("white")

    # Stats annotation
    stats = (
        f"Nodes: {G.number_of_nodes()}  |  "
        f"Edges: {G.number_of_edges()}  |  "
        f"Density: {nx.density(G):.4f}  |  "
        f"Layout: {layout}"
    )
    ax.text(
        0.5, 0.01, stats,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8, color="#aaaacc",
    )

    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────
#  Multi-panel comparison plot
# ─────────────────────────────────────────────

def visualize_multi_layout(
    G: nx.Graph,
    output_path: str = "multi_layout.png",
    color_attr: Optional[str] = "Class",
    layouts: List[str] = ("spring", "circular", "shell", "kamada_kawai"),
    figsize: Tuple[int, int] = (20, 16),
    dpi: int = 120,
    seed: int = 42,
) -> str:
    """
    Plot the same graph with multiple layouts side by side for comparison.
    """
    n = len(layouts)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="#0f0f1a")
    axes = axes.flatten() if n > 1 else [axes]

    color_map = _build_color_map(G, color_attr) if color_attr else {}
    node_colors = _node_colors(G, color_attr, color_map)
    sizes = _node_sizes_by_degree(G, min_size=15, max_size=200)
    weights = [data.get("weight", 1) for _, _, data in G.edges(data=True)]
    w_max = max(weights) if weights else 1
    w_min = min(weights) if weights else 1

    for i, (layout_name, ax) in enumerate(zip(layouts, axes)):
        ax.set_facecolor("#0f0f1a")
        ax.axis("off")
        pos = _get_layout(G, layout_name, seed=seed)

        if w_max > w_min:
            ew = [0.15 + 1.2 * (w - w_min) / (w_max - w_min) for w in weights]
        else:
            ew = [0.4] * len(weights)

        nx.draw_networkx_edges(G, pos, width=ew, alpha=0.18,
                               edge_color="#aaaaaa", ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=sizes, alpha=0.90,
                               linewidths=0.4, edgecolors="#ffffff", ax=ax)
        ax.set_title(layout_name.replace("_", " ").title(),
                     color="white", fontsize=13, fontweight="bold")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    if color_attr and color_map:
        patches = [
            mpatches.Patch(color=c, label=lbl)
            for lbl, c in color_map.items()
            if any(str(d.get(color_attr)) == lbl for _, d in G.nodes(data=True))
        ]
        fig.legend(
            handles=patches,
            title=color_attr,
            loc="lower center",
            ncol=min(len(patches), 6),
            framealpha=0.7,
            facecolor="#1a1a2e",
            edgecolor="#555577",
            labelcolor="white",
            title_fontsize=10,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.suptitle("Network Layout Comparison", color="white",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────
#  Attribute distribution plot
# ─────────────────────────────────────────────

def visualize_attribute_distribution(
    G: nx.Graph,
    attr: str,
    output_path: str = "attr_dist.png",
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 130,
) -> str:
    """Bar chart of how many nodes belong to each value of a categorical attribute."""
    from collections import Counter
    counts = Counter(str(d.get(attr, "N/A")) for _, d in G.nodes(data=True))
    labels_sorted = sorted(counts.keys())
    values = [counts[l] for l in labels_sorted]

    color_map = _build_color_map(G, attr)
    colors = [color_map.get(l, "#97c2fc") for l in labels_sorted]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    bars = ax.bar(labels_sorted, values, color=colors, edgecolor="#ffffff",
                  linewidth=0.5, alpha=0.9)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", color="white", fontsize=10)
    ax.set_xlabel(attr, color="white", fontsize=12)
    ax.set_ylabel("Count", color="white", fontsize=12)
    ax.set_title(f"Node distribution by {attr}", color="white",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path