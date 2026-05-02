"""
Phase 3: Layout Algorithms Module
Provides 4 layout strategies for network visualization.

Layouts implemented
-------------------
1. spring                — Force-directed; compact, shows community cores
2. fruchterman_reingold  — Force-directed (FR 1991); nodes spread wider
3. hierarchical (tree)   — Spanning-tree, layered top-down by BFS depth
4. radial                — BFS from highest-degree hub; concentric rings

FIX: compute_layout now always returns (pos_dict, elapsed_seconds) tuple.
"""

import math
import time
import warnings
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
#  Shared color palette
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLOR_MAP = {
    "1A": "#e6194b", "1B": "#f58231",
    "2A": "#3cb44b", "2B": "#bfef45",
    "3A": "#4363d8", "3B": "#42d4f4",
    "4A": "#911eb4", "4B": "#f032e6",
    "5A": "#9A6324", "5B": "#fabed4",
    "Teachers": "#aaffc3",
}
GENDER_COLOR_MAP = {"M": "#4363d8", "F": "#e6194b", "Unknown": "#888888"}
QUALITATIVE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#9A6324",
]


def _color_map_for(G: nx.Graph, attr: Optional[str]) -> Dict[str, str]:
    if attr == "Class":   return CLASS_COLOR_MAP
    if attr == "Gender":  return GENDER_COLOR_MAP
    if attr is None:      return {}
    vals = sorted({str(d.get(attr, "N/A")) for _, d in G.nodes(data=True)})
    return {v: QUALITATIVE[i % len(QUALITATIVE)] for i, v in enumerate(vals)}


def _node_colors(G: nx.Graph, attr: Optional[str], cmap: Dict) -> List[str]:
    fallback = "#97c2fc"
    if not attr:
        return [fallback] * G.number_of_nodes()
    return [cmap.get(str(d.get(attr, "N/A")), fallback) for _, d in G.nodes(data=True)]


def _degree_sizes(G: nx.Graph, lo=20, hi=280) -> List[float]:
    degs = [G.degree(n) for n in G.nodes()]
    mn, mx = min(degs), max(degs)
    if mx == mn:
        return [(lo + hi) / 2] * len(degs)
    return [lo + (hi - lo) * (d - mn) / (mx - mn) for d in degs]


# ─────────────────────────────────────────────────────────────────────────────
#  Individual layout computers
# ─────────────────────────────────────────────────────────────────────────────

def layout_spring(
    G: nx.Graph,
    seed: int = 42,
    iterations: int = 100,
    k_scale: float = 1.2,
) -> Dict[Any, Tuple[float, float]]:
    """Spring / force-directed — compact, shows community cores."""
    k = k_scale / math.sqrt(max(G.number_of_nodes(), 1))
    raw = nx.spring_layout(G, k=k, iterations=iterations, seed=seed, weight="weight")
    return {n: (float(p[0]), float(p[1])) for n, p in raw.items()}


def layout_fruchterman_reingold(
    G: nx.Graph,
    seed: int = 42,
    iterations: int = 150,
    k_scale: float = 1.8,
) -> Dict[Any, Tuple[float, float]]:
    """Fruchterman-Reingold force-directed layout. Nodes spread wider."""
    k = k_scale / math.sqrt(max(G.number_of_nodes(), 1))
    raw = nx.spring_layout(G, k=k, iterations=iterations, seed=seed, weight="weight")
    return {n: (float(p[0]), float(p[1])) for n, p in raw.items()}


def layout_hierarchical(
    G: nx.Graph,
    root: Optional[Any] = None,
    group_attr: Optional[str] = None,
    seed: int = 42,
) -> Dict[Any, Tuple[float, float]]:
    """
    Hierarchical tree layout: minimum spanning tree layered top-down by BFS depth.
    Also aliased as 'tree'.
    """
    ug = G.to_undirected() if G.is_directed() else G

    # Use MST for clean hierarchy
    try:
        T = nx.minimum_spanning_tree(ug, weight="weight")
    except Exception:
        T = ug

    if root is None:
        root = max(T.degree(), key=lambda x: x[1])[0]

    layers: List[List[Any]] = list(nx.bfs_layers(T, root))
    n_layers = len(layers)

    parent: Dict[Any, Optional[Any]] = {root: None}
    for layer in layers:
        for node in layer:
            for nbr in T.neighbors(node):
                if nbr not in parent:
                    parent[nbr] = node

    pos: Dict[Any, Tuple[float, float]] = {}
    max_width = max(len(layer) for layer in layers)

    for depth, layer in enumerate(layers):
        y = 1.0 - 2.0 * depth / max(n_layers - 1, 1)
        if group_attr:
            layer = sorted(
                layer,
                key=lambda n: (
                    pos.get(parent[n], (0.0, 0.0))[0],
                    str(G.nodes[n].get(group_attr, "")),
                ),
            )
        else:
            layer = sorted(
                layer,
                key=lambda n: pos.get(parent[n], (0.0, 0.0))[0],
            )
        n = len(layer)
        for i, node in enumerate(layer):
            x = (i - (n - 1) / 2) / max(max_width / 2, 1)
            pos[node] = (float(x), float(y))

    # Fallback for disconnected nodes not in BFS tree
    fallback = layout_spring(G, seed=seed)
    for node in G.nodes():
        if node not in pos:
            pos[node] = fallback[node]

    return pos


def layout_radial(
    G: nx.Graph,
    center_node: Optional[Any] = None,
    seed: int = 42,
) -> Dict[Any, Tuple[float, float]]:
    """Radial layout: most-connected hub at origin, others on concentric BFS rings."""
    if center_node is None or center_node not in G:
        center_node = max(G.degree(), key=lambda x: x[1])[0]

    distances = nx.single_source_shortest_path_length(G, center_node)
    rings: Dict[int, List[Any]] = defaultdict(list)
    for node, dist in distances.items():
        rings[dist].append(node)

    for dist in rings:
        rings[dist].sort(key=lambda n: -G.degree(n))

    pos: Dict[Any, Tuple[float, float]] = {}
    max_dist = max(rings.keys()) if rings else 1

    for dist, nodes in sorted(rings.items()):
        if dist == 0:
            pos[center_node] = (0.0, 0.0)
            continue
        radius = dist / max_dist
        n_ring = len(nodes)
        offset = (dist % 2) * (math.pi / n_ring / 2) if n_ring > 1 else 0
        angles = np.linspace(0, 2 * math.pi, n_ring, endpoint=False) + offset
        for node, angle in zip(nodes, angles):
            pos[node] = (float(radius * math.cos(angle)), float(radius * math.sin(angle)))

    # Fallback for nodes not reachable from center (disconnected components)
    fallback = layout_spring(G, seed=seed)
    for node in G.nodes():
        if node not in pos:
            pos[node] = fallback[node]

    return pos


# ─────────────────────────────────────────────────────────────────────────────
#  Layout registry — 4 layouts
# ─────────────────────────────────────────────────────────────────────────────

LAYOUTS = {
    "spring":               layout_spring,
    "fruchterman_reingold": layout_fruchterman_reingold,
    "hierarchical":         layout_hierarchical,
    "tree":                 layout_hierarchical,   # alias
    "radial":               layout_radial,
}

LAYOUT_DESCRIPTIONS = {
    "spring":               "Spring / force-directed — compact variant, shows community cores.",
    "fruchterman_reingold": "Force-directed (FR 1991) — nodes repel, edges attract. Reveals clusters.",
    "hierarchical":         "Layered tree — spanning tree from highest-degree root. Shows hierarchy.",
    "tree":                 "Layered tree — spanning tree from highest-degree root. Shows hierarchy.",
    "radial":               "Radial / BFS — hub at centre, rings by hop distance. Shows reach.",
}


def compute_layout(
    G: nx.Graph,
    name: str,
    seed: int = 42,
    group_attr: str = "Class",
    center_node: Optional[Any] = None,
    root_node: Optional[Any] = None,
) -> Tuple[Dict[Any, Tuple[float, float]], float]:
    """
    Compute a named layout.

    Returns
    -------
    (positions_dict, elapsed_seconds)
        positions_dict: {node: (x, y)}  — always plain (float, float) tuples
        elapsed_seconds: float
    """
    name_key = name.lower().replace(" ", "_").replace("-", "_")
    if name_key not in LAYOUTS:
        raise ValueError(
            f"Unknown layout '{name}'. Available: {list(LAYOUTS.keys())}"
        )

    t0 = time.time()

    if name_key == "radial":
        pos = layout_radial(G, center_node=center_node, seed=seed)
    elif name_key in ("hierarchical", "tree"):
        pos = layout_hierarchical(G, root=root_node, group_attr=group_attr, seed=seed)
    else:
        pos = LAYOUTS[name_key](G, seed=seed)

    elapsed = round(time.time() - t0, 3)

    # Guarantee all values are plain (float, float) tuples — never np.ndarray
    pos = {n: (float(p[0]), float(p[1])) for n, p in pos.items()}

    return pos, elapsed


# ─────────────────────────────────────────────────────────────────────────────
#  Single-layout render (static PNG)
# ─────────────────────────────────────────────────────────────────────────────

def render_layout(
    G: nx.Graph,
    layout_name: str,
    output_path: str,
    color_attr: Optional[str] = None,
    show_labels: bool = False,
    edge_alpha: float = 0.18,
    figsize: Tuple[int, int] = (14, 11),
    dpi: int = 150,
    seed: int = 42,
    group_attr: str = "Class",
    center_node: Optional[Any] = None,
    root_node: Optional[Any] = None,
    annotation: str = "",
) -> str:
    pos, elapsed = compute_layout(
        G, layout_name, seed=seed, group_attr=group_attr,
        center_node=center_node, root_node=root_node,
    )
    cmap   = _color_map_for(G, color_attr)
    colors = _node_colors(G, color_attr, cmap)
    sizes  = _degree_sizes(G)

    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    w_lo, w_hi = min(weights), max(weights)
    edge_widths = (
        [0.15 + 1.5 * (w - w_lo) / (w_hi - w_lo) for w in weights]
        if w_hi > w_lo else [0.4] * len(weights)
    )

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=edge_alpha,
                           edge_color="#bbbbbb", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                           alpha=0.92, linewidths=0.5, edgecolors="#ffffff", ax=ax)

    if show_labels:
        labels = {n: str(d.get("label", n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=5,
                                font_color="white", ax=ax)

    if color_attr and cmap:
        present_vals = {str(d.get(color_attr)) for _, d in G.nodes(data=True)}
        patches = [mpatches.Patch(color=c, label=lbl)
                   for lbl, c in cmap.items() if lbl in present_vals]
        leg = ax.legend(handles=patches, title=color_attr, loc="lower left",
                        framealpha=0.75, facecolor="#1a1a2e", edgecolor="#555577",
                        labelcolor="white", title_fontsize=9, fontsize=8)
        leg.get_title().set_color("white")

    desc      = LAYOUT_DESCRIPTIONS.get(layout_name.lower().replace(" ", "_"), "")
    title_str = layout_name.replace("_", " ").title()
    ax.set_title(title_str, color="white", fontsize=15, fontweight="bold", pad=10)

    footer = (f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  "
              f"Computed in {elapsed}s")
    if annotation:
        footer += f"  |  {annotation}"
    ax.text(0.5, 0.005, footer, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7.5, color="#aaaacc")
    ax.text(0.5, 0.97, desc, transform=ax.transAxes, ha="center",
            va="top", fontsize=8, color="#ccccee", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="#444466", alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  All-layouts comparison grid (2×2)
# ─────────────────────────────────────────────────────────────────────────────

def render_all_layouts(
    G: nx.Graph,
    output_path: str,
    color_attr: Optional[str] = None,
    layouts: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (24, 20),
    dpi: int = 120,
    seed: int = 42,
    group_attr: str = "Class",
) -> str:
    if layouts is None:
        # Use only the 4 primary layouts (exclude alias 'tree')
        layouts = ["spring", "fruchterman_reingold", "hierarchical", "radial"]

    n    = len(layouts)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="#0f0f1a")
    axes = np.array(axes).flatten()

    cmap    = _color_map_for(G, color_attr)
    colors  = _node_colors(G, color_attr, cmap)
    sizes   = _degree_sizes(G, lo=8, hi=120)
    weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    w_lo, w_hi = min(weights), max(weights)

    for i, name in enumerate(layouts):
        ax = axes[i]
        ax.set_facecolor("#0f0f1a")
        ax.axis("off")
        try:
            pos, elapsed = compute_layout(G, name, seed=seed, group_attr=group_attr)
        except Exception as e:
            ax.set_title(f"{name}\n[error: {e}]", color="red", fontsize=9)
            continue

        ew = ([0.1 + 0.8 * (w - w_lo) / (w_hi - w_lo) for w in weights]
              if w_hi > w_lo else [0.3] * len(weights))
        nx.draw_networkx_edges(G, pos, width=ew, alpha=0.15,
                               edge_color="#aaaaaa", ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes,
                               alpha=0.90, linewidths=0.3, edgecolors="#ffffff", ax=ax)
        label = name.replace("_", " ").title()
        ax.set_title(f"{label}\n({elapsed}s)", color="white", fontsize=10, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if color_attr and cmap:
        present = {str(d.get(color_attr)) for _, d in G.nodes(data=True)}
        patches = [mpatches.Patch(color=c, label=lbl)
                   for lbl, c in cmap.items() if lbl in present]
        fig.legend(handles=patches, title=color_attr, loc="lower center",
                   ncol=min(len(patches), 6), framealpha=0.75, facecolor="#1a1a2e",
                   edgecolor="#555577", labelcolor="white", title_fontsize=10,
                   fontsize=9, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Layout Algorithm Comparison",
                 color="white", fontsize=17, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.06, 1, 0.99])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    return output_path