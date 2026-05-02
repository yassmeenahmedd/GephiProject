"""
Graph Analytics Dashboard — Gephi-Style GUI
============================================
A Gephi-inspired interface with:
  • Live interactive network canvas (pan, zoom, click-to-select nodes)
  • Left sidebar: graph context panel with node inspector
  • Right sidebar: layout + appearance controls
  • Bottom panel: metrics overview bar
  • Tabbed analysis panels for Metrics, Filtering, etc.

Replaces static PNG-based visualization with a real-time matplotlib canvas
embedded in the PyQt6 window.
"""

import sys
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

# ── Matplotlib backend MUST be set before any pyplot import ─────────────────
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QFileDialog, QScrollArea, QFrame, QSplitter, QHeaderView,
    QSizePolicy, QGridLayout, QProgressBar, QStatusBar, QComboBox,
    QCheckBox, QSlider, QGroupBox, QLineEdit, QDoubleSpinBox,
    QSpinBox, QListWidget, QListWidgetItem, QMessageBox, QTextEdit,
    QRadioButton, QButtonGroup, QStackedWidget, QToolBar, QToolButton,
    QSizePolicy, QScrollArea
)
from PyQt6.QtGui import (
    QPixmap, QFont, QColor, QPalette, QIcon, QFontDatabase,
    QPainter, QLinearGradient, QBrush, QAction, QCursor
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QPoint

# ── Backend imports ──────────────────────────────────────────────────────────
try:
    from Data_Loader import (
        load_graph_from_csv, get_graph_summary, print_graph_summary,
        convert_graph, get_node_attribute_values, get_edge_attribute_values,
        list_node_attributes, list_edge_attributes, export_graph_to_csv,
    )
    from Metrics import (
        compute_global_metrics, compute_degree_distribution,
        compute_clustering, compute_path_lengths, compute_node_metrics_table,
        compute_rich_club, export_node_metrics_csv,
        render_metrics_dashboard, render_degree_distribution,
        render_clustering_analysis, render_path_length_analysis,
        render_rich_club,
    )
    from Layout import (
        compute_layout, render_layout, render_all_layouts,
        render_radial_annotated, render_hierarchical_annotated,
        LAYOUTS, LAYOUT_DESCRIPTIONS,
    )
    from Visualizer import (
        visualize_matplotlib, visualize_pyvis,
        visualize_multi_layout, visualize_attribute_distribution,
    )
    from Filtering import (
        compute_centralities, detect_communities,
        filter_by_centrality, filter_by_membership, filter_combined,
        get_filter_stats, run_filtering_pipeline,
        render_filter_comparison, render_centrality_distributions,
        render_centrality_scatter, render_community_membership,
    )
    BACKEND_AVAILABLE = True
except ImportError as _err:
    BACKEND_AVAILABLE = False
    _IMPORT_ERR = str(_err)

try:
    from Community_Detection import (
        ALGORITHMS as CD_ALGORITHMS,
        run_community_detection, evaluate_partition,
        partition_stats, partition_to_sets,
        render_community_graph,
        render_community_size_dist, render_comparison_table_img,
        run_all_algorithms,
    )
    CD_AVAILABLE = True
except ImportError as _cde:
    CD_AVAILABLE = False
    _CD_ERR = str(_cde)

try:
    from Link_Analysis import (
        compute_all_link_metrics, build_metrics_table,
        render_link_analysis_dashboard, render_centrality_comparison,
        render_hits_chart, get_top_nodes,
    )
    LA_AVAILABLE = True
except ImportError as _lae:
    LA_AVAILABLE = False
    _LA_ERR = str(_lae)


# ─────────────────────────────────────────────────────────────────────────────
#  Theme tokens
# ─────────────────────────────────────────────────────────────────────────────

BG           = "#0a0a12"
PANEL        = "#0f0f1c"
CARD         = "#141428"
BORDER       = "#1e1e38"
CYAN         = "#00d4ff"
CYAN_DIM     = "#00a8cc"
PURPLE       = "#7c3aed"
TEXT         = "#e0e0f0"
MUTED        = "#5a5a7a"
SUCCESS      = "#00e676"
WARNING      = "#ffb300"
DANGER       = "#ff5252"
PINK         = "#f472b6"

# Canvas background
CANVAS_BG    = "#060610"

STYLESHEET = f"""
* {{ box-sizing: border-box; }}
QMainWindow, QWidget {{
    background:{BG}; color:{TEXT};
    font-family:'Segoe UI','SF Pro Display','Helvetica Neue',sans-serif;
    font-size:12px;
}}
QSplitter::handle {{ background:{BORDER}; }}
QTabWidget::pane {{ border:1px solid {BORDER}; background:{PANEL}; border-top:none; }}
QTabBar::tab {{
    background:{BG}; color:{MUTED}; padding:8px 16px;
    border:none; border-bottom:2px solid transparent;
    font-size:10px; font-weight:600; letter-spacing:0.8px; text-transform:uppercase;
}}
QTabBar::tab:selected {{ color:{CYAN}; border-bottom:2px solid {CYAN}; background:{PANEL}; }}
QTabBar::tab:hover:!selected {{ color:{TEXT}; background:{PANEL}; }}
QPushButton#primary {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00b4d8,stop:1 #0077b6);
    color:#fff; border:none; border-radius:5px; padding:7px 18px;
    font-size:11px; font-weight:600;
}}
QPushButton#primary:hover {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {CYAN},stop:1 #0096c7); }}
QPushButton#secondary {{
    background:transparent; color:{CYAN}; border:1px solid {CYAN_DIM};
    border-radius:5px; padding:5px 12px; font-size:11px; font-weight:500;
}}
QPushButton#secondary:hover {{ background:{CYAN}18; border-color:{CYAN}; }}
QPushButton#tool {{
    background:{CARD}; color:{TEXT}; border:1px solid {BORDER};
    border-radius:4px; padding:4px 10px; font-size:10px; font-weight:500;
}}
QPushButton#tool:hover {{ background:{BORDER}; color:{CYAN}; }}
QPushButton#tool:checked {{ background:{CYAN}28; color:{CYAN}; border-color:{CYAN}; }}
QScrollBar:vertical {{ background:{BG}; width:5px; border-radius:2px; }}
QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:2px; min-height:20px; }}
QScrollBar::handle:vertical:hover {{ background:{CYAN_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:{BG}; height:5px; border-radius:2px; }}
QScrollBar::handle:horizontal {{ background:{BORDER}; border-radius:2px; min-width:20px; }}
QScrollBar::handle:horizontal:hover {{ background:{CYAN_DIM}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QTableWidget {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:6px;
    gridline-color:{BORDER}; selection-background-color:{CYAN}28;
    font-family:'JetBrains Mono','Cascadia Code','Consolas',monospace;
    font-size:10px; outline:none;
}}
QTableWidget::item {{ padding:4px 8px; border-bottom:1px solid {BORDER}; }}
QTableWidget::item:selected {{ background:{CYAN}22; color:{CYAN}; }}
QHeaderView::section {{
    background:{PANEL}; color:{MUTED}; padding:6px 8px; border:none;
    border-bottom:1px solid {BORDER}; font-size:9px; font-weight:700;
    letter-spacing:0.8px; text-transform:uppercase;
}}
QStatusBar {{ background:{BG}; color:{MUTED}; border-top:1px solid {BORDER}; font-size:10px; }}
QProgressBar {{ background:{BORDER}; border:none; border-radius:2px; height:2px; }}
QProgressBar::chunk {{ background:{CYAN}; border-radius:2px; }}
QComboBox {{
    background:{CARD}; color:{TEXT}; border:1px solid {BORDER};
    border-radius:4px; padding:4px 8px; font-size:11px;
}}
QComboBox::drop-down {{ border:none; width:18px; }}
QComboBox:hover {{ border-color:{CYAN_DIM}; }}
QComboBox QAbstractItemView {{
    background:{CARD}; color:{TEXT}; border:1px solid {BORDER};
    selection-background-color:{CYAN}30;
}}
QCheckBox {{ color:{TEXT}; font-size:11px; spacing:6px; }}
QCheckBox::indicator {{
    width:13px; height:13px; border-radius:3px;
    border:1px solid {BORDER}; background:{CARD};
}}
QCheckBox::indicator:checked {{ background:{CYAN}; border-color:{CYAN}; }}
QSlider::groove:horizontal {{ height:3px; background:{BORDER}; border-radius:2px; }}
QSlider::handle:horizontal {{
    background:{CYAN}; width:13px; height:13px; margin:-5px 0; border-radius:6px;
}}
QSlider::sub-page:horizontal {{ background:{CYAN_DIM}; border-radius:2px; }}
QGroupBox {{
    color:{MUTED}; border:1px solid {BORDER}; border-radius:7px;
    margin-top:8px; font-size:9px; font-weight:700;
    letter-spacing:1px; padding:10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:10px; top:0;
    padding:0 5px; color:{MUTED};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background:{CARD}; color:{TEXT}; border:1px solid {BORDER};
    border-radius:4px; padding:4px 8px; font-size:11px;
}}
QLineEdit:focus, QSpinBox:focus {{ border-color:{CYAN_DIM}; }}
QSpinBox::up-button, QSpinBox::down-button {{ background:{PANEL}; border:none; width:14px; }}
QTextEdit {{
    background:{CARD}; color:{MUTED}; border:1px solid {BORDER};
    border-radius:5px; font-family:'JetBrains Mono','Consolas',monospace;
    font-size:10px; padding:6px;
}}
QFrame#sidebar {{
    background:{PANEL}; border-right:1px solid {BORDER};
}}
QFrame#sidebar_right {{
    background:{PANEL}; border-left:1px solid {BORDER};
}}
QFrame#card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
}}
QFrame#metric_bar {{
    background:{PANEL}; border-top:1px solid {BORDER};
}}
QLabel#node_id {{
    font-family:'JetBrains Mono','Consolas',monospace;
    font-size:13px; font-weight:700; color:{CYAN};
}}
QLabel#section_lbl {{
    color:{MUTED}; font-size:9px; font-weight:700; letter-spacing:1.5px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Color palettes
# ─────────────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "1A": "#e6194b", "1B": "#f58231",
    "2A": "#3cb44b", "2B": "#bfef45",
    "3A": "#4363d8", "3B": "#42d4f4",
    "4A": "#911eb4", "4B": "#f032e6",
    "5A": "#9A6324", "5B": "#C8860A",
    "Teachers": "#2ec4b6",
}
GENDER_COLORS = {"M": "#4363d8", "F": "#e6194b", "Unknown": "#888"}
QUALITATIVE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#9A6324",
]

def _color_for_attr(G, attr):
    if attr == "Class":  return CLASS_COLORS
    if attr == "Gender": return GENDER_COLORS
    if attr is None:     return {}
    vals = sorted({str(d.get(attr,"?")) for _,d in G.nodes(data=True)})
    return {v: QUALITATIVE[i % len(QUALITATIVE)] for i,v in enumerate(vals)}


# ─────────────────────────────────────────────────────────────────────────────
#  Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class Worker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str, int)

    def __init__(self, task: str, kwargs: dict):
        super().__init__()
        self.task   = task
        self.kwargs = kwargs

    def run(self):
        try:
            result = {}
            k = self.kwargs
            t = self.task

            if t == "load":
                self.progress.emit("Loading CSV files …", 20)
                G = load_graph_from_csv(
                    k["nodes"], k["edges"],
                    directed=k.get("directed", False),
                    aggregate_duplicate_edges=k.get("aggregate", True),
                )
                self.progress.emit("Computing layout …", 60)
                pos = nx.spring_layout(G, seed=42, k=1.5/math.sqrt(max(G.number_of_nodes(),1)))
                self.progress.emit("Summary …", 85)
                summary = get_graph_summary(G)
                self.progress.emit("Done", 100)
                result = {"G": G, "pos": pos, "summary": summary}

            elif t == "layout":
                G   = k["G"]
                name = k["name"]
                seed = k.get("seed", 42)
                self.progress.emit(f"Computing {name} layout …", 30)
                if BACKEND_AVAILABLE:
                    pos = compute_layout(G, name, seed=seed)
                else:
                    pos = nx.spring_layout(G, seed=seed)
                self.progress.emit("Done", 100)
                result = {"pos": pos}

            elif t == "metrics":
                G = k["G"]
                self.progress.emit("Global metrics …", 10)
                gm    = compute_global_metrics(G)
                self.progress.emit("Degree dist …", 25)
                dd    = compute_degree_distribution(G)
                self.progress.emit("Clustering …", 40)
                cc    = compute_clustering(G)
                self.progress.emit("Paths …", 55)
                pl    = compute_path_lengths(G)
                self.progress.emit("Rich-club …", 68)
                rc    = compute_rich_club(G)
                self.progress.emit("Node table …", 78)
                table = compute_node_metrics_table(G)
                self.progress.emit("Rendering …", 85)
                render_metrics_dashboard(G, gm, dd, cc, pl, "_dash.png")
                render_degree_distribution(dd, "_deg.png")
                render_clustering_analysis(G, cc, "_clust.png")
                render_path_length_analysis(pl, "_paths.png")
                render_rich_club(rc, "_rc.png")
                self.progress.emit("Done", 100)
                result = {
                    "gm": gm, "table": table,
                    "images": {
                        "dashboard":  "_dash.png",
                        "degree":     "_deg.png",
                        "clustering": "_clust.png",
                        "paths":      "_paths.png",
                        "rich_club":  "_rc.png",
                    }
                }

            elif t == "filter":
                G = k["G"]
                self.progress.emit("Centralities …", 15)
                cents = compute_centralities(G)
                self.progress.emit("Communities …", 30)
                comms = detect_communities(G)
                mode  = k.get("mode","centrality")
                self.progress.emit(f"Filter ({mode}) …", 50)
                if mode == "centrality":
                    Gf = filter_by_centrality(
                        G, cents,
                        degree_range=k.get("deg_range",(0,1)),
                        betweenness_range=k.get("bet_range",(0,1)),
                        closeness_range=k.get("clo_range",(0,1)),
                    )
                elif mode == "membership":
                    Gf = filter_by_membership(
                        G, comms,
                        community_ids=k.get("community_ids"),
                        classes=k.get("classes"),
                    )
                else:
                    Gf = filter_combined(
                        G, cents, comms,
                        degree_range=k.get("deg_range",(0,1)),
                        betweenness_range=k.get("bet_range",(0,1)),
                        closeness_range=k.get("clo_range",(0,1)),
                        community_ids=k.get("community_ids"),
                        classes=k.get("classes"),
                        operator=k.get("operator","AND"),
                    )
                stats = get_filter_stats(G, Gf, cents, comms)
                self.progress.emit("Rendering filter charts …", 70)
                render_filter_comparison(G, Gf, "_fcomp.png", centralities=cents, communities=comms)
                render_centrality_distributions(G, cents, Gf, "_fdist.png")
                render_centrality_scatter(G, cents, Gf, "_fscatter.png")
                render_community_membership(G, comms, Gf, "_fcomm.png")
                self.progress.emit("Done", 100)
                result = {
                    "Gf": Gf, "stats": stats,
                    "centralities": cents, "communities": comms,
                    "images": {
                        "compare":   "_fcomp.png",
                        "cent_dist": "_fdist.png",
                        "scatter":   "_fscatter.png",
                        "community": "_fcomm.png",
                    }
                }


            elif t == "community":
                G = k["G"]
                algo = k.get("algorithm", "Vertex-Subgraph")
                self.progress.emit(f"Running {algo} …", 20)
                partition = run_community_detection(G, algo, **{key: val for key, val in k.items() if key not in ["G", "pos", "algorithm", "gt_attr"]})
                self.progress.emit("Evaluating …", 55)
                metrics_eval = evaluate_partition(G, partition,
                                                   ground_truth_attr=k.get("gt_attr","Class"))
                pstats = partition_stats(partition)
                self.progress.emit("Rendering …", 75)
                render_community_graph(G, partition, k.get("pos"),
                                        "_cd_graph.png", title=f"{algo} Communities")
                render_community_size_dist(partition, "_cd_sizes.png",
                                            title=f"{algo} — Community Sizes")
                self.progress.emit("Done", 100)
                result = {
                    "partition": partition,
                    "eval":      metrics_eval,
                    "stats":     pstats,
                    "images": {
                        "graph":  "_cd_graph.png",
                        "sizes":  "_cd_sizes.png",
                    }
                }

            elif t == "community_compare":
                import time
                G = k["G"]
                algos = k.get("algos", list(CD_ALGORITHMS.keys()))
                k_val = k.get("resolution", 1.0)
                n_comm = k.get("n_communities", 5)
                results = run_all_algorithms(G, resolution=k_val, n_communities=n_comm, seed=42)
                self.progress.emit("Rendering table …", 85)
                render_comparison_table_img(results, "_cd_compare.png")
                self.progress.emit("Done", 100)
                result = {"results": results, "compare_img": "_cd_compare.png"}

            elif t == "link_analysis":
                G = k["G"]
                self.progress.emit("Computing link metrics …", 20)
                metrics_la = compute_all_link_metrics(G)
                self.progress.emit("Building table …", 50)
                table_la   = build_metrics_table(G, metrics_la, top_n=0)
                self.progress.emit("Rendering dashboard …", 65)
                render_link_analysis_dashboard(G, metrics_la, k.get("pos"),
                                                "_la_dash.png",
                                                color_attr=k.get("color_attr","Class"))
                render_centrality_comparison(G, metrics_la, "_la_compare.png")
                hubs = metrics_la.get("hubs", {})
                auth = metrics_la.get("authorities", {})
                render_hits_chart(hubs, auth, "_la_hits.png")
                self.progress.emit("Done", 100)
                result = {
                    "metrics":    metrics_la,
                    "table":      table_la,
                    "images": {
                        "dashboard": "_la_dash.png",
                        "compare":   "_la_compare.png",
                        "hits":      "_la_hits.png",
                    }
                }

            self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
#  Live network canvas (Gephi-style)
# ─────────────────────────────────────────────────────────────────────────────

class NetworkCanvas(FigureCanvas):
    """
    Interactive matplotlib canvas that mimics Gephi's network view.
    Supports: pan (drag), zoom (scroll), node selection (click),
    dynamic color/size by attribute, edge weight encoding.
    """
    node_selected = pyqtSignal(object, dict)   # (node_id, attr_dict)
    node_deselected = pyqtSignal()

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=CANVAS_BG, tight_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(CANVAS_BG)
        self.ax.set_aspect("equal")
        self._clear_axes()

        # Graph state
        self.G: Optional[nx.Graph] = None
        self.pos: Dict = {}
        self.color_attr: Optional[str] = "Class"
        self.size_attr: str = "degree"
        self.show_labels: bool = False
        self.show_edges: bool = True
        self.edge_alpha: float = 0.25
        self.node_scale: float = 1.0
        self.selected_node = None
        self.highlighted_nodes: set = set()
        self.search_node = None

        # Interaction state
        self._drag_start = None
        self._xlim0 = None
        self._ylim0 = None
        self._is_panning = False

        # Connect events
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("scroll_event",         self._on_scroll)

    # ── Public API ──────────────────────────────────────────────────────────

    def set_graph(self, G: nx.Graph, pos: Dict):
        self.G   = G
        self.pos = pos
        self.selected_node = None
        self.highlighted_nodes = set()
        self.search_node = None
        self._draw()

    def set_layout(self, pos: Dict):
        self.pos = pos
        self._animate_transition(pos)

    def set_color_attr(self, attr: Optional[str]):
        self.color_attr = attr
        self._draw()

    def set_show_labels(self, v: bool):
        self.show_labels = v
        self._draw()

    def set_show_edges(self, v: bool):
        self.show_edges = v
        self._draw()

    def set_edge_alpha(self, v: float):
        self.edge_alpha = v
        self._draw()

    def set_node_scale(self, v: float):
        self.node_scale = v
        self._draw()

    def highlight_neighbors(self, node):
        if node is None or self.G is None:
            self.highlighted_nodes = set()
        else:
            self.highlighted_nodes = set(self.G.neighbors(node))
        self._draw()

    def search_and_highlight(self, node_id):
        self.search_node = node_id
        self._draw()

    def reset_view(self):
        if not self.pos:
            return
        xs = [p[0] for p in self.pos.values()]
        ys = [p[1] for p in self.pos.values()]
        pad_x = (max(xs)-min(xs))*0.12 or 0.2
        pad_y = (max(ys)-min(ys))*0.12 or 0.2
        self.ax.set_xlim(min(xs)-pad_x, max(xs)+pad_x)
        self.ax.set_ylim(min(ys)-pad_y, max(ys)+pad_y)
        self.draw_idle()

    # ── Drawing ─────────────────────────────────────────────────────────────

    def _clear_axes(self):
        self.ax.cla()
        self.ax.set_facecolor(CANVAS_BG)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)

    def _draw(self):
        if self.G is None or not self.pos:
            self._clear_axes()
            self.ax.text(0.5, 0.5, "Load a graph to begin",
                        ha="center", va="center", color=MUTED,
                        fontsize=14, transform=self.ax.transAxes,
                        fontfamily="monospace")
            self.draw_idle()
            return

        # Save view limits
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            had_limits = xlim != (0.0, 1.0)
        except Exception:
            had_limits = False

        self._clear_axes()

        G   = self.G
        pos = self.pos
        N   = G.number_of_nodes()

        # ── Build color map ─────────────────────────────────────────────
        cmap = _color_for_attr(G, self.color_attr)
        fallback = "#4a90d9"

        node_list  = list(G.nodes())
        node_index = {n: i for i, n in enumerate(node_list)}

        colors = []
        for n in node_list:
            d = G.nodes[n]
            val = str(d.get(self.color_attr, "?")) if self.color_attr else "?"
            colors.append(cmap.get(val, fallback))

        # ── Node sizes (degree-scaled) ───────────────────────────────────
        degs = [G.degree(n) for n in node_list]
        mn_d, mx_d = min(degs), max(degs)
        base_lo = max(12, 200 // max(N, 1))
        base_hi = max(60, 800 // max(N, 1))
        if mx_d == mn_d:
            sizes = [(base_lo + base_hi) / 2 * self.node_scale] * N
        else:
            sizes = [
                (base_lo + (base_hi - base_lo) * (d - mn_d) / (mx_d - mn_d)) * self.node_scale
                for d in degs
            ]

        # ── Draw edges ───────────────────────────────────────────────────
        if self.show_edges and G.number_of_edges() > 0:
            # Batch draw with LineCollection for speed
            segments = []
            ec_list  = []
            ea_list  = []
            for u, v, edata in G.edges(data=True):
                if u not in pos or v not in pos:
                    continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                segments.append([(x0, y0), (x1, y1)])

                # Highlight edges connecting selected node
                if self.selected_node in (u, v):
                    ec_list.append(CYAN)
                    ea_list.append(0.7)
                elif self.highlighted_nodes and (u in self.highlighted_nodes or v in self.highlighted_nodes):
                    ec_list.append("#ffb300")
                    ea_list.append(0.5)
                else:
                    ec_list.append("#2a2a4a")
                    ea_list.append(self.edge_alpha)

            if segments:
                lc = LineCollection(segments, colors=ec_list, linewidths=0.6,
                                    alpha=None, zorder=1)
                # Set per-edge alpha manually
                lc.set_alpha(None)
                rgba = []
                for col, alp in zip(ec_list, ea_list):
                    r, g, b = mcolors.to_rgb(col)
                    rgba.append((r, g, b, alp))
                lc.set_color(rgba)
                self.ax.add_collection(lc)

        # ── Draw nodes ───────────────────────────────────────────────────
        xs = [pos[n][0] for n in node_list if n in pos]
        ys = [pos[n][1] for n in node_list if n in pos]
        valid_colors = [colors[i] for i, n in enumerate(node_list) if n in pos]
        valid_sizes  = [sizes[i]  for i, n in enumerate(node_list) if n in pos]
        valid_nodes  = [n for n in node_list if n in pos]

        edge_colors = []
        lwidths     = []
        for n in valid_nodes:
            if n == self.selected_node:
                edge_colors.append("#ffffff")
                lwidths.append(2.0)
            elif n in self.highlighted_nodes:
                edge_colors.append("#ffb300")
                lwidths.append(1.5)
            elif n == self.search_node:
                edge_colors.append("#ff4444")
                lwidths.append(2.5)
            else:
                edge_colors.append("#0a0a20")
                lwidths.append(0.5)

        alphas = []
        for n in valid_nodes:
            if self.selected_node is not None:
                if n == self.selected_node or n in self.highlighted_nodes:
                    alphas.append(1.0)
                else:
                    alphas.append(0.25)
            else:
                alphas.append(0.92)

        # Draw in one scatter call with per-node alpha workaround
        # Group by alpha for performance
        groups = {}
        for i, (n, x, y, c, s, ec, lw, a) in enumerate(
            zip(valid_nodes, xs, ys, valid_colors, valid_sizes, edge_colors, lwidths, alphas)
        ):
            key = (ec, round(lw, 1), round(a, 2))
            groups.setdefault(key, {"xs":[],"ys":[],"cs":[],"ss":[]})
            groups[key]["xs"].append(x)
            groups[key]["ys"].append(y)
            groups[key]["cs"].append(c)
            groups[key]["ss"].append(s)

        for (ec, lw, alpha), g in groups.items():
            self.ax.scatter(
                g["xs"], g["ys"], s=g["ss"], c=g["cs"],
                edgecolors=ec, linewidths=lw, alpha=alpha,
                zorder=2
            )

        # ── Labels ───────────────────────────────────────────────────────
        if self.show_labels or self.selected_node is not None:
            show_set = set(valid_nodes) if self.show_labels else set()
            if self.selected_node:
                show_set.add(self.selected_node)
                show_set |= self.highlighted_nodes
            for n in show_set:
                if n not in pos:
                    continue
                x, y = pos[n]
                col = CYAN if n == self.selected_node else (
                    "#ffb300" if n in self.highlighted_nodes else "#ccccee"
                )
                fs = 7 if N > 200 else 8 if N > 100 else 9
                self.ax.text(
                    x, y, str(n),
                    fontsize=fs, color=col, ha="center", va="bottom",
                    fontfamily="monospace",
                    bbox=dict(facecolor=CANVAS_BG, alpha=0.6, pad=1, edgecolor="none"),
                    zorder=4
                )

        # ── Legend ───────────────────────────────────────────────────────
        if cmap and self.color_attr:
            # Only show if ≤ 15 categories
            if len(cmap) <= 15:
                handles = [
                    mpatches.Patch(color=c, label=k)
                    for k, c in cmap.items()
                    if any(
                        str(G.nodes[n].get(self.color_attr,"?")) == k
                        for n in G.nodes()
                    )
                ]
                if handles:
                    leg = self.ax.legend(
                        handles=handles, loc="lower left",
                        framealpha=0.8, facecolor="#10101e",
                        edgecolor=BORDER, labelcolor=TEXT,
                        fontsize=7, title=self.color_attr,
                        title_fontsize=8,
                    )
                    leg.get_title().set_color(CYAN)

        # ── Restore or fit view ──────────────────────────────────────────
        if had_limits and xlim != (0.0, 1.0):
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        else:
            if xs:
                pad_x = (max(xs)-min(xs))*0.1 or 0.3
                pad_y = (max(ys)-min(ys))*0.1 or 0.3
                self.ax.set_xlim(min(xs)-pad_x, max(xs)+pad_x)
                self.ax.set_ylim(min(ys)-pad_y, max(ys)+pad_y)

        self.draw_idle()

    def _animate_transition(self, new_pos: Dict, steps: int = 8):
        """Smooth morph between current and new layout."""
        if not self.pos:
            self.pos = new_pos
            self._draw()
            return

        old_pos = {n: self.pos.get(n, new_pos.get(n, (0,0))) for n in self.G.nodes()}
        for step in range(1, steps+1):
            t = step / steps
            t_smooth = t * t * (3 - 2*t)  # smoothstep
            interp = {
                n: (
                    old_pos[n][0] * (1-t_smooth) + new_pos[n][0] * t_smooth,
                    old_pos[n][1] * (1-t_smooth) + new_pos[n][1] * t_smooth
                )
                for n in self.G.nodes() if n in old_pos and n in new_pos
            }
            self.pos = interp
            self._draw()
            QApplication.processEvents()

        self.pos = new_pos
        self._draw()
        self.reset_view()

    # ── Mouse interaction ────────────────────────────────────────────────────

    def _on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._drag_start = (event.x, event.y, event.xdata, event.ydata)
            self._xlim0 = self.ax.get_xlim()
            self._ylim0 = self.ax.get_ylim()
            self._is_panning = False

    def _on_release(self, event):
        if event.button == 1 and event.inaxes == self.ax and not self._is_panning:
            self._try_select_node(event.xdata, event.ydata)
        self._drag_start = None
        self._is_panning = False

    def _on_motion(self, event):
        if self._drag_start and event.inaxes == self.ax:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            if abs(dx) > 4 or abs(dy) > 4:
                self._is_panning = True
                # Pan in data coordinates
                xd0 = self._drag_start[2]
                yd0 = self._drag_start[3]
                if event.xdata is None or event.ydata is None:
                    return
                delta_x = xd0 - event.xdata
                delta_y = yd0 - event.ydata
                xl = self._xlim0
                yl = self._ylim0
                self.ax.set_xlim(xl[0]+delta_x, xl[1]+delta_x)
                self.ax.set_ylim(yl[0]+delta_y, yl[1]+delta_y)
                self.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.85 if event.button == "up" else 1.15
        xl = list(self.ax.get_xlim())
        yl = list(self.ax.get_ylim())
        cx = event.xdata or (xl[0]+xl[1])/2
        cy = event.ydata or (yl[0]+yl[1])/2
        self.ax.set_xlim(cx + (xl[0]-cx)*factor, cx + (xl[1]-cx)*factor)
        self.ax.set_ylim(cy + (yl[0]-cy)*factor, cy + (yl[1]-cy)*factor)
        self.draw_idle()

    def _try_select_node(self, x, y):
        if self.G is None or x is None or y is None:
            return
        if not self.pos:
            return

        # Find the nearest node within a threshold
        xl = self.ax.get_xlim()
        yl = self.ax.get_ylim()
        span_x = xl[1] - xl[0]
        thresh = span_x * 0.03   # 3% of current view width

        best_node = None
        best_dist = float("inf")
        for n, (nx_, ny_) in self.pos.items():
            d = math.hypot(nx_ - x, ny_ - y)
            if d < best_dist and d < thresh:
                best_dist = d
                best_node = n

        if best_node is not None:
            self.selected_node = best_node
            self.highlight_neighbors(best_node)
            attrs = dict(self.G.nodes[best_node])
            self.node_selected.emit(best_node, attrs)
        else:
            self.selected_node = None
            self.highlighted_nodes = set()
            self._draw()
            self.node_deselected.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  Image viewer (for analysis panels)
# ─────────────────────────────────────────────────────────────────────────────

class ImageViewer(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("No image")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background:transparent;")
        self.setWidget(self._label)
        self._pixmap = None

    def load(self, path: str):
        if path and os.path.exists(path):
            self._pixmap = QPixmap(path)
            self._fit()

    def clear_image(self):
        self._pixmap = None
        self._label.clear()
        self._label.setText("Run analysis to see results")
        self._label.setStyleSheet(f"color:{MUTED}; font-size:13px; background:transparent;")

    def _fit(self):
        if self._pixmap:
            w = max(self.viewport().width()-20, 100)
            scaled = self._pixmap.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation)
            self._label.setPixmap(scaled)
            self._label.setStyleSheet("background:transparent;")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()


# ─────────────────────────────────────────────────────────────────────────────
#  Small reusable widgets
# ─────────────────────────────────────────────────────────────────────────────

class StatChip(QFrame):
    """Compact metric chip for the bottom bar."""
    def __init__(self, label: str, value: str = "—", color: str = CYAN):
        super().__init__()
        self.setObjectName("card")
        self.setFixedHeight(48)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(1)
        self._v = QLabel(value)
        self._v.setStyleSheet(
            f"color:{color}; font-size:14px; font-weight:700; "
            f"font-family:'JetBrains Mono','Consolas',monospace;"
        )
        self._l = QLabel(label.upper())
        self._l.setStyleSheet(f"color:{MUTED}; font-size:8px; letter-spacing:0.8px;")
        lay.addWidget(self._v)
        lay.addWidget(self._l)

    def set_value(self, v: str): self._v.setText(v)


class SectionLabel(QLabel):
    def __init__(self, text: str):
        super().__init__(text.upper())
        self.setObjectName("section_lbl")
        self.setStyleSheet(f"color:{MUTED}; font-size:9px; font-weight:700; "
                           f"letter-spacing:1.5px; padding:6px 0 2px 0;")


class RangeSlider(QWidget):
    def __init__(self, label: str, mn=0.0, mx=1.0):
        super().__init__()
        self._mn = mn; self._mx = mx
        v = QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(2)
        hdr = QHBoxLayout()
        self._lbl  = QLabel(label)
        self._lbl.setStyleSheet(f"color:{TEXT}; font-size:10px;")
        self._val_lbl = QLabel(f"[{mn:.2f}–{mx:.2f}]")
        self._val_lbl.setStyleSheet(f"color:{CYAN}; font-size:9px; "
                                     f"font-family:'JetBrains Mono','Consolas',monospace;")
        hdr.addWidget(self._lbl); hdr.addStretch(); hdr.addWidget(self._val_lbl)
        v.addLayout(hdr)
        row = QHBoxLayout(); row.setSpacing(4)
        self._lo = QSlider(Qt.Orientation.Horizontal)
        self._hi = QSlider(Qt.Orientation.Horizontal)
        for sl in (self._lo, self._hi):
            sl.setMinimum(0); sl.setMaximum(100); sl.setFixedHeight(18)
        self._hi.setValue(100)
        self._lo.valueChanged.connect(self._update)
        self._hi.valueChanged.connect(self._update)
        row.addWidget(self._lo); row.addWidget(self._hi)
        v.addLayout(row)

    def _update(self):
        lo = self._lo.value()/100*(self._mx-self._mn)+self._mn
        hi = self._hi.value()/100*(self._mx-self._mn)+self._mn
        self._val_lbl.setText(f"[{min(lo,hi):.2f}–{max(lo,hi):.2f}]")

    def get_range(self):
        lo = self._lo.value()/100*(self._mx-self._mn)+self._mn
        hi = self._hi.value()/100*(self._mx-self._mn)+self._mn
        return (min(lo,hi), max(lo,hi))


# ─────────────────────────────────────────────────────────────────────────────
#  Left sidebar — Context / Node Inspector
# ─────────────────────────────────────────────────────────────────────────────

class LeftSidebar(QFrame):
    search_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        self.setFixedWidth(220)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(0)

        # Title
        t = QLabel("CONTEXT")
        t.setStyleSheet(f"color:{CYAN}; font-size:10px; font-weight:700; "
                        f"letter-spacing:2px; padding-bottom:8px; "
                        f"border-bottom:1px solid {BORDER};")
        root.addWidget(t)

        # Search
        root.addWidget(SectionLabel("Find Node"))
        sr = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Node ID …")
        self._search_btn = QPushButton("→")
        self._search_btn.setObjectName("tool")
        self._search_btn.setFixedWidth(28)
        self._search_btn.clicked.connect(self._do_search)
        self._search_input.returnPressed.connect(self._do_search)
        sr.addWidget(self._search_input); sr.addWidget(self._search_btn)
        root.addLayout(sr)

        # Graph stats
        root.addWidget(SectionLabel("Graph"))
        self._graph_stats = QTextEdit()
        self._graph_stats.setReadOnly(True)
        self._graph_stats.setMaximumHeight(140)
        self._graph_stats.setPlainText("No graph loaded.")
        root.addWidget(self._graph_stats)

        # Node inspector
        root.addWidget(SectionLabel("Selected Node"))
        self._node_id_lbl = QLabel("—")
        self._node_id_lbl.setObjectName("node_id")
        self._node_id_lbl.setStyleSheet(f"color:{CYAN}; font-size:14px; font-weight:700; "
                                         f"font-family:'JetBrains Mono','Consolas',monospace;")
        root.addWidget(self._node_id_lbl)

        self._node_attrs = QTextEdit()
        self._node_attrs.setReadOnly(True)
        self._node_attrs.setMaximumHeight(160)
        self._node_attrs.setPlainText("Click a node to inspect.")
        root.addWidget(self._node_attrs)

        # Neighbors list
        root.addWidget(SectionLabel("Neighbors"))
        self._neighbors_list = QListWidget()
        self._neighbors_list.setMaximumHeight(120)
        self._neighbors_list.setStyleSheet(
            f"QListWidget {{ background:{CARD}; border:1px solid {BORDER}; "
            f"border-radius:4px; font-size:10px; }}"
        )
        root.addWidget(self._neighbors_list)
        root.addStretch()

    def update_graph_stats(self, summary: dict):
        lines = [
            f"Nodes:      {summary.get('nodes','?')}",
            f"Edges:      {summary.get('edges','?')}",
            f"Type:       {summary.get('type','?')}",
            f"Density:    {summary.get('density','?')}",
            f"Connected:  {'Yes' if summary.get('is_connected') else 'No'}",
            f"Components: {summary.get('components','?')}",
        ]
        self._graph_stats.setPlainText("\n".join(lines))

    def show_node(self, node_id, attrs: dict, neighbors: list):
        self._node_id_lbl.setText(str(node_id))
        lines = [f"{k}: {v}" for k, v in attrs.items()]
        self._node_attrs.setPlainText("\n".join(lines) or "(no attributes)")
        self._neighbors_list.clear()
        for nb in sorted(neighbors, key=str):
            self._neighbors_list.addItem(str(nb))

    def clear_node(self):
        self._node_id_lbl.setText("—")
        self._node_attrs.setPlainText("Click a node to inspect.")
        self._neighbors_list.clear()

    def _do_search(self):
        txt = self._search_input.text().strip()
        if txt:
            self.search_requested.emit(txt)


# ─────────────────────────────────────────────────────────────────────────────
#  Right sidebar — Appearance & Layout controls
# ─────────────────────────────────────────────────────────────────────────────

class RightSidebar(QFrame):
    layout_requested = pyqtSignal(str, int)  # (layout_name, seed)
    appearance_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar_right")
        self.setFixedWidth(220)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")

        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(4)

        t = QLabel("APPEARANCE")
        t.setStyleSheet(f"color:{CYAN}; font-size:10px; font-weight:700; "
                        f"letter-spacing:2px; padding-bottom:8px; "
                        f"border-bottom:1px solid {BORDER};")
        root.addWidget(t)

        # Layout
        root.addWidget(SectionLabel("Layout Algorithm"))
        self._layout_combo = QComboBox()
        layout_names = list(LAYOUTS.keys()) if BACKEND_AVAILABLE else [
            "spring","kamada_kawai","circular","shell","spectral","fruchterman_reingold",
            "radial","hierarchical"
        ]
        self._layout_combo.addItems(layout_names)
        root.addWidget(self._layout_combo)

        seed_row = QHBoxLayout()
        lbl_s = QLabel("Seed:")
        lbl_s.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 9999); self._seed_spin.setValue(42)
        seed_row.addWidget(lbl_s); seed_row.addWidget(self._seed_spin); seed_row.addStretch()
        root.addLayout(seed_row)

        apply_btn = QPushButton("▶  Apply Layout")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self._emit_layout)
        root.addWidget(apply_btn)

        # Color
        root.addWidget(SectionLabel("Color Nodes By"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["Class","Gender","None"])
        self._color_combo.currentTextChanged.connect(self._emit_appearance)
        root.addWidget(self._color_combo)

        # Display options
        root.addWidget(SectionLabel("Display"))
        self._labels_chk = QCheckBox("Show node labels")
        self._edges_chk  = QCheckBox("Show edges")
        self._edges_chk.setChecked(True)
        self._labels_chk.stateChanged.connect(self._emit_appearance)
        self._edges_chk.stateChanged.connect(self._emit_appearance)
        root.addWidget(self._labels_chk)
        root.addWidget(self._edges_chk)

        # Node size
        root.addWidget(SectionLabel("Node Scale"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setMinimum(10); self._size_slider.setMaximum(400)
        self._size_slider.setValue(100)
        self._size_lbl = QLabel("1.0×")
        self._size_lbl.setStyleSheet(f"color:{CYAN}; font-size:10px; font-family:monospace;")
        sl_row = QHBoxLayout()
        sl_row.addWidget(self._size_slider); sl_row.addWidget(self._size_lbl)
        root.addLayout(sl_row)
        self._size_slider.valueChanged.connect(self._on_size_changed)

        # Edge opacity
        root.addWidget(SectionLabel("Edge Opacity"))
        self._edge_slider = QSlider(Qt.Orientation.Horizontal)
        self._edge_slider.setMinimum(0); self._edge_slider.setMaximum(100)
        self._edge_slider.setValue(25)
        self._edge_lbl = QLabel("25%")
        self._edge_lbl.setStyleSheet(f"color:{CYAN}; font-size:10px; font-family:monospace;")
        ea_row = QHBoxLayout()
        ea_row.addWidget(self._edge_slider); ea_row.addWidget(self._edge_lbl)
        root.addLayout(ea_row)
        self._edge_slider.valueChanged.connect(self._on_edge_changed)

        # Reset view
        root.addWidget(SectionLabel("View"))
        reset_btn = QPushButton("⊙  Reset View")
        reset_btn.setObjectName("tool")
        reset_btn.clicked.connect(lambda: self.appearance_changed.emit({"action":"reset_view"}))
        root.addWidget(reset_btn)

        root.addStretch()
        scroll.setWidget(inner)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(scroll)

    def _emit_layout(self):
        self.layout_requested.emit(
            self._layout_combo.currentText(),
            self._seed_spin.value()
        )

    def _emit_appearance(self):
        col = self._color_combo.currentText()
        self.appearance_changed.emit({
            "color_attr":   None if col == "None" else col,
            "show_labels":  self._labels_chk.isChecked(),
            "show_edges":   self._edges_chk.isChecked(),
            "node_scale":   self._size_slider.value()/100,
            "edge_alpha":   self._edge_slider.value()/100,
        })

    def _on_size_changed(self, v):
        self._size_lbl.setText(f"{v/100:.1f}×")
        self._emit_appearance()

    def _on_edge_changed(self, v):
        self._edge_lbl.setText(f"{v}%")
        self._emit_appearance()

    def update_color_options(self, attrs: list):
        current = self._color_combo.currentText()
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        self._color_combo.addItems(attrs + ["None"])
        idx = self._color_combo.findText(current)
        self._color_combo.setCurrentIndex(max(0, idx))
        self._color_combo.blockSignals(False)


# ─────────────────────────────────────────────────────────────────────────────
#  Bottom metric bar
# ─────────────────────────────────────────────────────────────────────────────

class MetricBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("metric_bar")
        self.setFixedHeight(58)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(8)

        self.chips = {
            "nodes":      StatChip("Nodes",        "—",  CYAN),
            "edges":      StatChip("Edges",        "—",  CYAN),
            "density":    StatChip("Density",      "—",  "#7c3aed"),
            "apl":        StatChip("Avg Path Len", "—",  SUCCESS),
            "diameter":   StatChip("Diameter",     "—",  WARNING),
            "clustering": StatChip("Avg Clust.",   "—",  PINK),
            "sw_sigma":   StatChip("Small-World σ","—",  "#fb923c"),
            "components": StatChip("Components",   "—",  "#a3e635"),
        }
        for chip in self.chips.values():
            lay.addWidget(chip)
        lay.addStretch()

    def update_summary(self, s: dict):
        self.chips["nodes"].set_value(str(s.get("nodes","—")))
        self.chips["edges"].set_value(str(s.get("edges","—")))
        self.chips["density"].set_value(str(s.get("density","—")))
        self.chips["components"].set_value(str(s.get("components","—")))

    def update_metrics(self, gm: dict):
        self.chips["apl"].set_value(str(gm.get("avg_path_length","—")))
        self.chips["diameter"].set_value(str(gm.get("diameter","—")))
        self.chips["clustering"].set_value(str(gm.get("avg_clustering","—")))
        self.chips["sw_sigma"].set_value(str(gm.get("sw_sigma","—")))


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis panels (Metrics / Filtering) — shown in bottom tab area
# ─────────────────────────────────────────────────────────────────────────────

class MetricsPanel(QWidget):
    metrics_computed = pyqtSignal(dict)

    def __init__(self, status_cb):
        super().__init__()
        self._status = status_cb
        self._G = None
        self._worker = None
        self._build()

    def set_graph(self, G, _=None):
        self._G = G
        self._run_btn.setEnabled(True)

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(8,8,8,8); root.setSpacing(6)

        ctrl = QHBoxLayout()
        self._run_btn = QPushButton("▶  Compute All Metrics")
        self._run_btn.setObjectName("primary")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._export_btn = QPushButton("⬇  Export CSV")
        self._export_btn.setObjectName("secondary")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180); self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False); self._progress.hide()
        ctrl.addWidget(self._run_btn); ctrl.addWidget(self._export_btn)
        ctrl.addStretch(); ctrl.addWidget(self._progress)
        root.addLayout(ctrl)

        inner_tabs = QTabWidget()
        self._viewers = {}
        for label, key in [
            ("Dashboard","dashboard"), ("Degree","degree"),
            ("Clustering","clustering"), ("Paths","paths"), ("Rich-Club","rich_club")
        ]:
            w = QWidget()
            v = QVBoxLayout(w); v.setContentsMargins(4,4,4,4)
            viewer = ImageViewer(); viewer.clear_image()
            v.addWidget(viewer)
            inner_tabs.addTab(w, label)
            self._viewers[key] = viewer

        # Node table tab
        table_w = QWidget()
        tv = QVBoxLayout(table_w); tv.setContentsMargins(4,4,4,4)
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        tv.addWidget(self._table)
        inner_tabs.addTab(table_w, "Node Table")
        root.addWidget(inner_tabs)
        self._table_data = []

    def _run(self):
        if not self._G: return
        self._run_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        self._worker = Worker("metrics", {"G": self._G})
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res):
        self._run_btn.setEnabled(True)
        self._progress.hide()
        for key, viewer in self._viewers.items():
            viewer.load(res["images"][key])
        self._table_data = res["table"]
        self._fill_table(res["table"])
        self._export_btn.setEnabled(True)
        self.metrics_computed.emit(res["gm"])
        self._status("Metrics computed")

    def _on_error(self, msg):
        self._run_btn.setEnabled(True)
        self._progress.hide()
        self._status(f"Error: {msg[:80]}")
        QMessageBox.critical(self, "Metrics Error", msg)

    def _fill_table(self, data):
        if not data: return
        cols = list(data[0].keys())
        self._table.setRowCount(len(data))
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels([c.replace("_"," ").upper() for c in cols])
        for ri, row in enumerate(data):
            for ci, key in enumerate(cols):
                item = QTableWidgetItem(str(row[key]))
                self._table.setItem(ri, ci, item)
        self._table.resizeColumnsToContents()

    def _export_csv(self):
        if not self._table_data: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Metrics", "node_metrics.csv", "CSV (*.csv)")
        if path:
            export_node_metrics_csv(self._table_data, path)
            self._status(f"Exported: {os.path.basename(path)}")


class FilteringPanel(QWidget):
    def __init__(self, status_cb):
        super().__init__()
        self._status = status_cb
        self._G = None
        self._worker = None
        self._build()

    def set_graph(self, G, _=None):
        self._G = G
        self._run_btn.setEnabled(True)

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(8,8,8,8); root.setSpacing(6)

        # Controls
        ctrl_row = QHBoxLayout(); ctrl_row.setSpacing(12)

        mode_grp = QGroupBox("FILTER MODE")
        mg = QHBoxLayout(mode_grp)
        self._m_cent = QRadioButton("Centrality")
        self._m_memb = QRadioButton("Membership")
        self._m_comb = QRadioButton("Combined")
        self._m_cent.setChecked(True)
        for rb in (self._m_cent, self._m_memb, self._m_comb):
            mg.addWidget(rb)
        ctrl_row.addWidget(mode_grp)

        cent_grp = QGroupBox("CENTRALITY RANGES")
        cg = QVBoxLayout(cent_grp)
        self._deg_sl = RangeSlider("Degree")
        self._bet_sl = RangeSlider("Betweenness")
        self._clo_sl = RangeSlider("Closeness")
        for sl in (self._deg_sl, self._bet_sl, self._clo_sl):
            cg.addWidget(sl)
        ctrl_row.addWidget(cent_grp, stretch=1)

        mem_grp = QGroupBox("MEMBERSHIP")
        memg = QVBoxLayout(mem_grp)
        self._classes_edit = QLineEdit()
        self._classes_edit.setPlaceholderText("Classes e.g. 3A,4A (blank=all)")
        self._comms_edit = QLineEdit()
        self._comms_edit.setPlaceholderText("Community IDs e.g. 0,1 (blank=all)")
        memg.addWidget(QLabel("Classes:")); memg.addWidget(self._classes_edit)
        memg.addWidget(QLabel("Community IDs:")); memg.addWidget(self._comms_edit)
        ctrl_row.addWidget(mem_grp)

        root.addLayout(ctrl_row)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("▶  Apply Filter")
        self._run_btn.setObjectName("primary")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180); self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False); self._progress.hide()
        btn_row.addWidget(self._run_btn); btn_row.addStretch(); btn_row.addWidget(self._progress)
        root.addLayout(btn_row)

        # Stats chips
        stats_row = QHBoxLayout()
        self._fstats = {
            "n_original": StatChip("Original", "—", MUTED),
            "n_filtered": StatChip("Kept",     "—", CYAN),
            "pct_nodes":  StatChip("% Nodes",  "—", SUCCESS),
            "e_filtered": StatChip("Edges",     "—", PURPLE),
            "pct_edges":  StatChip("% Edges",   "—", WARNING),
        }
        for chip in self._fstats.values():
            stats_row.addWidget(chip)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # Image tabs
        tabs = QTabWidget()
        self._fviewers = {}
        for label, key in [
            ("Comparison","compare"),("Centrality Dists","cent_dist"),
            ("Scatter","scatter"),("Communities","community")
        ]:
            w = QWidget()
            v = QVBoxLayout(w); v.setContentsMargins(4,4,4,4)
            viewer = ImageViewer(); viewer.clear_image()
            v.addWidget(viewer)
            tabs.addTab(w, label)
            self._fviewers[key] = viewer
        root.addWidget(tabs)

    def _run(self):
        if not self._G: return
        mode = ("centrality" if self._m_cent.isChecked()
                else "membership" if self._m_memb.isChecked()
                else "combined")
        cls_text = self._classes_edit.text().strip()
        classes = {c.strip() for c in cls_text.split(",") if c.strip()} or None
        comm_text = self._comms_edit.text().strip()
        try:
            community_ids = {int(x.strip()) for x in comm_text.split(",") if x.strip()} or None
        except ValueError:
            community_ids = None
        kwargs = {
            "G": self._G, "mode": mode,
            "deg_range": self._deg_sl.get_range(),
            "bet_range": self._bet_sl.get_range(),
            "clo_range": self._clo_sl.get_range(),
            "classes": classes, "community_ids": community_ids, "operator": "AND",
        }
        self._run_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        self._worker = Worker("filter", kwargs)
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res):
        self._run_btn.setEnabled(True); self._progress.hide()
        s = res["stats"]
        self._fstats["n_original"].set_value(str(s.get("n_original","—")))
        self._fstats["n_filtered"].set_value(str(s.get("n_filtered","—")))
        self._fstats["pct_nodes"].set_value(f"{s.get('pct_nodes','—')}%")
        self._fstats["e_filtered"].set_value(str(s.get("e_filtered","—")))
        self._fstats["pct_edges"].set_value(f"{s.get('pct_edges','—')}%")
        imgs = res["images"]
        for key, viewer in self._fviewers.items():
            viewer.load(imgs[key])
        self._status(f"Filter: {s.get('n_filtered','?')} nodes kept ({s.get('pct_nodes','?')}%)")

    def _on_error(self, msg):
        self._run_btn.setEnabled(True); self._progress.hide()
        self._status(f"Error: {msg[:80]}")
        QMessageBox.critical(self, "Filter Error", msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
#  Community Detection Panel
# ─────────────────────────────────────────────────────────────────────────────

class CommunityPanel(QWidget):
    """Runs community detection algorithms and evaluates them."""

    partition_ready = pyqtSignal(dict)   # emits the partition for canvas coloring

    def __init__(self, status_cb):
        super().__init__()
        self._status  = status_cb
        self._G       = None
        self._pos     = {}
        self._worker  = None
        self._last_partition = {}
        self._build()

    def set_graph(self, G, pos=None):
        self._G   = G
        self._pos = pos or {}
        self._run_btn.setEnabled(True)
        self._cmp_btn.setEnabled(True)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Controls row
        ctrl = QHBoxLayout(); ctrl.setSpacing(10)

        algo_grp = QGroupBox("ALGORITHM")
        ag = QHBoxLayout(algo_grp)
        self._algo_combo = QComboBox()
        algo_names = (list(CD_ALGORITHMS.keys()) if CD_AVAILABLE
                      else ["Vertex-Subgraph","Girvan-Newman","K-Clique"])
        self._algo_combo.addItems(algo_names)
        self._algo_combo.currentTextChanged.connect(self._on_algo_changed)
        ag.addWidget(self._algo_combo)
        ctrl.addWidget(algo_grp)

        k_grp = QGroupBox("Resolution")
        kg = QHBoxLayout(k_grp)
        self._k_spin = QDoubleSpinBox()
        self._k_spin.setMinimum(0.1)
        self._k_spin.setMaximum(5.0)
        self._k_spin.setValue(1.0)
        self._k_spin.setSingleStep(0.1)
        self._k_spin.setToolTip("Resolution parameter for Louvain community detection")
        kg.addWidget(self._k_spin)
        ctrl.addWidget(k_grp)
        self._k_group = k_grp

        eval_grp = QGroupBox("GROUND TRUTH ATTR")
        eg = QHBoxLayout(eval_grp)
        self._gt_combo = QComboBox()
        self._gt_combo.addItems(["Class","Gender","None"])
        eg.addWidget(self._gt_combo)
        ctrl.addWidget(eval_grp)

        self._run_btn = QPushButton("▶  Detect Communities")
        self._run_btn.setObjectName("primary")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_single)

        self._cmp_btn = QPushButton("⊞  Compare All Algorithms")
        self._cmp_btn.setObjectName("secondary")
        self._cmp_btn.setEnabled(False)
        self._cmp_btn.clicked.connect(self._run_compare)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(180); self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False); self._progress.hide()

        ctrl.addWidget(self._run_btn)
        ctrl.addWidget(self._cmp_btn)
        ctrl.addStretch()
        ctrl.addWidget(self._progress)
        root.addLayout(ctrl)

        self._on_algo_changed(self._algo_combo.currentText())

        # Stats chips
        stats_row = QHBoxLayout()
        self._chips = {
            "n_comm":       StatChip("Communities",  "—", CYAN),
            "conductance":  StatChip("Conductance", "—", PINK),
            "intra_density":StatChip("Edge Density", "—", SUCCESS),
            "nmi":          StatChip("NMI",         "—", WARNING),
        }
        for chip in self._chips.values():
            stats_row.addWidget(chip)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # Image tabs
        tabs = QTabWidget()
        self._viewers = {}
        for label, key in [
            ("Graph","graph"),
            ("Sizes","sizes"), ("Compare","compare")
        ]:
            w = QWidget()
            v = QVBoxLayout(w); v.setContentsMargins(4,4,4,4)
            viewer = ImageViewer(); viewer.clear_image()
            v.addWidget(viewer)
            tabs.addTab(w, label)
            self._viewers[key] = viewer

        # Evaluation table tab
        tbl_w = QWidget()
        tv = QVBoxLayout(tbl_w); tv.setContentsMargins(4,4,4,4)
        self._eval_table = QTableWidget()
        self._eval_table.setAlternatingRowColors(True)
        self._eval_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._eval_table.verticalHeader().setVisible(False)
        self._eval_table.setSortingEnabled(True)
        tv.addWidget(self._eval_table)
        tabs.addTab(tbl_w, "Evaluation Table")

        root.addWidget(tabs)

    def _run_single(self):
        if not self._G: return
        gt = self._gt_combo.currentText()
        if gt == "None": gt = "Class"
        self._run_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        kwargs = {
            "G": self._G,
            "pos": self._pos,
            "algorithm": self._algo_combo.currentText(),
            "gt_attr": gt,
        }
        if self._algo_combo.currentText() == "Louvain":
            kwargs["resolution"] = self._k_spin.value()
        self._worker = Worker("community", kwargs)
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_single_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_algo_changed(self, algo):
        self._k_group.setVisible(algo == "Louvain")

    def _on_single_done(self, res):
        self._run_btn.setEnabled(True); self._progress.hide()
        ev = res["eval"]; st = res["stats"]
        self._chips["n_comm"].set_value(str(st.get("n_communities","—")))
        self._chips["conductance"].set_value(str(ev.get("conductance","—")))
        self._chips["intra_density"].set_value(str(ev.get("intra_density","—")))
        self._chips["nmi"].set_value(str(ev.get("nmi","—")))
        for key, viewer in self._viewers.items():
            if key in res["images"]:
                viewer.load(res["images"][key])
        # fill evaluation table
        self._fill_eval_table([{"metric": k, "value": v}
                                for k, v in ev.items()])
        self._last_partition = res["partition"]
        self.partition_ready.emit(res["partition"])
        self._status(f"Communities: {st.get('n_communities','?')}  "
                     f"NMI={ev.get('nmi','?')}")

    def _run_compare(self):
        if not self._G: return
        gt = self._gt_combo.currentText()
        if gt == "None": gt = "Class"
        self._cmp_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        self._worker = Worker("community_compare", {
            "G": self._G,
            "algos": (list(CD_ALGORITHMS.keys()) if CD_AVAILABLE
                      else ["Louvain","Vertex-Subgraph","Girvan-Newman"]),
            "gt_attr": gt,
            "resolution": self._k_spin.value(),  # Pass resolution value for Louvain
            "n_communities": None,  # Let Girvan-Newman find optimal communities
        })
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_compare_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_compare_done(self, res):
        self._cmp_btn.setEnabled(True); self._progress.hide()
        self._viewers["compare"].load(res["compare_img"])
        # Fill eval table with all results
        rows = []
        for r in res["results"]:
            row = {"algorithm": r["algorithm"]}
            row.update(r["metrics"])
            rows.append(row)
        self._fill_eval_table(rows, multi=True)
        self._status("Algorithm comparison complete")

    def _fill_eval_table(self, data, multi=False):
        if not data: return
        cols = list(data[0].keys())
        self._eval_table.setRowCount(len(data))
        self._eval_table.setColumnCount(len(cols))
        self._eval_table.setHorizontalHeaderLabels(
            [c.replace("_"," ").upper() for c in cols])
        numeric_keys = {"conductance","intra_density","nmi","value"}
        for ri, row in enumerate(data):
            for ci, key in enumerate(cols):
                val = row[key]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if key in numeric_keys else
                    Qt.AlignmentFlag.AlignLeft  | Qt.AlignmentFlag.AlignVCenter)
                if key in ("nmi",):
                    item.setForeground(QColor(CYAN))
                self._eval_table.setItem(ri, ci, item)
        self._eval_table.resizeColumnsToContents()

    def _on_error(self, msg):
        self._run_btn.setEnabled(True); self._cmp_btn.setEnabled(True)
        self._progress.hide()
        self._status(f"Error: {msg[:80]}")
        QMessageBox.critical(self, "Community Detection Error", msg)


# ─────────────────────────────────────────────────────────────────────────────
#  Link Analysis Panel
# ─────────────────────────────────────────────────────────────────────────────

class LinkAnalysisPanel(QWidget):
    """PageRank, Betweenness, HITS, Katz, Eigenvector + comparison."""

    def __init__(self, status_cb):
        super().__init__()
        self._status  = status_cb
        self._G       = None
        self._pos     = {}
        self._worker  = None
        self._build()

    def set_graph(self, G, pos=None):
        self._G   = G
        self._pos = pos or {}
        self._run_btn.setEnabled(True)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        ctrl = QHBoxLayout(); ctrl.setSpacing(10)

        col_grp = QGroupBox("COLOR NODES BY")
        cg = QHBoxLayout(col_grp)
        self._color_combo = QComboBox()
        self._color_combo.addItems(["Class","Gender","None"])
        cg.addWidget(self._color_combo)
        ctrl.addWidget(col_grp)

        self._run_btn = QPushButton("▶  Run Link Analysis")
        self._run_btn.setObjectName("primary")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)

        self._export_btn = QPushButton("⬇  Export Table CSV")
        self._export_btn.setObjectName("secondary")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(180); self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False); self._progress.hide()

        ctrl.addWidget(self._run_btn)
        ctrl.addWidget(self._export_btn)
        ctrl.addStretch()
        ctrl.addWidget(self._progress)
        root.addLayout(ctrl)

        # Metric chips
        stats_row = QHBoxLayout()
        self._chips = {
            "top_pr":     StatChip("Top PageRank Node", "—",  CYAN),
            "top_bet":    StatChip("Top Betweenness",   "—",  "e87040"),
            "top_eigen":  StatChip("Top Eigenvector",   "—",  SUCCESS),
        }
        for chip in self._chips.values():
            stats_row.addWidget(chip)
        stats_row.addStretch()
        root.addLayout(stats_row)

        # Image tabs
        tabs = QTabWidget()
        self._viewers = {}
        for label, key in [
            ("Dashboard","dashboard"),
            ("Centrality Comparison","compare"),
            ("HITS Hubs & Auth","hits"),
        ]:
            w = QWidget()
            v = QVBoxLayout(w); v.setContentsMargins(4,4,4,4)
            viewer = ImageViewer(); viewer.clear_image()
            v.addWidget(viewer)
            tabs.addTab(w, label)
            self._viewers[key] = viewer

        # Node ranking table tab
        tbl_w = QWidget()
        tv = QVBoxLayout(tbl_w); tv.setContentsMargins(4,4,4,4)
        self._tbl = QTableWidget()
        self._tbl.setAlternatingRowColors(True)
        self._tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setSortingEnabled(True)
        tv.addWidget(self._tbl)
        tabs.addTab(tbl_w, "Node Rankings")
        root.addWidget(tabs)
        self._table_data = []

    def _run(self):
        if not self._G: return
        ca = self._color_combo.currentText()
        if ca == "None": ca = "Class"
        self._run_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        self._worker = Worker("link_analysis", {
            "G": self._G, "pos": self._pos, "color_attr": ca,
        })
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, res):
        self._run_btn.setEnabled(True); self._progress.hide()
        metrics = res["metrics"]
        pr  = metrics.get("pagerank", {})
        bt  = metrics.get("betweenness", {})
        ev  = metrics.get("eigenvector", {})
        if pr:
            top_pr = max(pr, key=pr.get)
            self._chips["top_pr"].set_value(
                f"{top_pr} ({pr[top_pr]:.4f})")
        if bt:
            top_bt = max(bt, key=bt.get)
            self._chips["top_bet"].set_value(
                f"{top_bt} ({bt[top_bt]:.4f})")
        if ev:
            top_ev = max(ev, key=ev.get)
            self._chips["top_eigen"].set_value(
                f"{top_ev} ({ev[top_ev]:.4f})")
        imgs = res["images"]
        for key, viewer in self._viewers.items():
            if key in imgs: viewer.load(imgs[key])
        self._table_data = res["table"]
        self._fill_table(res["table"])
        self._export_btn.setEnabled(True)
        self._status("Link analysis complete")

    def _fill_table(self, data):
        if not data: return
        cols = list(data[0].keys())
        self._tbl.setRowCount(len(data))
        self._tbl.setColumnCount(len(cols))
        self._tbl.setHorizontalHeaderLabels([c.replace("_"," ").upper() for c in cols])
        num_keys = {"pagerank","betweenness","eigenvector","katz",
                    "closeness","hubs","authorities","degree"}
        for ri, row in enumerate(data):
            for ci, key in enumerate(cols):
                val = row[key]
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    if key in num_keys else
                    Qt.AlignmentFlag.AlignLeft  | Qt.AlignmentFlag.AlignVCenter)
                if key == "pagerank":    item.setForeground(QColor(CYAN))
                elif key == "betweenness": item.setForeground(QColor(WARNING))
                self._tbl.setItem(ri, ci, item)
        self._tbl.resizeColumnsToContents()

    def _export_csv(self):
        if not self._table_data: return
        import csv
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Link Analysis", "link_analysis.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._table_data[0].keys())
                writer.writeheader()
                writer.writerows(self._table_data)
            self._status(f"Exported: {os.path.basename(path)}")

    def _on_error(self, msg):
        self._run_btn.setEnabled(True); self._progress.hide()
        self._status(f"Error: {msg[:80]}")
        QMessageBox.critical(self, "Link Analysis Error", msg)


class GephiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graph Analytics — Gephi-Style")
        self.setMinimumSize(1440, 900)
        self.setStyleSheet(STYLESHEET)
        self._G   = None
        self._pos = {}
        self._summary = {}
        self._worker  = None
        self._build()

    def _build(self):
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0,0,0,0)
        root_lay.setSpacing(0)
        self.setCentralWidget(root)

        # ── Top toolbar ──────────────────────────────────────────────────
        root_lay.addWidget(self._build_toolbar())

        # ── Global progress bar ──────────────────────────────────────────
        self._global_prog = QProgressBar()
        self._global_prog.setFixedHeight(2)
        self._global_prog.setTextVisible(False)
        self._global_prog.hide()
        root_lay.addWidget(self._global_prog)

        # ── Main splitter: left sidebar | center | right sidebar ─────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)

        self._left  = LeftSidebar()
        self._right = RightSidebar()

        # Center: canvas on top, analysis tabs below
        center_widget = QWidget()
        center_lay = QVBoxLayout(center_widget)
        center_lay.setContentsMargins(0,0,0,0)
        center_lay.setSpacing(0)

        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(2)

        # Canvas
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(f"background:{CANVAS_BG};")
        cf_lay = QVBoxLayout(canvas_frame)
        cf_lay.setContentsMargins(0,0,0,0)
        self._canvas = NetworkCanvas()
        cf_lay.addWidget(self._canvas)

        # Canvas overlay toolbar
        cf_lay.addWidget(self._build_canvas_toolbar())

        # Analysis tabs
        self._analysis_tabs = QTabWidget()
        self._metrics_panel  = MetricsPanel(self._status)
        self._filter_panel   = FilteringPanel(self._status)
        self._analysis_tabs.addTab(self._metrics_panel, "  Metrics  ")
        self._analysis_tabs.addTab(self._filter_panel,  "  Filtering  ")
        self._community_panel = CommunityPanel(self._status)
        self._link_panel      = LinkAnalysisPanel(self._status)
        self._analysis_tabs.addTab(self._community_panel, "  Communities  ")
        self._analysis_tabs.addTab(self._link_panel,      "  Link Analysis  ")
        self._analysis_tabs.setFixedHeight(380)

        v_splitter.addWidget(canvas_frame)
        v_splitter.addWidget(self._analysis_tabs)
        v_splitter.setSizes([560, 380])
        center_lay.addWidget(v_splitter)

        main_splitter.addWidget(self._left)
        main_splitter.addWidget(center_widget)
        main_splitter.addWidget(self._right)
        main_splitter.setSizes([220, 1000, 220])

        root_lay.addWidget(main_splitter, stretch=1)

        # ── Metric bar ───────────────────────────────────────────────────
        self._metric_bar = MetricBar()
        root_lay.addWidget(self._metric_bar)

        # ── Status bar ───────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self._statusbar.setFixedHeight(24)
        self.setStatusBar(self._statusbar)
        self._status("Load CSV files to begin  ·  Scroll to zoom  ·  Drag to pan  ·  Click nodes to inspect")

        # ── Connect signals ──────────────────────────────────────────────
        self._canvas.node_selected.connect(self._on_node_selected)
        self._canvas.node_deselected.connect(self._left.clear_node)
        self._left.search_requested.connect(self._on_search)
        self._right.layout_requested.connect(self._on_layout_requested)
        self._right.appearance_changed.connect(self._on_appearance_changed)
        self._metrics_panel.metrics_computed.connect(self._metric_bar.update_metrics)

        if not BACKEND_AVAILABLE:
            self._status(f"⚠  Backend unavailable: {_IMPORT_ERR[:100]}")

    def _build_toolbar(self):
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        title = QLabel("GRAPH ANALYTICS")
        title.setStyleSheet(
            f"color:{TEXT}; font-size:13px; font-weight:700; "
            f"letter-spacing:3px; font-family:'JetBrains Mono','Consolas',monospace;"
        )
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{CYAN}; font-size:8px; margin:0 8px;")
        sub = QLabel("Gephi-Style Social Network Visualizer")
        sub.setStyleSheet(f"color:{MUTED}; font-size:11px;")

        # Load button
        self._load_btn = QPushButton("📂  Load Graph")
        self._load_btn.setObjectName("primary")
        self._load_btn.clicked.connect(self._load_dialog)

        self._directed_chk  = QCheckBox("Directed")
        self._aggregate_chk = QCheckBox("Aggregate edges")
        self._aggregate_chk.setChecked(True)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(200); self._progress.setFixedHeight(2)
        self._progress.setTextVisible(False); self._progress.hide()

        self._graph_info = QLabel("No graph loaded")
        self._graph_info.setStyleSheet(
            f"color:{MUTED}; font-size:10px; "
            f"font-family:'JetBrains Mono','Consolas',monospace;"
        )

        lay.addWidget(title)
        lay.addWidget(dot)
        lay.addWidget(sub)
        lay.addStretch()
        lay.addWidget(self._directed_chk)
        lay.addWidget(self._aggregate_chk)
        lay.addWidget(self._graph_info)
        lay.addWidget(self._load_btn)
        lay.addWidget(self._progress)
        return bar

    def _build_canvas_toolbar(self):
        """Small floating toolbar just below the canvas."""
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet(f"background:{PANEL}; border-top:1px solid {BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 2, 12, 2)
        lay.setSpacing(6)

        def mk(label, tip, slot):
            b = QPushButton(label)
            b.setObjectName("tool")
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        lay.addWidget(mk("⊙ Fit", "Reset view to fit all nodes", self._canvas.reset_view))
        lay.addWidget(mk("+ Zoom", "Zoom in", lambda: self._zoom(0.7)))
        lay.addWidget(mk("– Zoom", "Zoom out", lambda: self._zoom(1.4)))

        lay.addWidget(QLabel("|", ), 0, Qt.AlignmentFlag.AlignVCenter)

        # Export canvas
        lay.addWidget(mk("💾 Export PNG", "Save current canvas view", self._export_canvas))
        lay.addWidget(mk("🌐 Interactive HTML", "Open PyVis in browser", self._export_pyvis))
        lay.addStretch()

        self._canvas_info = QLabel("")
        self._canvas_info.setStyleSheet(f"color:{MUTED}; font-size:9px; font-family:monospace;")
        lay.addWidget(self._canvas_info)
        return bar

    def _zoom(self, factor):
        xl = list(self._canvas.ax.get_xlim())
        yl = list(self._canvas.ax.get_ylim())
        cx = (xl[0]+xl[1])/2; cy = (yl[0]+yl[1])/2
        self._canvas.ax.set_xlim(cx+(xl[0]-cx)*factor, cx+(xl[1]-cx)*factor)
        self._canvas.ax.set_ylim(cy+(yl[0]-cy)*factor, cy+(yl[1]-cy)*factor)
        self._canvas.draw_idle()

    # ── Load ─────────────────────────────────────────────────────────────────

    def _load_dialog(self):
        nodes_path, _ = QFileDialog.getOpenFileName(
            self, "Select Nodes CSV", "", "CSV Files (*.csv)")
        if not nodes_path:
            return
        edges_path, _ = QFileDialog.getOpenFileName(
            self, "Select Edges CSV", "", "CSV Files (*.csv)")
        if not edges_path:
            return
        self._do_load(nodes_path, edges_path)

    def _do_load(self, nodes_path: str, edges_path: str):
        self._load_btn.setEnabled(False)
        self._progress.show(); self._progress.setValue(0)
        self._worker = Worker("load", {
            "nodes": nodes_path, "edges": edges_path,
            "directed": self._directed_chk.isChecked(),
            "aggregate": self._aggregate_chk.isChecked(),
        })
        self._worker.progress.connect(lambda m,p: (self._progress.setValue(p), self._status(m)))
        self._worker.finished.connect(self._on_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.start()

    def _on_loaded(self, res):
        self._load_btn.setEnabled(True)
        self._progress.hide()
        self._G       = res["G"]
        self._pos     = res["pos"]
        self._summary = res["summary"]

        # Update canvas
        self._canvas.set_graph(self._G, self._pos)

        # Update sidebar
        self._left.update_graph_stats(self._summary)

        # Update metric bar
        self._metric_bar.update_summary(self._summary)

        # Update color attribute options in right sidebar
        attrs = list_node_attributes(self._G) if BACKEND_AVAILABLE else ["Class","Gender"]
        self._right.update_color_options(attrs)

        # Propagate to analysis panels
        self._metrics_panel.set_graph(self._G)
        self._filter_panel.set_graph(self._G)
        self._community_panel.set_graph(self._G, self._pos)
        self._link_panel.set_graph(self._G, self._pos)

        # Header info
        n = self._summary["nodes"]; e = self._summary["edges"]
        self._graph_info.setText(f"● {n} nodes  ·  {e} edges  ·  {self._summary['type']}")
        self._graph_info.setStyleSheet(
            f"color:{SUCCESS}; font-size:10px; "
            f"font-family:'JetBrains Mono','Consolas',monospace;"
        )
        self._canvas_info.setText(f"{n}N  {e}E")
        self._status(f"Graph loaded — {n} nodes, {e} edges")

    def _on_load_error(self, msg):
        self._load_btn.setEnabled(True)
        self._progress.hide()
        self._status(f"Error: {msg[:80]}")
        QMessageBox.critical(self, "Load Error", msg)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _on_layout_requested(self, name: str, seed: int):
        if not self._G:
            return
        self._status(f"Computing {name} layout …")
        self._worker = Worker("layout", {"G": self._G, "name": name, "seed": seed})
        self._worker.finished.connect(lambda res: self._canvas.set_layout(res["pos"]))
        self._worker.error.connect(lambda msg: self._status(f"Layout error: {msg[:60]}"))
        self._worker.start()

    # ── Appearance ───────────────────────────────────────────────────────────

    def _on_appearance_changed(self, opts: dict):
        if "action" in opts:
            if opts["action"] == "reset_view":
                self._canvas.reset_view()
            return
        self._canvas.set_color_attr(opts.get("color_attr", "Class"))
        self._canvas.set_show_labels(opts.get("show_labels", False))
        self._canvas.set_show_edges(opts.get("show_edges", True))
        self._canvas.set_node_scale(opts.get("node_scale", 1.0))
        self._canvas.set_edge_alpha(opts.get("edge_alpha", 0.25))

    # ── Node selection ───────────────────────────────────────────────────────

    def _on_node_selected(self, node_id, attrs: dict):
        if self._G is None:
            return
        neighbors = list(self._G.neighbors(node_id))
        self._left.show_node(node_id, attrs, neighbors)
        self._status(
            f"Node: {node_id}  ·  Degree: {self._G.degree(node_id)}  ·  "
            f"Neighbors: {len(neighbors)}"
        )

    def _on_search(self, query: str):
        if self._G is None:
            return
        # Try exact match first, then string match
        target = None
        for n in self._G.nodes():
            if str(n) == query:
                target = n; break
        if target is None:
            for n in self._G.nodes():
                if query.lower() in str(n).lower():
                    target = n; break
        if target is not None:
            self._canvas.selected_node = target
            self._canvas.search_node = target
            self._canvas.highlight_neighbors(target)
            attrs = dict(self._G.nodes[target])
            neighbors = list(self._G.neighbors(target))
            self._left.show_node(target, attrs, neighbors)
            self._status(f"Found: {target}")
        else:
            self._status(f"Node not found: {query}")

    # ── Export ───────────────────────────────────────────────────────────────

    def _export_canvas(self):
        if not self._G:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Canvas", "network.png", "PNG (*.png)")
        if path:
            self._canvas.fig.savefig(path, dpi=200, bbox_inches="tight",
                                     facecolor=CANVAS_BG)
            self._status(f"Canvas exported: {os.path.basename(path)}")

    def _export_pyvis(self):
        if not self._G:
            return
        out = "network_interactive.html"
        try:
            visualize_pyvis(self._G, out, color_attr="Class",
                            show_labels=True, title="Social Network")
            import webbrowser
            webbrowser.open(os.path.abspath(out))
            self._status(f"Interactive HTML opened: {out}")
        except Exception as e:
            self._status(f"PyVis error: {e}")

    def _status(self, msg: str):
        self._statusbar.showMessage(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,        QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,    QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,          QColor(CARD))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL))
    palette.setColor(QPalette.ColorRole.Text,          QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button,        QColor(PANEL))
    palette.setColor(QPalette.ColorRole.ButtonText,    QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,     QColor(CYAN))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(BG))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)

    win = GephiApp()
    win.show()
    sys.exit(app.exec())