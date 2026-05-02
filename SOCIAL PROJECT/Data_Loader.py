"""
Phase 1: Data Loading Module
Handles CSV ingestion and graph construction for the Social Network Analysis Tool.
Supports both directed and undirected graphs with full node/edge attribute preservation.
"""

import os
import networkx as nx
import pandas as pd
from typing import Optional, Tuple, Dict, Any


# ─────────────────────────────────────────────
#  Core loader
# ─────────────────────────────────────────────

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and BOM characters from column names, then auto-detect case."""
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    return df


def _resolve_col(df: pd.DataFrame, preferred: str) -> str:
    """
    Find the actual column name in df that matches `preferred` case-insensitively.
    Returns the original preferred name if no match found (validation will catch it).
    """
    mapping = {c.lower(): c for c in df.columns}
    return mapping.get(preferred.lower(), preferred)


def load_graph_from_csv(
    nodes_path: str,
    edges_path: str,
    directed: bool = False,
    node_id_col: str = "id",
    source_col: str = "source",
    target_col: str = "target",
    aggregate_duplicate_edges: bool = True,
) -> nx.Graph:
    """
    Load a graph from two CSV files (nodes + edges).

    Parameters
    ----------
    nodes_path               : path to the nodes CSV file
    edges_path               : path to the edges CSV file
    directed                 : True → DiGraph | False → Graph
    node_id_col              : column in nodes CSV for the unique node id
    source_col               : column in edges CSV for the source node id
    target_col               : column in edges CSV for the target node id
    aggregate_duplicate_edges: if True, duplicate (src, tgt) rows are collapsed
                            into one edge and a 'weight' attribute is added
                            equal to the number of times that pair appeared.
                            This handles temporal contact networks where the
                            same pair can appear many times.

    Returns
    -------
    A NetworkX Graph (or DiGraph) with all CSV columns attached as attributes.
    """
    # ── Validate files ──────────────────────────────────────────────────────
    for path, label in [(nodes_path, "nodes"), (edges_path, "edges")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {label} file: {path}")
        if not path.lower().endswith(".csv"):
            raise ValueError(f"Expected a .csv file for {label}, got: {path}")

    # ── Read & clean CSVs ───────────────────────────────────────────────────
    nodes_df = _clean_columns(pd.read_csv(nodes_path))
    edges_df = _clean_columns(pd.read_csv(edges_path))

    # Case-insensitive column resolution
    node_id_col = _resolve_col(nodes_df, node_id_col)
    source_col  = _resolve_col(edges_df, source_col)
    target_col  = _resolve_col(edges_df, target_col)

    # ── Validate required columns ───────────────────────────────────────────
    if node_id_col not in nodes_df.columns:
        raise ValueError(
            f"Nodes CSV must have an id column. "
            f"Found columns: {list(nodes_df.columns)}"
        )
    for col in [source_col, target_col]:
        if col not in edges_df.columns:
            raise ValueError(
                f"Edges CSV must have a '{col}' column. "
                f"Found: {list(edges_df.columns)}"
            )

    # ── Handle duplicate edges (temporal / contact networks) ─────────────────
    raw_edge_count = len(edges_df)
    if aggregate_duplicate_edges:
        extra_cols = [c for c in edges_df.columns if c not in [source_col, target_col]]
        if extra_cols:
            # Keep first occurrence of extra attrs, sum numeric ones as weight
            edges_agg = edges_df.groupby([source_col, target_col], sort=False).agg(
                {c: "first" for c in extra_cols}
            ).reset_index()
            # Add contact count as weight if not already present
            if "weight" not in edges_agg.columns:
                weight_series = edges_df.groupby([source_col, target_col]).size().reset_index(name="weight")
                edges_agg = edges_agg.merge(weight_series, on=[source_col, target_col])
        else:
            # No extra columns — just count occurrences as weight
            edges_agg = (
                edges_df.groupby([source_col, target_col], sort=False)
                .size()
                .reset_index(name="weight")
            )
        edges_df = edges_agg

    # ── Build graph ─────────────────────────────────────────────────────────
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes with all attributes from the CSV
    for _, row in nodes_df.iterrows():
        node_id = row[node_id_col]
        attrs = {k: v for k, v in row.items() if k != node_id_col and pd.notna(v)}
        G.add_node(node_id, **attrs)

    # Add edges with all attributes from the edges CSV
    for _, row in edges_df.iterrows():
        src = row[source_col]
        tgt = row[target_col]
        attrs = {
            k: v for k, v in row.items()
            if k not in [source_col, target_col] and pd.notna(v)
        }
        # Auto-add any node not present in the nodes file
        for n in [src, tgt]:
            if n not in G.nodes:
                G.add_node(n)
        G.add_edge(src, tgt, **attrs)

    # ── Attach metadata ─────────────────────────────────────────────────────
    G.graph["directed"]          = directed
    G.graph["nodes_file"]        = os.path.basename(nodes_path)
    G.graph["edges_file"]        = os.path.basename(edges_path)
    G.graph["node_id_col"]       = node_id_col
    G.graph["raw_edge_count"]    = raw_edge_count
    G.graph["aggregated_edges"]  = aggregate_duplicate_edges

    return G


# ─────────────────────────────────────────────
#  Graph info summary
# ─────────────────────────────────────────────

def get_graph_summary(G: nx.Graph) -> Dict[str, Any]:
    """Return a dictionary of basic graph properties."""
    is_directed = G.is_directed()

    # Collect all attribute keys
    node_attrs = set()
    for _, data in G.nodes(data=True):
        node_attrs.update(data.keys())

    edge_attrs = set()
    for _, _, data in G.edges(data=True):
        edge_attrs.update(data.keys())

    summary = {
        "type":           "Directed" if is_directed else "Undirected",
        "nodes":          G.number_of_nodes(),
        "edges":          G.number_of_edges(),
        "node_attributes": sorted(node_attrs),
        "edge_attributes": sorted(edge_attrs),
        "is_connected":   (
            nx.is_weakly_connected(G) if is_directed
            else nx.is_connected(G)
        ),
        "density":        round(nx.density(G), 6),
        "self_loops":     nx.number_of_selfloops(G),
        "components": (
            nx.number_weakly_connected_components(G) if is_directed
            else nx.number_connected_components(G)
        ),
    }
    return summary


def print_graph_summary(G: nx.Graph) -> None:
    """Pretty-print the graph summary to console."""
    s = get_graph_summary(G)
    raw = G.graph.get("raw_edge_count")
    print("\n" + "═" * 45)
    print("  GRAPH SUMMARY")
    print("═" * 45)
    print(f"  Type          : {s['type']}")
    print(f"  Nodes         : {s['nodes']}")
    print(f"  Edges         : {s['edges']}")
    if raw and raw != s['edges']:
        print(f"  Raw events    : {raw}  (aggregated → weighted edges)")
    print(f"  Connected     : {s['is_connected']}")
    print(f"  Components    : {s['components']}")
    print(f"  Density       : {s['density']}")
    print(f"  Self-loops    : {s['self_loops']}")
    print(f"  Node attrs    : {s['node_attributes']}")
    print(f"  Edge attrs    : {s['edge_attributes']}")
    print("═" * 45 + "\n")


# ─────────────────────────────────────────────
#  Graph type conversion
# ─────────────────────────────────────────────

def convert_graph(G: nx.Graph, to_directed: bool) -> nx.Graph:
    """
    Convert between directed and undirected.
    Preserves all node and edge attributes.
    """
    if to_directed and not G.is_directed():
        return G.to_directed()
    elif not to_directed and G.is_directed():
        return G.to_undirected()
    return G  # already the right type


# ─────────────────────────────────────────────
#  Node / edge attribute helpers
# ─────────────────────────────────────────────

def get_node_attribute_values(G: nx.Graph, attr: str) -> Dict[Any, Any]:
    """Return {node_id: attr_value} for a given attribute (missing → None)."""
    return {n: data.get(attr) for n, data in G.nodes(data=True)}


def get_edge_attribute_values(G: nx.Graph, attr: str) -> Dict[Tuple, Any]:
    """Return {(src, tgt): attr_value} for a given edge attribute."""
    return {(u, v): data.get(attr) for u, v, data in G.edges(data=True)}


def list_node_attributes(G: nx.Graph):
    """Return sorted list of all node attribute keys in the graph."""
    attrs = set()
    for _, data in G.nodes(data=True):
        attrs.update(data.keys())
    return sorted(attrs)


def list_edge_attributes(G: nx.Graph):
    """Return sorted list of all edge attribute keys in the graph."""
    attrs = set()
    for _, _, data in G.edges(data=True):
        attrs.update(data.keys())
    return sorted(attrs)


# ─────────────────────────────────────────────
#  Export helpers
# ─────────────────────────────────────────────

def export_graph_to_csv(G: nx.Graph, nodes_out: str, edges_out: str) -> None:
    """Export the current graph back to nodes.csv and edges.csv."""
    # Nodes
    node_rows = []
    for node_id, data in G.nodes(data=True):
        row = {"id": node_id}
        row.update(data)
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(nodes_out, index=False)

    # Edges
    edge_rows = []
    for u, v, data in G.edges(data=True):
        row = {"source": u, "target": v}
        row.update(data)
        edge_rows.append(row)
    pd.DataFrame(edge_rows).to_csv(edges_out, index=False)

    print(f"Exported nodes → {nodes_out}")
    print(f"Exported edges → {edges_out}")