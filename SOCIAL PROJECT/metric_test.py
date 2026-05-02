import networkx as nx
from Metrics import *
from Data_Loader import load_graph_from_csv, print_graph_summary

G = load_graph_from_csv("Nodes.csv", "Edges.csv", directed=False)

print_graph_summary(G)

# ─────────────────────────────
# TEST EACH FUNCTION
# ─────────────────────────────

# 1. Global metrics
gm = compute_global_metrics(G)
print_global_metrics(gm)

# 2. Degree distribution
dd = compute_degree_distribution(G)
print("Gamma:", dd["gamma"], "R2:", dd["log_r2"])

# 3. Clustering
cc = compute_clustering(G)
print("Avg clustering:", cc["avg_clustering"])

# 4. Path lengths
pl = compute_path_lengths(G)
print("APL:", pl["apl"], "Diameter:", pl["diameter"])

# 5. Node table
table = compute_node_metrics_table(G)
print("Top node:", table[0])

# 6. Rich club
rc = compute_rich_club(G)
print("Rich club sample:", list(zip(rc["k_vals"][:5], rc["phi"][:5])))

# ─────────────────────────────
# TEST RENDERING
# ─────────────────────────────

render_degree_distribution(dd, "degree.png")
render_clustering_analysis(G, cc, "clustering.png")
render_path_length_analysis(pl, "paths.png")
render_rich_club(rc, "richclub.png")

render_metrics_dashboard(
    G, gm, dd, cc, pl,
    "dashboard.png"
)

print("✅ All tests completed. Check PNG files.")