from Data_Loader import load_graph_from_csv, print_graph_summary

# Paths
nodes_file = "Nodes.csv"
edges_file = "Edges.csv"

# Load graph
G = load_graph_from_csv(
    nodes_file,
    edges_file,
    directed=False  # change to True to test directed
)

# Print summary
print_graph_summary(G)