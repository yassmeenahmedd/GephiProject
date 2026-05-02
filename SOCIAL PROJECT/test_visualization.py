from Data_Loader import load_graph_from_csv, print_graph_summary
from Visualizer import (
    visualize_pyvis,
    visualize_matplotlib,
    visualize_multi_layout,
    visualize_attribute_distribution
)

# Load graph
G = load_graph_from_csv("Nodes.csv", "Edges.csv", directed=False)

print_graph_summary(G)
html_path = visualize_pyvis(
    G,
    output_path="test_network.html",
    color_attr="Class",
    size_by="degree",
    show_labels=True,
    label_attr="ID",   # or None
    title="Test Network"
)

print("PyVis output:", html_path)

img_path = visualize_matplotlib(
    G,
    output_path="test_network.png",
    layout="spring",
    color_attr="Gender",
    size_by="degree",
    show_labels=False,
    title="Static Graph"
)

print("Matplotlib output:", img_path)
multi_path = visualize_multi_layout(
    G,
    output_path="multi_layout.png",
    color_attr="Class"
)

print("Multi layout output:", multi_path)
dist_path = visualize_attribute_distribution(
    G,
    attr="Class",
    output_path="class_distribution.png"
)

print("Distribution output:", dist_path)
visualize_pyvis(
    G,
    output_path="weighted_edges.html",
    edge_width_by_weight=True
) 