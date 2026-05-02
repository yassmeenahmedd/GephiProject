from Data_Loader import load_graph_from_csv, print_graph_summary
from Layout import (
    compute_layout,
    render_layout,
    render_all_layouts,
    render_radial_annotated,
    render_hierarchical_annotated
)

# Load graph
G = load_graph_from_csv("Nodes.csv", "Edges.csv", directed=False)

print_graph_summary(G)
pos, time_taken = compute_layout(G, "spring")

print("Computed layout in:", time_taken, "seconds")
print("Sample positions:", list(pos.items())[:3])

output = render_layout(
    G,
    layout_name="fruchterman_reingold",
    output_path="fr_layout.png",
    color_attr="Class",
    show_labels=False
)

print("Saved:", output)

output = render_all_layouts(
    G,
    output_path="all_layouts.png",
    color_attr="Class"
)

print("Saved:", output)

output = render_radial_annotated(
    G,
    output_path="radial_annotated.png"
)

print("Saved:", output)

output = render_hierarchical_annotated(
    G,
    output_path="hierarchical_annotated.png"
)

print("Saved:", output)

layouts = [
    "spring",
    "kamada_kawai",
    "radial",
    "hierarchical",
    "shell",
    "circular",
    "spectral"
]

for layout in layouts:
    out = render_layout(G, layout, f"{layout}.png")
    print("Done:", layout)