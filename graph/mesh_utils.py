"""
Mesh Utility Functions
======================
Helpers for HEC-RAS mesh inspection, validation, and reporting.
"""

from __future__ import annotations
import numpy as np


def inspect_hdf5(hdf_path: str, max_depth: int = 5) -> None:
    """Print the HDF5 tree structure. Run this first on any new HEC-RAS file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip install h5py")
    with h5py.File(hdf_path, "r") as f:
        _print_tree(f, max_depth)


def _print_tree(node, max_depth: int, depth: int = 0) -> None:
    import h5py
    prefix = "  " * depth
    for key in node.keys():
        item = node[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}[Dataset] {key:50s}  shape={str(item.shape):20s}  dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}[Group]   {key}/")
            if depth < max_depth - 1:
                _print_tree(item, max_depth, depth + 1)


def compute_graph_stats(graph) -> dict:
    """Return a dict of basic graph statistics."""
    N          = graph.num_nodes
    E          = graph.edge_index.shape[1] // 2
    cell_area  = graph.cell_area.cpu().numpy()
    face_len   = graph.face_length.cpu().numpy()
    src        = graph.edge_index[0].cpu().numpy()
    degrees    = np.bincount(src, minlength=N)

    return {
        "num_cells":           int(N),
        "num_interior_faces":  int(E),
        "avg_degree":          float(degrees.mean()),
        "max_degree":          int(degrees.max()),
        "min_degree":          int(degrees.min()),
        "cell_area_mean_m2":   float(cell_area.mean()),
        "cell_area_min_m2":    float(cell_area.min()),
        "cell_area_max_m2":    float(cell_area.max()),
        "face_length_mean_m":  float(face_len.mean()),
        "face_length_max_m":   float(face_len.max()),
    }


def print_graph_stats(graph) -> None:
    stats = compute_graph_stats(graph)
    print("\n" + "─" * 50)
    print("  Graph Statistics")
    print("─" * 50)
    for k, v in stats.items():
        print(f"  {k:<30} {v}")
    print("─" * 50 + "\n")


def validate_graph(graph) -> list[str]:
    """Run sanity checks. Returns list of warning strings (empty = OK)."""
    warnings = []
    N = graph.num_nodes

    if graph.edge_index.max() >= N:
        warnings.append(f"edge_index max {graph.edge_index.max()} >= num_nodes {N}")
    if graph.edge_index.min() < 0:
        warnings.append("edge_index has negative indices")

    src     = graph.edge_index[0].cpu().numpy()
    degrees = np.bincount(src, minlength=N)
    isolated = int((degrees == 0).sum())
    if isolated > 0:
        warnings.append(f"{isolated} isolated nodes (degree=0)")

    fci = graph.face_cell_idx.cpu().numpy()
    if fci.size > 0 and fci.max() >= N:
        warnings.append(f"face_cell_idx max {fci.max()} >= num_nodes {N}")

    if graph.x.isnan().any():
        warnings.append("NaN in node features (graph.x)")
    if graph.edge_attr.isnan().any():
        warnings.append("NaN in edge features (graph.edge_attr)")
    if graph.cell_z.isnan().any():
        warnings.append("NaN in cell_z")
    if (graph.cell_area.cpu().numpy() <= 0).any():
        n_bad = (graph.cell_area.cpu().numpy() <= 0).sum()
        warnings.append(f"{n_bad} cells with non-positive area")

    return warnings
