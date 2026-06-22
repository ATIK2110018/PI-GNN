"""
Graph Construction from HEC-RAS HDF5 Geometry
==============================================
Supports TWO HEC-RAS geometry formats:

Format A – Classic exported mesh:
  Geometry/2D Flow Areas/<area>/
    Cells Center Coordinate [N,2], Faces Cell Info [F,2], FacePoints Coordinate, etc.

Format B – RAS Mapper mesh (Plan.h5 or Geometry.hdf with /Geometry/2D Flow Areas/Mesh/):
  Geometry/2D Flow Areas/Mesh/
    Cell Coordinates [N,2], Face Data [F,4], Node Coordinates [Nnodes,2]
    Property Tables/Cell Minimum Elevation [N]

Builds a PyG Data object with:
  data.x            [N, 4]   node features: [x_norm, y_norm, z_norm, area_norm]
  data.edge_index   [2, 2F]  bidirectional interior face edges
  data.edge_attr    [2F, 6]  [nx, ny, L_norm, dx_norm, dy_norm, dist_norm]
  data.pos          [N, 2]   raw UTM coordinates
  data.cell_z       [N]      bed elevation
  data.face_length  [F_int]  face lengths (for FVM)
  data.face_normal  [F_int, 2]
  data.face_cell_idx[F_int, 2]
  data.cell_area    [N]
"""

from __future__ import annotations
import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Install PyTorch Geometric: pip install torch-geometric")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_hecras_geometry(
    hdf_path: str,
    area_name: str | None = None,
    max_cells: int | None = None,
) -> dict:
    """
    Load HEC-RAS geometry into a raw dict. Auto-detects format.

    Returns:
        cell_xy       [N, 2]   cell center UTM coordinates
        cell_z        [N]      bed elevation
        face_cell_idx [F, 2]   (c0, c1) per face; -1 = boundary
        face_normal   [F, 2]   unit normal pointing from c0 to c1
        face_length   [F]      face width (m)
        area_name     str
        N             int
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip install h5py")

    with h5py.File(hdf_path, "r") as f:
        # Detect format
        if "Geometry/2D Flow Areas/Mesh" in f:
            print("[Graph] Detected: Format B (RAS Mapper mesh)")
            geo = _load_format_b(f, hdf_path)
        else:
            print("[Graph] Detected: Format A (classic exported mesh)")
            geo = _load_format_a(f, hdf_path, area_name)

    N = geo["N"]
    if max_cells is not None and N > max_cells:
        geo = _subsample(geo, max_cells)

    return geo


def build_pyg_graph(geo: dict, bathy_tif: str | None = None) -> Data:
    """Convert geometry dict to a PyG Data object."""
    cell_xy       = geo["cell_xy"]
    cell_z        = geo["cell_z"]
    face_cell_idx = geo["face_cell_idx"]
    face_normal   = geo["face_normal"]
    face_length   = geo["face_length"]
    N             = geo["N"]

    # Keep only interior faces
    interior = (face_cell_idx[:, 0] >= 0) & (face_cell_idx[:, 1] >= 0)
    fci = face_cell_idx[interior]
    fn  = face_normal[interior]
    fl  = face_length[interior]
    print(f"[Graph] Interior faces: {fci.shape[0]}")

    # ── Normalise coordinates ──────────────────────────────────────────────
    x_off, y_off = cell_xy[:, 0].min(), cell_xy[:, 1].min()
    x_scale = (cell_xy[:, 0].max() - cell_xy[:, 0].min()) + 1e-8
    y_scale = (cell_xy[:, 1].max() - cell_xy[:, 1].min()) + 1e-8
    xn = (cell_xy[:, 0] - x_off) / x_scale
    yn = (cell_xy[:, 1] - y_off) / y_scale

    z_min, z_max = cell_z.min(), cell_z.max()
    z_scale      = max(z_max - z_min, 1e-8)
    zn           = (cell_z - z_min) / z_scale

    area_approx = _estimate_cell_areas(cell_xy, fci, fl)
    area_norm   = area_approx / (area_approx.max() + 1e-8)

    node_feats = np.stack([xn, yn, zn, area_norm], axis=1).astype(np.float32)

    # ── Bidirectional edge index ───────────────────────────────────────────
    src = np.concatenate([fci[:, 0], fci[:, 1]])
    dst = np.concatenate([fci[:, 1], fci[:, 0]])
    edge_index = np.stack([src, dst], axis=0)

    # ── Edge features ──────────────────────────────────────────────────────
    dx = (cell_xy[fci[:, 1], 0] - cell_xy[fci[:, 0], 0]) / x_scale
    dy = (cell_xy[fci[:, 1], 1] - cell_xy[fci[:, 0], 1]) / y_scale
    dist    = np.sqrt(dx**2 + dy**2 + 1e-16)
    fl_norm = fl / (fl.max() + 1e-8)
    nxf, nyf = fn[:, 0], fn[:, 1]

    edge_attr = np.stack([
        np.concatenate([nxf,  -nxf]),
        np.concatenate([nyf,  -nyf]),
        np.concatenate([fl_norm,  fl_norm]),
        np.concatenate([dx,   -dx]),
        np.concatenate([dy,   -dy]),
        np.concatenate([dist,  dist]),
    ], axis=1).astype(np.float32)

    # ── Optional raster bathymetry ─────────────────────────────────────────
    if bathy_tif is not None:
        try:
            cell_z_r = _sample_bathy_tif(bathy_tif, cell_xy)
            valid    = np.isfinite(cell_z_r)
            cell_z   = np.where(valid, cell_z_r, cell_z)
            zn2      = (cell_z - cell_z.min()) / max(cell_z.max() - cell_z.min(), 1e-8)
            node_feats[:, 2] = zn2.astype(np.float32)
            print(f"[Graph] Bathymetry from raster: {bathy_tif}")
        except Exception as e:
            print(f"[Graph] Warning: raster bathymetry failed ({e})")

    # ── Assemble PyG Data ──────────────────────────────────────────────────
    data = Data(
        x          = torch.from_numpy(node_feats),
        edge_index = torch.from_numpy(edge_index).long(),
        edge_attr  = torch.from_numpy(edge_attr),
        pos        = torch.from_numpy(cell_xy.astype(np.float32)),
    )
    data.cell_z        = torch.from_numpy(cell_z.astype(np.float32))
    data.face_length   = torch.from_numpy(fl.astype(np.float32))
    data.face_normal   = torch.from_numpy(fn.astype(np.float32))
    data.face_cell_idx = torch.from_numpy(fci.astype(np.int64))
    data.cell_area     = torch.from_numpy(area_approx.astype(np.float32))
    data.x_offset      = torch.tensor([x_off, y_off], dtype=torch.float32)
    data.x_scale       = torch.tensor([x_scale, y_scale], dtype=torch.float32)
    data.z_stats       = torch.tensor([z_min, z_scale], dtype=torch.float32)
    data.num_nodes     = N

    print(f"[Graph] PyG Data: nodes={N}  edges={edge_index.shape[1]}  "
          f"node_dim={node_feats.shape[1]}  edge_dim={edge_attr.shape[1]}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Format B loader  (Plan.h5 / RAS Mapper)
# ─────────────────────────────────────────────────────────────────────────────

def _load_format_b(f, hdf_path: str) -> dict:
    """
    Parse geometry from RAS Mapper Plan.h5.
    Geometry/2D Flow Areas/Mesh/
      Cell Coordinates   [N, 2]
      Face Data          [F, 4]   cols: node0, node1, left_cell, right_cell
      Node Coordinates   [Nnodes, 2]
      Property Tables/Cell Minimum Elevation  [N]
    """
    mesh = f["Geometry/2D Flow Areas/Mesh"]

    # Cell centres
    cell_xy = np.asarray(mesh["Cell Coordinates"][:], dtype=np.float64)[:, :2]
    N       = cell_xy.shape[0]
    print(f"[Graph] Cells: {N}")

    # Bed elevation
    try:
        cell_z = np.asarray(
            mesh["Property Tables/Cell Minimum Elevation"][:], dtype=np.float64).ravel()
        print(f"[Graph] Cell elevation loaded: min={cell_z.min():.2f}  max={cell_z.max():.2f}")
    except KeyError:
        cell_z = np.zeros(N, dtype=np.float64)
        print("[Graph] Warning: Cell Minimum Elevation not found; using zeros.")

    # Face data: [F, 4] → [node0, node1, left_cell, right_cell]
    face_data  = np.asarray(mesh["Face Data"][:], dtype=np.int32)    # [F, 4]
    node_coords = np.asarray(mesh["Node Coordinates"][:], dtype=np.float64)  # [Nnodes, 2]
    F = face_data.shape[0]
    print(f"[Graph] Faces: {F}")

    # In HEC-RAS 6.x Plan.h5: Face Data cols = [cell0, cell1, node0, node1]
    c_left  = face_data[:, 0].astype(np.int64)   # [F]  left  cell  (-1 = boundary)
    c_right = face_data[:, 1].astype(np.int64)   # [F]  right cell  (-1 = boundary)
    fn0     = face_data[:, 2].astype(np.int64)   # [F]  first node index
    fn1     = face_data[:, 3].astype(np.int64)   # [F]  second node index

    # Face cell connectivity [F, 2]
    face_cell_idx = np.stack([c_left, c_right], axis=1)

    # Face lengths from node pair distances (clip invalid node indices)
    fn0 = np.clip(fn0, 0, len(node_coords) - 1)
    fn1 = np.clip(fn1, 0, len(node_coords) - 1)
    p0  = node_coords[fn0]    # [F, 2]
    p1  = node_coords[fn1]    # [F, 2]
    face_length = np.linalg.norm(p1 - p0, axis=1)   # [F]
    face_length = np.where(face_length < 1e-6, 1.0, face_length)  # guard zeros

    # Face unit normals perpendicular to edge, pointing from c_left → c_right
    edge_vec = p1 - p0
    nx_raw   = -edge_vec[:, 1]
    ny_raw   =  edge_vec[:, 0]
    mag      = np.sqrt(nx_raw**2 + ny_raw**2 + 1e-16)
    nx, ny   = nx_raw / mag, ny_raw / mag

    # Flip normals that point away from c_right (for interior faces)
    interior   = (c_left >= 0) & (c_right >= 0)
    cl, cr     = c_left[interior], c_right[interior]
    d_expected = cell_xy[cr] - cell_xy[cl]
    dot        = nx[interior] * d_expected[:, 0] + ny[interior] * d_expected[:, 1]
    flip_mask  = dot < 0
    if flip_mask.any():
        nx[interior] = np.where(flip_mask, -nx[interior], nx[interior])
        ny[interior] = np.where(flip_mask, -ny[interior], ny[interior])

    face_normal = np.stack([nx, ny], axis=1)   # [F, 2]

    # Detect area name from Attributes if available
    area_name = "Mesh"
    try:
        attrs      = f["Geometry/2D Flow Areas/Attributes"][:]
        area_name  = attrs["Name"][0].decode("utf-8").strip()
    except Exception:
        pass

    return {
        "cell_xy":       cell_xy,
        "cell_z":        cell_z,
        "face_cell_idx": face_cell_idx,
        "face_normal":   face_normal,
        "face_length":   face_length,
        "area_name":     area_name,
        "N":             N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Format A loader  (classic exported mesh)
# ─────────────────────────────────────────────────────────────────────────────

def _load_format_a(f, hdf_path: str, area_name: str | None) -> dict:
    areas = list(f["Geometry/2D Flow Areas"].keys())
    if not areas:
        raise ValueError(f"No 2D Flow Areas in {hdf_path}")
    if area_name is None:
        area_name = areas[0]
    g         = f[f"Geometry/2D Flow Areas/{area_name}"]
    available = list(g.keys())
    print(f"[Graph] Area '{area_name}' | datasets: {available}")

    cell_xy = np.asarray(g["Cells Center Coordinate"][:], dtype=np.float64)[:, :2]
    N       = cell_xy.shape[0]
    print(f"[Graph] Cells: {N}")

    if "Cells Minimum Elevation" in available:
        cell_z = np.asarray(g["Cells Minimum Elevation"][:], dtype=np.float64).ravel()
    else:
        cell_z = np.zeros(N, dtype=np.float64)

    face_cell_idx = None
    for key in ["Faces Cell Info", "Face Cell Info", "Faces Cell Indexes"]:
        if key in available:
            face_cell_idx = np.asarray(g[key][:], dtype=np.int64)
            if face_cell_idx.ndim == 1:
                face_cell_idx = face_cell_idx.reshape(-1, 2)
            break
    if face_cell_idx is None:
        face_cell_idx = _build_connectivity_from_cells(g, N, available)

    face_normal, face_length = _compute_face_normals_classic(g, face_cell_idx, cell_xy, available)

    return {
        "cell_xy":       cell_xy,
        "cell_z":        cell_z,
        "face_cell_idx": face_cell_idx,
        "face_normal":   face_normal,
        "face_length":   face_length,
        "area_name":     area_name,
        "N":             N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(geo: dict, max_cells: int) -> dict:
    N = geo["N"]
    print(f"[Graph] Subsampling {N} → {max_cells} cells.")
    rng        = np.random.default_rng(42)
    keep       = np.sort(rng.choice(N, size=max_cells, replace=False))
    cell_map   = -np.ones(N, dtype=np.int64)
    cell_map[keep] = np.arange(max_cells)

    cell_xy       = geo["cell_xy"][keep]
    cell_z        = geo["cell_z"][keep]
    fci           = geo["face_cell_idx"]
    c0, c1        = fci[:, 0], fci[:, 1]
    keep_faces    = (c0 >= 0) & (c1 >= 0) & \
                    (cell_map[np.clip(c0, 0, N-1)] >= 0) & \
                    (cell_map[np.clip(c1, 0, N-1)] >= 0)
    new_fci       = np.stack([cell_map[c0[keep_faces]], cell_map[c1[keep_faces]]], axis=1)
    return {
        "cell_xy":       cell_xy,
        "cell_z":        cell_z,
        "face_cell_idx": new_fci,
        "face_normal":   geo["face_normal"][keep_faces],
        "face_length":   geo["face_length"][keep_faces],
        "area_name":     geo["area_name"],
        "N":             max_cells,
    }


def _estimate_cell_areas(cell_xy: np.ndarray, fci: np.ndarray, fl: np.ndarray) -> np.ndarray:
    N = len(cell_xy)
    area = np.zeros(N, dtype=np.float64)
    c0, c1 = fci[:, 0], fci[:, 1]
    dx = cell_xy[c1, 0] - cell_xy[c0, 0]
    dy = cell_xy[c1, 1] - cell_xy[c0, 1]
    dist = np.sqrt(dx**2 + dy**2 + 1e-16)
    contrib = fl * dist * 0.5
    np.add.at(area, c0, contrib)
    np.add.at(area, c1, contrib)
    return np.clip(area, 1.0, None)


def _build_connectivity_from_cells(g, N: int, available: list) -> np.ndarray:
    for key in ["Cells Face and Orientation Info", "Cell Face Info"]:
        if key in available:
            raw = g[key][:]
            if raw.ndim == 2:
                face_to_cells: dict[int, list] = {}
                for ci in range(min(N, raw.shape[0])):
                    for fi in raw[ci]:
                        if fi < 0: break
                        face_to_cells.setdefault(int(fi), []).append(ci)
                pairs = []
                for cells in face_to_cells.values():
                    pairs.append([cells[0], cells[1]] if len(cells) == 2 else [cells[0], -1])
                return np.array(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
    return np.zeros((0, 2), dtype=np.int64)


def _compute_face_normals_classic(g, fci, cell_xy, available):
    F = len(fci); N = len(cell_xy)
    for key in ["Faces Normal and Length", "Face Normal and Length"]:
        if key in available:
            arr = np.asarray(g[key][:], dtype=np.float64)
            if arr.shape[0] >= F:
                arr = arr[:F]
                nx = arr[:, 0]; ny = arr[:, 1]
                fl = arr[:, 2] if arr.shape[1] >= 3 else np.ones(F)
                return np.stack([nx, ny], axis=1), fl
    c0 = np.clip(fci[:, 0], 0, N-1); c1 = np.clip(fci[:, 1], 0, N-1)
    dx = cell_xy[c1, 0] - cell_xy[c0, 0]; dy = cell_xy[c1, 1] - cell_xy[c0, 1]
    dist = np.sqrt(dx**2 + dy**2 + 1e-16)
    return np.stack([dx/dist, dy/dist], axis=1), dist


def _sample_bathy_tif(tif_path: str, cell_xy: np.ndarray) -> np.ndarray:
    import rasterio
    with rasterio.open(tif_path) as src:
        coords  = list(zip(cell_xy[:, 0], cell_xy[:, 1]))
        sampled = [v[0] for v in src.sample(coords, indexes=1)]
        nodata  = src.nodata
    result = np.array(sampled, dtype=np.float64)
    if nodata is not None:
        result[result == nodata] = np.nan
    return result
