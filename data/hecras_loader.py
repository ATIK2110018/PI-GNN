"""
HEC-RAS Simulation Output Loader
=================================
Supports TWO HEC-RAS HDF5 output formats:

Format A – Classic (HEC-RAS 5.x / exported mesh):
  /Results/Unsteady/Output/Output Blocks/Base Output/
    Unsteady Time Series/2D Flow Areas/<area>/
      Depth [T,N], Velocity X Component [T,N], Velocity Y Component [T,N], Time [T]

Format B – RAS Mapper mesh (HEC-RAS 6.x Plan.h5):
  /Results/Output Blocks/Base Output/2D Flow Areas/Mesh/
      Cell Depth [T,N], Face Velocity [T,F]
  /Results/Output Blocks/Base Output/Time  [T]
  /Geometry/2D Flow Areas/Mesh/
      Cell Coordinates [N,2], Face Data [F,4]

This loader auto-detects which format is present.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch


# ── Classic format key variants ────────────────────────────────────────────
_DEPTH_KEYS = ["Depth", "Water Depth", "depth", "Cell Depth"]
_WSE_KEYS   = ["Water Surface", "WSE", "Water Surface Elevation"]
_VX_KEYS    = ["Velocity X Component", "Velocity - Velocity X Component", "Velocity X", "Vx"]
_VY_KEYS    = ["Velocity Y Component", "Velocity - Velocity Y Component", "Velocity Y", "Vy"]
_TIME_KEYS  = ["Time", "time", "Times"]
_MANN_KEYS  = ["Manning's n", "Mannings n", "Manning n", "Cell Manning's n"]


def _find_key(group, candidates: list[str]) -> str | None:
    available = set(group.keys())
    for k in candidates:
        if k in available:
            return k
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main loader — auto-detects format
# ─────────────────────────────────────────────────────────────────────────────

def load_hecras_results(
    results_hdf: str,
    area_name: str | None = None,
    cell_z: np.ndarray | None = None,
    cell_indices: np.ndarray | None = None,
    time_range: tuple[float, float] | None = None,
) -> dict:
    """
    Load HEC-RAS unsteady results. Auto-detects Format A (classic) or Format B (RAS Mapper).

    Returns dict:
        h_series  : [T, N]  water depth (m)
        u_series  : [T, N]  x-velocity (m/s)   (estimated from face velocity if needed)
        v_series  : [T, N]  y-velocity (m/s)
        t_series  : [T]     time (seconds from start)
        n_series  : [N] or None
        area_name : str
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip install h5py")

    with h5py.File(results_hdf, "r") as f:
        # ── Detect format ──────────────────────────────────────────────────
        if "Results/Output Blocks" in f:
            print("[Data] Detected: Format B (RAS Mapper / HEC-RAS 6.x Plan.h5)")
            return _load_format_b(f, results_hdf, cell_z, cell_indices, time_range)
        else:
            print("[Data] Detected: Format A (Classic HEC-RAS unsteady results)")
            return _load_format_a(f, results_hdf, area_name, cell_z, cell_indices, time_range)


# ─────────────────────────────────────────────────────────────────────────────
# Format B — RAS Mapper Plan.h5 (HEC-RAS 6.x)
# ─────────────────────────────────────────────────────────────────────────────

def _load_format_b(f, path: str, cell_z, cell_indices, time_range) -> dict:
    """
    Load from RAS Mapper Plan.h5 structure.
    Results are at: Results/Output Blocks/Base Output/2D Flow Areas/Mesh/
    """
    base   = f["Results/Output Blocks/Base Output"]
    mesh_g = base["2D Flow Areas/Mesh"]

    # ── Time ──────────────────────────────────────────────────────────────
    # Time is at Results/Output Blocks/Base Output/Time (hours)
    t_hours = np.asarray(base["Time"][:], dtype=np.float64)
    t_sec   = t_hours * 3600.0
    t_idx   = _time_filter(t_sec, time_range)
    t_sec   = t_sec[t_idx]
    print(f"[Data] Time steps: {len(t_sec)}  ({t_sec[0]:.0f}s – {t_sec[-1]:.0f}s)")

    # ── Depth ──────────────────────────────────────────────────────────────
    if "Cell Depth" in mesh_g:
        h_series = np.asarray(mesh_g["Cell Depth"][t_idx, :], dtype=np.float32)
        print(f"[Data] Depth loaded: shape={h_series.shape}")
    else:
        raise ValueError("'Cell Depth' not found in Results/Output Blocks/Base Output/2D Flow Areas/Mesh/")

    N = h_series.shape[1]

    # ── Velocities from face velocity ──────────────────────────────────────
    # Face Velocity shape [T, F]: normal velocity at each face
    # Project face velocities to cell-centred Vx, Vy using face normals & areas
    if "Face Velocity" in mesh_g:
        fv_series = np.asarray(mesh_g["Face Velocity"][t_idx, :], dtype=np.float32)  # [T, F]
        print(f"[Data] Face Velocity loaded: shape={fv_series.shape}")

        # Load face geometry from plan file geometry section
        geo_mesh = f["Geometry/2D Flow Areas/Mesh"]
        face_data = np.asarray(geo_mesh["Face Data"][:], dtype=np.int32)   # [F, 4]
        # HEC-RAS 6.x Plan.h5 Face Data cols: [cell0, cell1, node0, node1]
        node_coords = np.asarray(geo_mesh["Node Coordinates"][:], dtype=np.float64)  # [Nnodes,2]

        # Compute face normals from node pairs (cols 2 and 3)
        fn0 = np.clip(face_data[:, 2].astype(np.int64), 0, len(node_coords) - 1)
        fn1 = np.clip(face_data[:, 3].astype(np.int64), 0, len(node_coords) - 1)
        p0  = node_coords[fn0]  # [F, 2]
        p1  = node_coords[fn1]  # [F, 2]
        edge_vec = p1 - p0      # [F, 2]
        # Normal = perpendicular to edge (rotate 90°)
        nx_raw = -edge_vec[:, 1]
        ny_raw =  edge_vec[:, 0]
        mag    = np.sqrt(nx_raw**2 + ny_raw**2 + 1e-16)
        nx     = nx_raw / mag   # [F]
        ny     = ny_raw / mag   # [F]

        # Cell indices for each face (col0=left cell, col1=right cell)
        c_left  = face_data[:, 0].astype(np.int64)   # [F]
        c_right = face_data[:, 1].astype(np.int64)   # [F]

        # Project face-normal velocity → cell Vx, Vy via area-weighted average
        u_series, v_series = _project_face_to_cell(
            fv_series, nx, ny, c_left, c_right, N
        )
        print(f"[Data] Cell Vx/Vy projected from face velocities.")
    else:
        print("[Data] Warning: 'Face Velocity' not found; using zeros for u, v.")
        u_series = np.zeros_like(h_series)
        v_series = np.zeros_like(h_series)

    # ── Detect area name ───────────────────────────────────────────────────
    try:
        area_attrs = f["Geometry/2D Flow Areas/Attributes"][:]
        area_name  = area_attrs["Name"][0].decode("utf-8").strip()
    except Exception:
        area_name = "Mesh"

    # ── Manning's n (not typically in results; skip) ───────────────────────
    n_series = None

    # ── Cell subsetting ────────────────────────────────────────────────────
    if cell_indices is not None:
        h_series = h_series[:, cell_indices]
        u_series = u_series[:, cell_indices]
        v_series = v_series[:, cell_indices]

    print(f"[Data] Final shapes → h={h_series.shape}  u={u_series.shape}")
    return {
        "h_series":  h_series,
        "u_series":  u_series,
        "v_series":  v_series,
        "t_series":  t_sec,
        "n_series":  n_series,
        "area_name": area_name,
    }


def _project_face_to_cell(
    fv: np.ndarray,   # [T, F] face normal velocity
    nx: np.ndarray,   # [F]
    ny: np.ndarray,   # [F]
    c_left:  np.ndarray,  # [F]
    c_right: np.ndarray,  # [F]
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Area-weighted projection of face-normal velocities to cell-centred Vx, Vy.
    For each cell: Vx = Σ (fv * nx * |edge_len|) / Σ |edge_len|
    """
    T, F   = fv.shape
    u_cell = np.zeros((T, N), dtype=np.float32)
    v_cell = np.zeros((T, N), dtype=np.float32)
    count  = np.zeros(N, dtype=np.float32)

    valid_left  = c_left  >= 0
    valid_right = c_right >= 0

    cl = c_left[valid_left]
    cr = c_right[valid_right]
    nxl = nx[valid_left];  nyl = ny[valid_left]
    nxr = nx[valid_right]; nyr = ny[valid_right]
    fvl = fv[:, valid_left]   # [T, F_valid_left]
    fvr = fv[:, valid_right]  # [T, F_valid_right]

    np.add.at(count, cl, 1)
    np.add.at(count, cr, 1)

    for t in range(T):
        np.add.at(u_cell[t], cl,  fvl[t] * nxl)
        np.add.at(v_cell[t], cl,  fvl[t] * nyl)
        np.add.at(u_cell[t], cr, -fvr[t] * nxr)  # outward from right cell is flipped
        np.add.at(v_cell[t], cr, -fvr[t] * nyr)

    safe_count = np.maximum(count, 1.0)
    u_cell /= safe_count[np.newaxis, :]
    v_cell /= safe_count[np.newaxis, :]
    return u_cell, v_cell


# ─────────────────────────────────────────────────────────────────────────────
# Format A — Classic HEC-RAS unsteady results
# ─────────────────────────────────────────────────────────────────────────────

def _load_format_a(f, path, area_name, cell_z, cell_indices, time_range) -> dict:
    if area_name is None:
        area_name = _detect_area_classic(f)
    print(f"[Data] Area: '{area_name}'")

    grp       = _locate_ts_group_classic(f, area_name)
    available = list(grp.keys())
    print(f"[Data] Datasets: {available}")

    t_key = _find_key(grp, _TIME_KEYS)
    if t_key:
        t_sec = np.asarray(grp[t_key][:], dtype=np.float64) * 3600.0
    else:
        sample_key = available[0]
        t_sec = np.arange(grp[sample_key].shape[0], dtype=np.float64)

    t_idx = _time_filter(t_sec, time_range)
    t_sec = t_sec[t_idx]
    print(f"[Data] Time steps: {len(t_sec)}")

    h_key = _find_key(grp, _DEPTH_KEYS)
    if h_key:
        h_series = grp[h_key][t_idx, :].astype(np.float32)
    else:
        wse_key = _find_key(grp, _WSE_KEYS)
        if wse_key is None:
            raise ValueError(f"No depth/WSE dataset in {path}. Available: {available}")
        wse = grp[wse_key][t_idx, :].astype(np.float32)
        h_series = wse if cell_z is None else np.maximum(wse - cell_z[np.newaxis, :], 0.0)

    vx_key = _find_key(grp, _VX_KEYS)
    vy_key = _find_key(grp, _VY_KEYS)
    u_series = grp[vx_key][t_idx, :].astype(np.float32) if vx_key else np.zeros_like(h_series)
    v_series = grp[vy_key][t_idx, :].astype(np.float32) if vy_key else np.zeros_like(h_series)

    n_series = None
    mann_key = _find_key(grp, _MANN_KEYS)
    if mann_key:
        n_raw    = grp[mann_key][:].astype(np.float32)
        n_series = n_raw[t_idx] if n_raw.ndim == 2 else n_raw

    if cell_indices is not None:
        h_series = h_series[:, cell_indices]
        u_series = u_series[:, cell_indices]
        v_series = v_series[:, cell_indices]

    print(f"[Data] Final shapes → h={h_series.shape}  u={u_series.shape}")
    return {"h_series": h_series, "u_series": u_series, "v_series": v_series,
            "t_series": t_sec, "n_series": n_series, "area_name": area_name}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by both loaders
# ─────────────────────────────────────────────────────────────────────────────

def _time_filter(t_sec: np.ndarray, time_range) -> np.ndarray:
    if time_range is not None:
        mask = (t_sec >= time_range[0]) & (t_sec <= time_range[1])
    else:
        mask = np.ones(len(t_sec), dtype=bool)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No time steps in range {time_range}. "
                         f"File covers {t_sec[0]:.0f}s – {t_sec[-1]:.0f}s.")
    return idx


def _detect_area_classic(f) -> str:
    for path in ["Geometry/2D Flow Areas",
                 "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas"]:
        try:
            areas = list(f[path].keys())
            if areas:
                return areas[0]
        except KeyError:
            pass
    raise ValueError("Cannot auto-detect area name. Set area_name in config.yaml.")


def _locate_ts_group_classic(f, area_name: str):
    candidates = [
        f"Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{area_name}",
        f"Results/Unsteady/2D Flow Areas/{area_name}",
        f"Results/2D Flow Areas/{area_name}",
    ]
    for path in candidates:
        if path in f:
            return f[path]
    raise ValueError(f"Cannot find results for area '{area_name}'. Run --inspect-hdf.")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data for testing without HEC-RAS results
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_results(
    cell_xy: np.ndarray, t_series: np.ndarray,
    n_true: np.ndarray | None = None, n_default: float = 0.035,
) -> dict:
    """Generate synthetic h, u, v for testing when HEC-RAS output is unavailable."""
    N = len(cell_xy); T = len(t_series)
    if n_true is None:
        n_true = np.full(N, n_default, dtype=np.float32)

    y_n  = (cell_xy[:, 1] - cell_xy[:, 1].min()) / (cell_xy[:, 1].ptp() + 1e-8)
    h_base = 3.0 - 1.5 * y_n
    u_base = 0.5 + 1.0 * y_n

    h_series = np.zeros((T, N), dtype=np.float32)
    u_series = np.zeros((T, N), dtype=np.float32)
    v_series = np.zeros((T, N), dtype=np.float32)
    for ti, t in enumerate(t_series):
        wave = 0.3 * np.sin(2.0 * np.pi * t / 3600.0)
        h_series[ti] = (h_base + wave).clip(0.1).astype(np.float32)
        u_series[ti] = (u_base + 0.2 * wave).astype(np.float32)

    print(f"[Data] Synthetic data: T={T}, N={N}")
    return {"h_series": h_series, "u_series": u_series, "v_series": v_series,
            "t_series": t_series, "n_series": n_true, "area_name": "synthetic"}


def to_tensors(results: dict, device: torch.device) -> dict:
    tensors = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            if k in ["h_series", "u_series", "v_series"]:
                tensors[k] = torch.from_numpy(v)  # Keep on CPU to save VRAM
            else:
                tensors[k] = torch.from_numpy(v).to(device)
        else:
            tensors[k] = v
    return tensors


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-driven synthetic fields (no HEC-RAS results required)
# ─────────────────────────────────────────────────────────────────────────────

def load_boundary_driven_results(
    cell_xy:         np.ndarray,
    cell_z:          np.ndarray,
    bc:              dict,
    n_init:          float = 0.035,
    g:               float = 9.81,
    channel_width_m: float = 200.0,
    channel_slope:   float = 5.0e-5,
    hdf_path:        str | None = None,
    btc_wse_series:  np.ndarray | None = None,
    btc_cell_idx:    int | None = None,
) -> dict:
    """
    Generate physically-motivated synthetic h, u, v fields driven by real
    boundary conditions when HEC-RAS simulation results are unavailable.
    If hdf_path is provided, boundary cells are snapped to HEC-RAS BC lines.
    """
    t_sec    = bc["t_sec"]           # [T]
    Q_series = bc["discharge"]       # [T]  — upstream discharge (m³/s)
    H_down   = bc["stage_down"]      # [T]  — downstream WSE (m)

    T = len(t_sec)
    N = len(cell_xy)

    h_series = np.zeros((T, N), dtype=np.float32)
    u_series = np.zeros((T, N), dtype=np.float32)
    v_series = np.zeros((T, N), dtype=np.float32)

    use_poly = False
    if hdf_path is not None:
        import h5py
        from pathlib import Path
        geom_path = Path(hdf_path)
        bc_h5_path = geom_path.parent.parent / "Boundary Conditions" / "Boundary Condition.h5"
        if bc_h5_path.exists():
            try:
                with h5py.File(bc_h5_path, "r") as f:
                    poly = f["Boundary Conditions/BC Lines/Polyline"][:]
                
                a, b = poly[0], poly[1] # Upstream
                c, d = poly[2], poly[3] # Downstream

                def dist_to_segment(p, pt1, pt2):
                    ap = p - pt1
                    ab = pt2 - pt1
                    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
                    projection = pt1 + t[:, None] * ab
                    return np.linalg.norm(p - projection, axis=1)

                d_up = dist_to_segment(cell_xy, a, b)
                d_down = dist_to_segment(cell_xy, c, d)
                
                denom = d_up + d_down
                denom = np.where(denom < 1e-6, 1.0, denom)
                y_norm = d_down / denom
                
                upstream_bc_cell_idx = int(np.argmin(d_up))
                downstream_bc_cell_idx = int(np.argmin(d_down))
                
                z_north = float(cell_z[upstream_bc_cell_idx])
                z_south = float(cell_z[downstream_bc_cell_idx])
                use_poly = True
                print(f"[Data] Boundary condition successfully snapped to HEC-RAS BC Lines!")
                print(f"[Data] Upstream BC segment node #{upstream_bc_cell_idx} bed z={z_north:.2f}m")
                print(f"[Data] Downstream BC segment node #{downstream_bc_cell_idx} bed z={z_south:.2f}m")
            except Exception as e:
                print(f"[Data] Warning: Failed to load polyline BC ({e}). Falling back to spatial-extreme anchors.")

    if not use_poly:
        y_min = cell_xy[:, 1].min()
        y_max = cell_xy[:, 1].max()
        y_norm = (cell_xy[:, 1] - y_min) / max(y_max - y_min, 1.0)   # [N]  0=south/down
        south_cell_idx = int(np.argmin(cell_xy[:, 1]))
        z_south = float(cell_z[south_cell_idx])
        north_cell_idx = int(np.argmax(cell_xy[:, 1]))
        z_north = float(cell_z[north_cell_idx])

    W = channel_width_m
    S = channel_slope

    h_min_phys = 0.05   # minimum physical depth (m)

    log_every = max(T // 10, 1)
    for ti in range(T):
        if ti % log_every == 0:
            print(f"[Data] Boundary-driven fields: {ti}/{T} time steps", flush=True)
        Q     = float(Q_series[ti]) if not np.isnan(Q_series[ti]) else float(np.nanmean(Q_series))
        wse_d = float(H_down[ti])   if not np.isnan(H_down[ti])   else float(z_south + 2.0)

        # ── Upstream depth via Manning (wide-channel: R ≈ h) ────────────
        h_up = (max(Q, 0.0) * n_init / (W * np.sqrt(S + 1e-10))) ** 0.6
        h_up = max(h_up, h_min_phys)

        # ── Upstream WSE ─────────────────────────────────────────────────
        wse_u = z_north + h_up

        # ── Spatially interpolate WSE ─────────────────────────────────────
        # If BTC gauge is available and valid at this step, do piecewise interpolation.
        btc_wse_val = btc_wse_series[ti] if (btc_wse_series is not None and ti < len(btc_wse_series)) else np.nan
        if btc_cell_idx is not None and not np.isnan(btc_wse_val):
            y_btc = y_norm[btc_cell_idx]
            mask_down = y_norm <= y_btc
            wse_cell = np.zeros(N, dtype=np.float32)
            # Downstream to BTC
            wse_cell[mask_down] = wse_d + (y_norm[mask_down] / (y_btc + 1e-12)) * (btc_wse_val - wse_d)
            # BTC to Upstream
            wse_cell[~mask_down] = btc_wse_val + ((y_norm[~mask_down] - y_btc) / (1.0 - y_btc + 1e-12)) * (wse_u - btc_wse_val)
        else:
            wse_cell = wse_d + y_norm * (wse_u - wse_d)          # [N]

        # ── Depth ─────────────────────────────────────────────────────────
        h_cell = np.maximum(wse_cell - cell_z, h_min_phys).astype(np.float32)

        # ── Velocity ──────────────────────────────────────────────────────
        u_cell = (Q / (W * np.maximum(h_cell, h_min_phys))).astype(np.float32)
        u_cell = np.clip(u_cell, 0.0, 5.0)

        h_series[ti] = h_cell
        u_series[ti] = u_cell
        # v_series stays zero

    print(
        f"[Data] Boundary-driven synthetic fields: T={T}, N={N}  "
        f"h∈[{h_series.min():.2f}, {h_series.max():.2f}]  "
        f"u∈[{u_series.min():.2f}, {u_series.max():.2f}]"
    )
    return {
        "h_series":  h_series,
        "u_series":  u_series,
        "v_series":  v_series,
        "t_series":  t_sec,
        "n_series":  None,
        "area_name": "boundary_driven",
    }
