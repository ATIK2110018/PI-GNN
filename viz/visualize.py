"""
Visualization Utilities for PIGNN
===================================
Plots:
  1. Manning's n spatial field (triangulated, with error map if n_true given)
     — optionally overlays observation station markers (e.g. BTC)
  2. Training loss history (with BTC component)
  3. Flow field snapshot (h, speed, velocity vectors)
  4. FVM residual maps
  5. BTC observed vs. model-predicted stage time series comparison
  6. Boundary conditions time series (Ord Ferry Q + Colusa stage)
  7. Manning's n histogram with KDE
  8. Spatial n vs. bed elevation two-panel map
  9. Convergence detail (loss, BTC RMSE, n mean±std)
 10. Velocity field snapshot at peak flow
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Public API — existing plots (updated with obs_points + BTC args)
# ─────────────────────────────────────────────────────────────────────────────

def plot_n_field(
    cell_xy:    np.ndarray,
    n_pred:     np.ndarray,
    n_true:     np.ndarray | None = None,
    save_path:  str = "n_field.png",
    dpi:        int = 200,
    cmap:       str = "RdYlGn_r",
    title:      str = "Predicted Manning's n",
    obs_points: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Spatial Manning's n map. Optionally shows error panel if n_true provided.

    Parameters
    ----------
    obs_points : optional dict mapping station name → (x_utm, y_utm).
                 Each station is drawn as a red star marker with a label.
    """
    cell_xy, n_pred, n_true = _sample_spatial_arrays(cell_xy, n_pred, n_true)
    cell_xy, obs_points, x_label, y_label = _local_plot_coordinates(cell_xy, obs_points)
    x, y    = cell_xy[:, 0], cell_xy[:, 1]
    triang  = _triangulate(x, y)
    ncols   = 2 if n_true is not None else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 8), dpi=dpi)
    if ncols == 1:
        axes = [axes]

    # Panel 1 – predicted n
    ax  = axes[0]
    vmin, vmax = np.percentile(n_pred, [2, 98])
    tcf = ax.tripcolor(triang, n_pred, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    plt.colorbar(tcf, ax=ax, label="Manning's n")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    _stats_text(ax, n_pred)
    _plot_obs_markers(ax, obs_points)

    # Panel 2 – absolute error
    if n_true is not None:
        ax2  = axes[1]
        err  = np.abs(n_pred - n_true)
        rmse = np.sqrt(np.mean(err**2))
        tcf2 = ax2.tripcolor(triang, err, cmap="Reds", shading="gouraud")
        plt.colorbar(tcf2, ax=ax2, label="|Δn|")
        ax2.set_title(f"|n_pred − n_true|   (RMSE={rmse:.5f})",
                      fontsize=12, fontweight="bold")
        ax2.set_xlabel(x_label)
        ax2.set_aspect("equal")
        _plot_obs_markers(ax2, obs_points)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_loss_history(history: dict[str, list], save_path: str = "loss_history.png",
                      dpi: int = 150) -> None:
    """Plot training loss curves (log scale) and n statistics."""
    main_keys  = [k for k in ["total", "fvm", "obs", "smooth", "bound", "btc"] if history.get(k)]
    extra_keys = [k for k in ["n_mean", "n_std", "n_rmse", "btc_stage_err_m"] if history.get(k)]
    nrows = 2 if extra_keys else 1

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), dpi=dpi)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    for k in main_keys:
        vals = np.clip(np.array(history[k]), 1e-12, None)
        ax.semilogy(vals, label=k, linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss History", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    if extra_keys:
        ax2 = axes[1]
        for k in extra_keys:
            vals = np.array(history[k])
            if k == "btc_stage_err_m":
                ax2.plot(vals, label="BTC stage error (m)", linewidth=1.5, color="crimson")
            else:
                ax2.plot(vals, label=k, linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_title("Manning's n Statistics & BTC Stage Error", fontweight="bold")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_flow_snapshot(
    cell_xy: np.ndarray, h: np.ndarray, u: np.ndarray, v: np.ndarray,
    t: float, save_path: str = "snapshot.png", dpi: int = 150,
    obs_points: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Depth + velocity magnitude + quiver at one time step."""
    cell_xy, h, u, v = _sample_spatial_arrays(cell_xy, h, u, v)
    cell_xy, obs_points, x_label, y_label = _local_plot_coordinates(cell_xy, obs_points)
    x, y   = cell_xy[:, 0], cell_xy[:, 1]
    triang = _triangulate(x, y)
    speed  = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)

    tcf0 = axes[0].tripcolor(triang, h, cmap="Blues", shading="gouraud")
    plt.colorbar(tcf0, ax=axes[0], label="Depth h (m)")
    axes[0].set_title(f"Depth  t={t:.0f}s", fontweight="bold")
    axes[0].set_xlabel(x_label); axes[0].set_ylabel(y_label)
    axes[0].set_aspect("equal")
    _plot_obs_markers(axes[0], obs_points)

    tcf1 = axes[1].tripcolor(triang, speed, cmap="plasma", shading="gouraud")
    plt.colorbar(tcf1, ax=axes[1], label="Speed (m/s)")
    axes[1].set_title(f"Speed  t={t:.0f}s", fontweight="bold")
    axes[1].set_xlabel(x_label)
    axes[1].set_aspect("equal")

    step = max(1, len(x) // 500)
    axes[2].quiver(x[::step], y[::step], u[::step], v[::step],
                   speed[::step], cmap="plasma", alpha=0.8)
    axes[2].set_title(f"Velocity vectors  t={t:.0f}s", fontweight="bold")
    axes[2].set_xlabel(x_label)
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_residual_map(
    cell_xy: np.ndarray,
    R_cont: np.ndarray, R_momx: np.ndarray, R_momy: np.ndarray,
    save_path: str = "residual_map.png", dpi: int = 150,
) -> None:
    """Absolute FVM residual maps."""
    cell_xy, R_cont, R_momx, R_momy = _sample_spatial_arrays(
        cell_xy, R_cont, R_momx, R_momy
    )
    cell_xy, _, x_label, y_label = _local_plot_coordinates(cell_xy, None)
    x, y   = cell_xy[:, 0], cell_xy[:, 1]
    triang = _triangulate(x, y)
    titles = ["Continuity Residual", "X-Momentum Residual", "Y-Momentum Residual"]
    resids = [np.abs(R_cont), np.abs(R_momx), np.abs(R_momy)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=dpi)
    for ax, res, title in zip(axes, resids, titles):
        vmax = np.percentile(res, 95) + 1e-12
        tcf  = ax.tripcolor(triang, res.clip(0, vmax), cmap="hot_r", shading="gouraud")
        plt.colorbar(tcf, ax=ax, label="|Residual|")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_mesh_bc_obs(
    graph,
    obs_points: dict[str, tuple[float, float]] | None = None,
    btc_cell_idx: int | None = None,
    save_path: str = "mesh_bc_obs.png",
    dpi: int = 200,
) -> None:
    """Plot mesh overview plus a local inset around the snapped observation cell."""
    cell_xy = graph.pos.cpu().numpy()
    from matplotlib.collections import LineCollection
    import numpy as np

    N = len(cell_xy)
    
    fig, ax = plt.subplots(figsize=(8, 12), dpi=dpi)
    
    # Extract face lines if available, otherwise fallback to dual graph edges
    if hasattr(graph, "face_lines") and graph.face_lines is not None:
        segments = graph.face_lines.cpu().numpy()
    else:
        src, dst = graph.edge_index.cpu().numpy()
        mask = src < dst
        src, dst = src[mask], dst[mask]
        p1 = cell_xy[src]
        p2 = cell_xy[dst]
        segments = np.stack([p1, p2], axis=1)

    # Subsample segments for large meshes (>50k nodes) to avoid matplotlib freeze
    if N > 50000 and len(segments) > 50000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(segments), size=50000, replace=False)
        segments = segments[idx]
        print(f"[Viz] Mesh large ({N} nodes): subsampled to 50k faces for plotting")
    
    # Plot true mesh as a LineCollection
    lc = LineCollection(segments, colors="0.65", linewidths=0.15, alpha=0.5, rasterized=True)
    ax.add_collection(lc)
    ax.autoscale()

    if obs_points and "BTC" in obs_points:
        x_btc, y_btc = obs_points["BTC"]
        label = "USGS Butte City Gauge"
        if btc_cell_idx is not None:
            label += f" (Node #{btc_cell_idx})"
        ax.scatter(x_btc, y_btc, s=150, color="limegreen", edgecolor="black",
                   linewidth=1.0, zorder=12, label=label)
        ax.annotate(
            "BTC",
            xy=(x_btc, y_btc),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="darkred", alpha=0.9),
            zorder=13,
        )

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles = [
        Line2D([0], [0], color="0.55", lw=1.0, label="River Channel Banks"),
        Line2D([0], [0], color="blue", lw=3.0, label="Upstream BC Line (Ord Ferry)"),
        Line2D([0], [0], color="red", lw=3.0, label="Downstream BC Line (Colusa)"),
        *handles,
        Line2D([0], [0], color="green", lw=1.5, ls="--", label="Mesh Inset (Zoomed Section)"),
    ]

    ax.set_title("Sacramento River Reach - Unstructured 2D Mesh, BCs & Gauge",
                 fontweight="bold")
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0),
              fontsize=8, framealpha=0.9)

    if obs_points and "BTC" in obs_points:
        inset = fig.add_axes([0.58, 0.61, 0.34, 0.24])
        _plot_local_mesh_inset(inset, graph, cell_xy, cell_xy, obs_points["BTC"], btc_cell_idx)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# New study-critical visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_btc_stage_comparison(
    t_sec:          np.ndarray,
    btc_wse_obs:    np.ndarray,
    h_series:       np.ndarray | torch.Tensor,
    btc_cell_idx:   int,
    z_btc:          float,
    save_path:      str = "btc_stage_comparison.png",
    dpi:            int = 200,
    sim_start_str:  str = "2026-01-01",
) -> None:
    """
    Plot observed vs. model-predicted water surface elevation at BTC gauge.

    Panels:
      1. Observed BTC WSE (solid blue with markers) vs. predicted WSE (dashed red),
         with shaded residual area and RMSE/MAE annotations.
      2. Residual (pred − obs) with shaded over/under prediction regions.

    Parameters
    ----------
    t_sec          : [T] simulation time in seconds
    btc_wse_obs    : [T] observed WSE at BTC (m NAVD88); NaN where missing
    h_series       : [T, N] predicted water depth (m)
    btc_cell_idx   : mesh cell index for BTC gauge
    z_btc          : bed elevation at BTC cell (m NAVD88)
    """
    if isinstance(h_series, torch.Tensor):
        h_series = h_series.cpu().numpy()

    # Model-predicted WSE at BTC
    h_btc    = h_series[:, btc_cell_idx]   # [T]
    pred_wse = h_btc + z_btc               # [T]

    # Convert time axis to date-like hourly labels
    t_hours  = t_sec / 3600.0

    obs_mask = ~np.isnan(btc_wse_obs)
    residual = pred_wse - btc_wse_obs      # NaN where observation missing
    rmse     = float(np.sqrt(np.nanmean(residual**2))) if obs_mask.any() else float("nan")
    mae      = float(np.nanmean(np.abs(residual)))     if obs_mask.any() else float("nan")
    bias     = float(np.nanmean(residual))              if obs_mask.any() else float("nan")

    # Mark missing data gaps for annotation
    missing_mask = np.isnan(btc_wse_obs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=dpi, sharex=True)
    fig.suptitle(
        f"BTC Gauge: Observed vs. Model-Predicted WSE  —  January 2026\n"
        f"Cell #{btc_cell_idx}  |  z_bed={z_btc:.2f} m NAVD88",
        fontsize=12, fontweight="bold",
    )

    # ── Panel 1: Stage time series ────────────────────────────────────────
    ax = axes[0]
    # Shaded residual fill between obs and pred
    ax.fill_between(
        t_hours[obs_mask], btc_wse_obs[obs_mask], pred_wse[obs_mask],
        where=pred_wse[obs_mask] >= btc_wse_obs[obs_mask],
        color="red", alpha=0.12, label="_nolegend_",
    )
    ax.fill_between(
        t_hours[obs_mask], btc_wse_obs[obs_mask], pred_wse[obs_mask],
        where=pred_wse[obs_mask] < btc_wse_obs[obs_mask],
        color="blue", alpha=0.12, label="_nolegend_",
    )
    ax.plot(t_hours[obs_mask], btc_wse_obs[obs_mask], "b-o",
            linewidth=1.5, markersize=2, label="Observed WSE — CDEC BTC (m NAVD88)", alpha=0.9)
    ax.plot(t_hours, pred_wse, "r--",
            linewidth=1.8, label="Model-predicted WSE (h + z_bed)", alpha=0.85)

    # Mark missing data gaps as grey spans
    if missing_mask.any():
        in_gap = False
        gap_start = 0
        for i, m in enumerate(missing_mask):
            if m and not in_gap:
                gap_start = i; in_gap = True
            elif not m and in_gap:
                ax.axvspan(t_hours[gap_start], t_hours[i], color="grey",
                           alpha=0.2, label="_nolegend_")
                in_gap = False
        if in_gap:
            ax.axvspan(t_hours[gap_start], t_hours[-1], color="grey", alpha=0.2)

    # Metrics text box
    metrics_txt = f"RMSE = {rmse:.3f} m\nMAE  = {mae:.3f} m\nBias = {bias:+.3f} m"
    ax.text(0.98, 0.97, metrics_txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    ax.set_ylabel("Water Surface Elevation (m NAVD88)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)

    # ── Panel 2: Residual ─────────────────────────────────────────────────
    ax2 = axes[1]
    resid_valid = residual[obs_mask]
    ax2.plot(t_hours[obs_mask], resid_valid, "k-", linewidth=1.0, alpha=0.8, label="Residual")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.fill_between(t_hours[obs_mask], resid_valid, 0,
                     where=resid_valid >= 0, color="red", alpha=0.18, label="Over-predict")
    ax2.fill_between(t_hours[obs_mask], resid_valid, 0,
                     where=resid_valid < 0, color="blue", alpha=0.18, label="Under-predict")
    ax2.set_xlabel(f"Hours since {sim_start_str} (Jan 2026)", fontsize=11)
    ax2.set_ylabel("Residual  pred − obs  (m)", fontsize=11)
    ax2.set_title(f"Residual  RMSE={rmse:.3f} m   Bias={bias:+.3f} m", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}  (RMSE={rmse:.3f} m  MAE={mae:.3f} m  Bias={bias:+.3f} m)")


def plot_boundary_conditions(
    t_sec:        np.ndarray,
    discharge:    np.ndarray,
    stage_down:   np.ndarray,
    save_path:    str = "boundary_conditions.png",
    dpi:          int = 200,
    sim_start_str: str = "2026-01-01",
) -> None:
    """
    Two-panel time series of upstream and downstream boundary conditions.

    Top panel   : Ord Ferry upstream discharge (m³/s)
    Bottom panel: Colusa downstream stage (m NAVD88)
    Missing data gaps are shaded grey.

    Parameters
    ----------
    t_sec      : [T] simulation time in seconds
    discharge  : [T] upstream Q (m³/s)
    stage_down : [T] downstream WSE (m)
    """
    t_hours = t_sec / 3600.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=dpi, sharex=True)
    fig.suptitle(
        "Boundary Conditions — Sacramento River  |  January 2026",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Upstream discharge ───────────────────────────────────────
    ax1 = axes[0]
    q_nan = np.isnan(discharge)
    q_plot = np.where(q_nan, np.nan, discharge)
    ax1.plot(t_hours, q_plot, "steelblue", linewidth=1.8,
             label="Ord Ferry Discharge (m³/s)")
    ax1.fill_between(t_hours, q_plot, 0, where=~q_nan,
                     color="steelblue", alpha=0.15)
    _shade_missing(ax1, t_hours, q_nan)

    # Mark peak
    if not q_nan.all():
        peak_idx = int(np.nanargmax(discharge))
        ax1.axvline(t_hours[peak_idx], color="darkorange", linestyle="--",
                    linewidth=1.2, label=f"Peak Q={discharge[peak_idx]:.0f} m³/s @ t={t_hours[peak_idx]:.0f}h")
    ax1.set_ylabel("Discharge (m³/s)", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ── Panel 2: Downstream stage ─────────────────────────────────────────
    ax2 = axes[1]
    s_nan = np.isnan(stage_down)
    s_plot = np.where(s_nan, np.nan, stage_down)
    ax2.plot(t_hours, s_plot, "darkorange", linewidth=1.8,
             label="Colusa Stage (m NAVD88)")
    ax2.fill_between(t_hours, s_plot, np.nanmin(s_plot) - 0.1,
                     where=~s_nan, color="darkorange", alpha=0.15)
    _shade_missing(ax2, t_hours, s_nan)
    ax2.set_xlabel(f"Hours since {sim_start_str}", fontsize=11)
    ax2.set_ylabel("Stage — WSE (m NAVD88)", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_manning_n_histogram(
    n_pred:    np.ndarray,
    save_path: str = "manning_n_histogram.png",
    dpi:       int = 200,
) -> None:
    """
    Histogram + KDE of predicted Manning's n values across all mesh cells.

    Reference values are marked:
      n=0.025 (main channel), n=0.050 (floodplain), n=0.080 (dense vegetation)
    """
    from scipy.stats import gaussian_kde  # soft dep — catch if missing

    n_flat  = n_pred.ravel()
    n_mean  = float(np.mean(n_flat))
    n_std   = float(np.std(n_flat))
    n_min   = float(np.min(n_flat))
    n_max   = float(np.max(n_flat))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Histogram
    counts, bins, patches = ax.hist(
        n_flat, bins=80, density=True,
        color="steelblue", alpha=0.6, edgecolor="white", linewidth=0.4,
        label="Cell n values",
    )

    # KDE overlay
    try:
        kde = gaussian_kde(n_flat)
        x_kde = np.linspace(n_flat.min() - 0.002, n_flat.max() + 0.002, 300)
        ax.plot(x_kde, kde(x_kde), "k-", linewidth=2.0, label="KDE")
    except Exception:
        pass   # scipy not available — skip KDE

    # Reference vertical lines
    ref_vals  = [0.025, 0.050, 0.080]
    ref_labels = ["n=0.025\n(main channel)", "n=0.050\n(floodplain)", "n=0.080\n(vegetation)"]
    ref_colors = ["navy", "forestgreen", "saddlebrown"]
    for rv, rl, rc in zip(ref_vals, ref_labels, ref_colors):
        ax.axvline(rv, color=rc, linestyle="--", linewidth=1.6, alpha=0.8, label=rl)

    # Stats text box
    stats_txt = (
        f"N cells = {len(n_flat):,}\n"
        f"Mean  = {n_mean:.4f}\n"
        f"Std   = {n_std:.4f}\n"
        f"Min   = {n_min:.4f}\n"
        f"Max   = {n_max:.4f}"
    )
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="goldenrod", alpha=0.9))

    ax.set_xlabel("Manning's n", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of PIGNN-Predicted Manning's n", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_spatial_n_with_elevation(
    cell_xy:    np.ndarray,
    n_pred:     np.ndarray,
    cell_z:     np.ndarray,
    save_path:  str = "n_vs_elevation.png",
    dpi:        int = 200,
    cmap_n:     str = "RdYlGn_r",
    cmap_z:     str = "terrain",
    obs_points: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Two-panel spatial map:
      Left  : Predicted Manning's n field
      Right : Bed elevation z_bed (m NAVD88)

    Helps visualize correlation between roughness and topography.

    Parameters
    ----------
    cell_z     : [N] bed elevation (m NAVD88)
    obs_points : station markers dict {name: (x_utm, y_utm)}
    """
    cell_xy, n_pred, cell_z = _sample_spatial_arrays(cell_xy, n_pred, cell_z)
    cell_xy, obs_points, x_label, y_label = _local_plot_coordinates(cell_xy, obs_points)
    x, y   = cell_xy[:, 0], cell_xy[:, 1]
    triang = _triangulate(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=dpi)
    fig.suptitle("PIGNN: Manning's n vs. Bed Elevation", fontsize=13, fontweight="bold")

    # ── Left: Manning's n ─────────────────────────────────────────────────
    ax1 = axes[0]
    vmin_n, vmax_n = np.percentile(n_pred, [2, 98])
    tcf1 = ax1.tripcolor(triang, n_pred, cmap=cmap_n,
                         vmin=vmin_n, vmax=vmax_n, shading="gouraud")
    plt.colorbar(tcf1, ax=ax1, label="Manning's n")
    ax1.set_title("Predicted Manning's n", fontsize=11, fontweight="bold")
    ax1.set_xlabel(x_label); ax1.set_ylabel(y_label)
    ax1.set_aspect("equal")
    _stats_text(ax1, n_pred)
    _plot_obs_markers(ax1, obs_points)

    # ── Right: Bed elevation ──────────────────────────────────────────────
    ax2 = axes[1]
    vmin_z, vmax_z = np.percentile(cell_z, [2, 98])
    tcf2 = ax2.tripcolor(triang, cell_z, cmap=cmap_z,
                         vmin=vmin_z, vmax=vmax_z, shading="gouraud")
    plt.colorbar(tcf2, ax=ax2, label="z_bed (m NAVD88)")
    ax2.set_title("Bed Elevation z_bed", fontsize=11, fontweight="bold")
    ax2.set_xlabel(x_label)
    ax2.set_aspect("equal")
    _stats_text(ax2, cell_z)
    _plot_obs_markers(ax2, obs_points)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_convergence_detail(
    history:   dict[str, list],
    save_path: str = "convergence_detail.png",
    dpi:       int = 200,
) -> None:
    """
    Detailed 3-panel training convergence figure:
      Panel 1 : Total loss (log scale)
      Panel 2 : BTC stage RMSE in metres over training epochs
      Panel 3 : Manning's n mean ± std band over epochs
    """
    epochs = np.arange(1, len(history.get("total", [])) + 1)
    if len(epochs) == 0:
        warnings.warn("[Viz] No training history found for convergence plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=dpi, sharex=True)
    fig.suptitle("PIGNN Training Convergence Detail", fontsize=13, fontweight="bold")

    # ── Panel 1: Total loss ───────────────────────────────────────────────
    ax1 = axes[0]
    if history.get("total"):
        vals = np.clip(np.array(history["total"]), 1e-12, None)
        ax1.semilogy(epochs, vals, "b-", linewidth=1.5, label="Total loss")
        # Detect large LR decay events (loss plateau → sudden drop)
        if len(vals) > 20:
            diffs = np.diff(np.log10(vals))
            drops = np.where(diffs < -0.05)[0]
            for d in drops[:5]:   # mark at most 5 events
                ax1.axvline(epochs[d], color="orange", linestyle=":",
                            linewidth=1.0, alpha=0.7)
    for k, color in [("fvm", "green"), ("btc", "crimson"), ("smooth", "purple")]:
        if history.get(k):
            v = np.clip(np.array(history[k]), 1e-12, None)
            ax1.semilogy(epochs[:len(v)], v, "--", linewidth=1.0, color=color,
                         alpha=0.7, label=f"{k} loss")
    ax1.set_ylabel("Loss (log scale)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    # Add LR event legend entry
    ax1.plot([], [], ":", color="orange", linewidth=1.0, alpha=0.7, label="LR decay event")
    ax1.legend(fontsize=8)

    # ── Panel 2: BTC stage RMSE ───────────────────────────────────────────
    ax2 = axes[1]
    if history.get("btc_stage_err_m"):
        errs = np.array(history["btc_stage_err_m"])
        ax2.plot(epochs[:len(errs)], errs, "crimson", linewidth=1.5)
        ax2.fill_between(epochs[:len(errs)], errs, color="crimson", alpha=0.12)
        # Running 50-epoch smoothed line
        if len(errs) > 50:
            smooth = np.convolve(errs, np.ones(50) / 50, mode="valid")
            ax2.plot(epochs[49:49+len(smooth)], smooth, "darkred", linewidth=2.0,
                     linestyle="--", label="50-ep moving avg")
            ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No BTC gauge data", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=11, color="grey")
    ax2.set_ylabel("BTC Stage RMSE (m)")
    ax2.set_title("BTC Stage Error over Training", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Manning's n mean ± std ───────────────────────────────────
    ax3 = axes[2]
    n_mean_arr = np.array(history.get("n_mean", []))
    n_std_arr  = np.array(history.get("n_std", []))
    if len(n_mean_arr) > 0:
        ep = epochs[:len(n_mean_arr)]
        ax3.plot(ep, n_mean_arr, "k-", linewidth=1.8, label="n̄ (mean)")
        ax3.fill_between(ep,
                         n_mean_arr - n_std_arr,
                         n_mean_arr + n_std_arr,
                         color="grey", alpha=0.25, label="±σ")
        if history.get("n_rmse"):
            n_rmse_arr = np.array(history["n_rmse"])
            ax3_r = ax3.twinx()
            ax3_r.plot(epochs[:len(n_rmse_arr)], n_rmse_arr, "orangered",
                       linewidth=1.2, linestyle="--", label="n RMSE vs truth")
            ax3_r.set_ylabel("n RMSE vs. truth")
            ax3_r.legend(fontsize=8, loc="upper right")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Manning's n")
    ax3.set_title("Manning's n Evolution", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_velocity_field_snapshot(
    cell_xy:        np.ndarray,
    h_series:       np.ndarray | torch.Tensor,
    u_series:       np.ndarray | torch.Tensor,
    v_series:       np.ndarray | torch.Tensor,
    t_sec:          np.ndarray,
    discharge:      np.ndarray | None = None,
    save_path:      str = "velocity_field_peak.png",
    dpi:            int = 200,
    obs_points:     dict[str, tuple[float, float]] | None = None,
    sim_start_str:  str = "2026-01-01",
) -> None:
    """
    Velocity field snapshot at the peak-flow time step.

    Shows streamlines / quiver plot of velocity on a depth-colored background.
    Peak is determined by the time step with maximum upstream discharge.

    Parameters
    ----------
    h_series   : [T, N] water depth
    u_series   : [T, N] x-velocity
    v_series   : [T, N] y-velocity
    t_sec      : [T] time in seconds
    discharge  : [T] upstream discharge used to identify peak flow step
    """
    if isinstance(h_series, torch.Tensor): h_series = h_series.cpu().numpy()
    if isinstance(u_series, torch.Tensor): u_series = u_series.cpu().numpy()
    if isinstance(v_series, torch.Tensor): v_series = v_series.cpu().numpy()

    T = len(t_sec)
    if discharge is not None and not np.all(np.isnan(discharge)):
        peak_idx = int(np.nanargmax(discharge))
    else:
        # Fall back to middle of simulation
        peak_idx = T // 2

    h = h_series[peak_idx]
    u = u_series[peak_idx]
    v = v_series[peak_idx]
    speed = np.sqrt(u**2 + v**2)

    peak_h    = float(t_sec[peak_idx]) / 3600.0
    peak_label = f"t = {peak_h:.1f} h"
    if discharge is not None and not np.isnan(discharge[peak_idx]):
        peak_label += f"  (Q = {discharge[peak_idx]:.0f} m³/s)"

    cell_xy, h, u, v, speed = _sample_spatial_arrays(cell_xy, h, u, v, speed)
    cell_xy, obs_points, x_label, y_label = _local_plot_coordinates(cell_xy, obs_points)
    x, y   = cell_xy[:, 0], cell_xy[:, 1]
    triang = _triangulate(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=dpi)
    fig.suptitle(
        f"Peak-Flow Velocity Field — {peak_label}\nSacramento River PIGNN",
        fontsize=12, fontweight="bold",
    )

    # ── Left: Depth background with quiver ───────────────────────────────
    ax1 = axes[0]
    tcf = ax1.tripcolor(triang, h, cmap="Blues", shading="gouraud",
                        alpha=0.85, vmin=0)
    plt.colorbar(tcf, ax=ax1, label="Depth h (m)")
    step = max(1, len(x) // 600)
    qv = ax1.quiver(
        x[::step], y[::step], u[::step], v[::step],
        speed[::step], cmap="plasma", alpha=0.85,
        scale=None, scale_units="xy",
    )
    plt.colorbar(qv, ax=ax1, label="Speed (m/s)")
    ax1.set_title("Depth + Velocity Vectors", fontweight="bold")
    ax1.set_xlabel(x_label); ax1.set_ylabel(y_label)
    ax1.set_aspect("equal")
    _plot_obs_markers(ax1, obs_points)

    # ── Right: Speed magnitude ────────────────────────────────────────────
    ax2 = axes[1]
    tcf2 = ax2.tripcolor(triang, speed, cmap="plasma", shading="gouraud")
    plt.colorbar(tcf2, ax=ax2, label="Speed |V| (m/s)")
    ax2.set_title("Flow Speed", fontweight="bold")
    ax2.set_xlabel(x_label)
    ax2.set_aspect("equal")
    _plot_obs_markers(ax2, obs_points)
    _stats_text(ax2, speed)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[Viz] Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main make_all_plots() — orchestrates all outputs
# ─────────────────────────────────────────────────────────────────────────────

def make_all_plots(
    model, graph, results: dict, history: dict, run_dir: str,
    n_true:         np.ndarray | None = None,
    device=None,
    obs_points:     dict[str, tuple[float, float]] | None = None,
    btc_cell_idx:   int | None        = None,
    btc_wse_series: np.ndarray | None = None,
    z_btc:          float | None      = None,
    # Boundary condition arrays for BC plot
    bc_discharge:   np.ndarray | None = None,
    bc_stage_down:  np.ndarray | None = None,
) -> None:
    """
    Generate all standard output plots after training.

    Parameters
    ----------
    obs_points     : optional {name: (x_utm, y_utm)} for station markers on maps
    btc_cell_idx   : BTC cell index for stage comparison plot
    btc_wse_series : [T] observed BTC WSE (m NAVD88) for comparison plot
    z_btc          : BTC bed elevation for comparison plot
    bc_discharge   : [T] upstream Ord Ferry discharge (m³/s)
    bc_stage_down  : [T] downstream Colusa stage (m)
    """
    if device is None:
        device = next(model.parameters()).device
    run_dir  = Path(run_dir)
    cell_xy  = graph.pos.cpu().numpy()

    # Helper for numpy conversion
    def _np(arr, i=None):
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        return arr if i is None else arr[i]

    # ── Loss history ──────────────────────────────────────────────────────
    try:
        plot_loss_history(history, str(run_dir / "loss_history.png"))
    except Exception as e:
        warnings.warn(f"[Viz] plot_loss_history failed: {e}")

    # ── Convergence detail ────────────────────────────────────────────────
    try:
        plot_convergence_detail(history, str(run_dir / "convergence_detail.png"))
    except Exception as e:
        warnings.warn(f"[Viz] plot_convergence_detail failed: {e}")

    # ── Final n prediction ────────────────────────────────────────────────
    model.eval()
    T   = results["h_series"].shape[0]
    mid = T // 2
    h_in = results["h_series"][mid]
    u_in = results["u_series"][mid]
    v_in = results["v_series"][mid]
    if not isinstance(h_in, torch.Tensor):
        h_in = torch.from_numpy(h_in)
        u_in = torch.from_numpy(u_in)
        v_in = torch.from_numpy(v_in)
    h_in = h_in.to(device); u_in = u_in.to(device); v_in = v_in.to(device)

    with torch.no_grad():
        out = model(graph.to(device), h_in, u_in, v_in)
    n_pred = out["n"].cpu().numpy()

    # ── Manning's n spatial map ───────────────────────────────────────────
    try:
        plot_n_field(cell_xy, n_pred, n_true=n_true,
                     save_path=str(run_dir / "manning_n_field.png"),
                     title="PIGNN: Predicted Manning's n",
                     obs_points=obs_points)
    except Exception as e:
        warnings.warn(f"[Viz] plot_n_field failed: {e}")

    # ── Manning's n histogram ─────────────────────────────────────────────
    try:
        plot_manning_n_histogram(n_pred, str(run_dir / "manning_n_histogram.png"))
    except Exception as e:
        warnings.warn(f"[Viz] plot_manning_n_histogram failed: {e}")

    # ── Spatial n vs. bed elevation ───────────────────────────────────────
    try:
        cell_z_np = graph.cell_z.cpu().numpy() if hasattr(graph, "cell_z") else None
        if cell_z_np is not None:
            plot_spatial_n_with_elevation(
                cell_xy, n_pred, cell_z_np,
                save_path=str(run_dir / "n_vs_elevation.png"),
                obs_points=obs_points,
            )
    except Exception as e:
        warnings.warn(f"[Viz] plot_spatial_n_with_elevation failed: {e}")

    # ── Boundary conditions ───────────────────────────────────────────────
    try:
        t_arr = results["t_series"]
        if isinstance(t_arr, torch.Tensor):
            t_arr = t_arr.cpu().numpy()
        if bc_discharge is not None and bc_stage_down is not None:
            plot_boundary_conditions(
                t_arr, bc_discharge, bc_stage_down,
                save_path=str(run_dir / "boundary_conditions.png"),
            )
    except Exception as e:
        warnings.warn(f"[Viz] plot_boundary_conditions failed: {e}")

    # Determine flow series to plot (predict via GNN if surrogate_mode is True)
    if getattr(model, "surrogate_mode", False):
        print("[Viz] surrogate_mode is True → generating predicted flow fields selectively to optimize speed and memory …")
        # Identify peak-flow time step
        if bc_discharge is not None and not np.all(np.isnan(bc_discharge)):
            peak_idx = int(np.nanargmax(bc_discharge))
        else:
            peak_idx = T // 2

        # Key steps we want full spatial fields for (snapshots + peak velocity)
        save_steps = {0, T // 4, T // 2, 3 * T // 4, T - 1, peak_idx}

        # Stride for BTC gauge (predict BTC WSE every 4th step for plotting speed)
        stride = 4
        run_steps = sorted(list(save_steps.union(set(range(0, T, stride)))))

        full_h = {}
        full_u = {}
        full_v = {}

        btc_h_list = []
        btc_t_list = []

        model.eval()
        with torch.no_grad():
            for ti in run_steps:
                h_in = results["h_series"][ti]
                u_in = results["u_series"][ti]
                v_in = results["v_series"][ti]
                if not isinstance(h_in, torch.Tensor):
                    h_in = torch.from_numpy(h_in)
                    u_in = torch.from_numpy(u_in)
                    v_in = torch.from_numpy(v_in)
                h_in = h_in.to(device); u_in = u_in.to(device); v_in = v_in.to(device)

                out = model(graph.to(device), h_in, u_in, v_in)
                h_val = out["h"].cpu().numpy()
                u_val = out["u"].cpu().numpy()
                v_val = out["v"].cpu().numpy()

                if ti in save_steps:
                    full_h[ti] = h_val
                    full_u[ti] = u_val
                    full_v[ti] = v_val

                if btc_cell_idx is not None:
                    btc_h_list.append(float(h_val[btc_cell_idx]))
                    btc_t_list.append(ti)

        # Interpolate BTC water depths to all time steps
        if btc_cell_idx is not None:
            btc_h_all = np.interp(np.arange(T), btc_t_list, btc_h_list)
            h_plot = btc_h_all[:, None]
            plot_btc_idx = 0
        else:
            h_plot = np.zeros((T, 1), dtype=np.float32)
            plot_btc_idx = 0

        u_plot = None
        v_plot = None
    else:
        h_plot = _np(results["h_series"])
        u_plot = _np(results["u_series"])
        v_plot = _np(results["v_series"])
        full_h = h_plot
        full_u = u_plot
        full_v = v_plot
        plot_btc_idx = btc_cell_idx

    # ── Flow snapshots at 5 time points ──────────────────────────────────
    for ti in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
        if ti < 0 or ti >= T:
            continue
        try:
            plot_flow_snapshot(
                cell_xy, full_h[ti], full_u[ti], full_v[ti],
                float(t_arr[ti]),
                save_path=str(run_dir / f"snapshot_t{ti:04d}.png"),
                obs_points=obs_points,
            )
        except Exception as e:
            warnings.warn(f"[Viz] plot_flow_snapshot t={ti} failed: {e}")

    # ── BTC stage comparison ──────────────────────────────────────────────
    try:
        if btc_cell_idx is not None and btc_wse_series is not None and z_btc is not None:
            plot_btc_stage_comparison(
                t_sec        = t_arr,
                btc_wse_obs  = btc_wse_series,
                h_series     = h_plot,
                btc_cell_idx = plot_btc_idx,
                z_btc        = z_btc,
                save_path    = str(run_dir / "btc_stage_comparison.png"),
            )
    except Exception as e:
        warnings.warn(f"[Viz] plot_btc_stage_comparison failed: {e}")

    # ── Peak-flow velocity field ──────────────────────────────────────────
    try:
        plot_velocity_field_snapshot(
            cell_xy       = cell_xy,
            h_series      = full_h,
            u_series      = full_u,
            v_series      = full_v,
            t_sec         = t_arr,
            discharge     = bc_discharge,
            save_path     = str(run_dir / "velocity_field_peak.png"),
            obs_points    = obs_points,
        )
    except Exception as e:
        warnings.warn(f"[Viz] plot_velocity_field_snapshot failed: {e}")

    print(f"[Viz] All plots saved to {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_spatial_arrays(
    cell_xy: np.ndarray,
    *arrays: np.ndarray | None,
    max_points: int = 50000,
) -> tuple[np.ndarray, ...]:
    """
    Downsample large spatial arrays for plotting only.

    Full-resolution Delaunay triangulation on 200k+ mesh-cell centres can dominate
    runtime after training. A fixed stride keeps maps deterministic and fast while
    preserving the whole domain footprint for diagnostics.
    """
    n = len(cell_xy)
    if n <= max_points:
        return (cell_xy, *arrays)

    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=np.int64)
    sampled = [cell_xy[idx]]
    for arr in arrays:
        sampled.append(None if arr is None else arr[idx])
    print(f"[Viz] Plotting {len(idx):,}/{n:,} mesh cells for spatial map speed.")
    return tuple(sampled)


def _local_plot_coordinates(
    cell_xy: np.ndarray,
    obs_points: dict[str, tuple[float, float]] | None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]] | None, str, str]:
    """Convert large UTM coordinates to local plot offsets for readable ticks."""
    origin_x = np.floor(float(np.nanmin(cell_xy[:, 0])) / 1000.0) * 1000.0
    origin_y = np.floor(float(np.nanmin(cell_xy[:, 1])) / 1000.0) * 1000.0

    local_xy = cell_xy.copy()
    local_xy[:, 0] -= origin_x
    local_xy[:, 1] -= origin_y

    local_obs = None
    if obs_points:
        local_obs = {
            name: (x - origin_x, y - origin_y)
            for name, (x, y) in obs_points.items()
        }

    x_label = f"Easting - {origin_x:.0f} (m)"
    y_label = f"Northing - {origin_y:.0f} (m)"
    return local_xy, local_obs, x_label, y_label


def _triangulate(x: np.ndarray, y: np.ndarray) -> tri.Triangulation:
    """Delaunay triangulation with long-edge masking (removes domain boundary artefacts)."""
    triang  = tri.Triangulation(x, y)
    pts     = np.column_stack([x, y])
    tris    = triang.triangles
    p0, p1, p2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    longest = np.maximum.reduce([
        np.linalg.norm(p0 - p1, axis=1),
        np.linalg.norm(p1 - p2, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
    ])
    triang.set_mask(longest > np.percentile(longest, 95))
    return triang


def _stats_text(ax, values: np.ndarray) -> None:
    text = (
        f"min {values.min():.4f}\n"
        f"mean {values.mean():.4f}\n"
        f"max {values.max():.4f}"
    )
    ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    _format_spatial_axis(ax)


def _format_spatial_axis(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def _plot_obs_markers(
    ax,
    obs_points: dict[str, tuple[float, float]] | None,
) -> None:
    """Overlay observation station markers (red star) on a spatial axes."""
    _format_spatial_axis(ax)
    if not obs_points:
        return
    for name, (x, y) in obs_points.items():
        ax.plot(x, y, marker="*", markersize=12, color="red",
                markeredgecolor="darkred", markeredgewidth=0.8, zorder=10)
        ax.annotate(
            name, xy=(x, y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold", color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="darkred", alpha=0.8),
            zorder=11,
        )


def _plot_local_mesh_inset(
    ax,
    graph,
    cell_xy_raw: np.ndarray,
    cell_xy_local: np.ndarray,
    btc_xy_local: tuple[float, float],
    btc_cell_idx: int | None,
    radius_m: float = 450.0,
) -> None:
    """Draw a local mesh inset around the snapped BTC cell."""
    x_btc, y_btc = btc_xy_local
    dist = np.linalg.norm(cell_xy_local - np.array([x_btc, y_btc]), axis=1)
    nearby = dist <= radius_m
    nearby_idx = np.where(nearby)[0]

    from matplotlib.collections import LineCollection
    if hasattr(graph, "face_lines") and graph.face_lines is not None:
        segments = graph.face_lines.cpu().numpy()
        fci = graph.face_cell_idx.cpu().numpy()
        keep = nearby[fci[:, 0]] & nearby[fci[:, 1]]
        segments = segments[keep]
    else:
        src, dst = graph.edge_index.cpu().numpy()
        mask = (src < dst) & nearby[src] & nearby[dst]
        src, dst = src[mask], dst[mask]
        p1 = cell_xy_local[src]
        p2 = cell_xy_local[dst]
        segments = np.stack([p1, p2], axis=1)
        
    lc = LineCollection(segments, colors="0.55", linewidths=0.5, zorder=1)
    ax.add_collection(lc)

    ax.scatter(cell_xy_local[nearby_idx, 0], cell_xy_local[nearby_idx, 1],
               s=5, color="0.45", alpha=0.8, zorder=2)
    ax.scatter(x_btc, y_btc, s=90, color="limegreen", edgecolor="black",
               linewidth=0.9, zorder=4)
    if btc_cell_idx is not None:
        ax.annotate(
            f"BTC #{btc_cell_idx}",
            xy=(x_btc, y_btc),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="darkgreen",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="darkgreen", alpha=0.9),
            zorder=5,
        )

    ax.set_xlim(x_btc - radius_m, x_btc + radius_m)
    ax.set_ylim(y_btc - radius_m, y_btc + radius_m)
    ax.set_title("Mesh Detail Around Snapped BTC Cell", fontsize=9, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    _format_spatial_axis(ax)


def _shade_missing(ax, t_hours: np.ndarray, nan_mask: np.ndarray) -> None:
    """Shade time periods where data is missing (NaN) as semi-transparent grey spans."""
    if not nan_mask.any():
        return
    in_gap  = False
    gs      = 0
    for i, m in enumerate(nan_mask):
        if m and not in_gap:
            gs = i; in_gap = True
        elif not m and in_gap:
            ax.axvspan(t_hours[gs], t_hours[i], color="grey", alpha=0.25,
                       label="_nolegend_")
            in_gap = False
    if in_gap:
        ax.axvspan(t_hours[gs], t_hours[-1], color="grey", alpha=0.25,
                   label="_nolegend_")
