"""
PIGNN Main Entry Point
======================
Physics-Informed Graph Neural Network for Inverse Manning's n Estimation
on HEC-RAS Unstructured Meshes.

Usage:
  python main.py [options]

Options:
  --config        Path to config.yaml  (default: config.yaml)
  --results-hdf   Path to HEC-RAS results HDF5 (overrides config)
  --run-dir       Output directory (overrides config)
  --epochs        Number of training epochs (overrides config)
  --device        'cuda' or 'cpu' (overrides config)
  --inspect-hdf   Print HDF5 tree and exit
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from graph.build_graph  import load_hecras_geometry, build_pyg_graph
from graph.mesh_utils   import inspect_hdf5, print_graph_stats, validate_graph
from data.hecras_loader import load_hecras_results, load_synthetic_results, to_tensors, load_boundary_driven_results
from data.obs_loader    import load_btc_stage, load_boundary_conditions, snap_to_mesh
from model.pignn        import PIGNN
from train.trainer      import PINNTrainer, TrainConfig, set_seed
from viz.visualize      import make_all_plots, plot_mesh_bc_obs

print("[DIAG] All imports complete", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    print("[DIAG] main() started", flush=True)

    # ── 1. Configuration ──────────────────────────────────────────────────
    cfg = load_config(args.config or "config.yaml")
    if args.device:      cfg["training"]["device"]         = args.device
    if args.epochs:      cfg["training"]["epochs"]         = args.epochs
    if args.results_hdf: cfg["geometry"]["results_hdf"]    = args.results_hdf
    if args.run_dir:     cfg["training"]["checkpoint_dir"] = args.run_dir

    use_cuda = torch.cuda.is_available() and cfg["training"]["device"] == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")
    print("[DIAG] torch.cuda done", flush=True)
    print(f"\n[Main] Device: {device}", flush=True)

    # Memory Check & Safety Safeguard
    if use_cuda:
        try:
            free_vram, total_vram = torch.cuda.mem_get_info()
            free_gb = free_vram / (1024 ** 3)
            total_gb = total_vram / (1024 ** 3)
            print(f"[Main] GPU VRAM: Free = {free_gb:.2f} GB, Total = {total_gb:.2f} GB", flush=True)
            if free_gb < 2.0:
                print(f"[Main] WARNING: Low free VRAM ({free_gb:.2f} GB < 2.0 GB). Training might OOM.", flush=True)
            
            # Print safety check warning for large meshes on low VRAM
            if cfg.get("geometry", {}).get("max_cells") is None and free_gb < 3.0:
                print(f"[Main] WARNING: Free VRAM ({free_gb:.2f} GB) is less than 3 GB. Full-scale training on the entire HEC-RAS mesh might run out of memory.", flush=True)
        except Exception as e:
            print(f"[Main] Note: Could not query GPU memory info: {e}", flush=True)

    set_seed(cfg["training"]["seed"])

    run_dir = Path(cfg["training"]["checkpoint_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.dump(cfg, f)

    # ── 2. (Optional) HDF5 inspection ────────────────────────────────────
    if args.inspect_hdf:
        print("\n[Main] Geometry HDF5 structure:")
        inspect_hdf5(cfg["geometry"]["hdf_path"])
        results_hdf = cfg["geometry"].get("results_hdf")
        if results_hdf and Path(results_hdf).exists():
            print("\n[Main] Results HDF5 structure:")
            inspect_hdf5(results_hdf)
        return

    # ── 3. Build graph from HEC-RAS geometry ─────────────────────────────
    print("\n[Main] Building graph from HEC-RAS geometry …", flush=True)
    geo = load_hecras_geometry(
        hdf_path  = cfg["geometry"]["hdf_path"],
        area_name = cfg["geometry"].get("area_name"),
        max_cells = cfg["geometry"].get("max_cells"),
    )
    graph = build_pyg_graph(
        geo, 
        bathy_tif=cfg["geometry"].get("bathy_tif"),
        lulc_tif=cfg["geometry"].get("lulc_tif")
    )
    print_graph_stats(graph)

    warnings = validate_graph(graph)
    if warnings:
        print("[Main] ⚠  Graph warnings:", flush=True)
        for w in warnings:
            print(f"       • {w}", flush=True)
    else:
        print("[Main] ✓ Graph validation passed.", flush=True)

    if args.no_train:
        print("[Main] --no-train flag set. Exiting after graph construction.", flush=True)
        return

    # ── 4. Load simulation results & boundary conditions ──────────────────
    results_hdf = cfg["geometry"].get("results_hdf")
    boundary_cfg = cfg.get("boundary")
    obs_cfg = cfg.get("observations")

    bc_discharge = None
    bc_stage_down = None
    btc_wse_series = None
    btc_cell_idx = None
    obs_points = None

    # ── 4b. Load Butte City (BTC) observations & Snap ────────────────────
    if obs_cfg:
        btc_csv = obs_cfg.get("btc_csv")
        btc_lat = obs_cfg.get("btc_lat")
        btc_lon = obs_cfg.get("btc_lon")
        snap_radius = obs_cfg.get("btc_snap_radius_m", 5000.0)
        snap_offset_e = obs_cfg.get("btc_snap_offset_easting_m", 0.0)
        snap_offset_n = obs_cfg.get("btc_snap_offset_northing_m", 0.0)

        if btc_csv and Path(btc_csv).exists() and btc_lat is not None and btc_lon is not None:
            print(f"\n[Main] Loading Butte City observations from {btc_csv} ...", flush=True)
            btc_wse_series = load_btc_stage(btc_csv)
            btc_cell_idx = snap_to_mesh(
                lat               = btc_lat,
                lon               = btc_lon,
                cell_xy           = geo["cell_xy"],
                max_radius_m      = snap_radius,
                offset_easting_m  = snap_offset_e,
                offset_northing_m = snap_offset_n,
            )
            # Create obs_points for visualization overlays
            if btc_cell_idx is not None:
                btc_x = float(geo["cell_xy"][btc_cell_idx, 0])
                btc_y = float(geo["cell_xy"][btc_cell_idx, 1])
                obs_points = {"BTC": (btc_x, btc_y)}
                plot_mesh_bc_obs(
                    graph,
                    obs_points=obs_points,
                    btc_cell_idx=btc_cell_idx,
                    save_path=str(run_dir / "mesh_bc_obs.png"),
                )

    if boundary_cfg and not (results_hdf and Path(results_hdf).exists()):
        print("\n[Main] Running in boundary-driven data assimilation mode (no HEC-RAS results).", flush=True)
        # Load boundary conditions
        bc = load_boundary_conditions(
            discharge_csv = boundary_cfg["upstream_discharge_csv"],
            stage_csv     = boundary_cfg["downstream_stage_csv"],
        )
        bc_discharge = bc["discharge"]
        bc_stage_down = bc["stage_down"]

        # Generate boundary-driven synthetic flow fields using graph.cell_z with terrain bathymetry
        raw = load_boundary_driven_results(
            cell_xy  = geo["cell_xy"],
            cell_z   = graph.cell_z.cpu().numpy(),
            bc       = bc,
            n_init   = cfg["model"].get("n_init", 0.035),
            g        = cfg["physics"].get("g", 9.81),
            hdf_path = cfg["geometry"]["hdf_path"],
            btc_wse_series = btc_wse_series,
            btc_cell_idx   = btc_cell_idx,
        )
    elif results_hdf and Path(results_hdf).exists():
        print(f"\n[Main] Loading HEC-RAS results: {results_hdf}")
        raw = load_hecras_results(
            results_hdf  = results_hdf,
            area_name    = geo["area_name"],
            cell_z       = graph.cell_z.cpu().numpy(),
            time_range   = (cfg["data"].get("t_start", 0.0),
                            cfg["data"].get("t_end",   2678400.0)),
        )
    else:
        print("\n[Main] No results HDF5 or boundary config found → using synthetic testing data.")
        dt = cfg["data"].get("dt", 60.0)
        t_series = np.arange(
            cfg["data"].get("t_start", 0.0),
            cfg["data"].get("t_end",   3600.0) + dt,
            dt,
        )
        raw = load_synthetic_results(cell_xy=geo["cell_xy"], t_series=t_series)

    results = to_tensors(raw, device)

    # Ground-truth n for validation plots
    n_true_np = raw.get("n_series")
    if isinstance(n_true_np, torch.Tensor):
        n_true_np = n_true_np.cpu().numpy()
    if n_true_np is not None and n_true_np.ndim == 2:
        n_true_np = n_true_np.mean(axis=0)

    # ── 5. Build PIGNN model ──────────────────────────────────────────────
    print("\n[Main] Building PIGNN model …")
    mcfg  = cfg["model"]
    model = PIGNN(
        node_feat_dim  = graph.x.shape[1],
        edge_feat_dim  = graph.edge_attr.shape[1],
        flow_feat_dim  = 3,
        hidden_dim     = mcfg["hidden_dim"],
        n_layers       = mcfg["n_message_passing"],
        dropout        = mcfg.get("dropout", 0.0),
        n_min          = mcfg["n_min"],
        n_max          = mcfg["n_max"],
        use_attention  = mcfg.get("architecture", "encoder_process_decode") != "edgeconv",
        surrogate_mode = True,
        use_checkpoint = mcfg.get("use_checkpoint", True),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Main] Trainable parameters: {n_params:,}")

    # ── 6. Train ──────────────────────────────────────────────────────────
    tcfg = cfg["training"]
    pcfg = cfg["physics"]
    train_cfg = TrainConfig(
        lr                = tcfg.get("lr", 1e-3),
        lr_decay_patience = tcfg.get("lr_decay_patience", 300),
        lr_decay_factor   = tcfg.get("lr_decay_factor", 0.5),
        epochs            = tcfg.get("epochs", 3000),
        batch_time_steps  = tcfg.get("batch_time_steps", 8),
        w_fvm             = tcfg.get("w_fvm", 1.0),
        w_obs             = tcfg.get("w_obs", 10.0),
        w_smooth          = tcfg.get("w_smooth", 0.05),
        w_bound           = tcfg.get("w_bound", 0.1),
        w_btc             = tcfg.get("w_btc", 50.0),
        w_z               = tcfg.get("w_z", 10.0),
        flux_scheme       = pcfg.get("flux_scheme", "lax_friedrichs"),
        h_min             = pcfg.get("h_min", 1e-3),
        seed              = tcfg.get("seed", 42),
        log_every         = tcfg.get("log_every", 100),
        checkpoint_dir    = str(run_dir),
    )
    trainer = PINNTrainer(
        model, graph, results, train_cfg, device,
        btc_cell_idx   = btc_cell_idx,
        btc_wse_series = btc_wse_series,
    )
    history = trainer.train()

    # ── 7. Visualize ──────────────────────────────────────────────────────
    print("\n[Main] Generating plots …")
    z_btc = trainer.z_btc if hasattr(trainer, "z_btc") else None
    make_all_plots(
        model, graph, results, history,
        run_dir        = str(run_dir),
        n_true         = n_true_np,
        device         = device,
        obs_points     = obs_points,
        btc_cell_idx   = btc_cell_idx,
        btc_wse_series = btc_wse_series,
        z_btc          = z_btc,
        bc_discharge   = bc_discharge,
        bc_stage_down  = bc_stage_down,
    )

    # ── 8. Export final n field ───────────────────────────────────────────
    model.eval()
    T_tot = results["h_series"].shape[0]
    mid   = T_tot // 2
    with torch.no_grad():
        final = model(graph.to(device),
                      results["h_series"][mid].to(device),
                      results["u_series"][mid].to(device),
                      results["v_series"][mid].to(device))
    n_final = final["n"].cpu().numpy()
    np.save(str(run_dir / "n_field_final.npy"), n_final)
    np.savetxt(
        str(run_dir / "n_field_final.csv"),
        np.column_stack([graph.pos.cpu().numpy(), n_final[:, None]]),
        delimiter=",", header="x,y,n", comments="",
    )
    print(f"\n[Main] n field stats:  mean={n_final.mean():.4f}  "
          f"std={n_final.std():.4f}  min={n_final.min():.4f}  max={n_final.max():.4f}")
    print(f"[Main] All outputs → {run_dir}")
    print("[Main] Done. ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PIGNN: Inverse Manning's n estimation on HEC-RAS unstructured mesh."
    )
    p.add_argument("--config",       default="config.yaml",
                   help="YAML config file (default: config.yaml)")
    p.add_argument("--results-hdf",  default=None,
                   help="Path to HEC-RAS unsteady results HDF5")
    p.add_argument("--run-dir",      default=None,
                   help="Output directory (overrides config.yaml)")
    p.add_argument("--epochs",       type=int, default=None,
                   help="Training epochs (overrides config.yaml)")
    p.add_argument("--device",       default=None, choices=["cuda", "cpu"],
                   help="Compute device (overrides config.yaml)")
    p.add_argument("--inspect-hdf",  action="store_true",
                   help="Print HDF5 tree structure and exit")
    p.add_argument("--no-train",     action="store_true",
                   help="Build graph only; skip training")
    return p


if __name__ == "__main__":
    main(_parser().parse_args())
