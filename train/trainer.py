"""
PIGNN Trainer
=============
Orchestrates the inverse Manning's n estimation training loop.

Each epoch:
  1. Sample a random batch of consecutive time steps
  2. Forward pass → n_pred per cell
  3. FVM residuals for each time transition t → t+1
  4. BTC stage observation loss (sparse real gauge constraint)
  5. Backpropagate weighted loss, clip gradients, step optimizer
  6. Log and checkpoint best model

Data-assimilation mode (no HEC-RAS results):
  - h/u/v fields are boundary-driven synthetic (from load_boundary_driven_results)
  - BTC gauge provides the only real observation, weighted by w_btc (default 50.0)
  - FVM physics residuals enforce SWE consistency
"""

from __future__ import annotations
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from physics.fvm_residual import swe_fvm_residuals
from train.losses import compute_total_loss


@dataclass
class TrainConfig:
    # Optimisation
    lr:                float = 1e-3
    lr_decay_patience: int   = 300
    lr_decay_factor:   float = 0.5
    epochs:            int   = 3000
    batch_time_steps:  int   = 8
    accumulation_steps:int   = 1
    # Loss weights
    w_fvm:    float = 1.0
    w_obs:    float = 10.0
    w_smooth: float = 0.05
    w_bound:  float = 0.1
    w_btc:    float = 50.0   # BTC sparse gauge observation loss (high — only real obs)
    w_z:      float = 10.0   # Elevation correlation penalty
    # Physics
    flux_scheme: str   = "lax_friedrichs"
    h_min:       float = 1e-3
    # Logging / checkpointing
    seed:           int = 42
    log_every:      int = 100
    checkpoint_dir: str = "runs/pignn"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PINNTrainer:
    """
    Trainer for PIGNN inverse Manning's n problem.

    Args:
        model          : PIGNN model
        graph          : PyG Data object (on CPU; moved to device inside __init__)
        results        : dict from hecras_loader with h_series, u_series, v_series, t_series
        cfg            : TrainConfig
        device         : torch.device
        btc_cell_idx   : int or None — index of BTC gauge cell in the mesh
        btc_wse_series : np.ndarray [T] or None — observed WSE at BTC (m NAVD88)
    """

    def __init__(
        self,
        model,
        graph,
        results: dict,
        cfg: TrainConfig,
        device: torch.device,
        btc_cell_idx:   int | None       = None,
        btc_wse_series: np.ndarray | None = None,
    ):
        self.model   = model.to(device)
        self.graph   = graph.to(device)
        self.cfg     = cfg
        self.device  = device
        self.run_dir = Path(cfg.checkpoint_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        def _to_cpu(v):
            return torch.from_numpy(v) if isinstance(v, np.ndarray) else v

        self.h_series = _to_cpu(results["h_series"])   # [T, N] (stays on CPU)
        self.u_series = _to_cpu(results["u_series"])
        self.v_series = _to_cpu(results["v_series"])

        t_raw = results["t_series"]
        if isinstance(t_raw, np.ndarray):
            t_raw = torch.from_numpy(t_raw.astype(np.float32))
        self.t_series = t_raw.to(device)

        self.T = self.h_series.shape[0]
        self.N = self.h_series.shape[1]

        # dt (assume uniform)
        self.dt = float((self.t_series[1] - self.t_series[0]).item()) if self.T > 1 else 3600.0

        # Ground-truth n for validation (if available)
        self.n_true = None
        n_raw = results.get("n_series")
        if n_raw is not None:
            if isinstance(n_raw, np.ndarray):
                n_raw = torch.from_numpy(n_raw)
            n_raw = n_raw.to(device)
            self.n_true = n_raw.mean(dim=0) if n_raw.ndim == 2 else n_raw

        # ── BTC gauge observation ─────────────────────────────────────────
        self.btc_cell_idx = btc_cell_idx
        if btc_cell_idx is None:
            warnings.warn(
                "[Trainer] btc_cell_idx is None — BTC gauge loss will be SKIPPED. "
                "Training relies on FVM physics + smoothness only."
            )
            self.btc_wse_series = None
            self.z_btc = None
        else:
            # BTC WSE time series [T] on CPU (indexed per step)
            if btc_wse_series is not None:
                self.btc_wse_series = btc_wse_series.astype(np.float32)
            else:
                warnings.warn("[Trainer] btc_wse_series is None — BTC gauge loss disabled.")
                self.btc_wse_series = None

            # Bed elevation at BTC cell (scalar)
            cell_z = graph.cell_z if hasattr(graph, "cell_z") else None
            if cell_z is not None:
                z_np = cell_z.cpu().numpy() if isinstance(cell_z, torch.Tensor) else cell_z
                self.z_btc = float(z_np[btc_cell_idx])
                print(f"[Trainer] BTC cell #{btc_cell_idx}  z_btc={self.z_btc:.2f} m NAVD88")
            else:
                self.z_btc = 0.0
                warnings.warn("[Trainer] graph.cell_z not found — z_btc set to 0.0.")

        # Precompute bed slopes to save redundant computations on the GPU
        from physics.fvm_residual import _green_gauss_slope
        self.dz_dx, self.dz_dy = _green_gauss_slope(
            self.graph.cell_z,
            self.graph.face_cell_idx,
            self.graph.face_normal,
            self.graph.face_length,
            self.graph.cell_area,
            self.N,
            self.device
        )

        self.rng = np.random.default_rng(cfg.seed)
        self.history: dict[str, list] = {
            k: [] for k in ["total", "fvm", "obs", "smooth", "bound", "btc",
                             "btc_stage_err_m", "n_mean", "n_std", "val_loss"]
        }
        if self.n_true is not None:
            self.history["n_rmse"] = []

        self.best_loss = np.inf
        print(f"[Trainer] N={self.N}  T={self.T}  dt={self.dt:.1f}s  device={device}  "
              f"BTC={'enabled' if self.btc_cell_idx is not None else 'disabled'}")

    # ── Time-batch sampling ────────────────────────────────────────────────

    def _time_batch(self) -> list[int]:
        B = min(self.cfg.batch_time_steps, self.T - 1)
        t0 = int(self.rng.integers(0, max(1, self.T - B)))
        return list(range(t0, min(t0 + B, self.T - 1)))

    # ── Per-step loss ──────────────────────────────────────────────────────

    def _step_loss(
        self,
        t_idx: int,
        n_pred: Tensor,
        h_pred: Tensor | None = None,
        u_pred: Tensor | None = None,
        v_pred: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute per-time-step loss components:
          - FVM physics residuals (SWE)
          - BTC sparse gauge observation at this time step (if available)
          - Smoothness + bound penalties (via compute_total_loss)
        """
        h_t,  u_t,  v_t  = self.h_series[t_idx].to(self.device),   self.u_series[t_idx].to(self.device),   self.v_series[t_idx].to(self.device)
        h_tp1,u_tp1,v_tp1= self.h_series[t_idx+1].to(self.device), self.u_series[t_idx+1].to(self.device), self.v_series[t_idx+1].to(self.device)

        # Use GNN-predicted next state in surrogate mode to enable observation feedback
        h_next = h_pred if h_pred is not None else h_tp1
        u_next = u_pred if u_pred is not None else u_tp1
        v_next = v_pred if v_pred is not None else v_tp1

        R_cont, R_momx, R_momy = swe_fvm_residuals(
            h_t, u_t, v_t, h_next, u_next, v_next,
            n_pred,
            self.graph.cell_z,
            self.graph.face_cell_idx,
            self.graph.face_normal,
            self.graph.face_length,
            self.graph.cell_area,
            dt          = self.dt,
            flux_scheme = self.cfg.flux_scheme,
            h_min       = self.cfg.h_min,
            dz_dx       = self.dz_dx,
            dz_dy       = self.dz_dy,
        )

        # ── BTC gauge observation at time step t+1 ────────────────────────
        # Use t+1 so the loss aligns with the state we're trying to predict
        h_pred_btc  = None
        btc_wse_obs = None
        if (self.btc_cell_idx is not None
                and self.btc_wse_series is not None
                and self.z_btc is not None):
            # h_pred at BTC cell: use the model-predicted field at t+1
            # (h_tp1 here is our boundary-driven synthetic field; the model
            #  ultimately controls h through n and FVM consistency)
            h_pred_btc  = h_next[self.btc_cell_idx]   # scalar tensor
            raw_obs     = self.btc_wse_series[t_idx + 1] if (t_idx + 1) < len(self.btc_wse_series) else float("nan")
            btc_wse_obs = float(raw_obs)

        return compute_total_loss(
            R_cont, R_momx, R_momy,
            n_pred, self.graph.edge_index,
            self.model.n_min, self.model.n_max,
            w_fvm    = self.cfg.w_fvm,
            w_obs    = self.cfg.w_obs,
            w_smooth = self.cfg.w_smooth,
            w_bound  = self.cfg.w_bound,
            w_btc    = self.cfg.w_btc,
            # Dense observation (HEC-RAS h/u/v)
            h_pred=h_next, u_pred=u_next, v_pred=v_next,
            h_true=h_tp1, u_true=u_tp1, v_true=v_tp1,
            # BTC sparse gauge
            h_pred_btc  = h_pred_btc,
            z_btc       = self.z_btc,
            btc_wse_obs = btc_wse_obs,
            z           = self.graph.cell_z,
            w_z         = getattr(self.cfg, 'w_z', 10.0),
        )

    # ── Main training loop ─────────────────────────────────────────────────

    def evaluate_validation_loss(self) -> float:
        """Evaluate loss on the fixed validation window (first 2 steps, 0 NaNs)."""
        self.model.eval()
        val_indices = list(range(0, min(2, self.T - 1)))
        val_agg: dict[str, float] = {}
        with torch.no_grad():
            for t_idx in val_indices:
                h_in = self.h_series[t_idx].to(self.device)
                u_in = self.u_series[t_idx].to(self.device)
                v_in = self.v_series[t_idx].to(self.device)
                
                out = self.model(self.graph, h_in, u_in, v_in)
                n_pred = out["n"]
                
                h_pred = out.get("h")
                u_pred = out.get("u")
                v_pred = out.get("v")
                
                sl = self._step_loss(t_idx, n_pred, h_pred, u_pred, v_pred)
                for k, v in sl.items():
                    val_agg[k] = val_agg.get(k, 0.0) + v.item()
            
            v_size = len(val_indices)
            val_total = val_agg.get("total", 0.0) / v_size
        return val_total

    def train(self) -> dict[str, list]:
        model  = self.model
        cfg    = self.cfg
        device = self.device

        opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=cfg.lr_decay_patience, factor=cfg.lr_decay_factor, min_lr=1e-6
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        print(f"\n{'='*60}")
        print(f"  PIGNN Training  |  epochs={cfg.epochs}  |  device={device}")
        print(f"  BTC weight w_btc={cfg.w_btc}  |  FVM weight w_fvm={cfg.w_fvm}")
        print(f"{'='*60}\n")

        t0 = time.time()
        t_epoch = time.time()
        val_loss = 0.0
        for epoch in range(1, cfg.epochs + 1):
#             if device.type == "cuda":
#                 torch.cuda.empty_cache()

            dt_epoch = time.time() - t_epoch
            if epoch % max(1, cfg.log_every) == 0 or epoch == 1:
                if epoch > 1:
                    print(f"[{epoch:5d}/{cfg.epochs}] (last epoch: {dt_epoch:.0f}s)", flush=True)
                else:
                    print(f"[{epoch:5d}/{cfg.epochs}] Starting...", flush=True)

            model.train()
            t_epoch = time.time()
            opt.zero_grad()
            
            accum_steps = getattr(cfg, "accumulation_steps", 1)
            agg_epoch: dict[str, Tensor] = {}
            n_pred_epoch = None
            
            for step_i in range(accum_steps):
                time_batch = self._time_batch()
                agg_step: dict[str, Tensor] = {}
                
                # If surrogate mode, run GNN pass at each time step in the batch
                if getattr(model, "surrogate_mode", False):
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        for t_idx in time_batch:
                            h_in = self.h_series[t_idx].detach().to(device)
                            u_in = self.u_series[t_idx].detach().to(device)
                            v_in = self.v_series[t_idx].detach().to(device)
                            
                            out = model(self.graph, h_in, u_in, v_in)
                            n_pred = out["n"]
                            h_pred = out["h"]
                            u_pred = out["u"]
                            v_pred = out["v"]
                            
                            sl = self._step_loss(t_idx, n_pred, h_pred, u_pred, v_pred)
                            for k, v in sl.items():
                                agg_step[k] = agg_step[k] + v if k in agg_step else v
                        B   = len(time_batch)
                        agg_step = {k: v / B for k, v in agg_step.items()}
                else:
                    # Use midpoint time step as GNN context (h,u,v input)
                    mid    = time_batch[len(time_batch) // 2]
                    h_in   = self.h_series[mid].detach().to(device)
                    u_in   = self.u_series[mid].detach().to(device)
                    v_in   = self.v_series[mid].detach().to(device)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        out    = model(self.graph, h_in, u_in, v_in)
                        n_pred = out["n"]   # [N]

                        # Accumulate losses over time batch
                        for t_idx in time_batch:
                            sl = self._step_loss(t_idx, n_pred)
                            for k, v in sl.items():
                                agg_step[k] = agg_step[k] + v if k in agg_step else v
                        B   = len(time_batch)
                        agg_step = {k: v / B for k, v in agg_step.items()}

                # Scale loss by accum_steps so learning rate doesn't explode
                loss_scaled = agg_step["total"] / accum_steps
                scaler.scale(loss_scaled).backward()
                
                # Save first n_pred for logging, and accumulate logs
                if step_i == 0:
                    n_pred_epoch = n_pred.detach()
                for k, v in agg_step.items():
                    agg_epoch[k] = agg_epoch.get(k, 0.0) + (v.detach() / accum_steps)

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            # Assign aggregated metrics back to local variables for downstream logging
            agg = agg_epoch
            n_pred = n_pred_epoch

            # Evaluate validation loss
            if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                val_loss = self.evaluate_validation_loss()
            sched.step(val_loss)

            self._record(agg, n_pred, val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save(epoch, n_pred)

            if epoch % cfg.log_every == 0:
                self._log(epoch, agg, n_pred, val_loss, t0)

        self._save_history()
        print(f"\nTraining complete. Best loss: {self.best_loss:.4e}")
        return self.history

    # ── Helpers ───────────────────────────────────────────────────────────

    def _record(self, losses: dict, n_pred: Tensor, val_loss: float) -> None:
        for k in ["total", "fvm", "obs", "smooth", "bound", "btc"]:
            if k in losses:
                self.history[k].append(losses[k].item())

        # BTC stage error in metres (sqrt of L_btc for interpretability)
        btc_val = losses.get("btc")
        if btc_val is not None:
            err_m = float(btc_val.item() ** 0.5)
            self.history["btc_stage_err_m"].append(err_m)

        self.history["val_loss"].append(val_loss)

        with torch.no_grad():
            self.history["n_mean"].append(n_pred.mean().item())
            self.history["n_std"].append(n_pred.std().item())
            if self.n_true is not None and "n_rmse" in self.history:
                rmse = ((n_pred - self.n_true) ** 2).mean().sqrt().item()
                self.history["n_rmse"].append(rmse)

    def _log(self, epoch: int, losses: dict, n_pred: Tensor, val_loss: float, t0: float) -> None:
        elapsed = time.time() - t0
        btc_err_m = losses.get("btc")
        btc_str   = (f"  BTC_err={btc_err_m.item()**0.5:.3f}m"
                     if btc_err_m is not None and btc_err_m.item() > 0
                     else "")
        print(
            f"[{epoch:5d}/{self.cfg.epochs}] "
            f"Loss={losses['total'].item():.3e}  "
            f"Val_Loss={val_loss:.3e}  "
            f"FVM={losses['fvm'].item():.3e}  "
            f"Smooth={losses['smooth'].item():.3e}"
            f"{btc_str}  "
            f"n: {n_pred.mean().item():.4f}±{n_pred.std().item():.4f}  "
            f"({elapsed:.0f}s)",
            flush=True
        )

    def _save(self, epoch: int, n_pred: Tensor) -> None:
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "best_loss":   self.best_loss,
            "history":     self.history,
        }, self.run_dir / "pignn_checkpoint.pt")
        np.save(str(self.run_dir / "n_field.npy"),
                n_pred.detach().cpu().numpy())

    def _save_history(self) -> None:
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(
                {k: [float(x) for x in v] for k, v in self.history.items()},
                f, indent=2
            )
