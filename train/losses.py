"""
Loss Functions for PIGNN Inverse Training
==========================================
Total loss = w_fvm * L_fvm + w_obs * L_obs + w_smooth * L_smooth
           + w_bound * L_bound + w_btc * L_btc

L_fvm    : Mean-squared FVM SWE residuals  → physics consistency
L_obs    : MSE between predicted & HEC-RAS flow fields  → data fit
L_smooth : Graph Laplacian penalty on n  → spatial smoothness
L_bound  : Soft constraint keeping n ∈ [n_min, n_max]
L_btc    : Sparse point-observation loss at Butte City (BTC) gauge
"""

from __future__ import annotations
import torch
from torch import Tensor


def fvm_physics_loss(R_cont: Tensor, R_momx: Tensor, R_momy: Tensor) -> Tensor:
    """Mean-squared residuals across all three SWE equations."""
    return (R_cont**2 + R_momx**2 + R_momy**2).mean()


def observation_loss(
    h_pred: Tensor, u_pred: Tensor, v_pred: Tensor,
    h_true: Tensor, u_true: Tensor, v_true: Tensor,
    obs_mask: Tensor | None = None,
) -> Tensor:
    """MSE between predicted and observed (HEC-RAS) flow fields."""
    if obs_mask is not None:
        h_pred, h_true = h_pred[obs_mask], h_true[obs_mask]
        u_pred, u_true = u_pred[obs_mask], u_true[obs_mask]
        v_pred, v_true = v_pred[obs_mask], v_true[obs_mask]
    return ((h_pred - h_true)**2 + (u_pred - u_true)**2 + (v_pred - v_true)**2).mean()


def smoothness_loss(n_pred: Tensor, edge_index: Tensor) -> Tensor:
    """Graph Laplacian smoothness: penalise large n differences between neighbours."""
    src, dst = edge_index[0], edge_index[1]
    return ((n_pred[src] - n_pred[dst])**2).mean()


def bound_penalty(n_pred: Tensor, n_min: float, n_max: float) -> Tensor:
    """Soft ReLU penalty for n outside physical bounds."""
    below = torch.relu(n_min - n_pred)
    above = torch.relu(n_pred - n_max)
    return (below**2 + above**2).mean()


def btc_stage_loss(
    h_pred_btc: Tensor,
    z_btc: float,
    btc_wse_obs: float | Tensor,
) -> Tensor:
    """
    Sparse point-observation loss at the Butte City (BTC) gauge cell.

    Computes MSE between the model-predicted water surface elevation (WSE)
    and the observed BTC WSE at a single time step:

        L_btc = (h_pred_btc + z_btc - btc_wse_obs)^2

    Parameters
    ----------
    h_pred_btc  : scalar or [1] tensor — predicted water depth at BTC cell (m)
    z_btc       : float — bed elevation at BTC cell (m NAVD88)
    btc_wse_obs : float or scalar tensor — observed WSE at BTC (m NAVD88)

    Returns
    -------
    Scalar loss tensor.
    """
    pred_wse = h_pred_btc + z_btc
    if not isinstance(btc_wse_obs, Tensor):
        btc_wse_obs = torch.tensor(float(btc_wse_obs), dtype=pred_wse.dtype,
                                   device=pred_wse.device)
    return (pred_wse - btc_wse_obs) ** 2


def compute_total_loss(
    R_cont: Tensor, R_momx: Tensor, R_momy: Tensor,
    n_pred: Tensor, edge_index: Tensor,
    n_min: float, n_max: float,
    w_fvm: float = 1.0, w_obs: float = 0.0,
    w_smooth: float = 0.05, w_bound: float = 0.1,
    w_btc: float = 0.0,
    h_pred: Tensor | None = None, u_pred: Tensor | None = None, v_pred: Tensor | None = None,
    h_true: Tensor | None = None, u_true: Tensor | None = None, v_true: Tensor | None = None,
    obs_mask: Tensor | None = None,
    # BTC sparse gauge observation arguments
    h_pred_btc:  Tensor | None = None,
    z_btc:       float | None  = None,
    btc_wse_obs: float | Tensor | None = None,
) -> dict[str, Tensor]:
    """
    Compute all loss terms and weighted total.

    Returns dict with keys: 'total', 'fvm', 'obs', 'smooth', 'bound', 'btc'.

    BTC stage loss is only applied when w_btc > 0 and all BTC arguments are
    provided and btc_wse_obs is not NaN (missing observation).
    """
    losses: dict[str, Tensor] = {}
    losses["fvm"]    = fvm_physics_loss(R_cont, R_momx, R_momy)
    losses["smooth"] = smoothness_loss(n_pred, edge_index)
    losses["bound"]  = bound_penalty(n_pred, n_min, n_max)

    if w_obs > 0 and h_pred is not None and h_true is not None:
        losses["obs"] = observation_loss(h_pred, u_pred, v_pred,
                                         h_true, u_true, v_true, obs_mask)
    else:
        losses["obs"] = torch.zeros(1, device=n_pred.device).squeeze()

    # ── BTC sparse gauge observation loss ─────────────────────────────────
    # Skip if not configured or if the observed value is NaN (missing record)
    btc_obs_valid = (
        btc_wse_obs is not None
        and not (isinstance(btc_wse_obs, float) and (btc_wse_obs != btc_wse_obs))
        and not (isinstance(btc_wse_obs, Tensor) and btc_wse_obs.isnan().any())
    )
    if w_btc > 0 and h_pred_btc is not None and z_btc is not None and btc_obs_valid:
        losses["btc"] = btc_stage_loss(h_pred_btc, z_btc, btc_wse_obs)
    else:
        losses["btc"] = torch.zeros(1, device=n_pred.device).squeeze()

    losses["total"] = (w_fvm    * losses["fvm"]
                       + w_obs    * losses["obs"]
                       + w_smooth * losses["smooth"]
                       + w_bound  * losses["bound"]
                       + w_btc    * losses["btc"])
    return losses
