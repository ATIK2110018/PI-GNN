"""
FVM Physics Residuals for 2D Shallow Water Equations on Graph Mesh
===================================================================
Computes the 2D SWE residuals using Finite Volume Method (FVM) formulation
directly on the HEC-RAS unstructured mesh graph.

For each interior cell, residuals:
  R_cont_i  = (h_{t+1} - h_t) / dt  +  (1/A_i) Σ_f  F_h  · L_f
  R_momx_i  = (hu_{t+1} - hu_t) / dt + (1/A_i) Σ_f  F_hu · L_f  + g·h·∂z/∂x + τ_x
  R_momy_i  = (hv_{t+1} - hv_t) / dt + (1/A_i) Σ_f  F_hv · L_f  + g·h·∂z/∂y + τ_y

Flux schemes: Lax-Friedrichs (robust) or Roe (more accurate).
"""

from __future__ import annotations
import torch
from torch import Tensor

G = 9.81  # gravitational acceleration (m/s²)


# ─────────────────────────────────────────────────────────────────────────────
# Flux functions
# ─────────────────────────────────────────────────────────────────────────────

def lax_friedrichs_flux(
    h_i: Tensor, u_i: Tensor, v_i: Tensor,
    h_j: Tensor, u_j: Tensor, v_j: Tensor,
    nx: Tensor, ny: Tensor,
    h_min: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lax-Friedrichs numerical flux at each face [F]."""
    h_i = h_i.clamp(min=h_min)
    h_j = h_j.clamp(min=h_min)

    qn_i = u_i * nx + v_i * ny
    qn_j = u_j * nx + v_j * ny

    Fh_i  = h_i * qn_i
    Fhu_i = h_i * u_i * qn_i + 0.5 * G * h_i**2 * nx
    Fhv_i = h_i * v_i * qn_i + 0.5 * G * h_i**2 * ny

    Fh_j  = h_j * qn_j
    Fhu_j = h_j * u_j * qn_j + 0.5 * G * h_j**2 * nx
    Fhv_j = h_j * v_j * qn_j + 0.5 * G * h_j**2 * ny

    c_i   = torch.sqrt(G * h_i)
    c_j   = torch.sqrt(G * h_j)
    s_max = torch.maximum(qn_i.abs() + c_i, qn_j.abs() + c_j)

    F_h  = 0.5 * (Fh_i  + Fh_j)  - 0.5 * s_max * (h_j        - h_i)
    F_hu = 0.5 * (Fhu_i + Fhu_j) - 0.5 * s_max * (h_j * u_j  - h_i * u_i)
    F_hv = 0.5 * (Fhv_i + Fhv_j) - 0.5 * s_max * (h_j * v_j  - h_i * v_i)
    return F_h, F_hu, F_hv


def roe_flux(
    h_i: Tensor, u_i: Tensor, v_i: Tensor,
    h_j: Tensor, u_j: Tensor, v_j: Tensor,
    nx: Tensor, ny: Tensor,
    h_min: float = 1e-3,
) -> tuple[Tensor, Tensor, Tensor]:
    """Approximate Roe flux for 2D SWE [F]."""
    h_i = h_i.clamp(min=h_min)
    h_j = h_j.clamp(min=h_min)

    sqrt_hi = torch.sqrt(h_i)
    sqrt_hj = torch.sqrt(h_j)
    denom   = sqrt_hi + sqrt_hj + 1e-12

    h_roe   = 0.5 * (h_i + h_j)
    u_roe   = (sqrt_hi * u_i + sqrt_hj * u_j) / denom
    v_roe   = (sqrt_hi * v_i + sqrt_hj * v_j) / denom
    c_roe   = torch.sqrt(G * h_roe.clamp(min=h_min))
    qn_roe  = u_roe * nx + v_roe * ny

    lam1 = qn_roe - c_roe
    lam3 = qn_roe + c_roe

    dh  = h_j - h_i
    dqn = (u_j - u_i) * nx + (v_j - v_i) * ny
    dqt = -(v_j - v_i) * nx + (u_j - u_i) * ny

    alpha1 = 0.5 * (dh - h_roe * dqn / (c_roe + 1e-12))
    alpha2 = h_roe * dqt
    alpha3 = 0.5 * (dh + h_roe * dqn / (c_roe + 1e-12))

    # Physical fluxes at left and right
    qn_i  = u_i * nx + v_i * ny
    Fh_i  = h_i * qn_i
    Fhu_i = h_i * u_i * qn_i + 0.5 * G * h_i**2 * nx
    Fhv_i = h_i * v_i * qn_i + 0.5 * G * h_i**2 * ny
    qn_j  = u_j * nx + v_j * ny
    Fh_j  = h_j * qn_j
    Fhu_j = h_j * u_j * qn_j + 0.5 * G * h_j**2 * nx
    Fhv_j = h_j * v_j * qn_j + 0.5 * G * h_j**2 * ny

    # Roe dissipation
    diss_h  = alpha1 * lam1.abs() + alpha3 * lam3.abs()
    diss_hu = (alpha1 * lam1.abs() * (u_roe - c_roe * nx)
               + alpha2 * (qn_roe).abs() * (-ny)
               + alpha3 * lam3.abs() * (u_roe + c_roe * nx))
    diss_hv = (alpha1 * lam1.abs() * (v_roe - c_roe * ny)
               + alpha2 * (qn_roe).abs() * nx
               + alpha3 * lam3.abs() * (v_roe + c_roe * ny))

    F_h  = 0.5 * (Fh_i  + Fh_j)  - 0.5 * diss_h
    F_hu = 0.5 * (Fhu_i + Fhu_j) - 0.5 * diss_hu
    F_hv = 0.5 * (Fhv_i + Fhv_j) - 0.5 * diss_hv
    return F_h, F_hu, F_hv


# ─────────────────────────────────────────────────────────────────────────────
# Main residual function
# ─────────────────────────────────────────────────────────────────────────────

def swe_fvm_residuals(
    h_t:           Tensor,   # [N]
    u_t:           Tensor,   # [N]
    v_t:           Tensor,   # [N]
    h_tp1:         Tensor,   # [N]
    u_tp1:         Tensor,   # [N]
    v_tp1:         Tensor,   # [N]
    n_pred:        Tensor,   # [N]  Manning's n per cell
    cell_z:        Tensor,   # [N]  bed elevation
    face_cell_idx: Tensor,   # [F, 2]
    face_normal:   Tensor,   # [F, 2]
    face_length:   Tensor,   # [F]
    cell_area:     Tensor,   # [N]
    dt:            float,
    flux_scheme:   str   = "lax_friedrichs",
    h_min:         float = 1e-3,
    dz_dx:         Tensor | None = None,
    dz_dy:         Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute per-cell FVM SWE residuals (R_cont, R_momx, R_momy).
    Uses Crank-Nicolson time-averaging for stability.
    """
    N      = h_t.shape[0]
    device = h_t.device

    h_avg = 0.5 * (h_t + h_tp1)
    u_avg = 0.5 * (u_t + u_tp1)
    v_avg = 0.5 * (v_t + v_tp1)

    c0 = face_cell_idx[:, 0]
    c1 = face_cell_idx[:, 1]
    nx = face_normal[:, 0]
    ny = face_normal[:, 1]
    L  = face_length

    hi, ui, vi = h_avg[c0], u_avg[c0], v_avg[c0]
    hj, uj, vj = h_avg[c1], u_avg[c1], v_avg[c1]

    if flux_scheme == "roe":
        F_h, F_hu, F_hv = roe_flux(hi, ui, vi, hj, uj, vj, nx, ny, h_min)
    else:
        F_h, F_hu, F_hv = lax_friedrichs_flux(hi, ui, vi, hj, uj, vj, nx, ny, h_min)

    # Accumulate flux divergence (outward from c0, inward to c1)
    div_h  = torch.zeros(N, device=device)
    div_hu = torch.zeros(N, device=device)
    div_hv = torch.zeros(N, device=device)

    for div, F in [(div_h, F_h), (div_hu, F_hu), (div_hv, F_hv)]:
        div.scatter_add_(0, c0,  F * L)
        div.scatter_add_(0, c1, -F * L)

    inv_area = 1.0 / cell_area.clamp(min=1.0)

    # Bed slope (Green-Gauss)
    if dz_dx is None or dz_dy is None:
        dz_dx, dz_dy = _green_gauss_slope(cell_z, face_cell_idx, face_normal, face_length,
                                           cell_area, N, device)

    # Manning friction  τ = g·n²·|v|·v / h^(4/3)
    h_safe = h_avg.clamp(min=h_min)
    speed  = torch.sqrt(u_avg**2 + v_avg**2 + 1e-8)
    h43    = h_safe ** (4.0 / 3.0)
    fric_x = G * n_pred**2 * u_avg * speed / h43
    fric_y = G * n_pred**2 * v_avg * speed / h43

    R_cont = (h_tp1 - h_t) / dt + inv_area * div_h
    R_momx = ((h_tp1 * u_tp1 - h_t * u_t) / dt
              + inv_area * div_hu + G * h_avg * dz_dx + fric_x)
    R_momy = ((h_tp1 * v_tp1 - h_t * v_t) / dt
              + inv_area * div_hv + G * h_avg * dz_dy + fric_y)

    return R_cont, R_momx, R_momy


def _green_gauss_slope(
    z: Tensor, fci: Tensor, fn: Tensor, fl: Tensor,
    cell_area: Tensor, N: int, device
) -> tuple[Tensor, Tensor]:
    """Estimate ∂z/∂x, ∂z/∂y using the Green-Gauss theorem on graph faces."""
    c0, c1 = fci[:, 0], fci[:, 1]
    nx, ny  = fn[:, 0], fn[:, 1]
    z_face  = 0.5 * (z[c0] + z[c1])

    dzdx = torch.zeros(N, device=device)
    dzdy = torch.zeros(N, device=device)
    dzdx.scatter_add_(0, c0,  z_face * nx * fl)
    dzdx.scatter_add_(0, c1, -z_face * nx * fl)
    dzdy.scatter_add_(0, c0,  z_face * ny * fl)
    dzdy.scatter_add_(0, c1, -z_face * ny * fl)

    inv_area = 1.0 / cell_area.clamp(min=1.0)
    return dzdx * inv_area, dzdy * inv_area
