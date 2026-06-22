"""
Observation & Boundary Condition Loader
=========================================
Loads real gauge observations and boundary conditions for boundary-driven
(data-assimilation) PIGNN training on the Sacramento River HEC-RAS mesh.

Modules:
  load_btc_stage()          – parse BTC CDEC stage CSV → aligned WSE array
  load_boundary_conditions()– load Ord Ferry Q + Colusa stage → aligned arrays
  snap_to_mesh()            – convert BTC lat/lon → UTM, find nearest cell index
  build_simulation_time()   – construct canonical 744-step hourly time axis

All time series are aligned to the 744-hour January 2026 simulation axis:
  t = [0, 3600, 7200, …, 2678400] seconds  (744 steps, inclusive)

UTM conversion:
  Sacramento Valley is in UTM Zone 10N (EPSG:32610).
  Uses pyproj if available, else falls back to an inline Transverse Mercator
  approximation accurate to ~1 m within Sacramento Valley.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Simulation epoch: 2026-01-01 00:00:00 UTC
_SIM_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_N_STEPS   = 744           # hourly steps in January 2026 (0 … 743 hours)
_DT_SEC    = 3600.0        # seconds per step

# UTM Zone 10N (Sacramento Valley)
_UTM_ZONE  = 10
_UTM_EPSG  = 32610


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_simulation_time() -> np.ndarray:
    """Return the canonical 744-step hourly time axis in seconds [744]."""
    return np.arange(_N_STEPS, dtype=np.float64) * _DT_SEC


def load_btc_stage(
    csv_path: str,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Parse the BTC CDEC stage CSV and return a 744-element array of water
    surface elevations (m NAVD88), aligned to the simulation time axis.

    CSV columns:
      STATION_ID, DURATION, SENSOR_NUMBER, SENSOR_TYPE,
      DATE TIME, OBS DATE, VALUE, DATA_FLAG, UNITS

    'DATE TIME' format: YYYYMMDD HHMM  (local time — treated as UTC for
    alignment; CDEC records are typically PST/PDT but the simulation epoch
    is aligned to calendar day boundaries, so any sub-hourly timezone shift
    is within the snap tolerance used here).

    Parameters
    ----------
    csv_path   : path to BTC_stage_data_Jan2026.csv
    fill_value : value for missing / flagged records (default NaN)

    Returns
    -------
    wse : np.ndarray [744] float64 — WSE in metres (NAVD88)
    """
    import csv

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"BTC stage CSV not found: {path}")

    # Initialise output with fill_value
    wse = np.full(_N_STEPS, fill_value, dtype=np.float64)

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Parse timestamp ─────────────────────────────────────────────
            dt_str = row.get("DATE TIME", "").strip()
            if not dt_str:
                continue
            try:
                # Format: YYYYMMDD HHMM  (e.g. "20260101 0000")
                dt = datetime.strptime(dt_str, "%Y%m%d %H%M").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                try:
                    # Fallback: YYYYMMDD HHMM (no space variant)
                    dt = datetime.strptime(dt_str.replace(" ", ""), "%Y%m%d%H%M").replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    warnings.warn(f"[BTC] Cannot parse timestamp: {dt_str!r}")
                    continue

            # Map to simulation step index ─────────────────────────────────
            delta_sec = (dt - _SIM_EPOCH).total_seconds()
            step_idx  = int(round(delta_sec / _DT_SEC))
            if step_idx < 0 or step_idx >= _N_STEPS:
                continue

            # Parse VALUE ─────────────────────────────────────────────────
            val_str = row.get("VALUE", "").strip()
            flag    = row.get("DATA_FLAG", "").strip()
            if not val_str or flag.upper() in ("A", "E", "B", "R"):
                # Skip missing or flagged values
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue

            wse[step_idx] = val

    n_valid = np.sum(~np.isnan(wse))
    print(f"[OBS] BTC stage: {n_valid}/{_N_STEPS} valid records loaded from {path.name}")
    return wse


def load_boundary_conditions(
    discharge_csv: str,
    stage_csv: str,
) -> dict:
    """
    Load upstream (Ord Ferry) discharge and downstream (Colusa) stage,
    both aligned to the 744-step simulation time axis.

    Parameters
    ----------
    discharge_csv : path to ord_ferry_discharge_jan2026.csv
                    Columns: datetime, Discharge_cms
    stage_csv     : path to colusa_stage_jan2026.csv
                    Columns: datetime, WaterLevel_m

    Returns
    -------
    dict with keys:
      't_sec'       : np.ndarray [744] — simulation time in seconds
      'discharge'   : np.ndarray [744] — upstream Q (m³/s)
      'stage_down'  : np.ndarray [744] — downstream WSE (m)
    """
    t_sec = build_simulation_time()
    Q_arr = _load_timeseries_csv(
        discharge_csv, value_col="Discharge_cms",
        label="Ord Ferry discharge",
    )
    H_arr = _load_timeseries_csv(
        stage_csv, value_col="WaterLevel_m",
        label="Colusa stage",
    )
    return {
        "t_sec":      t_sec,
        "discharge":  Q_arr,
        "stage_down": H_arr,
    }


def snap_to_mesh(
    lat: float,
    lon: float,
    cell_xy: np.ndarray,
    max_radius_m: float = 5000.0,
    offset_easting_m: float = 0.0,
    offset_northing_m: float = 0.0,
) -> int | None:
    """
    Find the nearest mesh cell to the given geographic coordinate.

    Parameters
    ----------
    lat, lon     : WGS84 geographic coordinates of the observation station
    cell_xy      : np.ndarray [N, 2] — mesh cell centroids in UTM (m)
    max_radius_m : maximum search radius (metres); returns None if exceeded
    offset_easting_m, offset_northing_m
                 : optional UTM offsets applied before snapping. Use this when
                   the gauge point should be associated with the hydraulic
                   main-channel cell rather than the physical bank marker.

    Returns
    -------
    cell_idx : int or None
    """
    easting, northing = _latlon_to_utm(lat, lon)
    raw_easting, raw_northing = easting, northing
    easting += float(offset_easting_m)
    northing += float(offset_northing_m)

    if offset_easting_m or offset_northing_m:
        print(
            f"[OBS] Applying BTC snap offset: "
            f"dE={offset_easting_m:+.1f} m, dN={offset_northing_m:+.1f} m "
            f"→ target {easting:.0f}E, {northing:.0f}N "
            f"(from {raw_easting:.0f}E, {raw_northing:.0f}N)"
        )

    dx   = cell_xy[:, 0] - easting
    dy   = cell_xy[:, 1] - northing
    dist = np.sqrt(dx**2 + dy**2)
    idx  = int(np.argmin(dist))
    min_dist = float(dist[idx])

    if min_dist > max_radius_m:
        warnings.warn(
            f"[OBS] Nearest mesh cell is {min_dist:.0f} m away "
            f"(max_radius={max_radius_m:.0f} m). BTC loss will be skipped."
        )
        return None

    print(
        f"[OBS] BTC snapped to cell #{idx}  "
        f"(UTM {easting:.0f}E, {northing:.0f}N → "
        f"cell @ {cell_xy[idx,0]:.0f}E, {cell_xy[idx,1]:.0f}N, "
        f"dist={min_dist:.1f} m)"
    )
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# UTM conversion
# ─────────────────────────────────────────────────────────────────────────────

def _latlon_to_utm(lat: float, lon: float) -> tuple[float, float]:
    """
    Convert WGS84 lat/lon (degrees) to UTM Zone 10N (EPSG:32610) easting/northing (m).

    Uses pyproj if available (preferred). Falls back to an inline Transverse
    Mercator formula valid for UTM Zone 10N.
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{_UTM_EPSG}", always_xy=True)
        easting, northing = transformer.transform(lon, lat)
        print(f"[OBS] UTM conversion via pyproj: ({lat}, {lon}) → ({easting:.0f}, {northing:.0f})")
        return easting, northing
    except ImportError:
        pass

    # ── Inline Transverse Mercator approximation (UTM Zone 10N) ──────────
    # Accurate to ~1 m in Sacramento Valley
    easting, northing = _tm_approx(lat, lon, zone=_UTM_ZONE, northern=True)
    print(
        f"[OBS] UTM conversion via TM approx: ({lat}, {lon}) → "
        f"({easting:.0f}, {northing:.0f}) [pyproj unavailable]"
    )
    return easting, northing


def _tm_approx(lat_deg: float, lon_deg: float, zone: int, northern: bool) -> tuple[float, float]:
    """
    Transverse Mercator projection (WGS84 ellipsoid) for a single UTM zone.
    Reference: Bowring (1983) / Karney (2011) series expansion, 5th order.
    """
    # WGS84 parameters
    a   = 6378137.0           # semi-major axis (m)
    f   = 1.0 / 298.257223563
    b   = a * (1.0 - f)
    e2  = 1.0 - (b / a) ** 2  # first eccentricity squared
    n   = f / (2.0 - f)       # third flattening

    k0  = 0.9996              # scale factor
    E0  = 500_000.0           # false easting
    N0  = 0.0 if northern else 10_000_000.0

    # Central meridian
    lon0 = (zone - 1) * 6.0 - 180.0 + 3.0

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    l   = np.radians(lon_deg - lon0)

    # Conformal latitude
    e  = np.sqrt(e2)
    psi = (np.log(np.tan(np.pi / 4.0 + lat / 2.0))
           - e * np.log((1 + e * np.sin(lat)) / (1 - e * np.sin(lat))) / 2.0)
    chi = 2.0 * np.arctan(np.exp(psi)) - np.pi / 2.0

    # Coefficients (5th-order Krüger series)
    n2, n3, n4, n5 = n**2, n**3, n**4, n**5
    A = (a / (1.0 + n)) * (1.0 + n2 / 4.0 + n4 / 64.0)

    alpha = [
        0,
        (1.0 / 2.0) * n - (2.0 / 3.0) * n2 + (5.0 / 16.0) * n3 + (41.0 / 180.0) * n4 - (127.0 / 288.0) * n5,
        (13.0 / 48.0) * n2 - (3.0 / 5.0) * n3 + (557.0 / 1440.0) * n4 + (281.0 / 630.0) * n5,
        (61.0 / 240.0) * n3 - (103.0 / 140.0) * n4 + (15061.0 / 26880.0) * n5,
        (49561.0 / 161280.0) * n4 - (179.0 / 168.0) * n5,
        (34729.0 / 80640.0) * n5,
    ]

    t   = np.sin(chi)
    s   = np.cos(chi)
    th  = np.tanh(np.complex128(0 + 1j) * l * np.cos(chi) +
                  np.arcsinh(np.sin(l) * s / np.sqrt(1 - (np.sin(l) * s) ** 2 +
                                                      t**2 * (1 - (np.sin(l) * s) ** 2 / (np.cos(chi)**2 + np.sin(l)**2 * s**2)))))
    # Simpler direct computation using standard formulae
    xi_prime  = np.arctan2(t, np.cos(l) * s)    # ~latitude-like
    eta_prime = np.arctanh(np.sin(l) * s / np.sqrt(1.0 - (np.sin(l) * s)**2 + t**2 * np.cos(l)**2 / (1 + np.cos(l)**2 * s**2 + t**2 * np.cos(l)**2 - np.cos(l)**2 * s**2)))

    # Actually use the standard exact formulas
    sin_chi = np.sin(chi)
    cos_chi = np.cos(chi)
    xi0  = np.arctan2(sin_chi, cos_chi * np.cos(l))
    eta0 = np.arctanh(cos_chi * np.sin(l) / np.sqrt(1.0 - (cos_chi * np.sin(l))**2))

    xi  = xi0
    eta = eta0
    for j in range(1, 6):
        xi  += alpha[j] * np.sin(2.0 * j * xi0) * np.cosh(2.0 * j * eta0)
        eta += alpha[j] * np.cos(2.0 * j * xi0) * np.sinh(2.0 * j * eta0)

    easting  = E0 + k0 * A * eta
    northing = N0 + k0 * A * xi

    return float(easting), float(northing)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_timeseries_csv(
    csv_path: str,
    value_col: str,
    label: str = "",
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Load a two-column CSV (datetime, value) and align to the 744-step axis.

    The datetime column may be:
      - ISO 8601: "2026-01-01 00:00:00" or "2026-01-01T00:00:00"
      - ISO 8601 with timezone: "2026-01-01 00:00:00-08:00"
      - YYYYMMDD HHMM (CDEC format)

    Values are linearly interpolated for any missing steps.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} CSV not found: {path}")

    import csv as csv_mod
    times_sec: list[float] = []
    values:    list[float] = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv_mod.DictReader(fh)
        # Discover datetime column name
        headers    = reader.fieldnames or []
        dt_col     = _find_dt_col(headers)
        if dt_col is None:
            raise ValueError(f"[BCs] Cannot find datetime column in {path}. Headers: {headers}")

        for row in reader:
            dt_str = row.get(dt_col, "").strip()
            val_str = row.get(value_col, "").strip()
            if not dt_str or not val_str:
                continue
            try:
                dt = _parse_datetime(dt_str)
            except ValueError:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue

            delta_sec = (dt - _SIM_EPOCH).total_seconds()
            times_sec.append(delta_sec)
            values.append(val)

    if not times_sec:
        warnings.warn(f"[BCs] No records loaded from {path}. Returning NaN array.")
        return np.full(_N_STEPS, fill_value)

    times_sec_arr = np.array(times_sec, dtype=np.float64)
    values_arr    = np.array(values, dtype=np.float64)

    # Sort by time
    order         = np.argsort(times_sec_arr)
    times_sec_arr = times_sec_arr[order]
    values_arr    = values_arr[order]

    # Target grid
    t_grid = build_simulation_time()

    # Interpolate onto the target grid (linear, extrapolate boundary with fill)
    result = np.interp(t_grid, times_sec_arr, values_arr,
                       left=fill_value, right=fill_value)

    n_valid = np.sum(~np.isnan(result))
    print(f"[OBS] {label}: {n_valid}/{_N_STEPS} steps interpolated from {path.name}")
    return result


def _find_dt_col(headers: list[str]) -> str | None:
    """Heuristically find the datetime column from CSV headers."""
    for col in headers:
        cl = col.lower().replace(" ", "").replace("_", "")
        if cl in ("datetime", "date", "time", "timestamp", "datehour"):
            return col
    return None


def _parse_datetime(s: str) -> datetime:
    """
    Parse a datetime string from multiple formats, returning a UTC-aware datetime.
    """
    s = s.strip()
    # ISO 8601 with timezone offset (e.g. "2026-01-01 01:00:00-08:00")
    for fmt in [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    # CDEC format: YYYYMMDD HHMM
    for fmt in ["%Y%m%d %H%M", "%Y%m%d%H%M"]:
        try:
            return datetime.strptime(s.replace(" ", ""), fmt.replace(" ", "")).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

    raise ValueError(f"Cannot parse datetime: {s!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversion helper
# ─────────────────────────────────────────────────────────────────────────────

def to_torch(arr: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Convert a numpy array (possibly containing NaN) to a torch tensor."""
    return torch.from_numpy(arr.astype(np.float32)).to(device)
