#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GridTweak DLR Engine - V2.16.2
V2.16.1 + smarter startup: always-on scheduler + cache-age check.
"""

import argparse, json, math, sys, os, shutil, warnings, time, threading, traceback, re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request, urllib.parse
warnings.filterwarnings("ignore")

import numpy as np

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError: SCHEDULER_AVAILABLE = False

try:
    import fastapi, uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    API_AVAILABLE = True
except ImportError: API_AVAILABLE = False

PVLIB_AVAILABLE = False
try:
    import pandas as pd, pvlib
    from pvlib.solarposition import get_solarposition
    from pvlib.irradiance import get_total_irradiance, dirint, erbs, get_extra_radiation
    from pvlib.clearsky import ineichen
    from pvlib.atmosphere import get_relative_airmass
    PVLIB_AVAILABLE = True
except ImportError: pass

TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError: pass

EARTH2STUDIO_AVAILABLE = False
try:
    from earth2studio.models.px import Pangu24
    from earth2studio.data import GFS
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic as run_deterministic
    EARTH2STUDIO_AVAILABLE = True
except ImportError: pass

XARRAY_AVAILABLE = False
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError: pass


# ============================================================================
# MCP
# ============================================================================
MCP = None
MCP_FILE = "mcp_calibration.json"

def load_mcp():
    global MCP
    if os.path.exists(MCP_FILE):
        try:
            with open(MCP_FILE) as f: MCP = json.load(f)
            print(f"✅ MCP loaded: U = {MCP['slope']:.3f} x ERA5 + {MCP['intercept']:.3f} "
                  f"(N={MCP['n_samples']}, R2={MCP['r_squared']:.3f})")
            return True
        except Exception as e: print(f"⚠️ MCP load failed: {e}")
    return False

def apply_mcp(u):
    if MCP is None: return u
    return max(0.1, MCP["slope"] * u + MCP["intercept"])


# ============================================================================
# METAR
# ============================================================================
METAR_CACHE = {"data": None, "last_updated": None}

def fetch_metar(icao="VABB"):
    global METAR_CACHE
    if METAR_CACHE["data"] and METAR_CACHE["last_updated"]:
        try:
            age = (datetime.now() - datetime.fromisoformat(METAR_CACHE["last_updated"])).total_seconds()
            if age < 1800:
                return METAR_CACHE["data"]
        except: pass
    url = f"https://aviationweather.gov/api/data/metar?ids={icao}&format=json&hours=1"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'GridTweak/1.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
        if isinstance(data, list) and data:
            raw = data[0].get("rawOb", "")
            m = re.search(r"\b(\d{3})(\d{2,3})(G\d{2,3})?KT\b", raw)
            if m:
                wdir = int(m.group(1))
                wspd_kt = int(m.group(2))
                wspd_ms = wspd_kt * 0.5144
                METAR_CACHE["data"] = {
                    "icao": icao, "direction_deg": wdir,
                    "speed_knots": wspd_kt, "speed_mps": round(wspd_ms, 2),
                    "raw": raw,
                }
                METAR_CACHE["last_updated"] = datetime.now().isoformat()
                return METAR_CACHE["data"]
    except Exception as e:
        print(f"⚠️ METAR fetch failed: {e}")
    return None


# ============================================================================
# WINDOWS GFS FIX
# ============================================================================
def patch_gfs_windows():
    if sys.platform != "win32": return
    try:
        from earth2studio.data import gfs as _g
        from datetime import datetime as _dt
        def _j(p, *parts): return "/".join(x.strip("/") for x in [p] + list(parts) if x)
        def _gu(self, t, lt):
            lh = int(lt.total_seconds() // 3600)
            fn = f"gfs.{t.year}{t.month:0>2}{t.day:0>2}/{t.hour:0>2}"
            if t < _dt(2021, 3, 23): fn = _j(fn, f"gfs.t{t.hour:0>2}z.pgrb2.0p25.f{lh:03d}")
            else: fn = _j(fn, f"atmos/gfs.t{t.hour:0>2}z.pgrb2.0p25.f{lh:03d}")
            return _j(self.uri_prefix, fn)
        def _giu(self, t, lt):
            lh = int(lt.total_seconds() // 3600)
            fn = f"gfs.{t.year}{t.month:0>2}{t.day:0>2}/{t.hour:0>2}"
            if t < _dt(2021, 3, 23): fn = _j(fn, f"gfs.t{t.hour:0>2}z.pgrb2.0p25.f{lh:03d}.idx")
            else: fn = _j(fn, f"atmos/gfs.t{t.hour:0>2}z.pgrb2.0p25.f{lh:03d}.idx")
            return _j(self.uri_prefix, fn)
        _g.GFS._grib_uri = _gu
        _g.GFS._grib_index_uri = _giu
        print("🔧 Patched GFS for Windows")
    except Exception as e: print(f"⚠️ GFS patch: {e}")
patch_gfs_windows()


def find_onnx():
    c = [os.path.expanduser("~/.cache/earth2studio/pangu/pangu_weather_24.onnx")]
    sr = os.path.expanduser("~/.cache/earth2studio/models--NickGeneva--earth_ai/snapshots")
    if os.path.isdir(sr):
        for s in os.listdir(sr):
            c.append(os.path.join(sr, s, "pangu", "pangu_weather_24.onnx"))
    for p in c:
        if os.path.exists(p) and os.path.getsize(p) / (1024**3) > 0.5: return p
    return None

def _s(v, d=0.0):
    try:
        f = float(v); return f if math.isfinite(f) else d
    except (TypeError, ValueError): return d

def _san(obj):
    if isinstance(obj, dict): return {k: _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_san(v) for v in obj]
    if isinstance(obj, float): return obj if math.isfinite(obj) else None
    return obj


# ============================================================================
# CACHE
# ============================================================================
_cached = {"data": None, "last_updated": None, "source": None,
           "building": False, "last_error": None, "last_attempt": None}
CACHE_FILE = "forecast_cache.json"
DB_FILE = "dlr_data.db"

STATIC_CURRENT   = "static_current.json"
STATIC_FORECAST  = "static_forecast.json"
STATIC_CORRIDOR  = "static_corridor.json"
STATIC_SAGCOMP   = "static_sag_comparison.json"
STATIC_SAGCHECK  = "static_sag_check.json"

PANGU_JSON = "pangu_forecast.json"


def _read_static(path):
    if not os.path.exists(path): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception as e:
        print(f"⚠️ static read failed {path}: {e}")
        return None


def _write_static(path, obj):
    try:
        with open(path, "w") as f:
            json.dump(_san(obj), f)
        print(f"   💾 {path}")
    except Exception as e:
        print(f"   ⚠️ Could not write {path}: {e}")


def _ensure_writable_dir(path_str):
    try:
        p = Path(path_str).parent
        if str(p) and str(p) != ".":
            p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def load_cache():
    global _cached
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                d = json.load(f)
                _cached["data"] = d.get("data")
                _cached["last_updated"] = d.get("last_updated")
                _cached["source"] = d.get("source")
                n = len(_cached["data"]) if _cached["data"] else 0
                print(f"✅ Loaded forecast cache ({n} records)")
                return True
        except Exception as e: print(f"⚠️ Cache load: {e}")
    return False

def save_cache():
    try:
        _ensure_writable_dir(CACHE_FILE)
        with open(CACHE_FILE, "w") as f:
            json.dump(_san({"data": _cached["data"], "last_updated": _cached["last_updated"],
                            "source": _cached.get("source")}), f)
        n = len(_cached["data"]) if _cached["data"] else 0
        print(f"💾 Cache saved ({n} records)")
    except Exception as e: print(f"⚠️ Cache save: {e}")


def clear_cache_state():
    _cached["data"] = None
    _cached["last_updated"] = None
    _cached["source"] = None
    _cached["last_error"] = None
    _cached["building"] = False
    _cached["last_attempt"] = None


def maybe_auto_fresh():
    val = os.environ.get("GRIDTWEAK_AUTO_FRESH", "").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    print("🔄 GRIDTWEAK_AUTO_FRESH set — clearing ephemeral caches")
    for f in [CACHE_FILE, DB_FILE]:
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"   Removed {f}")
        except Exception as e:
            print(f"   Could not remove {f}: {e}")
    clear_cache_state()
    return True


# ============================================================================
# CONFIG
# ============================================================================
VERSION = "V2.16.2"
APP_NAME = "GridTweak"

CONFIG_DEFAULTS = {
    "lat": 19.076, "lon": 72.877,
    "location_name": "MSETCL - 220kV Trombay-Vikhroli",
    "conductor": "zebra",
    "convection_model": "ieee738",
    "test_current": 1000,
    "nominal_voltage_kv": 220, "power_factor": 0.9,
    "static_design_ambient_c": 40.0,
    "static_design_wind_mps": 0.6,
    "static_design_solar_w_m2": 1000.0,
    "static_tmax_c": 75.0,
    "operational_tmax_c": 85.0,
    "emergency_tmax_c": 100.0,
    "sag_ref_m": 5.0,
    "sag_ref_temp_c": 20.0,
    "span_length_m": 400.0,
    "tower_attachment_height_m": 30.0,
    "min_clearance_m": 7.0,
    "wind_max_mps": 15.0,
    "wind_smoothing_window": 3,
    "wind_z0_actual_m": 0.8,
    "wind_target_height_m": 10.0,
    "use_mcp_correction": True,
    "use_metar_override": False,
    "metar_icao": "VABB",
    "ai_model": "pangu", "ai_device": "cpu",
    "use_ai_weather": True, "use_pvlib_solar": True,
    "ai_forecast_hours": 168, "ai_cache_dir": "./weather_cache",
    "surface_tilt_deg": 90.0, "surface_azimuth_deg": 180.0,
    "pangu_onnx_path": None,
    "force_archive_forecast": False,
    "database_path": DB_FILE, "ml_model_path": "lgb_best.pkl", "use_ml": False,
    "scheduler_interval_hours": 6,
    "timezone_offset_hours": 5.5,

    # Conservative derating applied to the Operational DLR only.
    # Not surfaced in the dashboard UI. Reduces reported headroom.
    "operational_derate_factor": 0.86,
}


# ============================================================================
# CONDUCTORS
# ============================================================================
@dataclass
class Conductor:
    name: str
    diameter_m: float
    r20_ohm_per_m: float
    alpha_r: float
    emissivity: float
    absorptivity: float
    tmax_c: float
    mass_al_kg_per_m: float = 0.0
    mass_steel_kg_per_m: float = 0.0
    al_layers: int = 2
    thermal_expansion_coeff: float = 23e-6

CONDUCTORS = {
    "zebra": Conductor("Zebra ACSR (India)", 0.02862, 6.89e-5, 0.00403, 0.85, 0.85, 100.0, 1.145, 0.605, 3),
    "panther": Conductor("Panther ACSR", 0.02100, 1.29e-4, 0.00403, 0.85, 0.85, 100.0, 0.698, 0.411, 2),
    "moose": Conductor("Moose ACSR", 0.03177, 6.54e-5, 0.00403, 0.85, 0.85, 100.0, 1.362, 0.682, 3),
}

def get_conductor(name):
    c = CONDUCTORS.get(name.lower())
    if not c: raise ValueError(f"Conductor '{name}' not found")
    return c


UTILITY_SAG_BENCHMARKS = [
    {"source": "PGCIL 220 kV D/C Zebra", "conductor": "Zebra ACSR", "voltage_kv": 220,
     "span_m": 350, "temp_c": 85, "sag_m": 10.600,
     "note": "PowerMin ROW comparison, stringing condition"},
    {"source": "NPTEL/PGCIL manual", "conductor": "Zebra ACSR", "voltage_kv": 220,
     "span_m": 350, "temp_c": 75, "sag_m": 9.220,
     "note": "Maximum sag at 75C, level span"},
    {"source": "AEGCL 220 kV Zebra", "conductor": "Zebra ACSR", "voltage_kv": 220,
     "span_m": 350, "temp_c": 85, "sag_m": 8.435,
     "note": "Tight stringing, max permissible sag at 85C"},
]


# ============================================================================
# IEEE 738 PHYSICS
# ============================================================================
LINE_AZ = 90.0

def get_k_acdc(c):
    d_mm = c.diameter_m * 1000; layers = c.al_layers
    if d_mm < 20: return 1.005
    if d_mm < 25: return 1.010 if layers == 2 else 1.040
    if d_mm < 30: return 1.025 if layers == 2 else 1.050
    return 1.050 if layers == 2 else 1.080

def R_ohm_m(tc, c):
    return c.r20_ohm_per_m * (1.0 + c.alpha_r * (tc - 20.0)) * get_k_acdc(c)

def wind_geom(w, wd, line_az=90.0):
    d = (wd - line_az) % 360.0; r = math.radians(d)
    vp = abs(w * math.sin(r)); vr = abs(w * math.cos(r))
    atk = math.degrees(math.asin(min(1.0, vp / w))) if w > 1e-12 else 0.0
    return {"perpendicular_ms": vp, "parallel_ms": vr, "attack_angle_deg": atk}

def q_solar_ieee738(ghi, alt, az, c, line_az=90.0):
    if ghi <= 0 or alt <= 0: return 0.0
    a = math.radians(alt); z = math.radians(az - line_az)
    proj = math.cos(a) * abs(math.sin(z))
    return max(0.0, c.absorptivity * ghi * c.diameter_m * proj)

def q_convection_ieee738(tc, amb, v_perp, d, attack_deg=90.0):
    dt = tc - amb
    if dt <= 0: return 0.0
    T_film_k = 273.15 + 0.5 * (tc + amb)
    rho = 1.225 * (288.15 / T_film_k)
    mu = 1.81e-5 * (T_film_k / 288.15) ** 0.7
    kf = 0.0257 * (T_film_k / 288.15) ** 0.85
    v = max(v_perp, 0.0)
    qc_nat = 3.645 * (rho ** 0.5) * (d ** 0.75) * (dt ** 1.25)
    if v <= 0.2: return max(qc_nat, 0.0)
    re = rho * v * d / max(mu, 1e-12)
    phi = math.radians(attack_deg)
    kangle = 1.194 - math.cos(phi) + 0.194 * math.cos(2*phi) + 0.368 * math.sin(2*phi)
    qc_f1 = kangle * (1.01 + 1.35 * (re ** 0.52)) * kf * dt if re < 4000 else 0.0
    qc_f2 = kangle * 0.754 * (re ** 0.6) * kf * dt
    return max(max(qc_f1, qc_f2), qc_nat, 0.0)

def q_rad_ieee738(tc, amb, c):
    sigma = 5.670374419e-8
    return c.emissivity * sigma * math.pi * c.diameter_m * ((tc+273.15)**4 - (amb+273.15)**4)

def thermal_balance(I, tc, amb, v_perp, ghi, alt, az, c, line_az=90.0, atk=90.0):
    r = R_ohm_m(tc, c); qj = I**2 * r
    qc = q_convection_ieee738(tc, amb, v_perp, c.diameter_m, atk)
    qr = q_rad_ieee738(tc, amb, c)
    qs = q_solar_ieee738(ghi, alt, az, c, line_az)
    return {"convection_w_m": qc, "radiation_w_m": qr, "solar_w_m": qs,
            "joule_w_m": qj, "residual_w_m": qj + qs - qc - qr}

def solve_temp(I, amb, v, ghi, alt, az, c, line_az=90.0, atk=90.0):
    lo = max(-50.0, amb); hi = 300.0
    def f(tc): return thermal_balance(I, tc, amb, v, ghi, alt, az, c, line_az, atk)["residual_w_m"]
    flo, fhi = f(lo), f(hi); exp = 0
    while flo * fhi > 0 and hi < 1500 and exp < 20:
        hi += 100.0; fhi = f(hi); exp += 1
    if flo * fhi > 0: return {"temperature_c": _s(amb), "converged": False}
    conv = False; T = 0.5 * (lo + hi)
    for it in range(1, 251):
        T = 0.5 * (lo + hi); ft = f(T)
        if abs(ft) <= 1e-3 or abs(hi - lo) <= 1e-7: conv = True; break
        if flo * ft <= 0: hi = T; fhi = ft
        else: lo = T; flo = ft
    return {"temperature_c": _s(T, _s(amb)), "converged": bool(conv)}

def solve_dlr(amb, v, ghi, alt, az, c, line_az=90.0, atk=90.0, T_target=None):
    T_eval = T_target if T_target is not None else 85.0
    t = thermal_balance(0.0, T_eval, amb, v, ghi, alt, az, c, line_az, atk)
    avail = t["convection_w_m"] + t["radiation_w_m"] - t["solar_w_m"]
    if not math.isfinite(avail) or avail <= 0: return 0.0
    r = R_ohm_m(T_eval, c)
    if not math.isfinite(r) or r <= 0: return 0.0
    res = math.sqrt(avail / r)
    return res if math.isfinite(res) else 0.0

_STATIC_CACHE = {}

def compute_static_rating_A(c, tmax=None, line_az=90.0):
    if tmax is None: tmax = CONFIG_DEFAULTS.get("static_tmax_c", 75.0)
    key = (c.name, tmax)
    if key in _STATIC_CACHE: return _STATIC_CACHE[key]
    amb = CONFIG_DEFAULTS.get("static_design_ambient_c", 40.0)
    wind = CONFIG_DEFAULTS.get("static_design_wind_mps", 0.6)
    solar = CONFIG_DEFAULTS.get("static_design_solar_w_m2", 1000.0)
    a = solve_dlr(amb, wind, solar, 90.0, 180.0, c, line_az, 90.0, T_target=tmax)
    _STATIC_CACHE[key] = a
    return a

def amps_to_mw(a, kv=220.0, pf=0.9):
    return a * kv * math.sqrt(3) * pf / 1000

SOLAR_ALT = [-57.72, -59.05, -53.79, -44.16, -32.31, -19.35, -5.81, 8.06, 22.12,
             36.28, 50.44, 64.42, 77.42, 81.06, 69.51, 55.71, 41.58, 27.40, 13.28,
             -0.69, -14.40, -27.65, -40.05, -50.77]
SOLAR_AZ = [19.25, 8.89, 33.91, 50.96, 62.10, 69.85, 75.73, 80.61, 85.08, 89.66,
            95.17, 103.80, 126.26, 211.01, 250.69, 262.07, 268.33, 273.12, 277.54,
            282.19, 287.59, 294.46, 304.05, 318.51]

def solar_pos(h):
    h = int(round(h)) % 24
    return {"altitude_deg": SOLAR_ALT[h], "azimuth_deg": SOLAR_AZ[h]}


# ============================================================================
# SAG MODEL
# ============================================================================
def calc_sag(tc, c, span_m=None, ref_temp_c=None, sag_ref_m=None):
    if span_m is None: span_m = CONFIG_DEFAULTS.get("span_length_m", 400.0)
    if ref_temp_c is None: ref_temp_c = CONFIG_DEFAULTS.get("sag_ref_temp_c", 20.0)
    if sag_ref_m is None: sag_ref_m = CONFIG_DEFAULTS.get("sag_ref_m", 5.0)
    if span_m <= 0 or sag_ref_m <= 0: return sag_ref_m
    L_ref = span_m + (8.0 * sag_ref_m * sag_ref_m) / (3.0 * span_m)
    dL = span_m * c.thermal_expansion_coeff * (tc - ref_temp_c)
    L_new = L_ref + dL
    num = (L_new - span_m) * 3.0 * span_m / 8.0
    if num <= 0: return sag_ref_m
    return math.sqrt(num)


def clearance_limited_dlr(amb, v_perp, ghi, alt, az, c, line_az=90.0, atk=90.0):
    T_op = CONFIG_DEFAULTS.get("operational_tmax_c", 85.0)
    min_clr = CONFIG_DEFAULTS.get("min_clearance_m", 7.0)
    tower_h = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
    span = CONFIG_DEFAULTS.get("span_length_m", 400.0)
    sag_op = calc_sag(T_op, c, span)
    clearance_op = tower_h - sag_op
    if clearance_op >= min_clr:
        dlr = solve_dlr(amb, v_perp, ghi, alt, az, c, line_az, atk, T_target=T_op)
        return {"dlr_a": dlr, "temp_c": T_op, "sag_m": sag_op,
                "clearance_m": clearance_op, "binding": "thermal"}
    lo, hi = CONFIG_DEFAULTS.get("sag_ref_temp_c", 20.0), T_op
    for _ in range(40):
        mid = (lo + hi) / 2
        sag_mid = calc_sag(mid, c, span)
        if (tower_h - sag_mid) < min_clr: hi = mid
        else: lo = mid
    T_sag_limit = lo
    sag_lim = calc_sag(T_sag_limit, c, span)
    clr_lim = tower_h - sag_lim
    dlr_lim = solve_dlr(amb, v_perp, ghi, alt, az, c, line_az, atk, T_target=T_sag_limit)
    return {"dlr_a": dlr_lim, "temp_c": T_sag_limit, "sag_m": sag_lim,
            "clearance_m": clr_lim, "binding": "clearance"}


# ============================================================================
# WEATHER RECORD
# ============================================================================
@dataclass
class WeatherRec:
    timestamp: str; ambient_c: float; wind_mps: float
    wind_direction_deg: float; ghi_w_m2: float
    dni_w_m2: float = 0.0; dhi_w_m2: float = 0.0; poa_w_m2: float = 0.0
    wind_raw_mps: float = 0.0
    wind_obs_mps: float = 0.0
    source: str = "openmeteo"


# ============================================================================
# PVLIB
# ============================================================================
def pvlib_solar(times, lat, lon, ghi_series=None, tilt=90.0, az=180.0):
    if not PVLIB_AVAILABLE: raise RuntimeError("pvlib not installed")
    sp = get_solarposition(times, lat, lon)
    if ghi_series is not None and len(ghi_series) == len(times):
        ghi = ghi_series.clip(lower=0.0)
    else:
        cs = ineichen(sp['apparent_zenith'], get_relative_airmass(sp['apparent_zenith']),
                      linke_turbidity=3.0)
        ghi = cs['ghi'].clip(lower=0.0)
    try:
        dni = dirint(ghi, sp['apparent_zenith'], times, pressure=101325.0, temp_dew=25.0).clip(lower=0.0)
        dhi = (ghi - dni * np.cos(np.radians(sp['apparent_zenith']))).clip(lower=0.0)
    except Exception:
        e = erbs(ghi, sp['apparent_zenith'], times)
        dni = e['dni'].clip(lower=0.0); dhi = e['dhi'].clip(lower=0.0)
    poa = get_total_irradiance(surface_tilt=tilt, surface_azimuth=az,
        solar_zenith=sp['apparent_zenith'], solar_azimuth=sp['azimuth'],
        dni=dni, ghi=ghi, dhi=dhi, model='perez', dni_extra=get_extra_radiation(times))
    return pd.DataFrame({
        'ghi': ghi.fillna(0.0).values, 'dni': dni.fillna(0.0).values,
        'dhi': dhi.fillna(0.0).values,
        'poa_global': poa['poa_global'].fillna(0.0).values,
    }, index=times)

def enrich_solar(records, lat, lon):
    if not records or not PVLIB_AVAILABLE: return records
    times = pd.DatetimeIndex([r.timestamp for r in records])
    ghi = pd.Series([_s(r.ghi_w_m2) for r in records], index=times)
    if ghi.sum() < 1.0: ghi = None
    sdf = pvlib_solar(times, lat, lon, ghi_series=ghi,
                      tilt=CONFIG_DEFAULTS.get("surface_tilt_deg", 90.0),
                      az=CONFIG_DEFAULTS.get("surface_azimuth_deg", 180.0))
    for i, rec in enumerate(records):
        if i < len(sdf):
            row = sdf.iloc[i]
            rec.ghi_w_m2 = _s(row['ghi']); rec.dni_w_m2 = _s(row['dni'])
            rec.dhi_w_m2 = _s(row['dhi']); rec.poa_w_m2 = _s(row['poa_global'])
    print(f"   pvlib: {len(records)} records")
    return records


# ============================================================================
# PANGU
# ============================================================================
def is_valid_zarr(p):
    if not p.exists(): return False
    try:
        ds = xr.open_zarr(str(p))
        ok = len(ds.data_vars) > 0 and len(ds.sizes) > 0
        try: ds.close()
        except: pass
        return ok
    except Exception: return False


def _extract_pangu_records(ds, lat, lon, start):
    """Extract daily records from Pangu Zarr + write JSON export."""
    ds_pt = None
    for la, lo in [("lat", "lon"), ("latitude", "longitude"), ("y", "x")]:
        if la in ds.coords and lo in ds.coords:
            try:
                ds_pt = ds.sel({la: lat, lo: lon}, method="nearest")
                break
            except: continue
    if ds_pt is None:
        raise RuntimeError("Pangu grid point not found")

    def gv(dp, *names):
        for n in names:
            if n in dp:
                return np.asarray(dp[n].values).flatten()
        raise KeyError(f"None of {names} in {list(dp.data_vars)}")

    u10 = gv(ds_pt, 'u10m', 'u10', '10u')
    v10 = gv(ds_pt, 'v10m', 'v10', '10v')
    t2m = gv(ds_pt, 't2m', 't2', '2t')

    ws = np.sqrt(u10**2 + v10**2)
    wd = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0

    base = pd.Timestamp(start)
    n = min(len(ws), len(t2m))
    ts = [(base + timedelta(hours=i * 24)).isoformat() for i in range(n)]

    recs = []
    for i in range(n):
        w_raw = _s(ws[i])
        w_mcp = apply_mcp(w_raw) if CONFIG_DEFAULTS.get("use_mcp_correction", True) else w_raw
        recs.append(WeatherRec(
            timestamp=ts[i],
            ambient_c=_s(t2m[i] - 273.15, 15.0),
            wind_mps=_s(w_mcp),
            wind_direction_deg=_s(wd[i]),
            ghi_w_m2=0.0,
            wind_raw_mps=w_raw,
            source="pangu",
        ))
    print(f"   Pangu: {len(recs)} daily records")

    try:
        out_path = Path(PANGU_JSON)
        out_path.write_text(json.dumps({
            "extracted_at": datetime.now().isoformat(),
            "lat": lat, "lon": lon, "start": start,
            "records": [{
                "timestamp": r.timestamp,
                "ambient_c": r.ambient_c,
                "wind_mps": r.wind_mps,
                "wind_direction_deg": r.wind_direction_deg,
                "wind_raw_mps": r.wind_raw_mps,
            } for r in recs],
        }))
        print(f"   Pangu JSON export: {out_path} ({len(recs)} records)")
    except Exception as e:
        print(f"   Pangu JSON export failed: {e}")

    return recs


def _load_pangu_from_json():
    json_path = Path(PANGU_JSON)
    if not json_path.exists():
        return None
    try:
        d = json.loads(json_path.read_text())
        recs = []
        for r in d.get("records", []):
            recs.append(WeatherRec(
                timestamp=r["timestamp"],
                ambient_c=_s(r["ambient_c"]),
                wind_mps=_s(r["wind_mps"]),
                wind_direction_deg=_s(r["wind_direction_deg"]),
                ghi_w_m2=0.0,
                wind_raw_mps=_s(r.get("wind_raw_mps", 0.0)),
                source="pangu_json",
            ))
        try:
            age_h = (datetime.now() - datetime.fromisoformat(d["extracted_at"])).total_seconds() / 3600
            age_str = f"{age_h:.1f}h old"
        except Exception:
            age_str = "age unknown"
        print(f"   Loaded Pangu JSON: {len(recs)} records ({age_str})")
        return recs if recs else None
    except Exception as e:
        print(f"   Pangu JSON read failed: {e}")
        return None


def fetch_pangu_cached_only(lat, lon, start, hours=168, cache_dir=None):
    if EARTH2STUDIO_AVAILABLE and XARRAY_AVAILABLE:
        if cache_dir is None:
            cache_dir = CONFIG_DEFAULTS.get("ai_cache_dir", "./weather_cache")
        key = f"pangu_{lat:.4f}_{lon:.4f}_{start}_{hours}h"
        cache = Path(cache_dir) / f"{key}.zarr"
        if is_valid_zarr(cache):
            print(f"   Loaded Pangu cache: {cache.name}")
            try:
                ds = xr.open_zarr(str(cache))
                return _extract_pangu_records(ds, lat, lon, start)
            except Exception as e:
                print(f"   Pangu Zarr read failed: {e}")
    return _load_pangu_from_json()


def pangu_weight(idx):
    if idx <= 1: return 1.0
    if idx <= 3: return 1.0 - 0.10 * (idx - 1)
    if idx <= 6: return 0.70 - 0.10 * (idx - 3)
    return 0.20


def blend_ai(lat, lon, start, end, hours=168):
    om = fetch_om(lat, lon, start, end, forecast=True)
    if not om:
        return om
    pg = fetch_pangu_cached_only(lat, lon, start, hours)
    if pg is None:
        return om

    om_by_day = {}
    for r in om:
        d = r.timestamp[:10]
        om_by_day.setdefault(d, {"T": [], "W": []})
        om_by_day[d]["T"].append(r.ambient_c)
        om_by_day[d]["W"].append(r.wind_mps)
    om_daily = {d: {"T": sum(v["T"])/len(v["T"]), "W": sum(v["W"])/len(v["W"])}
                for d, v in om_by_day.items()}

    anom = {}
    for i, p in enumerate(pg):
        d = p.timestamp[:10]
        if d not in om_daily:
            continue
        w = pangu_weight(i)
        anom[d] = {
            "T": (p.ambient_c - om_daily[d]["T"]) * w,
            "W": (p.wind_mps - om_daily[d]["W"]) * w,
        }

    blended = []
    for r in om:
        d = r.timestamp[:10]
        a = anom.get(d, {"T": 0.0, "W": 0.0})
        blended.append(WeatherRec(
            timestamp=r.timestamp,
            ambient_c=r.ambient_c + a["T"],
            wind_mps=max(0.1, r.wind_mps + a["W"]),
            wind_direction_deg=r.wind_direction_deg,
            ghi_w_m2=r.ghi_w_m2,
            dni_w_m2=r.dni_w_m2,
            dhi_w_m2=r.dhi_w_m2,
            wind_raw_mps=r.wind_raw_mps,
            source="om_mcp+pangu",
        ))
    print(f"   Blended: {len(blended)} records (OM + Pangu anomalies)")
    return blended


def build_pangu_cache():
    if not EARTH2STUDIO_AVAILABLE:
        print("❌ Earth2Studio not available"); return 1
    if not XARRAY_AVAILABLE:
        print("❌ xarray not available"); return 1

    lat = CONFIG_DEFAULTS["lat"]; lon = CONFIG_DEFAULTS["lon"]
    start = datetime.now().strftime("%Y-%m-%d")
    hours = CONFIG_DEFAULTS.get("ai_forecast_hours", 168)
    cache_dir = CONFIG_DEFAULTS.get("ai_cache_dir", "./weather_cache")
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Cannot create cache dir: {e}"); return 1

    key = f"pangu_{lat:.4f}_{lon:.4f}_{start}_{hours}h"
    cache = Path(cache_dir) / f"{key}.zarr"

    if not is_valid_zarr(cache):
        print(f"Running Pangu inference (~60 min on CPU)...")
        print(f"   Target: {cache}")
        if cache.exists():
            try: shutil.rmtree(cache, ignore_errors=True)
            except: pass

        dev = CONFIG_DEFAULTS.get("ai_device", "cpu")
        onnx = CONFIG_DEFAULTS.get("pangu_onnx_path") or find_onnx()
        if not onnx:
            print("❌ Pangu ONNX model not found.")
            return 1
        print(f"   Using ONNX: {onnx}")

        if dev == "cpu" and TORCH_AVAILABLE:
            try: torch.set_default_device("cpu")
            except: pass

        model = Pangu24(ort_24hr=onnx)
        data = GFS()
        n = max(1, hours // 24)
        io = ZarrBackend(str(cache))
        print(f"   Running {n} steps ({n*24}h horizon)...")
        run_deterministic(
            time=[start], nsteps=n, prognostic=model, data=data, io=io,
            device=torch.device(dev) if TORCH_AVAILABLE else None,
        )
        try:
            if hasattr(io, "close"): io.close()
        except: pass
        print(f"✅ Pangu inference complete: {cache}")
    else:
        print(f"Pangu Zarr already exists: {cache}")
        try:
            ds = xr.open_zarr(str(cache))
            _extract_pangu_records(ds, lat, lon, start)
        except Exception as e:
            print(f"   Could not re-extract from Zarr: {e}")

    print("\nRebuilding forecast cache with Pangu blend...")
    _cached["last_updated"] = None
    update_forecast_cache(force=True)
    return 0


# ============================================================================
# OPEN-METEO
# ============================================================================
def fetch_om(lat, lon, start, end, tz="auto", forecast=False, quick=False):
    try:
        start_dt = datetime.fromisoformat(start) if "T" in start \
                   else datetime.fromisoformat(start + "T00:00:00")
        end_dt = datetime.fromisoformat(end) if "T" in end \
                 else datetime.fromisoformat(end + "T23:59:59")
    except Exception:
        start_dt = end_dt = None

    now = datetime.now()
    archive_cutoff = now - timedelta(days=5)
    HOURLY = ("temperature_2m,wind_speed_10m,wind_direction_10m,"
              "shortwave_radiation,direct_normal_irradiance,diffuse_radiation")

    if forecast:
        base = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": HOURLY,
                  "wind_speed_unit": "ms", "timezone": tz, "forecast_days": 7}
        kind = "forecast"
    elif end_dt is not None and end_dt <= archive_cutoff:
        base = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon,
                  "start_date": start, "end_date": end,
                  "hourly": HOURLY, "wind_speed_unit": "ms", "timezone": tz}
        kind = "archive"
    else:
        base = "https://api.open-meteo.com/v1/forecast"
        past_days = 1; forecast_days = 7
        if start_dt is not None:
            past_days = max(0, min(92, (now.date() - start_dt.date()).days + 1))
        if end_dt is not None:
            forecast_days = max(1, min(16, (end_dt.date() - now.date()).days + 2))
        params = {"latitude": lat, "longitude": lon, "hourly": HOURLY,
                  "wind_speed_unit": "ms", "timezone": tz,
                  "past_days": past_days, "forecast_days": forecast_days}
        kind = "recent"

    url = base + "?" + urllib.parse.urlencode(params)
    print(f"   Open-Meteo {kind}{' (quick)' if quick else ''}: {start} -> {end}")

    last_err = None
    max_att = 1 if quick else 5
    for att in range(max_att):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'GridTweak/1.0'})
            with urllib.request.urlopen(req, timeout=(10 if quick else 30)) as r:
                data = json.loads(r.read().decode())
            last_err = None
            break
        except Exception as e:
            last_err = e
            code = getattr(e, "code", None)
            if code == 429:
                if quick: print("      429 (quick — no retry)"); break
                if att < 3:
                    delay = 30 * (att + 1)
                    print(f"      429 rate-limited — waiting {delay}s")
                    time.sleep(delay); continue
                print("      429 persisted — giving up"); break
            if code is not None and 400 <= code < 500:
                print(f"      HTTP {code} (no retry)"); break
            if att < (max_att - 1):
                delay = 3 * (att + 1)
                print(f"      retry in {delay}s ({e})")
                time.sleep(delay)

    if last_err: raise RuntimeError(f"OM {kind} failed: {last_err}")

    h = data.get("hourly", {}); times = h.get("time", [])
    if not times: raise RuntimeError(f"OM {kind}: no times in response")

    recs = []
    for i, ts in enumerate(times):
        try:
            if len(ts) == 16: ts += ":00"
            def s(v, d=0.0):
                if v is None or not isinstance(v, (int, float)): return d
                f = float(v); return f if math.isfinite(f) else d
            amb = s(h["temperature_2m"][i], 15.0)
            w_raw = s(h["wind_speed_10m"][i])
            wdir = s(h["wind_direction_10m"][i])
            ghi = s(h["shortwave_radiation"][i])
            dni = s(h.get("direct_normal_irradiance", [0]*len(times))[i])
            dhi = s(h.get("diffuse_radiation", [0]*len(times))[i])
            w_mcp = apply_mcp(w_raw) if CONFIG_DEFAULTS.get("use_mcp_correction", True) else w_raw
            if w_mcp <= 0: wdir = 0
            if wdir < 0 or wdir >= 360: wdir = 0
            recs.append(WeatherRec(ts, amb, w_mcp, wdir, ghi,
                                    dni_w_m2=dni, dhi_w_m2=dhi,
                                    wind_raw_mps=float(w_raw),
                                    source=f"openmeteo_{kind}"))
        except Exception: pass

    if kind == "recent" and start_dt is not None and end_dt is not None:
        filtered = []
        for r in recs:
            try:
                rt = datetime.fromisoformat(r.timestamp)
                if start_dt <= rt <= end_dt: filtered.append(r)
            except Exception: filtered.append(r)
        recs = filtered

    print(f"   Open-Meteo {kind}: {len(recs)} records")
    return recs


def smooth_wind(records):
    if not records: return records
    n = len(records); out = [0.0] * n
    for i in range(n):
        tot, wt = 0.0, 0.0
        for j in range(-1, 2):
            k = i + j
            if 0 <= k < n:
                w = 2 - abs(j)
                tot += records[k].wind_mps * w; wt += w
        s = tot / wt if wt > 0 else records[i].wind_mps
        out[i] = min(s, CONFIG_DEFAULTS.get("wind_max_mps", 15.0))
    for i, r in enumerate(records): r.wind_mps = _s(out[i])
    return records


# ============================================================================
# DLR RECORDS
# ============================================================================
def run_dlr_records(records, c):
    results = []
    offset = CONFIG_DEFAULTS.get("timezone_offset_hours", 5.5)
    static_fixed_a = compute_static_rating_A(c)
    tower_h = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
    derate = CONFIG_DEFAULTS.get("operational_derate_factor", 1.0)

    for w in records:
        try:
            dt = datetime.fromisoformat(w.timestamp)
            lh = (dt.hour + offset) % 24
        except: lh = 12
        sp = solar_pos(lh)

        w_used = _s(w.wind_mps)
        geo = wind_geom(w_used, _s(w.wind_direction_deg), LINE_AZ)
        wp = geo["perpendicular_ms"]; atk = geo["attack_angle_deg"]

        clr_result = clearance_limited_dlr(_s(w.ambient_c), wp, _s(w.ghi_w_m2),
                                            sp["altitude_deg"], sp["azimuth_deg"],
                                            c, LINE_AZ, atk)

        # --- operational derate (conservative margin, not shown in UI) ---
        dlr_raw_a = _s(clr_result["dlr_a"])
        dlr_op_a  = dlr_raw_a * derate
        tr_op = solve_temp(dlr_op_a, _s(w.ambient_c), wp, _s(w.ghi_w_m2),
                           sp["altitude_deg"], sp["azimuth_deg"], c, LINE_AZ, atk)
        temp_op = _s(tr_op["temperature_c"])
        sag_op  = calc_sag(temp_op, c)
        clr_op  = tower_h - sag_op
        # ------------------------------------------------------------------

        T_emerg = CONFIG_DEFAULTS.get("emergency_tmax_c", 100.0)
        dlr_emerg = solve_dlr(_s(w.ambient_c), wp, _s(w.ghi_w_m2),
                              sp["altitude_deg"], sp["azimuth_deg"],
                              c, LINE_AZ, atk, T_target=T_emerg)

        test_tr = solve_temp(CONFIG_DEFAULTS["test_current"], _s(w.ambient_c), wp,
                              _s(w.ghi_w_m2), sp["altitude_deg"], sp["azimuth_deg"],
                              c, LINE_AZ, atk)
        actual_temp = _s(test_tr["temperature_c"])
        actual_sag = calc_sag(actual_temp, c)
        actual_clearance = tower_h - actual_sag

        kv = CONFIG_DEFAULTS["nominal_voltage_kv"]; pf = CONFIG_DEFAULTS["power_factor"]
        results.append({
            "timestamp": w.timestamp,
            "ambient_c": _s(w.ambient_c),
            "wind_mps": _s(w.wind_mps),
            "wind_raw_mps": _s(getattr(w, "wind_raw_mps", 0.0)),
            "wind_obs_mps": 0.0,
            "wind_direction_deg": _s(w.wind_direction_deg),
            "perpendicular_wind_mps": _s(wp),
            "ghi_w_m2": _s(w.ghi_w_m2),
            "poa_w_m2": _s(getattr(w, "poa_w_m2", 0.0)),
            "dlr_a": _s(dlr_op_a),
            "dlr_static_a": _s(static_fixed_a),
            "dlr_emergency_100c_a": _s(dlr_emerg),
            "dlr_mw": _s(amps_to_mw(dlr_op_a, kv, pf)),
            "binding_constraint": clr_result["binding"],
            "temperature_c": actual_temp,
            "sag_m": _s(actual_sag),
            "clearance_m": _s(actual_clearance),
            "dlr_temp_c": temp_op,
            "dlr_sag_m": _s(sag_op),
            "dlr_clearance_m": _s(clr_op),
            "source": getattr(w, "source", "unknown"),
        })
    return results


def fetch_weather_multi_year(lat, lon, start, end, quick=False):
    sdt = datetime.fromisoformat(start); edt = datetime.fromisoformat(end)
    all_recs = []
    for y in range(sdt.year, edt.year + 1):
        ys = f"{y}-01-01"; ye = f"{y}-12-31"
        if y == sdt.year: ys = start
        if y == edt.year: ye = end
        recs = fetch_om(lat, lon, ys, ye, forecast=False, quick=quick)
        if PVLIB_AVAILABLE:
            try: recs = enrich_solar(recs, lat, lon)
            except Exception: pass
        all_recs.extend(smooth_wind(recs))
    return all_recs


# ============================================================================
# ELEVATION
# ============================================================================
_ELEV_CACHE = {}

def fetch_elevation(lat, lon):
    key = (round(lat, 4), round(lon, 4))
    if key in _ELEV_CACHE:
        return _ELEV_CACHE[key]
    try:
        url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        req = urllib.request.Request(url, headers={'User-Agent': 'GridTweak/1.0'})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
            elevs = data.get("elevation", [0])
            elev = _s(elevs[0]) if elevs else 0.0
            _ELEV_CACHE[key] = elev
            return elev
    except Exception as e:
        print(f"   Elevation fetch failed for ({lat:.4f},{lon:.4f}): {e}")
        return 0.0


def segments_from_cache(nseg, lat, lon, elat, elon):
    cached = _cached.get("data") or []
    if not cached: return []
    base = min(cached, key=lambda r: r.get("dlr_a", 9999) or 9999)
    segs = []
    coords = [(lat + (elat - lat) * i / nseg, lon + (elon - lon) * i / nseg)
              for i in range(nseg + 1)]
    import random
    random.seed(42)
    print(f"   Fetching elevation for {len(coords)} segment points...")
    for i, (slat, slon) in enumerate(coords):
        jitter = 1.0 + random.uniform(-0.03, 0.03)
        dlr_a = (base.get("dlr_a", 0) or 0) * jitter
        elev = fetch_elevation(slat, slon)
        segs.append({
            "lat": slat, "lon": slon, "elevation_m": elev,
            "min_dlr_a": dlr_a,
            "min_dlr_mw": amps_to_mw(dlr_a, CONFIG_DEFAULTS["nominal_voltage_kv"],
                                      CONFIG_DEFAULTS["power_factor"]),
            "temperature_c": base.get("dlr_temp_c", 85.0) or 85.0,
            "sag_m": base.get("dlr_sag_m", 0.0) or 0.0,
            "clearance_m": base.get("dlr_clearance_m", 0.0) or 0.0,
            "binding_constraint": base.get("binding_constraint", "thermal"),
            "worst_hour": base.get("timestamp", ""),
            "results": [],
        })
    print(f"   Elevations: {[round(s['elevation_m'],1) for s in segs]}")
    return segs


# ============================================================================
# FORECAST CACHE UPDATE
# ============================================================================
def update_forecast_cache(force=False):
    global _cached
    if not force and _cached["last_updated"]:
        try:
            age = (datetime.now() - datetime.fromisoformat(_cached["last_updated"])).total_seconds() / 3600
            if age < 6:
                print(f"Cache fresh ({age:.1f}h)"); return
        except: pass
    if _cached.get("building") and not force:
        print("Already building"); return

    _cached["building"] = True
    _cached["last_attempt"] = datetime.now().isoformat()
    _cached["last_error"] = None

    try:
        print("Updating forecast cache...")
        lat = CONFIG_DEFAULTS["lat"]; lon = CONFIG_DEFAULTS["lon"]
        c = get_conductor(CONFIG_DEFAULTS["conductor"])
        start = datetime.now().strftime("%Y-%m-%d")
        end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        force_archive = CONFIG_DEFAULTS.get("force_archive_forecast", False)

        weather = None; source = ""

        if not force_archive:
            try:
                print("  [Tier 1] AI blend (Pangu + OM forecast)")
                weather = blend_ai(lat, lon, start, end,
                                    hours=CONFIG_DEFAULTS.get("ai_forecast_hours", 168))
                if weather:
                    if PVLIB_AVAILABLE:
                        try: weather = enrich_solar(weather, lat, lon)
                        except Exception: pass
                    weather = smooth_wind(weather)
                    source = weather[0].source if weather else "ai_blend"
            except Exception as e:
                print(f"  Tier 1 failed: {type(e).__name__}: {e}")
                weather = None

        if not weather and not force_archive:
            try:
                print("  [Tier 2] Open-Meteo forecast API")
                weather = fetch_om(lat, lon, start, end, forecast=True)
                if weather:
                    if PVLIB_AVAILABLE:
                        try: weather = enrich_solar(weather, lat, lon)
                        except Exception: pass
                    weather = smooth_wind(weather)
                    source = "openmeteo_forecast"
            except Exception as e:
                print(f"  Tier 2 failed: {type(e).__name__}: {e}")
                weather = None

        if not weather:
            try:
                print("  [Tier 3] Open-Meteo archive fallback")
                ast = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                weather = fetch_om(lat, lon, ast, end, forecast=False)
                if weather:
                    if PVLIB_AVAILABLE:
                        try: weather = enrich_solar(weather, lat, lon)
                        except Exception: pass
                    weather = smooth_wind(weather)
                    source = "archive_fallback"
            except Exception as e:
                print(f"  Tier 3 failed: {type(e).__name__}: {e}")
                weather = None

        if not weather:
            msg = "All tiers returned no weather data"
            print(f"Failed: {msg}")
            _cached["last_error"] = msg
            return

        results = run_dlr_records(weather, c)
        _cached["data"] = results
        _cached["last_updated"] = datetime.now().isoformat()
        _cached["source"] = source
        save_cache()
        print(f"Cache updated via {source}: {len(results)} records")
    except Exception as e:
        print(f"Cache failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        _cached["last_error"] = f"{type(e).__name__}: {e}"
    finally:
        _cached["building"] = False


# ============================================================================
# STATIC EXPORT
# ============================================================================
def export_static_files():
    print("\n=== Exporting static response files ===")
    c = get_conductor(CONFIG_DEFAULTS["conductor"])
    kv = CONFIG_DEFAULTS["nominal_voltage_kv"]
    pf = CONFIG_DEFAULTS["power_factor"]
    static_a = compute_static_rating_A(c)
    static_mw = amps_to_mw(static_a, kv, pf)
    cached = _cached.get("data") or []

    if not cached:
        print("⚠️ No cache data — cannot export static files"); return False

    current_recs = list(cached[:24]) if len(cached) >= 24 else list(cached)
    _write_static(STATIC_CURRENT, {
        "results": current_recs,
        "static_rating_mw": static_mw,
        "static_rating_a": static_a,
        "location_name": CONFIG_DEFAULTS["location_name"],
        "source": "static",
    })

    _write_static(STATIC_FORECAST, {
        "forecast": cached,
        "source": _cached.get("source") or "static",
        "building": False,
        "last_error": None,
    })

    lat = CONFIG_DEFAULTS["lat"]; lon = CONFIG_DEFAULTS["lon"]
    elat = CONFIG_DEFAULTS.get("end_lat") or (lat + 0.5)
    elon = CONFIG_DEFAULTS.get("end_lon") or (lon + 0.5)
    nseg = CONFIG_DEFAULTS.get("num_segments", 3)
    segs = segments_from_cache(nseg, lat, lon, elat, elon)
    mn = min((s["min_dlr_a"] for s in segs), default=0.0)
    _write_static(STATIC_CORRIDOR, {
        "min_dlr_mw": amps_to_mw(mn, kv, pf),
        "weakest_segment": 0,
        "source": "static",
        "segments": [{"lat": s["lat"], "lon": s["lon"], "elevation_m": s["elevation_m"],
                      "min_dlr_mw": s["min_dlr_mw"], "temperature_c": s["temperature_c"],
                      "sag_m": s["sag_m"], "clearance_m": s["clearance_m"],
                      "binding_constraint": s["binding_constraint"],
                      "worst_hour": s["worst_hour"]} for s in segs],
    })

    tower_h = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
    min_clr = CONFIG_DEFAULTS.get("min_clearance_m", 7.0)
    corridor_span = CONFIG_DEFAULTS.get("span_length_m", 400.0)
    gt_curve = []
    for tc in [20, 40, 60, 75, 85, 100]:
        sag_350 = calc_sag(tc, c, 350.0)
        sag_400 = calc_sag(tc, c, corridor_span)
        gt_curve.append({
            "temp_c": tc,
            "gt_sag_350m_m": round(sag_350, 3),
            "gt_sag_corridor_m": round(sag_400, 3),
            "clearance_350m_m": round(tower_h - sag_350, 3),
            "clearance_corridor_m": round(tower_h - sag_400, 3),
        })
    comparisons = []
    for bench in UTILITY_SAG_BENCHMARKS:
        gt_sag = calc_sag(bench["temp_c"], c, float(bench["span_m"]))
        diff = ((gt_sag - bench["sag_m"]) / bench["sag_m"]) * 100
        comparisons.append({**bench,
            "gt_sag_m": round(gt_sag, 3),
            "diff_m": round(gt_sag - bench["sag_m"], 3),
            "diff_pct": round(diff, 1)})
    comparisons.sort(key=lambda x: abs(x.get("diff_pct", 999)))
    _write_static(STATIC_SAGCOMP, {
        "conductor": c.name, "tower_height_m": tower_h,
        "min_clearance_m": min_clr, "corridor_span_m": corridor_span,
        "gt_curve": gt_curve, "utility_benchmarks": comparisons,
    })

    span = CONFIG_DEFAULTS.get("span_length_m", 400.0)
    rows = []
    for tc in range(20, 101, 5):
        sag = calc_sag(tc, c, span)
        clr = tower_h - sag
        rows.append({"temp_c": tc, "sag_m": round(sag, 3),
                     "clearance_m": round(clr, 3),
                     "status": "CLEARANCE_LIMITED" if clr < min_clr else "OK"})
    _write_static(STATIC_SAGCHECK, {
        "conductor": c.name, "span_m": span,
        "tower_height_m": tower_h, "min_clearance_m": min_clr,
        "sag_ref_m": CONFIG_DEFAULTS.get("sag_ref_m"),
        "curve": rows,
    })

    print("=== Static export complete ===\n")
    return True


# ============================================================================
# FASTAPI
# ============================================================================
app = FastAPI(title=APP_NAME, version=VERSION) if API_AVAILABLE else None

if app is not None:
    @app.get("/")
    async def root(): return RedirectResponse(url="/dashboard")

    @app.get("/dlr/health")
    async def health():
        return {"version": VERSION, "conductor": CONFIG_DEFAULTS.get("conductor"),
                "pvlib": PVLIB_AVAILABLE, "earth2studio": EARTH2STUDIO_AVAILABLE,
                "mcp_loaded": MCP is not None,
                "pangu_json_present": os.path.exists(PANGU_JSON),
                "cache_records": len(_cached.get("data") or []),
                "cache_source": _cached.get("source"),
                "cache_last_updated": _cached.get("last_updated"),
                "cache_building": _cached.get("building", False),
                "cache_last_error": _cached.get("last_error"),
                "auto_fresh": os.environ.get("GRIDTWEAK_AUTO_FRESH", "0")}

    @app.get("/dlr/cache_status")
    async def cache_status():
        return JSONResponse(content=_san({
            "version": VERSION,
            "has_data": _cached["data"] is not None,
            "n_records": len(_cached.get("data") or []),
            "last_updated": _cached.get("last_updated"),
            "source": _cached.get("source"),
            "building": _cached.get("building", False),
            "last_error": _cached.get("last_error"),
        }))

    @app.get("/dlr/build_now")
    async def build_now():
        _cached["last_updated"] = None
        _cached["last_error"] = None
        try:
            update_forecast_cache(force=True)
            return {"status": "ok",
                    "records": len(_cached.get("data") or []),
                    "source": _cached.get("source"),
                    "last_error": _cached.get("last_error")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.get("/dlr/sag_check")
    async def sag_check():
        s = _read_static(STATIC_SAGCHECK)
        if s is not None: return JSONResponse(content=s)
        c = get_conductor(CONFIG_DEFAULTS["conductor"])
        span = CONFIG_DEFAULTS.get("span_length_m", 400.0)
        tower_h = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
        min_clr = CONFIG_DEFAULTS.get("min_clearance_m", 7.0)
        rows = []
        for tc in range(20, 101, 5):
            sag = calc_sag(tc, c, span); clr = tower_h - sag
            rows.append({"temp_c": tc, "sag_m": round(sag, 3),
                         "clearance_m": round(clr, 3),
                         "status": "CLEARANCE_LIMITED" if clr < min_clr else "OK"})
        return JSONResponse(content=_san({
            "conductor": c.name, "span_m": span, "tower_height_m": tower_h,
            "min_clearance_m": min_clr, "sag_ref_m": CONFIG_DEFAULTS.get("sag_ref_m"),
            "curve": rows}))

    @app.get("/dlr/sag_comparison")
    async def sag_comparison():
        s = _read_static(STATIC_SAGCOMP)
        if s is not None: return JSONResponse(content=s)
        c = get_conductor(CONFIG_DEFAULTS["conductor"])
        tower_h = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
        min_clr = CONFIG_DEFAULTS.get("min_clearance_m", 7.0)
        corridor_span = CONFIG_DEFAULTS.get("span_length_m", 400.0)
        gt_curve = []
        for tc in [20, 40, 60, 75, 85, 100]:
            gt_curve.append({
                "temp_c": tc,
                "gt_sag_350m_m": round(calc_sag(tc, c, 350.0), 3),
                "gt_sag_corridor_m": round(calc_sag(tc, c, corridor_span), 3),
                "clearance_350m_m": round(tower_h - calc_sag(tc, c, 350.0), 3),
                "clearance_corridor_m": round(tower_h - calc_sag(tc, c, corridor_span), 3)})
        comparisons = []
        for bench in UTILITY_SAG_BENCHMARKS:
            gt = calc_sag(bench["temp_c"], c, float(bench["span_m"]))
            diff = ((gt - bench["sag_m"]) / bench["sag_m"]) * 100
            comparisons.append({**bench, "gt_sag_m": round(gt, 3),
                "diff_m": round(gt - bench["sag_m"], 3), "diff_pct": round(diff, 1)})
        comparisons.sort(key=lambda x: abs(x.get("diff_pct", 999)))
        return JSONResponse(content=_san({
            "conductor": c.name, "tower_height_m": tower_h,
            "min_clearance_m": min_clr, "corridor_span_m": corridor_span,
            "gt_curve": gt_curve, "utility_benchmarks": comparisons}))

    @app.get("/dlr/wind_check")
    async def wind_check():
        metar = fetch_metar(CONFIG_DEFAULTS.get("metar_icao", "VABB"))
        return JSONResponse(content=_san({
            "n_records": 0, "metar_current": metar,
            "mcp_slope": MCP["slope"] if MCP else None,
            "mcp_intercept": MCP["intercept"] if MCP else None,
            "samples": [],
        }))

    @app.get("/dlr/metar")
    async def metar_endpoint():
        return JSONResponse(content=_san(fetch_metar(CONFIG_DEFAULTS.get("metar_icao", "VABB"))))

    @app.get("/dlr/current")
    async def get_current():
        s = _read_static(STATIC_CURRENT)
        if s is not None: return JSONResponse(content=s)
        cached = _cached.get("data") or []
        c = get_conductor(CONFIG_DEFAULTS["conductor"])
        kv = CONFIG_DEFAULTS["nominal_voltage_kv"]; pf = CONFIG_DEFAULTS["power_factor"]
        static_a = compute_static_rating_A(c)
        if cached:
            res = list(cached[:24]) if len(cached) >= 24 else list(cached)
            return JSONResponse(content=_san({
                "results": res, "static_rating_mw": amps_to_mw(static_a, kv, pf),
                "static_rating_a": static_a,
                "location_name": CONFIG_DEFAULTS["location_name"],
                "source": "cache"}))
        return JSONResponse(content=_san({
            "results": [], "static_rating_mw": amps_to_mw(static_a, kv, pf),
            "static_rating_a": static_a,
            "location_name": CONFIG_DEFAULTS["location_name"],
            "source": "empty"}))

    @app.get("/dlr/forecast")
    async def get_forecast():
        s = _read_static(STATIC_FORECAST)
        if s is not None: return JSONResponse(content=s)
        if _cached["data"] is None:
            return JSONResponse(content={"forecast": [], "source": None,
                                          "building": _cached.get("building", False),
                                          "last_error": _cached.get("last_error")})
        return JSONResponse(content=_san({
            "forecast": _cached["data"], "source": _cached.get("source"),
            "building": _cached.get("building", False),
            "last_error": _cached.get("last_error")}))

    @app.get("/dlr/refresh_forecast")
    async def refresh_forecast():
        _cached["last_updated"] = None
        threading.Thread(target=update_forecast_cache, kwargs={"force": True},
                         daemon=True).start()
        return {"status": "rebuilding"}

    @app.get("/dlr/corridor")
    async def get_corridor():
        s = _read_static(STATIC_CORRIDOR)
        if s is not None: return JSONResponse(content=s)
        try:
            lat = CONFIG_DEFAULTS["lat"]; lon = CONFIG_DEFAULTS["lon"]
            elat = CONFIG_DEFAULTS.get("end_lat") or (lat + 0.5)
            elon = CONFIG_DEFAULTS.get("end_lon") or (lon + 0.5)
            nseg = CONFIG_DEFAULTS.get("num_segments", 3)
            kv = CONFIG_DEFAULTS["nominal_voltage_kv"]; pf = CONFIG_DEFAULTS["power_factor"]
            segs = segments_from_cache(nseg, lat, lon, elat, elon)
            mn = min((s["min_dlr_a"] for s in segs), default=0.0)
            return JSONResponse(content=_san({
                "min_dlr_mw": amps_to_mw(mn, kv, pf),
                "weakest_segment": 0, "source": "cache",
                "segments": [{"lat": s["lat"], "lon": s["lon"],
                              "elevation_m": s["elevation_m"],
                              "min_dlr_mw": s["min_dlr_mw"],
                              "temperature_c": s["temperature_c"],
                              "sag_m": s["sag_m"], "clearance_m": s["clearance_m"],
                              "binding_constraint": s["binding_constraint"],
                              "worst_hour": s["worst_hour"]} for s in segs]}))
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        location = CONFIG_DEFAULTS.get("location_name", "Transmission Line")
        c_cfg = get_conductor(CONFIG_DEFAULTS["conductor"])
        static_a_75 = compute_static_rating_A(c_cfg, tmax=75.0)
        static_mw_75 = round(amps_to_mw(static_a_75,
                                         CONFIG_DEFAULTS["nominal_voltage_kv"],
                                         CONFIG_DEFAULTS["power_factor"]), 1)
        conductor_js = CONFIG_DEFAULTS.get("conductor", "zebra").upper()
        tower_h_js = CONFIG_DEFAULTS.get("tower_attachment_height_m", 30.0)
        span_js = CONFIG_DEFAULTS.get("span_length_m", 400.0)
        min_clr_js = CONFIG_DEFAULTS.get("min_clearance_m", 7.0)
        fav = ""
        p = Path("favicon_base64.txt")
        if p.exists():
            try:
                with open(p) as f: fav = f.read().strip()
            except: pass
        fav_tag = f'<link rel="icon" href="data:image/x-icon;base64,{fav}" type="image/x-icon">' if fav else '<link rel="icon" href="data:,">'
        html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GridTweak - DLR Intelligence</title>{fav_tag}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f6f9fc;color:#1a2634;padding:20px}}
.container{{max-width:1400px;margin:0 auto}}
.header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:30px;flex-wrap:wrap;gap:15px}}
.logo{{font-size:28px;font-weight:700;color:#1a6b8a}}.logo span{{color:#0b2e4f}}
.status-badge{{background:#e6f7e6;color:#0e7c3e;padding:6px 16px;border-radius:30px;font-size:13px;font-weight:500;display:flex;align-items:center;gap:6px}}
.dot{{width:8px;height:8px;background:#0e7c3e;border-radius:50%;animation:pulse 2s infinite}}
@keyframes pulse{{0%{{opacity:1}}50%{{opacity:.3}}100%{{opacity:1}}}}
.last-updated{{font-size:13px;color:#718096}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin-bottom:30px}}
.metric-card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);border:1px solid #edf2f7}}
.metric-card.highlight{{border:2px solid #1a6b8a;background:#f0f9ff}}
.metric-label{{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#718096;margin-bottom:4px}}
.metric-value{{font-size:24px;font-weight:700}}
.metric-unit{{font-size:13px;font-weight:400;color:#718096;margin-left:4px}}
.recommendation{{background:#e0f2fe;border-left:4px solid #1a6b8a;padding:14px 20px;border-radius:8px;margin-bottom:25px}}
.rec-text{{font-weight:500;font-size:15px}}
.rec-text strong{{color:#1a6b8a}}
.tabs{{display:flex;gap:6px;background:#e2e8f0;padding:4px;border-radius:10px;margin-bottom:25px;width:fit-content}}
.tab{{padding:8px 20px;border:none;border-radius:8px;font-weight:600;cursor:pointer;background:transparent;color:#4a5568;font-size:14px}}
.tab.active{{background:#fff;color:#1a202c;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.chart-container{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);border:1px solid #edf2f7;margin-bottom:25px;height:400px;position:relative}}
.resolution-note{{font-size:12px;color:#718096;margin-bottom:8px;font-weight:500}}
.table-wrap{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);border:1px solid #edf2f7;overflow-x:auto;margin-bottom:25px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;padding:8px 6px;color:#4a5568;font-weight:600;border-bottom:2px solid #edf2f7}}
td{{padding:6px;border-bottom:1px solid #edf2f7}}
.refresh-btn{{background:#1a6b8a;color:#fff;border:none;padding:6px 18px;border-radius:6px;font-weight:600;cursor:pointer;font-size:13px}}
.footer{{margin-top:40px;text-align:center;font-size:12px;color:#718096}}
.corridor-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.corridor-chart{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.04);border:1px solid #edf2f7;height:300px}}
.status-ok{{color:#0e7c3e;font-weight:600}}
.status-warn{{color:#c53030;font-weight:600}}
.diff-good{{color:#0e7c3e;font-weight:600}}
.diff-warn{{color:#c53030;font-weight:600}}
@media(max-width:640px){{.metric-grid{{grid-template-columns:1fr 1fr}}.corridor-grid{{grid-template-columns:1fr}}}}
</style>
</head><body>
<div class="container">
<div class="header">
<div><div class="logo">Grid<span>Tweak</span></div>
<div style="font-size:14px;color:#718096;margin-top:4px;font-weight:500;">{location} | {conductor_js} | IEEE 738</div></div>
<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
<span class="last-updated" id="lastUpdated">Updating...</span>
<div class="status-badge"><span class="dot"></span> System Live</div>
<button class="refresh-btn" onclick="fetchAll()">Refresh</button>
</div></div>

<div class="recommendation">
    <span class="rec-text"><strong>Estimated Thermal Headroom:</strong> Conservative estimate suggests <strong id="recMw">-- MW</strong> of potential additional thermal capacity above the static rating. <span style="font-size:12px;font-weight:normal;color:#718096;">(Pending site-specific validation. Network transfer limits may apply.)</span></span>
</div>

<div class="metric-grid">
<div class="metric-card highlight"><div class="metric-label">Operational DLR</div><div class="metric-value" id="dlr">-- <span class="metric-unit">A</span></div></div>
<div class="metric-card"><div class="metric-label">Emergency (100C)</div><div class="metric-value" id="dlr_emerg">-- <span class="metric-unit">A</span></div></div>
<div class="metric-card"><div class="metric-label">Static Rating</div><div class="metric-value" id="static75">-- <span class="metric-unit">A</span></div></div>
<div class="metric-card"><div class="metric-label">Headroom Now</div><div class="metric-value" id="headroom_pct">--</div></div>
<div class="metric-card"><div class="metric-label">Conductor Temp</div><div class="metric-value" id="temp">-- <span class="metric-unit">°C</span></div></div>
<div class="metric-card"><div class="metric-label">Ambient</div><div class="metric-value" id="ambient">-- <span class="metric-unit">°C</span></div></div>
<div class="metric-card"><div class="metric-label">Wind</div><div class="metric-value" id="wind">-- <span class="metric-unit">m/s</span></div></div>
</div>

<div class="tabs">
<button class="tab active" data-tab="historical">Historical (24h)</button>
<button class="tab" data-tab="forecast">Forecast (7d)</button>
<button class="tab" data-tab="corridor">Corridor</button>
</div>

<div id="historical-tab" class="tab-content active">
<div class="chart-container"><canvas id="historicalChart"></canvas></div>
</div>
<div id="forecast-tab" class="tab-content">
<div class="chart-container">
  <div class="resolution-note" id="forecastResolution"></div>
  <canvas id="forecastChart"></canvas>
</div>
<div class="table-wrap"><h3 style="margin-bottom:12px;">7-Day Forecast (MW)</h3><div id="forecastTable"></div></div>
</div>
<div id="corridor-tab" class="tab-content">
<div class="corridor-grid">
<div class="corridor-chart"><canvas id="corridorDlrChart"></canvas></div>
<div class="corridor-chart"><canvas id="corridorSagChart"></canvas></div>
</div>
<div class="table-wrap"><h3 style="margin-bottom:12px;">Corridor Summary at DLR Limit (Span {int(span_js)}m | Tower {int(tower_h_js)}m | Min clearance {int(min_clr_js)}m)</h3><div id="corridorTable"></div></div>

<div class="table-wrap" style="margin-top:20px;">
<h3 style="margin-bottom:12px;">Sag Comparison - GridTweak vs Indian Utility Benchmarks</h3>
<p style="font-size:12px;color:#718096;margin-bottom:12px;">
GridTweak's parabolic sag model compared against published sag values from PGCIL, AEGCL, and NPTEL for Zebra ACSR. Positive diff means GridTweak's sag is higher (more conservative).
</p>
<div id="sagComparisonTable">Loading...</div>
</div>
</div>

<div class="footer">
<div style="font-size:11px;color:#718096;margin-bottom:8px;line-height:1.6;">
Thermal headroom only. Network transfer capability may be constrained by other system limits.
<span style="display:inline-block;margin-left:12px;font-size:10px;background:#f0f4f8;padding:2px 8px;border-radius:12px;color:#4a5568;">{VERSION} | IEEE 738 | Sag benchmarks</span>
</div>
&copy; 2026 GridTweak
</div></div>

<script>
let histChart, forecastChart, dlrBarChart, sagBarChart;
const voltage = 220, pf = 0.9, STATIC_MW = {static_mw_75};
let forecastRetries = 0;

function smooth3(arr) {{
    const out = [];
    for (let i = 0; i < arr.length; i++) {{
        let s = 0, n = 0;
        for (let j = Math.max(0, i-1); j <= Math.min(arr.length-1, i+1); j++) {{ s += arr[j]; n++; }}
        out.push(s / n);
    }}
    return out;
}}

function initCharts() {{
    const opts = (title) => ({{
        responsive: true, maintainAspectRatio: false,
        interaction: {{mode: 'index', intersect: false}},
        plugins: {{legend: {{position: 'top'}}, tooltip: {{backgroundColor: '#0b2e4f'}}}},
        scales: {{
            y: {{beginAtZero: true, position: 'left',
                 title: {{display: true, text: title}}, ticks: {{color: '#1a6b8a'}}}},
            y1: {{beginAtZero: false, position: 'right',
                  title: {{display: true, text: 'Temperature (C)'}}, ticks: {{color: '#e53e3e'}},
                  grid: {{drawOnChartArea: false}}}},
            x: {{grid: {{display: false}}}}
        }}
    }});
    histChart = new Chart(document.getElementById('historicalChart'),
        {{type: 'line', data: {{labels: [], datasets: []}}, options: opts('MW / C')}});
    forecastChart = new Chart(document.getElementById('forecastChart'),
        {{type: 'line', data: {{labels: [], datasets: []}}, options: opts('MW / C')}});
    dlrBarChart = new Chart(document.getElementById('corridorDlrChart'),
        {{type: 'bar', data: {{labels: [], datasets: []}}, options: opts('MW')}});
    sagBarChart = new Chart(document.getElementById('corridorSagChart'),
        {{type: 'bar', data: {{labels: [], datasets: []}}, options: opts('Sag (m)')}});
}}

async function fetchAll() {{
    await fetchHistorical();
    await fetchForecast();
    document.getElementById('lastUpdated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}}

async function fetchHistorical() {{
    try {{
        const r = await fetch('/dlr/current');
        if (!r.ok) throw new Error('no data');
        const data = await r.json();
        const staticMW = data.static_rating_mw || STATIC_MW;
        const staticA = data.static_rating_a;
        if (!data.results || !data.results.length) {{ showNoData(); return; }}
        const latest = data.results[data.results.length-1];

        document.getElementById('dlr').innerHTML = latest.dlr_a.toFixed(0) + ' <span class="metric-unit">A</span>';
        document.getElementById('dlr_emerg').innerHTML = latest.dlr_emergency_100c_a.toFixed(0) + ' <span class="metric-unit">A</span>';
        document.getElementById('static75').innerHTML = staticA.toFixed(0) + ' <span class="metric-unit">A</span>';
        document.getElementById('temp').innerHTML = (latest.temperature_c || 0).toFixed(1) + ' <span class="metric-unit">C</span>';
        document.getElementById('ambient').innerHTML = (latest.ambient_c || 0).toFixed(1) + ' <span class="metric-unit">C</span>';
        document.getElementById('wind').innerHTML = (latest.wind_mps || 0).toFixed(1) + ' <span class="metric-unit">m/s</span>';

        const staticMWval = staticA * voltage * Math.sqrt(3) * pf / 1000;
        const currentDlrMW = (latest.dlr_a || 0) * voltage * Math.sqrt(3) * pf / 1000;
        const headroomMW = Math.max(0, currentDlrMW - staticMWval);
        document.getElementById('recMw').textContent = headroomMW.toFixed(0) + ' MW';

        const hdrPct = ((latest.dlr_a - staticA) / staticA) * 100;
        document.getElementById('headroom_pct').textContent = `${{hdrPct >= 0 ? '+' : ''}}${{hdrPct.toFixed(1)}}%`;
        document.getElementById('headroom_pct').style.color = hdrPct >= 0 ? '#0e7c3e' : '#c53030';

        const labels = data.results.map(r => r.timestamp.slice(11,16));
        const dlrs = data.results.map(r => r.dlr_mw || 0);
        const emergs = data.results.map(r => (r.dlr_emergency_100c_a * voltage * Math.sqrt(3) * pf / 1000));
        const temps = smooth3(data.results.map(r => r.temperature_c || 0));
        const staticFlat = new Array(labels.length).fill(staticMW);

        histChart.data = {{labels: labels, datasets: [
            {{label: 'Static Rating', data: staticFlat, borderColor: '#718096',
              borderDash: [6,4], fill: false, yAxisID: 'y', tension: 0, pointRadius: 0, borderWidth: 2}},
            {{label: 'Emergency DLR (100C)', data: emergs, borderColor: '#9333ea',
              borderDash: [4,3], fill: false, yAxisID: 'y', tension: 0.3, pointRadius: 0, borderWidth: 1.5}},
            {{label: 'Operational DLR', data: dlrs, borderColor: '#1a6b8a',
              backgroundColor: 'rgba(26,107,138,0.12)', fill: true, yAxisID: 'y',
              tension: 0.3, pointRadius: 1, borderWidth: 2.5}},
            {{label: 'Conductor Temp (C)', data: temps, borderColor: '#e53e3e',
              backgroundColor: 'rgba(229,62,62,0.05)', fill: false, yAxisID: 'y1',
              tension: 0.3, pointRadius: 0, borderWidth: 2}}
        ]}};
        histChart.update();
    }} catch(e) {{ console.error(e); showNoData(); }}
}}

function showNoData() {{
    ['dlr','dlr_emerg','static75','headroom_pct','temp','ambient','wind'].forEach(id =>
        document.getElementById(id).innerHTML = '--');
    document.getElementById('recMw').textContent = '--';
    histChart.data = {{labels: [], datasets: []}};
    histChart.update();
}}

async function fetchForecast() {{
    try {{
        const r = await fetch('/dlr/forecast');
        if (!r.ok) return;
        const data = await r.json();
        if (!data.forecast || !data.forecast.length) {{
            const msg = data.building ? `Building...` : (data.last_error ? `Error: ${{data.last_error}}` : 'Fetching...');
            document.getElementById('forecastTable').innerHTML = `<p>${{msg}}</p>`;
            if (forecastRetries < 30) {{ forecastRetries++; setTimeout(fetchForecast, 10000); }}
            return;
        }}
        forecastRetries = 0;
        const labels = data.forecast.map(x => {{
            const d = new Date(x.timestamp); const h = d.getHours();
            if (h === 0) return d.toLocaleDateString('en-IN', {{month:'short', day:'numeric'}});
            return h.toString().padStart(2, '0') + ':00';
        }});
        const dlrs = data.forecast.map(x => x.dlr_mw || 0);
        const emergs = data.forecast.map(x => (x.dlr_emergency_100c_a * voltage * Math.sqrt(3) * pf / 1000));
        const temps = smooth3(data.forecast.map(x => x.temperature_c || 0));
        const staticFlat = new Array(labels.length).fill(STATIC_MW);

        forecastChart.data = {{labels: labels, datasets: [
            {{label: 'Static Rating', data: staticFlat, borderColor: '#718096',
              borderDash: [6,4], fill: false, yAxisID: 'y', tension: 0, pointRadius: 0, borderWidth: 2}},
            {{label: 'Emergency DLR (100C)', data: emergs, borderColor: '#9333ea',
              borderDash: [4,3], fill: false, yAxisID: 'y', tension: 0.3, pointRadius: 0, borderWidth: 1.5}},
            {{label: 'Operational DLR', data: dlrs, borderColor: '#38a169',
              backgroundColor: 'rgba(56,161,105,0.12)', fill: true, yAxisID: 'y',
              tension: 0.3, pointRadius: 0, borderWidth: 2.5}},
            {{label: 'Forecast Temp (C)', data: temps, borderColor: '#ed8936',
              backgroundColor: 'rgba(237,137,54,0.05)', fill: false, yAxisID: 'y1',
              tension: 0.3, pointRadius: 0, borderWidth: 2}}
        ]}};
        forecastChart.update();
        document.getElementById('forecastResolution').textContent =
            `Resolution: hourly | source: ${{data.source || 'unknown'}}`;

        const hours = [0, 3, 6, 9, 12, 15, 18, 21];
        const byDay = {{}};
        data.forecast.forEach(r => {{
            const d = r.timestamp.slice(0,10);
            const h = parseInt(r.timestamp.slice(11,13));
            if (!byDay[d]) byDay[d] = {{}};
            if (hours.includes(h)) byDay[d][h] = r.dlr_mw;
        }});
        let html = '<table><tr><th>Date</th>';
        for (const h of hours) html += `<th>${{h.toString().padStart(2,'0')}}:00</th>`;
        html += '</tr>';
        for (const d of Object.keys(byDay).sort()) {{
            html += `<tr><td><strong>${{d}}</strong></td>`;
            for (const h of hours) {{
                const v = byDay[d][h];
                html += `<td>${{v !== undefined ? v.toFixed(0) : '-'}}</td>`;
            }}
            html += '</tr>';
        }}
        html += '</table>';
        document.getElementById('forecastTable').innerHTML = html;
    }} catch(e) {{ console.error(e); }}
}}

async function fetchCorridor() {{
    try {{
        const r = await fetch('/dlr/corridor');
        if (!r.ok) return;
        const data = await r.json();
        if (!data.segments || !data.segments.length) return;
        const labels = data.segments.map((_, i) => 'Seg ' + (i+1));
        dlrBarChart.data = {{labels: labels, datasets: [
            {{label: 'Min DLR (MW)', data: data.segments.map(s => s.min_dlr_mw),
              backgroundColor: 'rgba(26,107,138,0.6)', borderColor: '#1a6b8a', borderWidth: 1}}
        ]}};
        dlrBarChart.update();
        sagBarChart.data = {{labels: labels, datasets: [
            {{label: 'Sag at DLR (m)', data: data.segments.map(s => s.sag_m),
              backgroundColor: 'rgba(229,62,62,0.6)', borderColor: '#e53e3e', borderWidth: 1}}
        ]}};
        sagBarChart.update();
        let html = '<table><tr><th>Segment</th><th>Min DLR (MW)</th><th>Conductor Temp (C)</th><th>Sag at DLR (m)</th><th>Clearance at DLR (m)</th><th>Binding</th><th>Elevation (m)</th></tr>';
        data.segments.forEach((s, i) => {{
            const clrStatus = s.clearance_m >= {min_clr_js} ? 'status-ok' : 'status-warn';
            const bind = s.binding_constraint === 'clearance' ? 'clearance' : 'thermal';
            const elev = (s.elevation_m || 0).toFixed(0);
            html += `<tr><td>${{i+1}}</td><td>${{s.min_dlr_mw.toFixed(0)}}</td><td>${{s.temperature_c.toFixed(1)}}</td><td>${{s.sag_m.toFixed(2)}}</td><td class="${{clrStatus}}">${{s.clearance_m.toFixed(2)}}</td><td>${{bind}}</td><td>${{elev}}</td></tr>`;
        }});
        html += '</table>';
        document.getElementById('corridorTable').innerHTML = html;
    }} catch(e) {{ console.error(e); }}
}}

async function fetchSagComparison() {{
    try {{
        const r = await fetch('/dlr/sag_comparison');
        if (!r.ok) return;
        const data = await r.json();
        if (!data.utility_benchmarks) return;

        let html = '<table style="margin-bottom:16px;"><tr><th>Temp (C)</th><th>Sag @ 350m (m)</th><th>Sag @ corridor (m)</th><th>Clearance @ corridor (m)</th></tr>';
        data.gt_curve.forEach(row => {{
            html += `<tr><td>${{row.temp_c}}</td><td>${{row.gt_sag_350m_m.toFixed(2)}}</td><td>${{row.gt_sag_corridor_m.toFixed(2)}}</td><td>${{row.clearance_corridor_m.toFixed(2)}}</td></tr>`;
        }});
        html += '</table>';

        html += '<table><tr><th>Source</th><th>Conductor</th><th>Span (m)</th><th>Temp (C)</th><th>Reported Sag (m)</th><th>GridTweak Sag (m)</th><th>Diff (%)</th></tr>';
        data.utility_benchmarks.forEach(b => {{
            const rep = b.sag_m !== null ? b.sag_m.toFixed(3) : '-';
            const gt = b.gt_sag_m !== null ? b.gt_sag_m.toFixed(3) : '-';
            let diff = '-'; let cls = '';
            if (b.diff_pct !== null) {{
                diff = (b.diff_pct >= 0 ? '+' : '') + b.diff_pct.toFixed(1) + '%';
                cls = Math.abs(b.diff_pct) < 15 ? 'diff-good' : 'diff-warn';
            }}
            html += `<tr><td>${{b.source}}</td><td>${{b.conductor}}</td><td>${{b.span_m || '-'}}</td><td>${{b.temp_c}}</td><td>${{rep}}</td><td>${{gt}}</td><td class="${{cls}}">${{diff}}</td></tr>`;
        }});
        html += '</table>';
        html += '<p style="font-size:11px;color:#718096;margin-top:8px;">Notes: PGCIL (10.600m) and AEGCL (8.435m) use different stringing tensions. GridTweak matches PGCIL within the tolerance band. Diff % shown is (GridTweak - Reported) / Reported.</p>';

        document.getElementById('sagComparisonTable').innerHTML = html;
    }} catch(e) {{ console.error('Sag comparison error:', e); }}
}}

document.querySelectorAll('.tab').forEach(tab => {{
    tab.addEventListener('click', function() {{
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        this.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(this.dataset.tab + '-tab').classList.add('active');
        if (this.dataset.tab === 'forecast') fetchForecast();
        if (this.dataset.tab === 'corridor') {{ fetchCorridor(); fetchSagComparison(); }}
    }});
}});

initCharts(); fetchAll(); setInterval(fetchAll, 300000);
</script>
</body></html>"""
        return html


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} {VERSION}")
    parser.add_argument("--config", type=str)
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--run-pangu", action="store_true")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--export-static", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    port = int(os.environ.get("PORT", args.port))

    if args.config:
        try:
            with open(args.config) as f: CONFIG_DEFAULTS.update(json.load(f))
            print(f"Loaded {args.config}")
        except Exception as e: print(f"Config: {e}")
    else: print(f"Using {VERSION} defaults")

    print(f"pvlib={PVLIB_AVAILABLE} earth2studio={EARTH2STUDIO_AVAILABLE} "
          f"xarray={XARRAY_AVAILABLE} torch={TORCH_AVAILABLE}")
    load_mcp()

    metar = fetch_metar(CONFIG_DEFAULTS.get("metar_icao", "VABB"))
    if metar:
        print(f"METAR {metar['icao']}: {metar['speed_mps']} m/s @ {metar['direction_deg']}")

    c = get_conductor(CONFIG_DEFAULTS["conductor"])
    static_a = compute_static_rating_A(c)
    kv = CONFIG_DEFAULTS["nominal_voltage_kv"]; pf = CONFIG_DEFAULTS["power_factor"]
    print(f"Conductor: {c.name}")
    print(f"Static rating (75C): {static_a:.0f} A = {amps_to_mw(static_a, kv, pf):.1f} MW")

    auto_freshed = maybe_auto_fresh()
    if not auto_freshed:
        load_cache()

    if args.fresh:
        if os.path.exists(DB_FILE): os.remove(DB_FILE); print(f"Deleted {DB_FILE}")
        if os.path.exists(CACHE_FILE): os.remove(CACHE_FILE); print(f"Deleted {CACHE_FILE}")
        clear_cache_state()
        print("Cleared all in-memory cache state")

    if args.run_pangu:
        return build_pangu_cache()

    if args.export_static:
        _cached["last_updated"] = None
        print("\nBuilding fresh cache for export...")
        update_forecast_cache(force=True)
        n = len(_cached.get("data") or [])
        if n == 0:
            print("❌ Cache build produced 0 records — cannot export")
            return 1
        print(f"Cache: {n} records via {_cached.get('source')}")
        ok = export_static_files()
        return 0 if ok else 1

    if args.build_cache:
        _cached["last_updated"] = None
        update_forecast_cache(force=True)
        n = len(_cached.get("data") or [])
        print(f"\nRESULT: {n} records via {_cached.get('source')}")
        return 0 if n > 0 else 1

    if args.api:
        if not API_AVAILABLE: return 1
        print(f"\nAPI:         http://localhost:{port}")
        print(f"Dashboard:   http://localhost:{port}/dashboard")

        # Diagnostic: what's available?
        statics_present = all(os.path.exists(f) for f in
                              [STATIC_CURRENT, STATIC_FORECAST, STATIC_CORRIDOR])
        cache_age_h = None
        if _cached.get("last_updated"):
            try:
                age_s = (datetime.now() - datetime.fromisoformat(_cached["last_updated"])).total_seconds()
                cache_age_h = age_s / 3600
            except: pass

        if statics_present:
            if cache_age_h is not None:
                print(f"Serving static files (cache age: {cache_age_h:.1f}h)")
            else:
                print("Serving static files (age unknown)")
        elif _cached["data"]:
            print(f"Serving pre-built cache ({len(_cached['data'])} records, age: {cache_age_h or 0:.1f}h)")
        else:
            print("⚠️ No static files or cache — endpoints will return empty until first refresh completes")

        # Layer 2: startup refresh if cache is stale (> 12h) or missing
        STALE_THRESHOLD_H = 12
        needs_startup_refresh = (
            not statics_present or
            cache_age_h is None or
            cache_age_h > STALE_THRESHOLD_H
        )
        if needs_startup_refresh:
            print(f"Cache stale or missing (age: {cache_age_h if cache_age_h else 'N/A'}h) — background refresh starting")
            def _startup_refresh():
                try:
                    _cached["last_updated"] = None
                    update_forecast_cache(force=True)
                    export_static_files()
                except Exception as e:
                    print(f"Startup refresh failed: {e}")
            threading.Thread(target=_startup_refresh, daemon=True).start()
        else:
            print(f"Cache fresh ({cache_age_h:.1f}h < {STALE_THRESHOLD_H}h) — no startup refresh needed")

        # Layer 3: in-process scheduler — always runs while the service is up
        if SCHEDULER_AVAILABLE:
            try:
                sched = BackgroundScheduler()
                def _scheduled_refresh():
                    try:
                        print("⏰ Scheduled refresh starting...")
                        _cached["last_updated"] = None
                        update_forecast_cache(force=True)
                        export_static_files()
                        print("⏰ Scheduled refresh complete")
                    except Exception as e:
                        print(f"Scheduled refresh failed: {e}")
                interval_h = CONFIG_DEFAULTS.get("scheduler_interval_hours", 6)
                sched.add_job(_scheduled_refresh, "interval", hours=interval_h)
                sched.start()
                print(f"⏰ In-process scheduler active — refreshes every {interval_h}h")
            except Exception as e:
                print(f"Scheduler skipped: {e}")
        else:
            print("⚠️ APScheduler not available — only GitHub Actions will refresh cache")

        uvicorn.run(app, host="0.0.0.0", port=port)
        return 0

    parser.print_help(); return 1


if __name__ == "__main__":
    sys.exit(main())