#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GridTweak DLR Engine - V28 (Production Dashboard)
------------------------------------------------
Full DLR engine with corridor, sag, and auto‑computed static rating.
Root URL redirects to /dashboard.
Historical data is limited to 24 hours.
Zebra/Panther conductors added.
Wind Speed Clamp + Weighted Smoothing + Correction Factor applied.
DLR Scaling Factor for safety margin.
Location name displayed on dashboard.
Favicon loaded from favicon_base64.txt (if present).
Forecast: file‑based caching with archive fallback – no on‑request fetching.
"""

import argparse
import csv
import json
import math
import sys
import os
import sqlite3
import smtplib
import pickle
import warnings
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.parse
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI, Query, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ML (LightGBM)
ML_AVAILABLE = False
try:
    import lightgbm as lgb
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# GLOBAL FORECAST CACHE
# ============================================================================

_cached_forecast = {
    "data": None,
    "last_updated": None
}

FORECAST_CACHE_FILE = "forecast_cache.json"

def load_forecast_cache_from_file():
    """Load cached forecast from disk on startup."""
    global _cached_forecast
    if os.path.exists(FORECAST_CACHE_FILE):
        try:
            with open(FORECAST_CACHE_FILE, 'r') as f:
                data = json.load(f)
                _cached_forecast["data"] = data.get("data")
                _cached_forecast["last_updated"] = data.get("last_updated")
                record_count = len(_cached_forecast['data']) if _cached_forecast['data'] else 0
                print(f"✅ Loaded forecast cache from disk ({record_count} records, updated {_cached_forecast['last_updated']})")
                return True
        except Exception as e:
            print(f"⚠️ Could not load forecast cache from disk: {e}")
    return False

def save_forecast_cache_to_file():
    """Save cached forecast to disk."""
    global _cached_forecast
    try:
        with open(FORECAST_CACHE_FILE, 'w') as f:
            json.dump({
                "data": _cached_forecast["data"],
                "last_updated": _cached_forecast["last_updated"]
            }, f)
        record_count = len(_cached_forecast['data']) if _cached_forecast['data'] else 0
        print(f"💾 Forecast cache saved to disk ({record_count} records)")
    except Exception as e:
        print(f"⚠️ Could not save forecast cache to disk: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

VERSION = "V28"
APP_NAME = "GridTweak"

CONFIG_DEFAULTS = {
    "lat": 19.076,
    "lon": 72.877,
    "location_name": "MSETCL – 220kV Trombay-Vikhroli",
    "conductor": "panther",
    "convection_model": "ieee738",
    "test_current": 1000,
    "thresholds": [1200, 1500, 2000, 2500, 3000],
    "alert_threshold_dlr": 2500,
    "alert_threshold_temp": 80,
    "email_enabled": False,
    "email_smtp_server": "smtp.gmail.com",
    "email_smtp_port": 587,
    "email_from": "",
    "email_to": "",
    "email_password": "",
    "slack_enabled": False,
    "slack_webhook_url": "",
    "scheduler_interval_hours": 1,
    "database_path": "dlr_data.db",
    "ml_model_path": "lgb_best.pkl",
    "use_ml": True,
    "nominal_voltage_kv": 220,
    "power_factor": 0.9,
    "static_rating_mw": 600,
    "num_segments": 3,
    "end_lat": None,
    "end_lon": None,
    "corridor_mode": False,
    "span_length_m": 400,
    "sag_ref_m": 5.0,
    "thermal_expansion_coeff": 23e-6,
    "ref_temp_sag": 20.0,
    # --- WIND CORRECTION SETTINGS ---
    "wind_max_mps": 7.0,
    "wind_smoothing_window": 3,
    "wind_correction_factor": 0.4,
    # --- DLR CALIBRATION SETTINGS ---
    "dlr_scaling_factor": 0.80
}

# ============================================================================
# CONDUCTOR LIBRARY
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
    thermal_expansion_coeff: float = 23e-6

CONDUCTOR_LIBRARY = {
    "drake": Conductor("795 kcmil 26/7 Drake ACSR", 0.02814, 1.086e-5, 0.00403, 0.50, 0.50, 100.0),
    "cardinal": Conductor("954 kcmil 54/7 Cardinal ACSR", 0.03038, 8.90e-6, 0.00403, 0.50, 0.50, 100.0),
    "linnet": Conductor("336.4 kcmil 26/7 Linnet ACSR", 0.01849, 2.58e-5, 0.00403, 0.50, 0.50, 100.0),
    "rail": Conductor("1033.5 kcmil 54/7 Rail ACSR", 0.03208, 8.22e-6, 0.00403, 0.50, 0.50, 100.0),
    "zebra": Conductor("Zebra ACSR (India)", 0.03175, 8.50e-6, 0.00403, 0.50, 0.50, 100.0),
    "panther": Conductor("Panther ACSR", 0.02538, 1.29e-5, 0.00403, 0.50, 0.50, 100.0),
    "moose": Conductor("Moose ACSR", 0.03315, 7.20e-6, 0.00403, 0.50, 0.50, 100.0),
}

def get_conductor(name, overrides=None):
    c = CONDUCTOR_LIBRARY.get(name.lower())
    if not c:
        raise ValueError(f"Conductor '{name}' not found. Available: {list(CONDUCTOR_LIBRARY.keys())}")
    if overrides:
        for k, v in overrides.items():
            if hasattr(c, k):
                setattr(c, k, v)
    return c

# ============================================================================
# PHYSICS ENGINE
# ============================================================================

LINE_AZIMUTH_DEG = 90.0
ROUGHNESS_M = 0.0005
TURB_INTENSITY = 0.1

def wind_geometry(wind_mps, wind_direction_deg, line_azimuth_deg=90.0):
    delta = (wind_direction_deg - line_azimuth_deg) % 360.0
    delta_rad = math.radians(delta)
    v_perp = abs(wind_mps * math.sin(delta_rad))
    v_parallel = abs(wind_mps * math.cos(delta_rad))
    if wind_mps > 1e-12:
        attack_deg = math.degrees(math.asin(min(1.0, v_perp / wind_mps)))
    else:
        attack_deg = 0.0
    return {"perpendicular_ms": v_perp, "parallel_ms": v_parallel, "delta_deg": delta, "attack_angle_deg": attack_deg}

def resistance_ohm_per_m(temp_c, conductor):
    return conductor.r20_ohm_per_m * (1.0 + conductor.alpha_r * (temp_c - 20.0))

def solar_heat_w_m(ghi_w_m2, solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg=90.0):
    if ghi_w_m2 <= 0.0 or solar_altitude_deg <= 0.0:
        return 0.0
    alt_rad = math.radians(solar_altitude_deg)
    az_rad = math.radians(solar_azimuth_deg - line_azimuth_deg)
    projection = math.cos(alt_rad) * abs(math.sin(az_rad))
    return max(0.0, conductor.absorptivity * ghi_w_m2 * conductor.diameter_m * projection)

def convection_w_m(conductor_temp_c, ambient_c, wind_perp_mps, diameter_m, attack_angle_deg=90.0,
                   model="ieee738", roughness_m=0.0005, turb_intensity=0.1):
    delta_t = conductor_temp_c - ambient_c
    if delta_t <= 0.0:
        return 0.0
    d = diameter_m
    film_c = 0.5 * (conductor_temp_c + ambient_c)
    film_k = 273.15 + film_c
    rho = 1.225 * (288.15 / film_k)
    mu = 1.81e-5 * (film_k / 288.15) ** 0.7
    k_air = 0.0257 * (film_k / 288.15) ** 0.85
    cp = 1006.0
    v = max(wind_perp_mps, 0.0)
    re = rho * v * d / max(mu, 1e-12)
    nu = mu / max(rho, 1e-12)
    alpha = k_air / max(rho * cp, 1e-12)
    pr = nu / max(alpha, 1e-12)
    beta = 1.0 / film_k
    gr = 9.81 * beta * abs(delta_t) * d**3 / max(nu**2, 1e-20)
    ra = max(gr * pr, 0.0)
    nu_nat = (0.60 + 0.387 * ra**(1/6) / (1 + (0.559 / max(pr, 1e-12))**(9/16))**(8/27))**2
    q_nat = nu_nat * k_air * math.pi * delta_t
    if v <= 1e-12:
        return max(q_nat, 0.0)
    if re < 4.0e3:
        nu_forced = 0.193 * re**0.618 * pr**(1/3)
    else:
        nu_forced = 0.0266 * re**0.805 * pr**(1/3)
    q_forced = nu_forced * k_air * math.pi * delta_t
    if model in ("enhanced", "cfd-surrogate"):
        if re > 0:
            roughness_factor = 1 + 0.5 * (roughness_m / d) * re**0.2
        else:
            roughness_factor = 1.0
        q_forced *= roughness_factor
        alpha_rad = math.radians(attack_angle_deg)
        angle_factor = 0.42 + 0.58 * math.sin(alpha_rad)**1.5
        q_forced *= angle_factor
        q_forced *= (1 + 0.1 * turb_intensity)
    return math.sqrt(max(q_nat, 0.0)**2 + max(q_forced, 0.0)**2)

def radiation_w_m(conductor_temp_c, ambient_c, conductor):
    sigma = 5.670374419e-8
    tc = conductor_temp_c + 273.15
    ta = ambient_c + 273.15
    return conductor.emissivity * sigma * math.pi * conductor.diameter_m * (tc**4 - ta**4)

def thermal_balance(current_a, conductor_temp_c, ambient_c, wind_perp_mps, ghi_w_m2,
                    solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg=90.0,
                    attack_angle_deg=90.0, convection_model="ieee738", roughness_m=0.0005, turb_intensity=0.1):
    r = resistance_ohm_per_m(conductor_temp_c, conductor)
    qj = current_a**2 * r
    qconv = convection_w_m(conductor_temp_c, ambient_c, wind_perp_mps, conductor.diameter_m,
                           attack_angle_deg, convection_model, roughness_m, turb_intensity)
    qrad = radiation_w_m(conductor_temp_c, ambient_c, conductor)
    qsol = solar_heat_w_m(ghi_w_m2, solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg)
    residual = qj + qsol - qconv - qrad
    return {"convection_w_m": qconv, "radiation_w_m": qrad, "solar_w_m": qsol, "joule_w_m": qj, "residual_w_m": residual}

def solve_temperature(current_a, ambient_c, wind_perp_mps, ghi_w_m2, solar_altitude_deg, solar_azimuth_deg,
                      conductor, line_azimuth_deg=90.0, attack_angle_deg=90.0,
                      convection_model="ieee738", roughness_m=0.0005, turb_intensity=0.1):
    lo = max(-50.0, ambient_c)
    hi = 300.0
    def residual_at(tc):
        return thermal_balance(current_a, tc, ambient_c, wind_perp_mps, ghi_w_m2,
                               solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg,
                               attack_angle_deg, convection_model, roughness_m, turb_intensity)["residual_w_m"]
    flo = residual_at(lo)
    fhi = residual_at(hi)
    expansion = 0
    while flo * fhi > 0 and hi < 1500 and expansion < 20:
        hi += 100.0
        fhi = residual_at(hi)
        expansion += 1
    if flo * fhi > 0:
        raise RuntimeError(f"Cannot bracket root: f({lo})={flo}, f({hi})={fhi}")
    converged = False
    temp = 0.5 * (lo + hi)
    iter_count = 0
    for iter_count in range(1, 251):
        temp = 0.5 * (lo + hi)
        ft = residual_at(temp)
        if abs(ft) <= 1e-3 or abs(hi - lo) <= 1e-7:
            converged = True
            break
        if flo * ft <= 0:
            hi = temp; fhi = ft
        else:
            lo = temp; flo = ft
    terms = thermal_balance(current_a, temp, ambient_c, wind_perp_mps, ghi_w_m2,
                            solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg,
                            attack_angle_deg, convection_model, roughness_m, turb_intensity)
    scale = max(abs(terms["joule_w_m"]) + abs(terms["solar_w_m"]),
                abs(terms["convection_w_m"]) + abs(terms["radiation_w_m"]), 1.0)
    rel_res = abs(terms["residual_w_m"]) / scale
    if abs(terms["residual_w_m"]) <= 1e-3:
        converged = True
    return {
        "temperature_c": temp,
        "converged": converged,
        "iterations": iter_count,
        "residual_w_m": terms["residual_w_m"],
        "relative_residual": rel_res,
        "convection_w_m": terms["convection_w_m"],
        "radiation_w_m": terms["radiation_w_m"],
        "solar_w_m": terms["solar_w_m"],
        "joule_w_m": terms["joule_w_m"],
    }

def solve_dlr(ambient_c, wind_perp_mps, ghi_w_m2, solar_altitude_deg, solar_azimuth_deg,
              conductor, line_azimuth_deg=90.0, attack_angle_deg=90.0,
              convection_model="ieee738", roughness_m=0.0005, turb_intensity=0.1):
    terms = thermal_balance(0.0, conductor.tmax_c, ambient_c, wind_perp_mps, ghi_w_m2,
                            solar_altitude_deg, solar_azimuth_deg, conductor, line_azimuth_deg,
                            attack_angle_deg, convection_model, roughness_m, turb_intensity)
    available = terms["convection_w_m"] + terms["radiation_w_m"] - terms["solar_w_m"]
    if available <= 0.0:
        return 0.0
    r = resistance_ohm_per_m(conductor.tmax_c, conductor)
    return math.sqrt(available / r)

def amps_to_mw(amps, voltage_kv, pf=0.9):
    return amps * voltage_kv * math.sqrt(3) * pf / 1000

SOLAR_ALTITUDE = [
    -57.72, -59.05, -53.79, -44.16, -32.31, -19.35,
    -5.81, 8.06, 22.12, 36.28, 50.44, 64.42,
    77.42, 81.06, 69.51, 55.71, 41.58, 27.40,
    13.28, -0.69, -14.40, -27.65, -40.05, -50.77
]
SOLAR_AZIMUTH = [
    19.25, 8.89, 33.91, 50.96, 62.10, 69.85,
    75.73, 80.61, 85.08, 89.66, 95.17, 103.80,
    126.26, 211.01, 250.69, 262.07, 268.33, 273.12,
    277.54, 282.19, 287.59, 294.46, 304.05, 318.51
]

def solar_position(hour):
    if not 0 <= hour < 24:
        raise ValueError(f"Invalid hour: {hour}")
    return {"altitude_deg": SOLAR_ALTITUDE[hour], "azimuth_deg": SOLAR_AZIMUTH[hour]}

# ============================================================================
# SAG MODELLING
# ============================================================================

def calculate_sag(conductor_temp_c, conductor, span_length_m, ref_temp=20.0, sag_ref=5.0):
    alpha = conductor.thermal_expansion_coeff
    delta_t = conductor_temp_c - ref_temp
    sag = sag_ref * (1 + alpha * delta_t)
    return sag

# ============================================================================
# STATIC RATING COMPUTATION
# ============================================================================

def compute_static_rating(conductor, line_azimuth_deg=90.0, convection_model="ieee738"):
    ambient = 40.0
    wind = 0.6
    ghi = 1000.0
    solar_alt = 90.0
    solar_az = 180.0
    attack_angle = 90.0
    dlr = solve_dlr(ambient, wind, ghi, solar_alt, solar_az,
                    conductor, line_azimuth_deg, attack_angle,
                    convection_model, ROUGHNESS_M, TURB_INTENSITY)
    return dlr

# ============================================================================
# WEATHER FETCH & CORRECTION
# ============================================================================

@dataclass
class WeatherRecord:
    timestamp: str
    ambient_c: float
    wind_mps: float
    wind_direction_deg: float
    ghi_w_m2: float

def apply_wind_correction(records: List[WeatherRecord], config: Dict) -> List[WeatherRecord]:
    if not records:
        return records

    window = config.get("wind_smoothing_window", 3)
    max_mps = config.get("wind_max_mps", 8.0)
    factor = config.get("wind_correction_factor", 1.0)

    if window < 1:
        window = 1

    if factor != 1.0:
        for r in records:
            r.wind_mps = r.wind_mps * factor

    n = len(records)
    smoothed_winds = [0.0] * n

    if window % 2 == 0:
        window += 1
    half = window // 2
    weights = []
    for i in range(-half, half + 1):
        weight = half + 1 - abs(i)
        weights.append(weight)
    weight_sum = sum(weights)

    for i in range(n):
        total = 0.0
        w_sum = 0.0
        for j, w in enumerate(weights):
            idx = i + (j - half)
            if 0 <= idx < n:
                total += records[idx].wind_mps * w
                w_sum += w
        smoothed = total / w_sum if w_sum > 0 else records[i].wind_mps
        if smoothed > max_mps:
            smoothed = max_mps
        smoothed_winds[i] = smoothed

    for i, r in enumerate(records):
        r.wind_mps = smoothed_winds[i]

    print(f"🔧 Wind Correction Applied: Factor={factor}, Max={max_mps}m/s, Window={window}hrs")
    return records

def fetch_weather_from_openmeteo(lat, lon, start_date, end_date, timezone="auto", forecast=False):
    if forecast:
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon,
                  "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation",
                  "timezone": timezone, "forecast_days": 7}
    else:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon,
                  "start_date": start_date, "end_date": end_date,
                  "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation",
                  "timezone": timezone}
    
    url = base_url + "?" + urllib.parse.urlencode(params)
    print(f"Fetching weather from: {url}")
    
    headers = {'User-Agent': 'GridTweak/1.0 (DLR Platform; contact: hello@gridtweak.com)'}
    
    max_retries = 5
    base_delay = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            break
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"Attempt {attempt+1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to fetch weather after {max_retries} attempts: {e}")
    
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("No times returned from API")
    
    records = []
    for i, ts in enumerate(times):
        try:
            if len(ts) == 16:
                ts += ":00"
            amb = hourly["temperature_2m"][i]
            wind = hourly["wind_speed_10m"][i]
            wdir = hourly["wind_direction_10m"][i]
            ghi = hourly["shortwave_radiation"][i]
            
            def sanitise(val, default=0.0):
                if val is None or not isinstance(val, (int, float)):
                    return default
                return float(val)
            
            amb = sanitise(amb, 15.0)
            wind = sanitise(wind, 0.0)
            wdir = sanitise(wdir, 0.0)
            ghi = sanitise(ghi, 0.0)
            
            if wind == 0.0:
                wdir = 0.0
            if wdir < 0 or wdir >= 360:
                wdir = 0.0
            
            records.append(WeatherRecord(
                timestamp=ts, 
                ambient_c=amb, 
                wind_mps=wind,
                wind_direction_deg=wdir, 
                ghi_w_m2=ghi
            ))
        except (KeyError, IndexError, ValueError) as e:
            print(f"Warning: skipping record at index {i} due to {e}")
    
    if forecast and start_date and end_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            filtered = []
            for r in records:
                try:
                    dt = datetime.fromisoformat(r.timestamp)
                    if start_dt <= dt <= end_dt:
                        filtered.append(r)
                except:
                    continue
            records = filtered
        except Exception as e:
            print(f"Warning: date filtering failed: {e}")
    
    records = apply_wind_correction(records, CONFIG_DEFAULTS)
    return records

def fetch_weather_multi_year(lat, lon, start_date, end_date):
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    all_records = []
    years = range(start_dt.year, end_dt.year + 1)
    for year in years:
        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        if year == start_dt.year:
            year_start = start_date
        if year == end_dt.year:
            year_end = end_date
        try:
            recs = fetch_weather_from_openmeteo(lat, lon, year_start, year_end, forecast=False)
            all_records.extend(recs)
            print(f"  Fetched {len(recs)} records for {year}")
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
            raise
    return all_records

# ============================================================================
# FORECAST CACHE UPDATE
# ============================================================================

def update_forecast_cache():
    """Fetch forecast from Open-Meteo. If that fails, use archive data (last 7 days)."""
    global _cached_forecast

    # Check if cache is fresh (< 6 hours old)
    if _cached_forecast["last_updated"]:
        try:
            last_updated = datetime.fromisoformat(_cached_forecast["last_updated"])
            age = (datetime.now() - last_updated).total_seconds() / 3600
            if age < 6:
                print(f"⏭️ Forecast cache is fresh ({age:.1f} hours old). Skipping fetch.")
                return
        except:
            pass

    try:
        print("🌤️ Updating forecast cache...")
        lat = CONFIG_DEFAULTS["lat"]
        lon = CONFIG_DEFAULTS["lon"]
        conductor = get_conductor(CONFIG_DEFAULTS["conductor"])
        days = 7
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        weather = None
        data_source = ""

        # ---- Try forecast first ----
        try:
            weather = fetch_weather_from_openmeteo(lat, lon, start_date, end_date, forecast=True)
            if weather:
                data_source = "Open-Meteo forecast"
                print(f"✅ Forecast fetched ({len(weather)} records)")
        except Exception as e:
            print(f"⚠️ Forecast failed ({e}), trying archive...")

        # ---- Fallback: use archive data (last 7 days) ----
        if not weather:
            try:
                archive_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                weather = fetch_weather_from_openmeteo(lat, lon, archive_start, end_date, forecast=False)
                if weather:
                    data_source = "Archive (fallback)"
                    print(f"✅ Archive fallback fetched ({len(weather)} records)")
            except Exception as e:
                print(f"❌ Archive fallback also failed: {e}")

        if not weather:
            print("❌ All weather sources failed. Cache not updated.")
            return

        # ---- Process weather into DLR results ----
        results = []
        for w in weather:
            try:
                dt = datetime.fromisoformat(w.timestamp)
                hour = dt.hour
            except:
                hour = 12
            sp = solar_position(hour)
            geo = wind_geometry(w.wind_mps, w.wind_direction_deg, LINE_AZIMUTH_DEG)
            wind_perp = geo["perpendicular_ms"]
            attack_deg = geo["attack_angle_deg"]

            dlr = solve_dlr(w.ambient_c, wind_perp, w.ghi_w_m2, sp["altitude_deg"], sp["azimuth_deg"],
                            conductor, LINE_AZIMUTH_DEG, attack_deg,
                            CONFIG_DEFAULTS["convection_model"], ROUGHNESS_M, TURB_INTENSITY)
            scaling = CONFIG_DEFAULTS.get("dlr_scaling_factor", 1.0)
            dlr = dlr * scaling

            temp_res = solve_temperature(CONFIG_DEFAULTS["test_current"], w.ambient_c, wind_perp, w.ghi_w_m2,
                                         sp["altitude_deg"], sp["azimuth_deg"],
                                         conductor, LINE_AZIMUTH_DEG, attack_deg,
                                         CONFIG_DEFAULTS["convection_model"], ROUGHNESS_M, TURB_INTENSITY)
            voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
            pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
            mw = amps_to_mw(dlr, voltage, pf)
            sag_m = calculate_sag(
                temp_res["temperature_c"],
                conductor,
                CONFIG_DEFAULTS.get("span_length_m", 400),
                ref_temp=CONFIG_DEFAULTS.get("ref_temp_sag", 20.0),
                sag_ref=CONFIG_DEFAULTS.get("sag_ref_m", 5.0)
            )
            results.append({
                "timestamp": w.timestamp,
                "dlr_a": dlr,
                "dlr_mw": mw,
                "temperature_c": temp_res["temperature_c"],
                "ambient_c": w.ambient_c,
                "wind_mps": w.wind_mps,
                "ghi_w_m2": w.ghi_w_m2,
                "sag_m": sag_m,
                "status": "OK" if temp_res["temperature_c"] <= conductor.tmax_c else "OVER"
            })

        # ---- Save to cache and disk ----
        _cached_forecast["data"] = results
        _cached_forecast["last_updated"] = datetime.now().isoformat()
        save_forecast_cache_to_file()
        print(f"✅ Forecast cache updated ({data_source}): {len(results)} records at {_cached_forecast['last_updated']}")

    except Exception as e:
        print(f"❌ Forecast cache update failed: {e}. Keeping old cache.")

# ============================================================================
# ML MODEL LOADING
# ============================================================================

def load_ml_model(path):
    if not ML_AVAILABLE:
        return None, None
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        scaler_path = path.replace(".pkl", "_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        return model, scaler
    except Exception as e:
        print(f"Error loading ML model: {e}")
        return None, None

# ============================================================================
# CORRIDOR MODELLING
# ============================================================================

def generate_segment_coords(start_lat, start_lon, end_lat, end_lon, num_segments):
    coords = []
    for i in range(num_segments + 1):
        frac = i / num_segments
        lat = start_lat + (end_lat - start_lat) * frac
        lon = start_lon + (end_lon - start_lon) * frac
        coords.append((lat, lon))
    return coords

def fetch_elevation(lat, lon):
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("elevation", [0])[0]
    except:
        return 0

def apply_terrain_correction(ambient_c, wind_mps, elevation_m, base_elevation=0):
    delta_e = (elevation_m - base_elevation) / 1000.0
    temp_corrected = ambient_c - 6.5 * delta_e
    wind_corrected = wind_mps * (1 + 0.05 * delta_e)
    return temp_corrected, wind_corrected

def run_dlr_for_period_segment(weather_records, conductor, convection_model,
                               test_current, ml_model, use_ml):
    results = []
    if ml_model is not None:
        if isinstance(ml_model, tuple):
            model, scaler = ml_model
        else:
            model = ml_model
            scaler = None
    else:
        model = None
        scaler = None

    for i, w in enumerate(weather_records):
        try:
            dt = datetime.fromisoformat(w.timestamp)
            hour = dt.hour
        except:
            hour = 12
        sp = solar_position(hour)
        geo = wind_geometry(w.wind_mps, w.wind_direction_deg, LINE_AZIMUTH_DEG)
        wind_perp = geo["perpendicular_ms"]
        attack_deg = geo["attack_angle_deg"]

        use_ml_for_this = False
        if use_ml and model is not None and i >= 24:
            use_ml_for_this = True

        if use_ml_for_this:
            prev_results = results[-24:] if len(results) >= 24 else results
            def single_features(w, prev_results):
                dt = datetime.fromisoformat(w.timestamp)
                feats = [
                    w.ambient_c, w.wind_mps, w.wind_direction_deg, w.ghi_w_m2,
                    dt.hour, dt.timetuple().tm_yday,
                    math.sin(2*math.pi*dt.hour/24), math.cos(2*math.pi*dt.hour/24)
                ]
                for lag in [1, 2, 3, 6, 12, 24]:
                    if len(prev_results) >= lag:
                        feats.append(prev_results[-lag]['dlr_a'])
                    else:
                        feats.append(0.0)
                return np.array(feats).reshape(1, -1)
            X = single_features(w, prev_results)
            if scaler is not None:
                X = scaler.transform(X)
            dlr = model.predict(X)[0]
            temp_res = solve_temperature(test_current, w.ambient_c, wind_perp, w.ghi_w_m2,
                                         sp["altitude_deg"], sp["azimuth_deg"],
                                         conductor, LINE_AZIMUTH_DEG, attack_deg,
                                         convection_model, ROUGHNESS_M, TURB_INTENSITY)
        else:
            dlr = solve_dlr(w.ambient_c, wind_perp, w.ghi_w_m2, sp["altitude_deg"], sp["azimuth_deg"],
                            conductor, LINE_AZIMUTH_DEG, attack_deg, convection_model,
                            ROUGHNESS_M, TURB_INTENSITY)
            temp_res = solve_temperature(test_current, w.ambient_c, wind_perp, w.ghi_w_m2,
                                         sp["altitude_deg"], sp["azimuth_deg"],
                                         conductor, LINE_AZIMUTH_DEG, attack_deg,
                                         convection_model, ROUGHNESS_M, TURB_INTENSITY)

        # ✅ Apply DLR scaling factor
        scaling = CONFIG_DEFAULTS.get("dlr_scaling_factor", 1.0)
        dlr = dlr * scaling

        margin_c = conductor.tmax_c - temp_res["temperature_c"]
        status = "OK" if temp_res["temperature_c"] <= conductor.tmax_c else "OVER"
        sag_m = calculate_sag(
            temp_res["temperature_c"],
            conductor,
            CONFIG_DEFAULTS.get("span_length_m", 400),
            ref_temp=CONFIG_DEFAULTS.get("ref_temp_sag", 20.0),
            sag_ref=CONFIG_DEFAULTS.get("sag_ref_m", 5.0)
        )
        result = {
            "timestamp": w.timestamp,
            "ambient_c": w.ambient_c,
            "wind_mps": w.wind_mps,
            "wind_direction_deg": w.wind_direction_deg,
            "perpendicular_wind_mps": wind_perp,
            "ghi_w_m2": w.ghi_w_m2,
            "solar_altitude_deg": sp["altitude_deg"],
            "solar_azimuth_deg": sp["azimuth_deg"],
            "convection_w_m": temp_res["convection_w_m"],
            "radiation_w_m": temp_res["radiation_w_m"],
            "solar_w_m": temp_res["solar_w_m"],
            "joule_w_m": temp_res["joule_w_m"],
            "dlr_a": dlr,
            "temperature_c": temp_res["temperature_c"],
            "residual_w_m": temp_res["residual_w_m"],
            "relative_residual": temp_res["relative_residual"],
            "converged": temp_res["converged"],
            "temp_margin_c": margin_c,
            "status": status,
            "ml_used": use_ml_for_this,
            "sag_m": sag_m,
        }
        results.append(result)
    return results

def run_corridor_dlr(lat, lon, end_lat, end_lon, num_segments, conductor, convection_model,
                     test_current, ml_model, use_ml, start_date, end_date,
                     base_elevation=0, use_forecast=False):
    segment_coords = generate_segment_coords(lat, lon, end_lat, end_lon, num_segments)
    segment_results = []
    min_dlr = float('inf')
    weakest_idx = 0
    for idx, (slat, slon) in enumerate(segment_coords):
        elev = fetch_elevation(slat, slon)
        if use_forecast:
            weather = fetch_weather_from_openmeteo(slat, slon, start_date, end_date, forecast=True)
        else:
            weather = fetch_weather_multi_year(slat, slon, start_date, end_date)
        corrected_weather = []
        for w in weather:
            tc, wc = apply_terrain_correction(w.ambient_c, w.wind_mps, elev, base_elevation)
            corrected_w = WeatherRecord(
                timestamp=w.timestamp,
                ambient_c=tc,
                wind_mps=wc,
                wind_direction_deg=w.wind_direction_deg,
                ghi_w_m2=w.ghi_w_m2
            )
            corrected_weather.append(corrected_w)
        results = run_dlr_for_period_segment(
            corrected_weather, conductor, convection_model, test_current,
            ml_model, use_ml
        )
        min_seg_dlr = min(r["dlr_a"] for r in results)
        min_seg_hour = next(r for r in results if r["dlr_a"] == min_seg_dlr)
        segment_results.append({
            "lat": slat,
            "lon": slon,
            "elevation_m": elev,
            "min_dlr_a": min_seg_dlr,
            "min_dlr_mw": amps_to_mw(min_seg_dlr, CONFIG_DEFAULTS.get("nominal_voltage_kv", 220), CONFIG_DEFAULTS.get("power_factor", 0.9)),
            "temperature_at_min_dlr": min_seg_hour["temperature_c"],
            "sag_at_min_dlr": min_seg_hour["sag_m"],
            "results": results
        })
        if min_seg_dlr < min_dlr:
            min_dlr = min_seg_dlr
            weakest_idx = idx
    return segment_results, min_dlr, weakest_idx

# ============================================================================
# DATABASE, ALERTS, SCHEDULER
# ============================================================================

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            conductor TEXT NOT NULL,
            convection_model TEXT NOT NULL,
            test_current REAL NOT NULL,
            use_ml INTEGER NOT NULL,
            results_json TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_timestamp TEXT NOT NULL,
            run_id INTEGER,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_run(db_path, run_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO runs (run_timestamp, period_start, period_end, lat, lon,
                          conductor, convection_model, test_current, use_ml, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_data["run_timestamp"],
        run_data["period_start"],
        run_data["period_end"],
        run_data["lat"],
        run_data["lon"],
        run_data["conductor"],
        run_data["convection_model"],
        run_data["test_current"],
        1 if run_data["use_ml"] else 0,
        json.dumps(run_data["results"])
    ))
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def save_alert(db_path, alert_data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (alert_timestamp, run_id, type, message, acknowledged)
        VALUES (?, ?, ?, ?, ?)
    """, (
        alert_data["alert_timestamp"],
        alert_data.get("run_id"),
        alert_data["type"],
        alert_data["message"],
        0
    ))
    conn.commit()
    conn.close()

def get_latest_run(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "run_timestamp": row[1],
            "period_start": row[2],
            "period_end": row[3],
            "lat": row[4],
            "lon": row[5],
            "conductor": row[6],
            "convection_model": row[7],
            "test_current": row[8],
            "use_ml": bool(row[9]),
            "results": json.loads(row[10])
        }
    return None

def get_historical_runs(db_path, limit=100):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    runs = []
    for row in rows:
        runs.append({
            "id": row[0],
            "run_timestamp": row[1],
            "period_start": row[2],
            "period_end": row[3],
            "lat": row[4],
            "lon": row[5],
            "conductor": row[6],
            "convection_model": row[7],
            "test_current": row[8],
            "use_ml": bool(row[9]),
            "results": json.loads(row[10])
        })
    return runs

def check_alerts(results, config, db_path, run_id):
    alerts = []
    if not results:
        return
    min_dlr = min(r["dlr_a"] for r in results)
    max_temp = max(r["temperature_c"] for r in results)
    worst_dlr = next(r for r in results if r["dlr_a"] == min_dlr)
    worst_temp = next(r for r in results if r["temperature_c"] == max_temp)
    if min_dlr < config.get("alert_threshold_dlr", 2500):
        msg = f"DLR dropped to {min_dlr:.2f} A at {worst_dlr['timestamp']} (threshold: {config['alert_threshold_dlr']} A)"
        alerts.append({"type": "dlr", "message": msg})
    if max_temp > config.get("alert_threshold_temp", 80):
        msg = f"Conductor temperature reached {max_temp:.2f} °C at {worst_temp['timestamp']} (threshold: {config['alert_threshold_temp']} °C)"
        alerts.append({"type": "temperature", "message": msg})
    for alert in alerts:
        alert_data = {
            "alert_timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "type": alert["type"],
            "message": alert["message"]
        }
        save_alert(db_path, alert_data)
        if config.get("email_enabled"):
            send_email_alert(config, alert["message"])
        if config.get("slack_enabled"):
            send_slack_alert(config, alert["message"])
    return alerts

def send_email_alert(config, message):
    if not config.get("email_enabled"):
        return
    smtp_server = config.get("email_smtp_server")
    port = config.get("email_smtp_port")
    sender = config.get("email_from")
    password = config.get("email_password")
    receiver = config.get("email_to")
    subject = "GridTweak DLR Alert"
    body = f"Alert: {message}\n\nGridTweak DLR System"
    email_msg = f"Subject: {subject}\n\n{body}"
    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, email_msg)
    except Exception as e:
        print(f"Email send failed: {e}")

def send_slack_alert(config, message):
    if not config.get("slack_enabled") or not REQUESTS_AVAILABLE:
        return
    webhook = config.get("slack_webhook_url")
    payload = {"text": f"🚨 GridTweak DLR Alert: {message}"}
    try:
        requests.post(webhook, json=payload)
    except Exception as e:
        print(f"Slack send failed: {e}")

def run_scheduled_job(config, db_path, ml_model, ml_scaler):
    lat = config["lat"]
    lon = config["lon"]
    conductor = get_conductor(config["conductor"])
    convection_model = config["convection_model"]
    test_current = config["test_current"]
    use_ml = config["use_ml"] and ml_model is not None
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Running scheduled job for {start_date} to {end_date}")
    try:
        weather = fetch_weather_multi_year(lat, lon, start_date, end_date)
        results = run_dlr_for_period_segment(
            weather, conductor, convection_model, test_current,
            (ml_model, ml_scaler) if ml_model else None,
            use_ml
        )
        run_data = {
            "run_timestamp": datetime.now().isoformat(),
            "period_start": start_date,
            "period_end": end_date,
            "lat": lat,
            "lon": lon,
            "conductor": config["conductor"],
            "convection_model": convection_model,
            "test_current": test_current,
            "use_ml": use_ml,
            "results": results
        }
        run_id = save_run(db_path, run_data)
        check_alerts(results, config, db_path, run_id)
        print(f"Scheduled job completed. Run ID: {run_id}")
    except Exception as e:
        print(f"Scheduled job failed: {e}")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title=f"{APP_NAME} DLR API", version=VERSION) if API_AVAILABLE else None

def enrich_with_mw(results, voltage_kv, pf):
    for r in results:
        mw = amps_to_mw(r["dlr_a"], voltage_kv, pf)
        r["dlr_mw"] = mw
    return results

if app is not None:
    from fastapi.responses import RedirectResponse

    @app.get("/")
    async def root():
        return RedirectResponse(url="/dashboard")

    @app.get("/dlr/current")
    async def get_current():
        db_path = CONFIG_DEFAULTS.get("database_path", "dlr_data.db")
        run = get_latest_run(db_path)
        if not run:
            raise HTTPException(status_code=404, detail="No data available")
        results = run["results"]
        if len(results) > 24:
            results = results[-24:]
        run["results"] = results
        voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
        pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
        run["results"] = enrich_with_mw(run["results"], voltage, pf)
        conductor = get_conductor(CONFIG_DEFAULTS["conductor"])
        static_mw_override = CONFIG_DEFAULTS.get("static_rating_mw")
        if static_mw_override is not None:
            static_rating_mw = static_mw_override
        else:
            static_rating_amp = compute_static_rating(conductor, LINE_AZIMUTH_DEG, CONFIG_DEFAULTS["convection_model"])
            static_rating_mw = amps_to_mw(static_rating_amp, voltage, pf)
        run["static_rating_mw"] = static_rating_mw
        return JSONResponse(content=run)

    @app.get("/dlr/historical")
    async def get_historical(limit: int = Query(100, ge=1, le=1000)):
        db_path = CONFIG_DEFAULTS.get("database_path", "dlr_data.db")
        runs = get_historical_runs(db_path, limit=limit)
        voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
        pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
        for run in runs:
            run["results"] = enrich_with_mw(run["results"], voltage, pf)
        return JSONResponse(content=runs)

    @app.get("/dlr/forecast")
    async def get_forecast():
        """Return cached forecast data. No live fetching here."""
        global _cached_forecast
        if _cached_forecast["data"] is None:
            return JSONResponse(content={"forecast": []})
        return JSONResponse(content={"forecast": _cached_forecast["data"]})

    @app.get("/dlr/corridor")
    async def get_corridor():
        try:
            lat = CONFIG_DEFAULTS.get("lat", 19.076)
            lon = CONFIG_DEFAULTS.get("lon", 72.877)
            end_lat = CONFIG_DEFAULTS.get("end_lat")
            end_lon = CONFIG_DEFAULTS.get("end_lon")
            if end_lat is None:
                end_lat = lat + 0.5
            if end_lon is None:
                end_lon = lon + 0.5
            num_segments = CONFIG_DEFAULTS.get("num_segments", 3)
            conductor = get_conductor(CONFIG_DEFAULTS["conductor"])
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            seg_results, min_dlr, weakest_idx = run_corridor_dlr(
                lat, lon, end_lat, end_lon, num_segments,
                conductor, CONFIG_DEFAULTS["convection_model"],
                CONFIG_DEFAULTS["test_current"],
                None, CONFIG_DEFAULTS["use_ml"],
                start_date, end_date, use_forecast=False
            )
            voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
            pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
            segments = []
            for s in seg_results:
                segments.append({
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "elevation_m": s["elevation_m"],
                    "min_dlr_mw": s["min_dlr_mw"],
                    "temperature_c": s["temperature_at_min_dlr"],
                    "sag_m": s["sag_at_min_dlr"],
                })
            return JSONResponse(content={
                "min_dlr_mw": amps_to_mw(min_dlr, voltage, pf),
                "weakest_segment": weakest_idx,
                "segments": segments
            })
        except Exception as e:
            import traceback
            print(f"Corridor endpoint error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/dlr/forecast/corridor")
    async def get_forecast_corridor():
        try:
            lat = CONFIG_DEFAULTS.get("lat", 19.076)
            lon = CONFIG_DEFAULTS.get("lon", 72.877)
            end_lat = CONFIG_DEFAULTS.get("end_lat")
            end_lon = CONFIG_DEFAULTS.get("end_lon")
            if end_lat is None:
                end_lat = lat + 0.5
            if end_lon is None:
                end_lon = lon + 0.5
            num_segments = CONFIG_DEFAULTS.get("num_segments", 3)
            conductor = get_conductor(CONFIG_DEFAULTS["conductor"])
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            seg_results, min_dlr, weakest_idx = run_corridor_dlr(
                lat, lon, end_lat, end_lon, num_segments,
                conductor, CONFIG_DEFAULTS["convection_model"],
                CONFIG_DEFAULTS["test_current"],
                None, CONFIG_DEFAULTS["use_ml"],
                start_date, end_date, use_forecast=True
            )
            voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
            pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
            weakest = seg_results[weakest_idx]
            segments = []
            for s in seg_results:
                segments.append({
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "elevation_m": s["elevation_m"],
                    "min_dlr_mw": s["min_dlr_mw"],
                    "temperature_c": s["temperature_at_min_dlr"],
                    "sag_m": s["sag_at_min_dlr"],
                })
            return JSONResponse(content={
                "min_dlr_mw": amps_to_mw(min_dlr, voltage, pf),
                "weakest_segment": weakest_idx,
                "segments": segments,
                "forecast": [{"timestamp": r["timestamp"], "dlr_mw": amps_to_mw(r["dlr_a"], voltage, pf), "sag_m": r["sag_m"]} for r in weakest["results"]]
            })
        except Exception as e:
            import traceback
            print(f"Forecast corridor error: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # DASHBOARD HTML (with favicon from external file)
    # ========================================================================
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        location = CONFIG_DEFAULTS.get("location_name", "Transmission Line")

        # Read favicon base64 from file if it exists
        favicon_base64 = ""
        favicon_path = Path("favicon_base64.txt")
        if favicon_path.exists():
            try:
                with open(favicon_path, "r") as f:
                    favicon_base64 = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read favicon file: {e}")

        favicon_tag = ""
        if favicon_base64:
            favicon_tag = f'<link rel="icon" href="data:image/x-icon;base64,{favicon_base64}" type="image/x-icon">'
        else:
            favicon_tag = '<link rel="icon" href="data:,">'

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GridTweak – DLR Intelligence</title>
    {favicon_tag}
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f6f9fc;
            color: #1a2634;
            padding: 20px;
        }}
        body.dark {{
            background: #0b1a26;
            color: #e2e8f0;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .logo {{ font-size: 28px; font-weight: 700; color: #1a6b8a; }}
        .logo span {{ color: #0b2e4f; }}
        .dark .logo {{ color: #60a5fa; }}
        .dark .logo span {{ color: #93c5fd; }}
        .status-badge {{
            background: #e6f7e6;
            color: #0e7c3e;
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dark .status-badge {{ background: #1a3a2a; color: #6fcf97; }}
        .dot {{ width: 8px; height: 8px; background: #0e7c3e; border-radius: 50%; animation: pulse 2s infinite; }}
        .dark .dot {{ background: #6fcf97; }}
        @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.3; }} 100% {{ opacity: 1; }} }}
        .last-updated {{ font-size: 13px; color: #718096; }}
        .dark .last-updated {{ color: #a0aec0; }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #edf2f7;
            transition: 0.2s;
        }}
        .dark .metric-card {{
            background: #1a202c;
            border-color: #2d3748;
        }}
        .metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
        .metric-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #718096;
            margin-bottom: 4px;
        }}
        .dark .metric-label {{ color: #a0aec0; }}
        .metric-value {{
            font-size: 24px;
            font-weight: 700;
        }}
        .metric-unit {{
            font-size: 13px;
            font-weight: 400;
            color: #718096;
            margin-left: 4px;
        }}
        .dark .metric-unit {{ color: #a0aec0; }}
        .recommendation {{
            background: #e0f2fe;
            border-left: 4px solid #1a6b8a;
            padding: 14px 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .dark .recommendation {{ background: #1a2a3a; border-color: #60a5fa; }}
        .rec-text {{ font-weight: 500; font-size: 15px; }}
        .rec-text strong {{ color: #1a6b8a; }}
        .dark .rec-text strong {{ color: #60a5fa; }}
        .tabs {{
            display: flex;
            gap: 6px;
            background: #e2e8f0;
            padding: 4px;
            border-radius: 10px;
            margin-bottom: 25px;
            width: fit-content;
        }}
        .dark .tabs {{ background: #2d3748; }}
        .tab {{
            padding: 8px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            background: transparent;
            color: #4a5568;
            transition: 0.2s;
            font-size: 14px;
        }}
        .dark .tab {{ color: #a0aec0; }}
        .tab.active {{
            background: white;
            color: #1a202c;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .dark .tab.active {{ background: #1a202c; color: #e2e8f0; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #edf2f7;
            margin-bottom: 25px;
            height: 340px;
        }}
        .dark .chart-container {{
            background: #1a202c;
            border-color: #2d3748;
        }}
        .table-wrap {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #edf2f7;
            overflow-x: auto;
            margin-bottom: 25px;
        }}
        .dark .table-wrap {{ background: #1a202c; border-color: #2d3748; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th {{ text-align: left; padding: 8px 6px; color: #4a5568; font-weight: 600; border-bottom: 2px solid #edf2f7; }}
        .dark th {{ color: #a0aec0; border-bottom-color: #2d3748; }}
        td {{ padding: 6px; border-bottom: 1px solid #edf2f7; }}
        .dark td {{ border-bottom-color: #2d3748; }}
        .refresh-btn {{
            background: #1a6b8a;
            color: white;
            border: none;
            padding: 6px 18px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
            font-size: 13px;
        }}
        .refresh-btn:hover {{ background: #0f4a62; }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #718096;
        }}
        .dark .footer {{ color: #a0aec0; }}
        .corridor-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .corridor-chart {{
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            border: 1px solid #edf2f7;
            height: 280px;
        }}
        .dark .corridor-chart {{
            background: #1a202c;
            border-color: #2d3748;
        }}
        @media (max-width: 640px) {{
            .metric-grid {{ grid-template-columns: 1fr 1fr; }}
            .metric-value {{ font-size: 20px; }}
            .header {{ flex-direction: column; align-items: flex-start; }}
            .corridor-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <!-- Header -->
    <div class="header">
        <div>
            <div class="logo">Grid<span>Tweak</span></div>
            <div style="font-size: 14px; color: #718096; margin-top: 4px; font-weight: 500;">
                📍 {location}
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            <span class="last-updated" id="lastUpdated">Updating...</span>
            <div class="status-badge"><span class="dot"></span> System Live</div>
            <button class="refresh-btn" onclick="fetchAll()">↻ Refresh</button>
        </div>
    </div>

    <!-- Recommendation -->
    <div class="recommendation" id="recommendation">
        <span class="rec-text">💡 <strong>Operational Recommendation:</strong> You can safely push <strong id="recMw">--</strong> MW of additional power through this line.</span>
        <span style="font-size:14px;font-weight:500;" id="recStatus">✅ Within limits</span>
    </div>

    <!-- Metric Grid -->
    <div class="metric-grid" id="metricGrid">
        <div class="metric-card"><div class="metric-label">Available Capacity</div><div class="metric-value" id="dlr">-- <span class="metric-unit">MW</span></div></div>
        <div class="metric-card"><div class="metric-label">Headroom vs Static</div><div class="metric-value" id="headroom">-- <span class="metric-unit">MW</span></div></div>
        <div class="metric-card"><div class="metric-label">Capacity Factor</div><div class="metric-value" id="utilization">--</div></div>
        <div class="metric-card"><div class="metric-label">Conductor Temp</div><div class="metric-value" id="temp">-- <span class="metric-unit">°C</span></div></div>
        <div class="metric-card"><div class="metric-label">Ambient</div><div class="metric-value" id="ambient">-- <span class="metric-unit">°C</span></div></div>
        <div class="metric-card"><div class="metric-label">Wind Speed</div><div class="metric-value" id="wind">-- <span class="metric-unit">m/s</span></div></div>
        <div class="metric-card"><div class="metric-label">Sag</div><div class="metric-value" id="sag">-- <span class="metric-unit">m</span></div></div>
    </div>

    <!-- Tabs -->
    <div class="tabs">
        <button class="tab active" data-tab="historical">Historical (24h)</button>
        <button class="tab" data-tab="forecast">Forecast (7d)</button>
        <button class="tab" data-tab="corridor">Corridor</button>
    </div>

    <!-- Historical Tab -->
    <div id="historical-tab" class="tab-content active">
        <div class="chart-container"><canvas id="historicalChart"></canvas></div>
    </div>
    <!-- Forecast Tab -->
    <div id="forecast-tab" class="tab-content">
        <div class="chart-container"><canvas id="forecastChart"></canvas></div>
        <div class="table-wrap"><h3 style="margin-bottom:12px;">📈 7‑Day Forecast (MW)</h3><div id="forecastTable"></div></div>
    </div>
    <!-- Corridor Tab -->
    <div id="corridor-tab" class="tab-content">
        <div class="corridor-grid">
            <div class="corridor-chart"><canvas id="corridorDlrChart"></canvas></div>
            <div class="corridor-chart"><canvas id="corridorSagChart"></canvas></div>
        </div>
        <div class="table-wrap"><h3 style="margin-bottom:12px;">📊 Corridor Summary</h3><div id="corridorTable"></div></div>
    </div>

    <div class="footer">&copy; 2026 GridTweak — AI‑Powered Dynamic Line Rating</div>
</div>

<script>
    let histChart, forecastChart, dlrBarChart, sagBarChart;
    const voltage = 220, pf = 0.9;

    function initCharts() {{
        const opts = (title) => ({{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', labels: {{ font: {{ size: 12 }} }} }},
                tooltip: {{ backgroundColor: '#0b2e4f', titleFont: {{ weight: '600' }} }}
            }},
            scales: {{
                y: {{ beginAtZero: true, grid: {{ color: '#edf2f7' }}, title: {{ display: true, text: title, font: {{ size: 12 }} }} }},
                x: {{ grid: {{ display: false }} }}
            }}
        }});
        histChart = new Chart(document.getElementById('historicalChart'), {{ type: 'line', data: {{ labels: [], datasets: [] }}, options: opts('MW / °C') }});
        forecastChart = new Chart(document.getElementById('forecastChart'), {{ type: 'line', data: {{ labels: [], datasets: [] }}, options: opts('MW / °C') }});
        dlrBarChart = new Chart(document.getElementById('corridorDlrChart'), {{ type: 'bar', data: {{ labels: [], datasets: [] }}, options: {{...opts('MW'), plugins: {{ legend: {{ display: false }} }} }} }});
        sagBarChart = new Chart(document.getElementById('corridorSagChart'), {{ type: 'bar', data: {{ labels: [], datasets: [] }}, options: {{...opts('Sag (m)'), plugins: {{ legend: {{ display: false }} }} }} }});
    }}

    function ampsToMW(a) {{ return a * voltage * Math.sqrt(3) * pf / 1000; }}

    async function fetchAll() {{
        await fetchHistorical();
        await fetchForecast();
        await fetchCorridor();
        document.getElementById('lastUpdated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
    }}

    async function fetchHistorical() {{
        try {{
            const resp = await fetch('/dlr/current');
            if (!resp.ok) throw new Error('No data');
            const data = await resp.json();
            const staticMW = data.static_rating_mw || 350;
            if (!data || !data.results || data.results.length === 0) {{
                showNoData();
                return;
            }}
            const results = data.results;
            const latest = results[results.length-1] || results[0];
            const dlrMW = latest.dlr_mw || ampsToMW(latest.dlr_a);
            const headroom = dlrMW - staticMW;
            const capFactor = (dlrMW / staticMW * 100);
            document.getElementById('dlr').innerHTML = dlrMW.toFixed(0) + ' <span class="metric-unit">MW</span>';
            document.getElementById('headroom').innerHTML = headroom.toFixed(0) + ' <span class="metric-unit">MW</span>';
            document.getElementById('utilization').textContent = capFactor.toFixed(0) + '%';
            document.getElementById('temp').innerHTML = (latest.temperature_c || 0).toFixed(1) + ' <span class="metric-unit">°C</span>';
            document.getElementById('ambient').innerHTML = (latest.ambient_c || 0).toFixed(1) + ' <span class="metric-unit">°C</span>';
            document.getElementById('wind').innerHTML = (latest.wind_mps || 0).toFixed(1) + ' <span class="metric-unit">m/s</span>';
            document.getElementById('sag').innerHTML = (latest.sag_m || 0).toFixed(2) + ' <span class="metric-unit">m</span>';
            document.getElementById('recMw').textContent = headroom.toFixed(0);
            document.getElementById('recStatus').textContent = headroom > 0 ? '✅ Within limits' : '⚠️ Approaching thermal limit';

            const timestamps = results.map(r => r.timestamp.slice(11,16));
            const dlrs = results.map(r => r.dlr_mw || ampsToMW(r.dlr_a));
            const temps = results.map(r => r.temperature_c || 0);
            histChart.data = {{
                labels: timestamps,
                datasets: [
                    {{ label: 'Available Capacity (MW)', data: dlrs, borderColor: '#1a6b8a', fill: true, backgroundColor: 'rgba(26,107,138,0.1)' }},
                    {{ label: 'Temperature (°C)', data: temps, borderColor: '#e53e3e', fill: true, backgroundColor: 'rgba(229,62,62,0.1)', yAxisID: 'y1' }}
                ]
            }};
            histChart.update();
        }} catch(e) {{
            console.error('Historical error:', e);
            showNoData();
        }}
    }}

    function showNoData() {{
        const vals = ['dlr','headroom','utilization','temp','ambient','wind','sag'];
        vals.forEach(id => {{
            document.getElementById(id).innerHTML = '--';
        }});
        document.getElementById('recMw').textContent = '--';
        document.getElementById('recStatus').textContent = '⏳ No data yet';
        histChart.data = {{ labels: [], datasets: [] }};
        histChart.update();
    }}

    async function fetchForecast() {{
        try {{
            const resp = await fetch('/dlr/forecast');
            if (!resp.ok) {{
                const errText = await resp.text();
                console.error('Forecast API error:', resp.status, errText);
                document.getElementById('forecastTable').innerHTML = `<p style="color:red;">⚠️ Forecast unavailable (${{resp.status}})</p>`;
                return;
            }}
            const data = await resp.json();
            console.log('✅ Forecast data (cached):', data);
            if (!data || !data.forecast || data.forecast.length === 0) {{
                document.getElementById('forecastTable').innerHTML = '<p>No forecast data available for this location.</p>';
                return;
            }}
            const results = data.forecast;
            const timestamps = results.map(r => r.timestamp.slice(11,16));
            const dlrs = results.map(r => r.dlr_mw);
            const temps = results.map(r => r.temperature_c);
            forecastChart.data = {{
                labels: timestamps,
                datasets: [
                    {{ label: 'Forecast Capacity (MW)', data: dlrs, borderColor: '#38a169', fill: true, backgroundColor: 'rgba(56,161,105,0.1)' }},
                    {{ label: 'Forecast Temperature (°C)', data: temps, borderColor: '#ed8936', fill: true, backgroundColor: 'rgba(237,137,54,0.1)', yAxisID: 'y1' }}
                ]
            }};
            forecastChart.update();
            let tableHtml = `<table><tr><th>Date</th><th>00:00</th><th>06:00</th><th>12:00</th><th>18:00</th></tr>`;
            const days = {{}};
            results.forEach(r => {{
                const date = r.timestamp.slice(0,10);
                if (!days[date]) days[date] = {{}};
                const hour = parseInt(r.timestamp.slice(11,13));
                if ([0,6,12,18].includes(hour)) days[date][hour] = r.dlr_mw;
            }});
            for (const [date, hours] of Object.entries(days)) {{
                tableHtml += `<tr><td>${{date}}</td>`;
                for (const h of [0,6,12,18]) {{
                    const val = hours[h] !== undefined ? hours[h].toFixed(0) : '--';
                    tableHtml += `<td>${{val}}</td>`;
                }}
                tableHtml += `</tr>`;
            }}
            tableHtml += `</table>`;
            document.getElementById('forecastTable').innerHTML = tableHtml;
        }} catch(e) {{
            console.error('Forecast fetch error:', e);
            document.getElementById('forecastTable').innerHTML = `<p style="color:red;">⚠️ Failed to load forecast: ${{e.message}}</p>`;
        }}
    }}

    async function fetchCorridor() {{
        try {{
            const resp = await fetch('/dlr/corridor');
            if (!resp.ok) throw new Error('No corridor data');
            const data = await resp.json();
            if (!data || !data.segments || data.segments.length === 0) return;
            const segs = data.segments;
            const labels = segs.map((_, i) => 'Seg ' + (i+1));
            const dlrs = segs.map(s => s.min_dlr_mw);
            const sags = segs.map(s => s.sag_m);
            dlrBarChart.data = {{ labels, datasets: [{{ label: 'DLR (MW)', data: dlrs, backgroundColor: 'rgba(26,107,138,0.6)', borderColor: '#1a6b8a', borderWidth: 1 }}] }};
            dlrBarChart.update();
            sagBarChart.data = {{ labels, datasets: [{{ label: 'Sag (m)', data: sags, backgroundColor: 'rgba(229,62,62,0.6)', borderColor: '#e53e3e', borderWidth: 1 }}] }};
            sagBarChart.update();
            let tableHtml = `<table><tr><th>Segment</th><th>DLR (MW)</th><th>Temp (°C)</th><th>Sag (m)</th></tr>`;
            segs.forEach((s, i) => {{
                tableHtml += `<tr><td>${{i+1}}</td><td>${{s.min_dlr_mw.toFixed(0)}}</td><td>${{s.temperature_c.toFixed(1)}}</td><td>${{s.sag_m.toFixed(2)}}</td></tr>`;
            }});
            tableHtml += `</table>`;
            document.getElementById('corridorTable').innerHTML = tableHtml;
        }} catch(e) {{
            console.error('Corridor error:', e);
        }}
    }}

    document.querySelectorAll('.tab').forEach(tab => {{
        tab.addEventListener('click', function() {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(this.dataset.tab + '-tab').classList.add('active');
            if (this.dataset.tab === 'forecast') fetchForecast();
            if (this.dataset.tab === 'corridor') fetchCorridor();
        }});
    }});

    initCharts();
    fetchAll();
    setInterval(fetchAll, 60000);
</script>
</body>
</html>
"""
        return html

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} DLR Engine {VERSION}")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--run-now", action="store_true", help="Run the engine once")
    parser.add_argument("--scheduler", action="store_true", help="Start background scheduler")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--init-db", action="store_true", help="Initialize database")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--corridor", action="store_true", help="Enable corridor modelling")
    parser.add_argument("--end-lat", type=float, help="End latitude for corridor")
    parser.add_argument("--end-lon", type=float, help="End longitude for corridor")
    parser.add_argument("--segments", type=int, default=3, help="Number of segments")
    parser.add_argument("--forecast", action="store_true", help="Use forecast weather instead of archive")
    args = parser.parse_args()

    config = CONFIG_DEFAULTS.copy()
    if args.config:
        try:
            with open(args.config, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
                CONFIG_DEFAULTS.update(config)
                print(f"✅ Loaded config from {args.config}")
                print(f"   static_rating_mw = {CONFIG_DEFAULTS.get('static_rating_mw')}")
                print(f"   location_name = {CONFIG_DEFAULTS.get('location_name')}")
                print(f"   wind_correction_factor = {CONFIG_DEFAULTS.get('wind_correction_factor')}")
                print(f"   dlr_scaling_factor = {CONFIG_DEFAULTS.get('dlr_scaling_factor')}")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}. Using defaults.")
    else:
        print("ℹ️ No config file provided. Using default values.")

    db_path = CONFIG_DEFAULTS["database_path"]

    if args.init_db or not os.path.exists(db_path):
        init_db(db_path)
        print(f"Database initialized at {db_path}")

    ml_model = None
    ml_scaler = None
    if CONFIG_DEFAULTS.get("use_ml", False):
        model_path = CONFIG_DEFAULTS.get("ml_model_path", "lgb_best.pkl")
        if os.path.exists(model_path):
            ml_model, ml_scaler = load_ml_model(model_path)
            if ml_model:
                print(f"ML model loaded from {model_path}")
            else:
                print("Failed to load ML model. Falling back to physics.")
        else:
            print(f"ML model {model_path} not found. Falling back to physics.")

    if args.run_now:
        if args.corridor and args.end_lat is not None and args.end_lon is not None:
            lat = CONFIG_DEFAULTS["lat"]
            lon = CONFIG_DEFAULTS["lon"]
            end_lat = args.end_lat
            end_lon = args.end_lon
            num_segments = args.segments
            conductor = get_conductor(CONFIG_DEFAULTS["conductor"])
            convection_model = CONFIG_DEFAULTS["convection_model"]
            test_current = CONFIG_DEFAULTS["test_current"]
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") if not args.forecast else datetime.now().strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d") if not args.forecast else (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            seg_results, min_dlr, weakest_idx = run_corridor_dlr(
                lat, lon, end_lat, end_lon, num_segments,
                conductor, convection_model, test_current,
                (ml_model, ml_scaler) if ml_model else None,
                CONFIG_DEFAULTS["use_ml"],
                start_date, end_date,
                use_forecast=args.forecast
            )
            voltage = CONFIG_DEFAULTS.get("nominal_voltage_kv", 220)
            pf = CONFIG_DEFAULTS.get("power_factor", 0.9)
            print(f"Corridor DLR: min = {min_dlr:.2f} A ({amps_to_mw(min_dlr, voltage, pf):.1f} MW) at segment {weakest_idx+1}")
            weak_seg = seg_results[weakest_idx]
            print(f"Weakest segment sag: {weak_seg['sag_at_min_dlr']:.2f} m at {weak_seg['temperature_at_min_dlr']:.1f} °C")
            return 0

        run_scheduled_job(CONFIG_DEFAULTS, db_path, ml_model, ml_scaler)
        return 0

    if args.scheduler:
        if not SCHEDULER_AVAILABLE:
            print("APScheduler not installed. Install with: pip install apscheduler")
            return 1
        scheduler = BackgroundScheduler()
        interval = CONFIG_DEFAULTS.get("scheduler_interval_hours", 1)
        scheduler.add_job(
            run_scheduled_job,
            trigger=IntervalTrigger(hours=interval),
            args=[CONFIG_DEFAULTS, db_path, ml_model, ml_scaler],
            id="dlr_job"
        )
        scheduler.start()
        print(f"Scheduler started. Running every {interval} hour(s). Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            scheduler.shutdown()
            print("Scheduler stopped.")
        return 0

    if args.api:
        if not API_AVAILABLE:
            print("FastAPI not installed. Install with: pip install fastapi uvicorn")
            return 1
        
        print(f"Starting {APP_NAME} API server at http://localhost:{args.port}")
        print(f"Dashboard: http://localhost:{args.port}/dashboard")
        
        # --- Load forecast cache from disk (instant, no API call) ---
        cache_loaded = load_forecast_cache_from_file()
        
        # --- If cache is missing or very old, trigger a background update ---
        if not cache_loaded:
            print("🌤️ No disk cache found. Attempting initial fetch...")
            try:
                update_forecast_cache()
            except Exception as e:
                print(f"⚠️ Initial forecast fetch failed: {e}")
        else:
            # Check if cache is stale (> 6 hours) and update in background later
            try:
                if _cached_forecast["last_updated"]:
                    last_updated = datetime.fromisoformat(_cached_forecast["last_updated"])
                    age = (datetime.now() - last_updated).total_seconds() / 3600
                    if age > 6:
                        print(f"⏳ Disk cache is {age:.1f} hours old. Will refresh on next scheduled run.")
            except:
                pass
        
        # --- Schedule periodic refresh (every 6 hours) ---
        if SCHEDULER_AVAILABLE:
            try:
                scheduler = BackgroundScheduler()
                scheduler.add_job(update_forecast_cache, 'interval', hours=6, id='forecast_cache_job')
                scheduler.start()
                print("🔄 Forecast cache will refresh every 6 hours (if needed).")
            except Exception as e:
                print(f"⚠️ Could not start scheduler: {e}")
        else:
            print("ℹ️ APScheduler not available. Forecast cache will not auto-refresh.")
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return 0

    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())