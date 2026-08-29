#!/usr/bin/env python3
"""
Generate forecast.json for GridTweak
Run this locally to create the initial forecast file.
"""
import json
from datetime import datetime, timedelta
from dlr_v28 import (
    CONFIG_DEFAULTS, fetch_weather_from_openmeteo, get_conductor,
    solve_dlr, solve_temperature, solar_position, wind_geometry,
    amps_to_mw, calculate_sag, LINE_AZIMUTH_DEG, ROUGHNESS_M, TURB_INTENSITY
)

def main():
    config = CONFIG_DEFAULTS
    lat = config["lat"]
    lon = config["lon"]
    conductor = get_conductor(config["conductor"])
    days = 7
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

    print("🌤️ Fetching forecast from Open-Meteo...")
    try:
        weather = fetch_weather_from_openmeteo(lat, lon, start_date, end_date, forecast=True)
        if not weather:
            print("⚠️ Forecast empty, trying archive fallback...")
            archive_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            weather = fetch_weather_from_openmeteo(lat, lon, archive_start, end_date, forecast=False)
    except Exception as e:
        print(f"⚠️ Forecast failed ({e}), trying archive fallback...")
        archive_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        weather = fetch_weather_from_openmeteo(lat, lon, archive_start, end_date, forecast=False)

    if not weather:
        print("❌ No weather data available.")
        return

    print(f"✅ Fetched {len(weather)} weather records")

    results = []
    for w in weather:
        try:
            dt = datetime.fromisoformat(w.timestamp)
            offset = config.get("timezone_offset_hours", 5.5)
            local_hour = int(round((dt.hour + offset) % 24)) % 24
            sp = solar_position(local_hour)
            geo = wind_geometry(w.wind_mps, w.wind_direction_deg, LINE_AZIMUTH_DEG)
            wind_perp = geo["perpendicular_ms"]
            attack_deg = geo["attack_angle_deg"]

            dlr = solve_dlr(w.ambient_c, wind_perp, w.ghi_w_m2, sp["altitude_deg"], sp["azimuth_deg"],
                            conductor, LINE_AZIMUTH_DEG, attack_deg,
                            config["convection_model"], ROUGHNESS_M, TURB_INTENSITY)
            scaling = config.get("conservative_derating_factor", 1.0)
            dlr = dlr * scaling

            temp_res = solve_temperature(config["test_current"], w.ambient_c, wind_perp, w.ghi_w_m2,
                                         sp["altitude_deg"], sp["azimuth_deg"],
                                         conductor, LINE_AZIMUTH_DEG, attack_deg,
                                         config["convection_model"], ROUGHNESS_M, TURB_INTENSITY)
            voltage = config.get("nominal_voltage_kv", 220)
            pf = config.get("power_factor", 0.9)
            mw = amps_to_mw(dlr, voltage, pf)
            sag_m = calculate_sag(temp_res["temperature_c"], conductor,
                                  config.get("span_length_m", 400),
                                  ref_temp=config.get("ref_temp_sag", 20.0),
                                  sag_ref=config.get("sag_ref_m", 5.0))
            results.append({
                "timestamp": w.timestamp,
                "dlr_mw": mw,
                "temperature_c": temp_res["temperature_c"],
                "ambient_c": w.ambient_c,
                "wind_mps": w.wind_mps,
                "sag_m": sag_m,
                "status": "OK" if temp_res["temperature_c"] <= conductor.tmax_c else "OVER"
            })
        except Exception as e:
            print(f"⚠️ Skipping record {w.timestamp} due to {e}")
            continue

    with open("forecast.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ forecast.json written ({len(results)} records)")

if __name__ == "__main__":
    main()