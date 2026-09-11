"""
api_client.py
--------------
Fetches live data from two public APIs and turns it into clean numbers
that renderer.py can use to drive the artwork.

Data sources:
  1. USGS Earthquake API  -> magnitude, depth, latitude, longitude
  2. NOAA Space Weather    -> solar wind speed, Bz (magnetic field)

Reliability strategy:
  - Try the live API first (5 second timeout, so a dead network doesn't
    hang the whole program).
  - If it succeeds, save the parsed result to a local cache file
    (cache_usgs.json / cache_noaa.json).
  - If it fails (no internet, API down, empty data), fall back to the
    last cache file we saved.
  - If even the cache doesn't exist yet (first ever run, no internet),
    fall back to fixed safe default numbers so the program NEVER crashes
    just because data wasn't available.
"""

import json
import os
import requests

# ---------------------------------------------------------------------
# Where cache files live: same folder as this script, so it works no
# matter what folder you launch main.py from.
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_USGS_PATH = os.path.join(SCRIPT_DIR, "cache_usgs.json")
CACHE_NOAA_PATH = os.path.join(SCRIPT_DIR, "cache_noaa.json")

USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
NOAA_WIND_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
NOAA_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"

REQUEST_TIMEOUT_SECONDS = 5

# Fixed safe values used ONLY if there is no live data AND no cache at all.
USGS_HARD_DEFAULT = {"magnitude": 2.5, "depth": 10.0, "latitude": 0.0, "longitude": 0.0}
NOAA_HARD_DEFAULT = {"speed": 400.0, "bz": 0.0}


def _save_cache(path, data):
    """Write a dict to a local JSON file. Never lets a cache-write error
    crash the program -- worst case, we just don't have a cache this time."""
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def _load_cache(path):
    """Read a dict back from a local JSON file. Returns None if the file
    doesn't exist or is corrupted."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fetch_usgs():
    """
    Returns a dict: {"magnitude": float, "depth": float,
                      "latitude": float, "longitude": float}
    Picks the HIGHEST-magnitude earthquake in the last 24 hours, since
    that's the most visually/dramatically interesting one to map to art.
    """
    try:
        response = requests.get(USGS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])

        if not features:
            raise ValueError("USGS returned zero earthquakes for this window")

        # Each feature = one earthquake. properties.mag = magnitude,
        # geometry.coordinates = [longitude, latitude, depth_km].
        best = max(features, key=lambda f: (f["properties"].get("mag") or -999))

        mag = best["properties"].get("mag")
        lon, lat, depth = best["geometry"]["coordinates"]

        if mag is None:
            raise ValueError("Strongest USGS feature had no magnitude value")

        result = {
            "magnitude": float(mag),
            "depth": float(depth),
            "latitude": float(lat),
            "longitude": float(lon),
        }
        _save_cache(CACHE_USGS_PATH, result)
        return result

    except (requests.RequestException, ValueError, KeyError, TypeError) as err:
        print(f"[api_client] USGS live fetch failed ({err}); trying cache...")
        cached = _load_cache(CACHE_USGS_PATH)
        if cached is not None:
            print("[api_client] Using cached USGS data.")
            return cached
        print("[api_client] No USGS cache available; using hard default values.")
        return dict(USGS_HARD_DEFAULT)


def _most_recent_value(records, field_name):
    """
    NOAA's RTSW feeds return a list of reading dicts, newest first, mixing
    several spacecraft (SOLAR1, IMAP, ACE...). Each record has an "active"
    flag marking which spacecraft is the officially designated source at
    that moment. We prefer the newest ACTIVE reading with a real number;
    if none exists, we fall back to the newest reading from ANY spacecraft
    that has a real number for that field.
    """
    fallback_value = None
    for record in records:
        value = record.get(field_name)
        if value is None:
            continue
        if fallback_value is None:
            fallback_value = value  # newest non-null value seen so far, any spacecraft
        if record.get("active") is True:
            return float(value)  # best case: newest reading from the active spacecraft
    if fallback_value is not None:
        return float(fallback_value)
    return None


def fetch_noaa():
    """
    Returns a dict: {"speed": float, "bz": float}
    """
    result = {}

    # --- Solar wind speed ---
    try:
        response = requests.get(NOAA_WIND_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        records = response.json()
        speed_value = _most_recent_value(records, "proton_speed")

        if speed_value is None:
            raise ValueError("No usable proton_speed reading found")

        result["speed"] = speed_value

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as err:
        print(f"[api_client] NOAA wind (speed) fetch failed ({err}); will fall back.")
        result["speed"] = None

    # --- Bz (magnetic field, GSM) ---
    try:
        response = requests.get(NOAA_MAG_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        records = response.json()
        bz_value = _most_recent_value(records, "bz_gsm")

        if bz_value is None:
            raise ValueError("No usable bz_gsm reading found")

        result["bz"] = bz_value

    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as err:
        print(f"[api_client] NOAA mag (Bz) fetch failed ({err}); will fall back.")
        result["bz"] = None

    # If BOTH live calls worked, cache and return.
    if result["speed"] is not None and result["bz"] is not None:
        _save_cache(CACHE_NOAA_PATH, result)
        return result

    # Otherwise fall back to cache, then hard defaults -- but only for
    # whichever piece is missing, so a partial live success isn't wasted.
    cached = _load_cache(CACHE_NOAA_PATH)
    if result["speed"] is None:
        result["speed"] = (cached or {}).get("speed", NOAA_HARD_DEFAULT["speed"])
    if result["bz"] is None:
        result["bz"] = (cached or {}).get("bz", NOAA_HARD_DEFAULT["bz"])

    return result


def get_all_data():
    """Convenience function: fetches both sources and returns one combined dict.
    This is the function main.py will actually call."""
    usgs = fetch_usgs()
    noaa = fetch_noaa()
    return {
        "magnitude": usgs["magnitude"],
        "depth": usgs["depth"],
        "latitude": usgs["latitude"],
        "longitude": usgs["longitude"],
        "solar_wind_speed": noaa["speed"],
        "bz": noaa["bz"],
    }


if __name__ == "__main__":
    print("Fetching live data...")
    data = get_all_data()
    print("\nFinal combined data dict:")
    for key, value in data.items():
        print(f"  {key}: {value}")