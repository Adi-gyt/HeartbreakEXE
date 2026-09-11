"""
Heartbreak.exe - API Client

Fetches live Earth and space-weather data and converts it into
clean numeric values for renderer.py.

Data sources:
    USGS  -> earthquake data
    NOAA  -> solar wind, magnetic field, geomagnetic Kp

Reliability:
    LIVE -> CACHE -> DEFAULT

The renderer only needs to call:

    data = get_all_data()

and receives one clean dictionary of numeric values.
"""

import json
import math
import os

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_USGS_PATH = os.path.join(SCRIPT_DIR, "cache_usgs.json")
CACHE_WIND_PATH = os.path.join(SCRIPT_DIR, "cache_noaa_wind.json")
CACHE_MAG_PATH = os.path.join(SCRIPT_DIR, "cache_noaa_mag.json")
CACHE_KP_PATH = os.path.join(SCRIPT_DIR, "cache_noaa_kp.json")

REQUEST_TIMEOUT_SECONDS = 5


# ============================================================================
# API URLS
# ============================================================================

USGS_URL = (
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
    "all_day.geojson"
)

NOAA_WIND_URL = (
    "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
)

NOAA_MAG_URL = (
    "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
)

NOAA_KP_URL = (
    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
)


# ============================================================================
# SAFE DEFAULTS
#
# These are used ONLY when both live data and cache are unavailable.
# They are not used to manufacture variation.
# ============================================================================

USGS_DEFAULT = {
    "magnitude": 2.5,
    "depth": 10.0,
    "latitude": 0.0,
    "longitude": 0.0,
}

WIND_DEFAULT = {
    "solar_wind_speed": 400.0,
    "solar_wind_density": 5.0,
    "solar_wind_temperature": 100000.0,
}

MAG_DEFAULT = {
    "bz": 0.0,
    "by": 0.0,
    "bt": 0.0,
}

KP_DEFAULT = {
    "kp": 0.0,
}


# ============================================================================
# CACHE HELPERS
# ============================================================================

def _save_cache(path, data):
    """
    Save a dictionary to JSON.

    Cache errors are intentionally ignored. A cache failure should
    never crash the actual application.
    """
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except (OSError, TypeError):
        pass


def _load_cache(path):
    """
    Load a JSON cache.

    Returns:
        dict if the cache is valid
        None if it doesn't exist or is corrupted
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return None


# ============================================================================
# VALUE VALIDATION
# ============================================================================

def _to_float(value):
    """
    Convert a value into a usable finite float.

    Returns None for missing, invalid, NaN or infinite values.
    """
    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def _most_recent_value(records, field_name):
    """
    Find the newest usable value for a field in a NOAA record list.

    NOAA RTSW feeds can contain readings from multiple spacecraft.

    We prefer the newest ACTIVE spacecraft reading.
    If there is no active reading, we use the newest valid reading
    from any spacecraft.
    """
    if not isinstance(records, list):
        return None

    fallback = None

    for record in records:
        if not isinstance(record, dict):
            continue

        value = _to_float(record.get(field_name))

        if value is None:
            continue

        if fallback is None:
            fallback = value

        if record.get("active") is True:
            return value

    return fallback


# ============================================================================
# USGS EARTHQUAKE
# ============================================================================

def fetch_usgs():
    """
    Fetch the strongest usable earthquake from the last 24 hours.

    Returns:
        (
            data_dict,
            status_dict,
            cache_updated
        )
    """

    default_data = dict(USGS_DEFAULT)

    try:
        response = requests.get(
            USGS_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("USGS response is not a JSON object")

        features = payload.get("features")

        if not isinstance(features, list) or not features:
            raise ValueError("USGS returned no earthquake features")

        usable = []

        for feature in features:
            if not isinstance(feature, dict):
                continue

            properties = feature.get("properties")
            geometry = feature.get("geometry")

            if not isinstance(properties, dict):
                continue

            if not isinstance(geometry, dict):
                continue

            magnitude = _to_float(properties.get("mag"))

            coordinates = geometry.get("coordinates")

            if not isinstance(coordinates, list) or len(coordinates) < 3:
                continue

            longitude = _to_float(coordinates[0])
            latitude = _to_float(coordinates[1])
            depth = _to_float(coordinates[2])

            if (
                magnitude is None
                or longitude is None
                or latitude is None
                or depth is None
            ):
                continue

            usable.append(
                {
                    "magnitude": magnitude,
                    "depth": depth,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            )

        if not usable:
            raise ValueError("USGS contained no usable earthquake")

        # Strongest earthquake is preferred because it is the most
        # meaningful event for the visual system.
        best = max(
            usable,
            key=lambda earthquake: earthquake["magnitude"],
        )

        _save_cache(CACHE_USGS_PATH, best)

        status = {
            "magnitude": "LIVE",
            "depth": "LIVE",
            "latitude": "LIVE",
            "longitude": "LIVE",
        }

        return best, status, True

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
    ) as error:

        print(
            f"[api_client] USGS live fetch failed: {error}"
        )

        cached = _load_cache(CACHE_USGS_PATH)

        if cached:
            result = {}

            for key in USGS_DEFAULT:
                value = _to_float(cached.get(key))

                if value is None:
                    result[key] = USGS_DEFAULT[key]
                else:
                    result[key] = value

            status = {}

            for key in USGS_DEFAULT:
                if _to_float(cached.get(key)) is None:
                    status[key] = "DEFAULT"
                else:
                    status[key] = "CACHE"

            return result, status, False

        return (
            dict(default_data),
            {key: "DEFAULT" for key in default_data},
            False,
        )


# ============================================================================
# NOAA SOLAR WIND
# ============================================================================

def fetch_noaa_wind():
    """
    Fetch:

        solar_wind_speed
        solar_wind_density
        solar_wind_temperature

    Each value is handled independently.

    If one value fails, the others remain live.
    """

    fields = {
        "solar_wind_speed": "proton_speed",
        "solar_wind_density": "proton_density",
        "solar_wind_temperature": "proton_temperature",
    }

    result = {}
    status = {}
    cache_updated = False

    try:
        response = requests.get(
            NOAA_WIND_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        records = response.json()

        if not isinstance(records, list):
            raise ValueError("NOAA wind response is not a list")

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as error:

        print(
            f"[api_client] NOAA wind fetch failed: {error}"
        )

        records = None

    cached = _load_cache(CACHE_WIND_PATH)

    if not isinstance(cached, dict):
        cached = {}

    for output_key, api_key in fields.items():

        value = None

        if records is not None:
            value = _most_recent_value(records, api_key)

        if value is not None:
            result[output_key] = value
            status[output_key] = "LIVE"

        else:
            cached_value = _to_float(cached.get(output_key))

            if cached_value is not None:
                result[output_key] = cached_value
                status[output_key] = "CACHE"

            else:
                result[output_key] = WIND_DEFAULT[output_key]
                status[output_key] = "DEFAULT"

    # Save all currently successful usable values.
    # This also preserves older cached values for variables that
    # failed during this particular request.
    cache_to_save = dict(cached)

    for key in fields:
        if status[key] == "LIVE":
            cache_to_save[key] = result[key]

    if any(status[key] == "LIVE" for key in fields):
        _save_cache(CACHE_WIND_PATH, cache_to_save)
        cache_updated = True

    return result, status, cache_updated


# ============================================================================
# NOAA MAGNETIC FIELD
# ============================================================================

def fetch_noaa_mag():
    """
    Fetch:

        bz
        by
        bt

    Each magnetic-field variable is handled independently.
    """

    fields = {
        "bz": "bz_gsm",
        "by": "by_gsm",
        "bt": "bt",
    }

    result = {}
    status = {}
    cache_updated = False

    try:
        response = requests.get(
            NOAA_MAG_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        records = response.json()

        if not isinstance(records, list):
            raise ValueError("NOAA magnetic response is not a list")

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as error:

        print(
            f"[api_client] NOAA magnetic fetch failed: {error}"
        )

        records = None

    cached = _load_cache(CACHE_MAG_PATH)

    if not isinstance(cached, dict):
        cached = {}

    for output_key, api_key in fields.items():

        value = None

        if records is not None:
            value = _most_recent_value(records, api_key)

        if value is not None:
            result[output_key] = value
            status[output_key] = "LIVE"

        else:
            cached_value = _to_float(cached.get(output_key))

            if cached_value is not None:
                result[output_key] = cached_value
                status[output_key] = "CACHE"

            else:
                result[output_key] = MAG_DEFAULT[output_key]
                status[output_key] = "DEFAULT"

    cache_to_save = dict(cached)

    for key in fields:
        if status[key] == "LIVE":
            cache_to_save[key] = result[key]

    if any(status[key] == "LIVE" for key in fields):
        _save_cache(CACHE_MAG_PATH, cache_to_save)
        cache_updated = True

    return result, status, cache_updated


# ============================================================================
# NOAA GEOMAGNETIC Kp
# ============================================================================

def fetch_kp():
    """
    Fetch the newest usable planetary Kp value.
    """

    try:
        response = requests.get(
            NOAA_KP_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        records = response.json()

        if not isinstance(records, list):
            raise ValueError("NOAA Kp response is not a list")

        kp = _most_recent_value(records, "kp_index")

        if kp is None:
            raise ValueError("No usable Kp value found")

        result = {"kp": kp}

        _save_cache(CACHE_KP_PATH, result)

        return result, {"kp": "LIVE"}, True

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as error:

        print(
            f"[api_client] NOAA Kp fetch failed: {error}"
        )

        cached = _load_cache(CACHE_KP_PATH)

        if isinstance(cached, dict):
            cached_kp = _to_float(cached.get("kp"))

            if cached_kp is not None:
                return (
                    {"kp": cached_kp},
                    {"kp": "CACHE"},
                    False,
                )

        return (
            dict(KP_DEFAULT),
            {"kp": "DEFAULT"},
            False,
        )


# ============================================================================
# MAIN RENDERER-FACING INTERFACE
# ============================================================================

def get_all_data():
    """
    Fetch all Earth and space-weather data.

    This is the ONLY function renderer.py needs to call.

    Returns exactly these numeric values:

        magnitude
        depth
        latitude
        longitude

        solar_wind_speed
        solar_wind_density
        solar_wind_temperature

        bz
        by
        bt

        kp
    """

    usgs, _, _ = fetch_usgs()
    wind, _, _ = fetch_noaa_wind()
    magnetic, _, _ = fetch_noaa_mag()
    kp, _, _ = fetch_kp()

    return {
        # Earthquake
        "magnitude": float(usgs["magnitude"]),
        "depth": float(usgs["depth"]),
        "latitude": float(usgs["latitude"]),
        "longitude": float(usgs["longitude"]),

        # Solar wind
        "solar_wind_speed": float(wind["solar_wind_speed"]),
        "solar_wind_density": float(wind["solar_wind_density"]),
        "solar_wind_temperature": float(
            wind["solar_wind_temperature"]
        ),

        # Magnetic field
        "bz": float(magnetic["bz"]),
        "by": float(magnetic["by"]),
        "bt": float(magnetic["bt"]),

        # Geomagnetic activity
        "kp": float(kp["kp"]),
    }


# ============================================================================
# DIAGNOSTIC MODE
# ============================================================================

def _print_value(label, value, status, unit=""):
    """
    Print one clean diagnostic value.
    """
    suffix = f" {unit}" if unit else ""
    print(f"  {label}: {value}{suffix:<12} [{status}]")


def _print_diagnostics():
    """
    Run a complete live fetch and print a human-readable report.
    """

    print("Fetching live data...")
    print()

    usgs, usgs_status, usgs_cache = fetch_usgs()
    wind, wind_status, wind_cache = fetch_noaa_wind()
    magnetic, magnetic_status, magnetic_cache = fetch_noaa_mag()
    kp, kp_status, kp_cache = fetch_kp()

    print("EARTHQUAKE")

    _print_value(
        "Magnitude",
        usgs["magnitude"],
        usgs_status["magnitude"],
    )

    _print_value(
        "Depth",
        usgs["depth"],
        usgs_status["depth"],
        "km",
    )

    _print_value(
        "Latitude",
        usgs["latitude"],
        usgs_status["latitude"],
    )

    _print_value(
        "Longitude",
        usgs["longitude"],
        usgs_status["longitude"],
    )

    print(
        f"  Cache updated: {'YES' if usgs_cache else 'NO'}"
    )

    print()
    print("SOLAR WIND")

    _print_value(
        "Speed",
        wind["solar_wind_speed"],
        wind_status["solar_wind_speed"],
        "km/s",
    )

    _print_value(
        "Density",
        wind["solar_wind_density"],
        wind_status["solar_wind_density"],
        "p/cm³",
    )

    _print_value(
        "Temperature",
        wind["solar_wind_temperature"],
        wind_status["solar_wind_temperature"],
        "K",
    )

    print(
        f"  Cache updated: {'YES' if wind_cache else 'NO'}"
    )

    print()
    print("MAGNETIC FIELD")

    _print_value(
        "Bz",
        magnetic["bz"],
        magnetic_status["bz"],
        "nT",
    )

    _print_value(
        "By",
        magnetic["by"],
        magnetic_status["by"],
        "nT",
    )

    _print_value(
        "Bt",
        magnetic["bt"],
        magnetic_status["bt"],
        "nT",
    )

    print(
        f"  Cache updated: {'YES' if magnetic_cache else 'NO'}"
    )

    print()
    print("GEOMAGNETIC")

    _print_value(
        "Kp",
        kp["kp"],
        kp_status["kp"],
    )

    print(
        f"  Cache updated: {'YES' if kp_cache else 'NO'}"
    )

    print()
    print("Final combined data dict:")

    final_data = {
        "magnitude": float(usgs["magnitude"]),
        "depth": float(usgs["depth"]),
        "latitude": float(usgs["latitude"]),
        "longitude": float(usgs["longitude"]),
        "solar_wind_speed": float(wind["solar_wind_speed"]),
        "solar_wind_density": float(wind["solar_wind_density"]),
        "solar_wind_temperature": float(
            wind["solar_wind_temperature"]
        ),
        "bz": float(magnetic["bz"]),
        "by": float(magnetic["by"]),
        "bt": float(magnetic["bt"]),
        "kp": float(kp["kp"]),
    }

    for key, value in final_data.items():
        print(f"  {key}: {value}")


# ============================================================================
# RUN DIRECTLY
# ============================================================================

if __name__ == "__main__":
    _print_diagnostics()