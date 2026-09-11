"""
Heartbreak.exe - Parameter Mapper (parameter_mapper.py)

Translates exactly 11 raw Earth and space-weather data fields into
8 artistic visual parameters for the rendering engine.

Mappings are strictly artistic and arbitrary, prioritizing visual 
aesthetics and stable behavior over scientific causation.
"""

NUM_REFERENCE_BUCKETS = 4

def _clamp(value, min_val, max_val):
    """Restricts a value to be strictly within min_val and max_val."""
    return max(min_val, min(value, max_val))

def map_api_to_visuals(api_data):
    """
    Transforms the 11-value API data dictionary into an 8-value
    visual parameter dictionary with guaranteed bounds.
    """
    
    # 1. HUE SHIFT (-180 to +180)
    # Driven by solar_wind_temperature (approx. baseline 10k to 500k)
    temp = _clamp(api_data.get("solar_wind_temperature", 100000.0), 10000.0, 500000.0)
    temp_norm = (temp - 10000.0) / 490000.0
    hue_shift = (temp_norm * 360.0) - 180.0

    # 2. SATURATION SCALE (0.5 to 1.5)
    # Driven by solar_wind_density (approx. baseline 0 to 50)
    density = _clamp(api_data.get("solar_wind_density", 5.0), 0.0, 50.0)
    density_norm = density / 50.0
    saturation_scale = 0.5 + (density_norm * 1.0)

    # 3. LIGHT ANGLE (0 to 360)
    # Driven by bz (approx. baseline -50 to +50)
    bz = _clamp(api_data.get("bz", 0.0), -50.0, 50.0)
    bz_norm = (bz + 50.0) / 100.0
    light_angle = bz_norm * 360.0

    # 4. LIGHT WARMTH (-1.0 to +1.0)
    # Driven by by (approx. baseline -50 to +50)
    by_val = _clamp(api_data.get("by", 0.0), -50.0, 50.0)
    light_warmth = by_val / 50.0

    # 5. INTENSITY (0.0 to 1.0)
    # Driven by bt (0 to 30) + kp (0 to 9)
    bt = _clamp(api_data.get("bt", 0.0), 0.0, 30.0)
    kp = _clamp(api_data.get("kp", 0.0), 0.0, 9.0)
    bt_norm = bt / 30.0
    kp_norm = kp / 9.0
    intensity = (bt_norm + kp_norm) / 2.0

    # 6. ATMOSPHERE DENSITY (0.0 to 1.0)
    # Driven by kp (0 to 9)
    atmosphere_density = kp_norm

    # 7. DEFORM STRENGTH (0.0 to 1.0)
    # Driven by earthquake magnitude (approx. 1.0 to 9.0)
    mag = _clamp(api_data.get("magnitude", 2.5), 1.0, 9.0)
    deform_strength = (mag - 1.0) / 8.0

    # 8. REFERENCE BUCKET (0 to NUM_REFERENCE_BUCKETS - 1)
    # Driven by a robust, stable deterministic integer-mixing hash method using both latitude and longitude
    lat = api_data.get("latitude", 0.0)
    lon = api_data.get("longitude", 0.0)
    
    # Scale to preserve precision and convert to 32-bit unsigned integers safely across platforms and signs
    lat_fixed = int(round(lat * 100000.0)) & 0xFFFFFFFF
    lon_fixed = int(round(lon * 100000.0)) & 0xFFFFFFFF

    # Bitwise integer hash mixer combining both components independently and thoroughly
    h = (lat_fixed * 374761393) ^ (lon_fixed * 668265261)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    
    reference_bucket = abs(h) % NUM_REFERENCE_BUCKETS

    return {
        "hue_shift": float(hue_shift),
        "saturation_scale": float(saturation_scale),
        "light_angle": float(light_angle),
        "light_warmth": float(light_warmth),
        "intensity": float(intensity),
        "atmosphere_density": float(atmosphere_density),
        "reference_bucket": int(reference_bucket),
        "deform_strength": float(deform_strength)
    }


if __name__ == "__main__":
    # Test 1: Construct a complete 11-value API data input dictionary
    test_api_data = {
        "magnitude": 6.5,
        "depth": 12.0,                  # Unused by mapper, but safely ignored
        "latitude": 34.0522,
        "longitude": -118.2437,
        "solar_wind_speed": 450.0,      # Unused by mapper, but safely ignored
        "solar_wind_density": 25.0,
        "solar_wind_temperature": 255000.0,
        "bz": -12.5,
        "by": 10.0,
        "bt": 15.0,
        "kp": 4.5
    }

    print("--- Input Data ---")
    for k, v in test_api_data.items():
        print(f"{k}: {v}")
    
    # Generate Output
    result = map_api_to_visuals(test_api_data)
    
    print("\n--- Mapped Artistic Output ---")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Test 2: Verify all expected output keys exist and count is exactly 8
    expected_keys = {
        "hue_shift", "saturation_scale", "light_angle", "light_warmth",
        "intensity", "atmosphere_density", "reference_bucket", "deform_strength"
    }
    assert set(result.keys()) == expected_keys, "Output keys do not match exactly!"
    assert len(result) == 8, "Output did not return exactly 8 parameters!"

    # Test 3: Output Bounds Validation
    assert -180.0 <= result["hue_shift"] <= 180.0, "hue_shift out of bounds!"
    assert 0.5 <= result["saturation_scale"] <= 1.5, "saturation_scale out of bounds!"
    assert 0.0 <= result["light_angle"] <= 360.0, "light_angle out of bounds!"
    assert -1.0 <= result["light_warmth"] <= 1.0, "light_warmth out of bounds!"
    assert 0.0 <= result["intensity"] <= 1.0, "intensity out of bounds!"
    assert 0.0 <= result["atmosphere_density"] <= 1.0, "atmosphere_density out of bounds!"
    assert 0 <= result["reference_bucket"] < NUM_REFERENCE_BUCKETS, "reference_bucket out of bounds!"
    assert 0.0 <= result["deform_strength"] <= 1.0, "deform_strength out of bounds!"

    # Test 4: Deterministic Behavior
    result_run_two = map_api_to_visuals(test_api_data)
    assert result == result_run_two, "Mapper is not deterministic across identical calls!"

    # Test 5: Reference Bucket Stability
    # Testing that it doesn't break or go out of bounds on weird edge-case coordinates
    edge_cases = [
        {"latitude": 0.0, "longitude": 0.0},
        {"latitude": -90.0, "longitude": 180.0},
        {"latitude": 90.0, "longitude": -180.0},
        {"latitude": 45.1234567, "longitude": -45.1234567},
    ]
    for i, coords in enumerate(edge_cases):
        bucket = map_api_to_visuals(coords)["reference_bucket"]
        assert 0 <= bucket < NUM_REFERENCE_BUCKETS, f"Edge case {i} failed bucket bounds check."
    
    print("\n[SUCCESS] All Heartbreak.exe mapping constraints validated successfully.")