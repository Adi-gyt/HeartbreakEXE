"""
Heartbreak.exe - Region Mutation Planner (region_mapper.py)

Sits between parameter_mapper.py and mutation_engine.py in the pipeline:

    api_client3.py -> parameter_mapper.py -> region_mapper.py -> mutation_engine.py -> renderer.py

Given a region "knowledge" JSON for a reference image, the 8 artistic
parameters produced by parameter_mapper.py, and the set of regions/masks
actually available from the Step 4 extraction system, this module produces
a deterministic list of MUTATION INSTRUCTIONS describing which regions may
be mutated, how, and how strongly.

This module does NOT:
    - generate pixel masks
    - alter any image
    - perform actual color/lighting/shape math on pixels
    - use any randomness, timestamps, or UUIDs

It behaves as a safety-aware planner: it reads mutation_safety metadata and
only proposes mutations that are explicitly permitted for a given region.
"""

import json


# ============================================================================
# CONSTANTS
# ============================================================================

REQUIRED_ARTISTIC_PARAMS = {
    "hue_shift": (-180.0, 180.0),
    "saturation_scale": (0.5, 1.5),
    "light_angle": (0.0, 360.0),
    "light_warmth": (-1.0, 1.0),
    "intensity": (0.0, 1.0),
    "atmosphere_density": (0.0, 1.0),
    "reference_bucket": (0, 3),
    "deform_strength": (0.0, 1.0),
}

VALID_MUTATION_SAFETY = {
    "color_safe",
    "color+light_safe",
    "shape_sensitive",
    "atmosphere_only",
    "exclude",
}

VALID_MUTATION_TYPES = {"color", "light", "atmosphere", "shape"}

ALLOWED_MUTATION_PARAMS = {
    "color": {"hue_shift", "saturation_scale"},
    "light": {"light_angle", "light_warmth"},
    "atmosphere": {"atmosphere_density"},
    "shape": {"deform_strength"},
}

# Conservative damper applied when lighting is applied to a merely
# "color_safe" region rather than a "color+light_safe" region (Rule 3).
_CONSERVATIVE_LIGHT_FACTOR = 0.35


class RegionMapperError(ValueError):
    """Raised when input data to the region mapper is invalid or malformed."""


# ============================================================================
# SMALL HELPERS
# ============================================================================

def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def load_knowledge(path):
    """Load and return a region knowledge JSON file as a dict."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_artistic_params(artistic_params):
    if not isinstance(artistic_params, dict):
        raise RegionMapperError("artistic_params must be a dict")

    missing = [key for key in REQUIRED_ARTISTIC_PARAMS if key not in artistic_params]
    if missing:
        raise RegionMapperError(
            f"Missing required artistic parameters: {sorted(missing)}"
        )

    for key, (lo, hi) in REQUIRED_ARTISTIC_PARAMS.items():
        value = artistic_params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegionMapperError(f"artistic parameter '{key}' must be numeric")
        if not (lo <= value <= hi):
            raise RegionMapperError(
                f"artistic parameter '{key}'={value} out of range [{lo}, {hi}]"
            )


def _validate_mutation_safety(value, context=""):
    if value not in VALID_MUTATION_SAFETY:
        suffix = f" for region '{context}'" if context else ""
        raise RegionMapperError(
            f"Unknown mutation_safety value {value!r}{suffix}. "
            f"Expected one of {sorted(VALID_MUTATION_SAFETY)}"
        )


def _validate_output_mutations(mutations):
    for mutation in mutations:
        region_id = mutation.get("region_id")
        if not isinstance(region_id, str) or not region_id:
            raise RegionMapperError(f"Invalid region_id in mutation: {mutation}")

        mutation_type = mutation.get("mutation")
        if mutation_type not in VALID_MUTATION_TYPES:
            raise RegionMapperError(f"Invalid mutation type in: {mutation}")

        strength = mutation.get("strength")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise RegionMapperError(f"Invalid strength type in: {mutation}")
        if not (0.0 <= strength <= 1.0):
            raise RegionMapperError(f"Strength out of [0,1] in: {mutation}")

        parameters = mutation.get("parameters")
        if not isinstance(parameters, dict):
            raise RegionMapperError(f"Invalid parameters in: {mutation}")

        extra = set(parameters.keys()) - ALLOWED_MUTATION_PARAMS[mutation_type]
        if extra:
            raise RegionMapperError(
                f"Unexpected parameters {sorted(extra)} for mutation "
                f"type '{mutation_type}' in: {mutation}"
            )


# ============================================================================
# REGION AVAILABILITY (Step 4 extracted regions/masks)
# ============================================================================

def _available_region_labels(regions):
    """
    Normalize the `regions` input (Step 4 extracted regions/masks) into a
    set of identifiers we can check knowledge region_name/pattern_name
    values against.

    Accepts:
        - None / empty  -> returns None, meaning "no filtering applied"
          (every knowledge region is treated as having an available mask)
        - dict           -> keys are treated as available identifiers
        - list/tuple/set -> each item is either a string identifier, or a
          dict with a 'region_name', 'pattern_name', 'semantic_label', or
          'region_id' field used as the identifier

    This is intentionally conservative and non-fuzzy: it never guesses at
    matching differently-named labels. If the extraction system's naming
    does not line up with the knowledge JSON's region_name/pattern_name,
    that region simply will not be considered available.
    """
    if not regions:
        return None

    if isinstance(regions, dict):
        return {str(key) for key in regions.keys()}

    if isinstance(regions, (list, tuple, set)):
        labels = set()
        for entry in regions:
            if isinstance(entry, str):
                labels.add(entry)
            elif isinstance(entry, dict):
                identifier = (
                    entry.get("region_name")
                    or entry.get("pattern_name")
                    or entry.get("semantic_label")
                    or entry.get("region_id")
                )
                if identifier is not None:
                    labels.add(str(identifier))
            else:
                raise RegionMapperError(
                    f"Unsupported region entry type in `regions`: {type(entry)!r}"
                )
        return labels

    raise RegionMapperError("`regions` must be a dict, list, tuple, set, or None")


def _region_is_available(name, available_labels):
    if available_labels is None:
        return True
    return name in available_labels


# ============================================================================
# KNOWLEDGE ITERATION
# ============================================================================

def _iter_knowledge_regions(knowledge):
    """
    Yield (name, entry) for every semantic_region and pattern_region across
    all images listed in the knowledge JSON.
    """
    images = knowledge.get("images")
    if not isinstance(images, list) or not images:
        raise RegionMapperError("knowledge must contain a non-empty 'images' list")

    for image in images:
        for entry in image.get("semantic_regions", []) or []:
            name = entry.get("region_name")
            if not name:
                raise RegionMapperError("semantic_region entry missing 'region_name'")
            yield name, entry

        for entry in image.get("pattern_regions", []) or []:
            name = entry.get("pattern_name")
            if not name:
                raise RegionMapperError("pattern_region entry missing 'pattern_name'")
            yield name, entry


def _get_image_id(knowledge):
    images = knowledge.get("images")
    if isinstance(images, list) and images:
        return images[0].get("image_id", "unknown")
    return "unknown"


# ============================================================================
# ELIGIBILITY / STRENGTH CALCULATION (Rule 5, Rule 7)
# ============================================================================

def _eligibility_strength(entry):
    """
    Combine visual_importance, mutation_priority, and confidence into a base
    eligibility strength in [0, 1].

    confidence acts as a conservative damper: a low-confidence region never
    gets boosted just because its bbox or visual_importance is large (Rule 7).
    Missing fields fall back to a neutral 0.5 rather than crashing, since
    pattern_regions in practice do not always carry visual_importance /
    mutation_priority.
    """
    visual_importance = _clamp(float(entry.get("visual_importance", 0.5)))
    mutation_priority = _clamp(float(entry.get("mutation_priority", 0.5)))
    confidence = _clamp(float(entry.get("confidence", 0.5)))

    base = (visual_importance * 0.4) + (mutation_priority * 0.6)
    return _clamp(base * confidence)


def _final_strength(domain_base, intensity):
    """
    Rule 5: intensity is a global multiplier, never a mutation on its own.
    final_strength = domain_base * intensity, clamped to [0, 1].
    """
    return round(_clamp(_clamp(domain_base) * _clamp(intensity)), 6)


# ============================================================================
# CORE PLANNER
# ============================================================================

def build_mutation_plan(knowledge, artistic_params, regions=None):
    """
    Build a deterministic, safety-aware list of mutation instructions.

    Args:
        knowledge: parsed knowledge JSON dict for one reference image
                   (contains 'images' -> [{'semantic_regions', 'pattern_regions', ...}]).
        artistic_params: the 8-value dict produced by parameter_mapper.map_api_to_visuals.
        regions: the Step 4 extracted regions/masks (see _available_region_labels
                 for accepted shapes). Optional; if omitted, all knowledge
                 regions are treated as having an available mask.

    Returns:
        {
            "image_id": str,
            "reference_bucket": int,
            "mutations": [
                {"region_id": str, "mutation": str, "strength": float, "parameters": dict},
                ...
            ]
        }
    """
    if not isinstance(knowledge, dict):
        raise RegionMapperError("knowledge must be a dict")

    _validate_artistic_params(artistic_params)

    available_labels = _available_region_labels(regions)

    hue_shift = artistic_params["hue_shift"]
    saturation_scale = artistic_params["saturation_scale"]
    light_angle = artistic_params["light_angle"]
    light_warmth = artistic_params["light_warmth"]
    intensity = artistic_params["intensity"]
    atmosphere_density = artistic_params["atmosphere_density"]
    reference_bucket = artistic_params["reference_bucket"]
    deform_strength = artistic_params["deform_strength"]

    mutations = []

    for name, entry in _iter_knowledge_regions(knowledge):
        safety = entry.get("mutation_safety")
        _validate_mutation_safety(safety, context=name)

        # Rule 9: never touch a region explicitly marked exclude.
        if safety == "exclude":
            continue

        # Only plan mutations for regions Step 4 actually extracted a mask for.
        if not _region_is_available(name, available_labels):
            continue

        eligibility = _eligibility_strength(entry)

        # ---- Rule 1 & 2: color (hue_shift + saturation_scale) ----
        if safety in ("color_safe", "color+light_safe"):
            strength = _final_strength(eligibility, intensity)
            if strength > 0.0:
                mutations.append({
                    "region_id": name,
                    "mutation": "color",
                    "strength": strength,
                    "parameters": {
                        "hue_shift": round(hue_shift, 4),
                        "saturation_scale": round(saturation_scale, 4),
                    },
                })

        # ---- Rule 3: lighting ----
        if safety == "color+light_safe":
            strength = _final_strength(eligibility, intensity)
            if strength > 0.0:
                mutations.append({
                    "region_id": name,
                    "mutation": "light",
                    "strength": strength,
                    "parameters": {
                        "light_angle": round(light_angle, 4),
                        "light_warmth": round(light_warmth, 4),
                    },
                })
        elif safety == "color_safe":
            # "may affect color_safe only very conservatively"
            strength = _final_strength(eligibility * _CONSERVATIVE_LIGHT_FACTOR, intensity)
            if strength > 0.0:
                mutations.append({
                    "region_id": name,
                    "mutation": "light",
                    "strength": strength,
                    "parameters": {
                        "light_angle": round(light_angle, 4),
                        "light_warmth": round(light_warmth, 4),
                    },
                })

        # ---- Rule 4: atmosphere ----
        if safety == "atmosphere_only":
            domain_base = eligibility * atmosphere_density
            strength = _final_strength(domain_base, intensity)
            if strength > 0.0:
                mutations.append({
                    "region_id": name,
                    "mutation": "atmosphere",
                    "strength": strength,
                    "parameters": {
                        "atmosphere_density": round(atmosphere_density, 4),
                    },
                })

        # ---- Rule 6: shape / deformation ----
        if safety == "shape_sensitive":
            domain_base = eligibility * deform_strength
            strength = _final_strength(domain_base, intensity)
            if strength > 0.0:
                mutations.append({
                    "region_id": name,
                    "mutation": "shape",
                    "strength": strength,
                    "parameters": {
                        "deform_strength": round(deform_strength, 4),
                    },
                })

    _validate_output_mutations(mutations)

    return {
        "image_id": _get_image_id(knowledge),
        "reference_bucket": int(reference_bucket),
        "mutations": mutations,
    }


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":

    def _mock_knowledge():
        return {
            "dataset_version": "1.0",
            "images": [
                {
                    "image_id": "mock_image_01",
                    "semantic_regions": [
                        {
                            "region_name": "sky_and_clouds",
                            "visual_importance": 0.8,
                            "mutation_priority": 0.8,
                            "confidence": 0.95,
                            "mutation_safety": "color+light_safe",
                        },
                        {
                            "region_name": "midground_trees",
                            "visual_importance": 0.7,
                            "mutation_priority": 0.6,
                            "confidence": 0.9,
                            "mutation_safety": "color_safe",
                        },
                        {
                            "region_name": "overhang_roof",
                            "visual_importance": 0.75,
                            "mutation_priority": 0.1,
                            "confidence": 0.95,
                            "mutation_safety": "exclude",
                        },
                        {
                            "region_name": "bus_shelter_left",
                            "visual_importance": 0.85,
                            "mutation_priority": 0.2,
                            "confidence": 0.95,
                            "mutation_safety": "shape_sensitive",
                        },
                    ],
                    "pattern_regions": [
                        {
                            "pattern_name": "rain_streaks",
                            "confidence": 0.95,
                            "mutation_safety": "atmosphere_only",
                        },
                        {
                            "pattern_name": "power_lines",
                            "confidence": 0.95,
                            "mutation_safety": "exclude",
                        },
                        {
                            "pattern_name": "city_windows",
                            "confidence": 0.9,
                            "mutation_safety": "color+light_safe",
                        },
                    ],
                }
            ],
        }

    def _mock_artistic_params(**overrides):
        base = {
            "hue_shift": 83.0,
            "saturation_scale": 1.18,
            "light_angle": 210.0,
            "light_warmth": -0.3,
            "intensity": 0.75,
            "atmosphere_density": 0.6,
            "reference_bucket": 2,
            "deform_strength": 0.4,
        }
        base.update(overrides)
        return base

    def _mock_regions():
        # Availability filter matching the mock knowledge's region_name /
        # pattern_name values exactly.
        return [
            "sky_and_clouds", "midground_trees", "overhang_roof",
            "bus_shelter_left", "rain_streaks", "power_lines", "city_windows",
        ]

    passed = 0
    failed = 0

    def check(label, condition):
        global passed, failed
        if condition:
            print(f"[PASS] {label}")
            passed += 1
        else:
            print(f"[FAIL] {label}")
            failed += 1

    knowledge = _mock_knowledge()
    params = _mock_artistic_params()
    regions = _mock_regions()

    # 1. Basic plan generation works.
    plan = build_mutation_plan(knowledge, params, regions)
    check("plan has required top-level keys",
          {"image_id", "reference_bucket", "mutations"} <= set(plan.keys()))
    check("plan generated at least one mutation", len(plan["mutations"]) > 0)

    # 2. Every mutation has a valid region_id.
    known_names = {"sky_and_clouds", "midground_trees", "overhang_roof",
                    "bus_shelter_left", "rain_streaks", "power_lines", "city_windows"}
    check("every mutation has a valid region_id",
          all(isinstance(m["region_id"], str) and m["region_id"] in known_names
              for m in plan["mutations"]))

    # 3. Every mutation type is valid.
    check("every mutation type is valid",
          all(m["mutation"] in VALID_MUTATION_TYPES for m in plan["mutations"]))

    # 4. Every strength is between 0 and 1.
    check("every strength is within [0, 1]",
          all(0.0 <= m["strength"] <= 1.0 for m in plan["mutations"]))

    # 5. No "exclude" region receives a mutation.
    check("no exclude region receives a mutation",
          all(m["region_id"] not in ("overhang_roof", "power_lines")
              for m in plan["mutations"]))

    # 6. No "atmosphere_only" region receives color/light/shape.
    rain_mutations = [m for m in plan["mutations"] if m["region_id"] == "rain_streaks"]
    check("atmosphere_only region only receives atmosphere mutations",
          rain_mutations and all(m["mutation"] == "atmosphere" for m in rain_mutations))

    # 7. No non-shape-sensitive region receives shape.
    check("only shape_sensitive regions receive shape mutations",
          all(m["mutation"] != "shape" or m["region_id"] == "bus_shelter_left"
              for m in plan["mutations"]))

    # 8. Determinism.
    plan_again = build_mutation_plan(knowledge, params, regions)
    check("build_mutation_plan is deterministic", plan == plan_again)

    # 9. intensity=0 produces zero mutation strength or no mutations.
    zero_intensity_params = _mock_artistic_params(intensity=0.0)
    zero_intensity_plan = build_mutation_plan(knowledge, zero_intensity_params, regions)
    check("intensity=0 yields no mutations or all-zero strengths",
          len(zero_intensity_plan["mutations"]) == 0
          or all(m["strength"] == 0.0 for m in zero_intensity_plan["mutations"]))

    # 10. deform_strength=0 produces no shape mutations.
    zero_deform_params = _mock_artistic_params(deform_strength=0.0)
    zero_deform_plan = build_mutation_plan(knowledge, zero_deform_params, regions)
    check("deform_strength=0 yields no shape mutations",
          all(m["mutation"] != "shape" for m in zero_deform_plan["mutations"]))

    # 11. Missing required artistic parameters raises a clear error.
    incomplete_params = _mock_artistic_params()
    del incomplete_params["intensity"]
    try:
        build_mutation_plan(knowledge, incomplete_params, regions)
        check("missing artistic parameter raises RegionMapperError", False)
    except RegionMapperError as error:
        check("missing artistic parameter raises RegionMapperError", "intensity" in str(error))

    # 12. Unknown mutation safety values raise a clear error.
    bad_knowledge = _mock_knowledge()
    bad_knowledge["images"][0]["semantic_regions"][0]["mutation_safety"] = "totally_made_up"
    try:
        build_mutation_plan(bad_knowledge, params, regions)
        check("unknown mutation_safety raises RegionMapperError", False)
    except RegionMapperError as error:
        check("unknown mutation_safety raises RegionMapperError",
              "totally_made_up" in str(error))

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
