"""
Heartbreak.exe - Mutation Engine (mutation_engine.py)

L1 -- SEMANTIC COLOR MUTATION ONLY.

Pipeline position:

    api_client3.py -> parameter_mapper.py -> region_mapper.py
        -> mutation_engine.py (THIS MODULE) -> renderer.py

Inputs consumed:
    - the original RGB image
    - the Step 4 exact per-pixel region label map (image_01_region_labels.png)
    - the Step 4 region JSON (image_01_regions.json) -- the per-region
      execution/safety authority at runtime
    - the mutation plan produced by region_mapper.build_mutation_plan()

DATA CONTRACT (clarified):
region_mapper.py's mutation plan addresses regions by SEMANTIC CATEGORY
NAME as "region_id" (e.g. "sky_and_clouds"), not by a single Step 4
integer label. A semantic region_id is resolved as:

    mutation.region_id == step4_region_entry["semantic_label"]

A single semantic mutation can therefore match MANY Step 4 integer
`region_label` values. Each matching Step 4 entry's OWN mutation_safety
is checked independently -- eligible entries are mutated, ineligible
ones are skipped, and neither outcome fails the whole mutation unless
NO Step 4 region matches the semantic label at all.

This module does NOT:
    - implement L2 (lighting), L3 (atmosphere), or L4 (shape)
    - fetch API data, compute artistic parameters, or build mutation plans
    - use bounding boxes, reconstructed masks, or morphology
    - touch any pixel outside the resolved, safety-filtered union mask
    - use randomness, timestamps, or UUIDs
"""

import json

import cv2
import numpy as np
from PIL import Image


# ============================================================================
# CONSTANTS
# ============================================================================

VALID_MUTATION_TYPES = {"color", "light", "atmosphere", "shape"}
COLOR_ELIGIBLE_SAFETY = {"color_safe", "color+light_safe"}


class MutationEngineError(ValueError):
    """Raised when input data or the mutation plan is invalid/malformed."""


# ============================================================================
# LOADERS
# ============================================================================

def load_image(path):
    """Load an image as an RGB uint8 numpy array (alpha dropped, if any)."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_region_label_map(path):
    """Load the Step 4 exact per-pixel label map as a 2D integer array."""
    arr = np.array(Image.open(path))
    if arr.ndim != 2:
        raise MutationEngineError(
            f"Region label map must be single-channel, got shape {arr.shape}"
        )
    return arr


def load_region_data(path):
    """Load the Step 4 region JSON (a list of region entries)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise MutationEngineError("Step 4 region JSON must be a list of region entries")
    return data


# ============================================================================
# SEMANTIC INDEX
# ============================================================================

def _build_semantic_index(region_data):
    """
    Group Step 4 region entries by semantic_label, since region_mapper.py
    addresses regions by semantic category name rather than by individual
    Step 4 integer region_label.
    """
    index = {}
    for entry in region_data:
        for field in ("semantic_label", "mutation_safety", "region_label"):
            if field not in entry:
                raise MutationEngineError(
                    f"Step 4 region entry missing required field {field!r}: {entry}"
                )
        label = entry["region_label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise MutationEngineError(
                f"Step 4 region_label must be an int, got {label!r} in: {entry}"
            )
        index.setdefault(entry["semantic_label"], []).append(entry)
    return index


# ============================================================================
# VALIDATION
# ============================================================================

def validate_inputs(image, label_map, region_data, mutation_plan):
    """
    Validate every input before any pixel is touched. Raises
    MutationEngineError with a clear message on any problem.
    Returns the semantic index (semantic_label -> [step4 entries]) built
    from region_data, for reuse by the caller.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise MutationEngineError(f"Image must be RGB (H,W,3), got shape {image.shape}")

    if label_map.shape != image.shape[:2]:
        raise MutationEngineError(
            f"Label map dimensions {label_map.shape} != image dimensions {image.shape[:2]}"
        )

    if not np.issubdtype(label_map.dtype, np.integer):
        raise MutationEngineError(
            f"Label map must be integer-valued, got dtype {label_map.dtype}"
        )

    semantic_index = _build_semantic_index(region_data)

    if not isinstance(mutation_plan, dict) or "mutations" not in mutation_plan:
        raise MutationEngineError("mutation_plan must be a dict containing a 'mutations' list")

    mutations = mutation_plan["mutations"]
    if not isinstance(mutations, list):
        raise MutationEngineError("mutation_plan['mutations'] must be a list")

    seen_color_region_ids = set()

    for m in mutations:
        if not isinstance(m, dict):
            raise MutationEngineError(f"Each mutation must be a dict, got {m!r}")

        region_id = m.get("region_id")
        mutation_type = m.get("mutation")
        strength = m.get("strength")

        if not isinstance(region_id, str) or not region_id:
            raise MutationEngineError(f"Mutation region_id must be a non-empty string: {m!r}")

        if mutation_type not in VALID_MUTATION_TYPES:
            raise MutationEngineError(f"Unsupported mutation type {mutation_type!r} in: {m!r}")

        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise MutationEngineError(f"Mutation strength must be numeric: {m!r}")
        if not (0.0 <= strength <= 1.0):
            raise MutationEngineError(f"Mutation strength out of [0,1]: {m!r}")

        if mutation_type != "color":
            # L2/L3/L4 -- out of scope for this engine.
            continue

        parameters = m.get("parameters")
        if not isinstance(parameters, dict):
            raise MutationEngineError(f"Color mutation missing 'parameters' dict: {m!r}")

        hue_shift = parameters.get("hue_shift")
        saturation_scale = parameters.get("saturation_scale")

        if isinstance(hue_shift, bool) or not isinstance(hue_shift, (int, float)):
            raise MutationEngineError(f"hue_shift must be numeric: {m!r}")
        if isinstance(saturation_scale, bool) or not isinstance(saturation_scale, (int, float)):
            raise MutationEngineError(f"saturation_scale must be numeric: {m!r}")

        if region_id in seen_color_region_ids:
            raise MutationEngineError(f"Duplicate color mutation for region_id {region_id!r}")
        seen_color_region_ids.add(region_id)

        if region_id not in semantic_index:
            raise MutationEngineError(
                f"Color mutation refers to semantic label {region_id!r} which has no "
                f"corresponding Step 4 regions in the region JSON"
            )

    return semantic_index


# ============================================================================
# RESOLUTION: semantic region_id -> eligible / skipped Step 4 labels
# ============================================================================

def _resolve_color_mutation_labels(region_id, semantic_index):
    """
    Resolve a mutation's semantic region_id into the Step 4 integer
    region_label values that are eligible for color mutation, per each
    matching Step 4 entry's OWN mutation_safety (the Step 4 JSON is the
    runtime execution authority, not the knowledge-JSON-level safety
    already baked into the plan).
    """
    matching_entries = semantic_index[region_id]
    eligible = []
    skipped = []

    for entry in matching_entries:
        label = entry["region_label"]
        safety = entry["mutation_safety"]
        if safety in COLOR_ELIGIBLE_SAFETY:
            eligible.append(label)
        else:
            skipped.append({"region_label": label, "reason": f"mutation_safety={safety}"})

    return matching_entries, eligible, skipped


# ============================================================================
# L1 COLOR TRANSFORM
# ============================================================================

def _hue_saturation_transform(image, hue_shift, saturation_scale):
    """
    Whole-image RGB -> HSV -> (hue shift + saturation scale) -> RGB.
    Deterministic and purely per-pixel: no neighbor access, no
    interpolation/resampling. Value (V) is left untouched to avoid
    brightness/intensity changes, per spec.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h_channel = hsv[:, :, 0].astype(np.int32)
    s_channel = hsv[:, :, 1].astype(np.float64)
    v_channel = hsv[:, :, 2]

    # OpenCV's 8-bit hue channel is [0, 179] representing 0-360 degrees.
    hue_shift_cv = hue_shift * (179.0 / 360.0)
    new_h = np.mod(h_channel + np.round(hue_shift_cv), 180).astype(np.uint8)

    new_s = np.clip(np.round(s_channel * saturation_scale), 0, 255).astype(np.uint8)

    transformed_hsv = np.dstack([new_h, new_s, v_channel]).astype(np.uint8)
    return cv2.cvtColor(transformed_hsv, cv2.COLOR_HSV2RGB)


def apply_color_mutation(image, label_map, eligible_labels, hue_shift, saturation_scale, strength):
    """
    Return (new_image, affected_pixel_count). new_image is a copy of
    `image` with the color transform blended in, restricted to exactly
    the pixels where label_map is one of eligible_labels. Every other
    pixel is byte-identical to the input.
    """
    output = image.copy()

    if not eligible_labels:
        return output, 0

    mask = np.isin(label_map, eligible_labels)
    affected = int(np.count_nonzero(mask))

    if affected == 0 or strength == 0:
        return output, affected

    transformed = _hue_saturation_transform(image, hue_shift, saturation_scale)

    blended = image.astype(np.float64) * (1.0 - strength) + transformed.astype(np.float64) * strength
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    output[mask] = blended[mask]
    return output, affected


# ============================================================================
# ORCHESTRATION
# ============================================================================

def apply_mutation_plan(image, label_map, region_data, mutation_plan):
    """
    Validate inputs, then apply every L1 color mutation in the plan.
    Non-color mutations (light/atmosphere/shape) are present in
    region_mapper.py's plan but are intentionally not executed here.

    Returns (output_image, report). report is a list of dicts, one per
    color mutation, describing resolution, eligibility, and pixel counts.
    """
    semantic_index = validate_inputs(image, label_map, region_data, mutation_plan)

    output = image.copy()
    report = []

    for m in mutation_plan["mutations"]:
        if m["mutation"] != "color":
            continue

        region_id = m["region_id"]
        strength = m["strength"]
        hue_shift = m["parameters"]["hue_shift"]
        saturation_scale = m["parameters"]["saturation_scale"]

        matching_entries, eligible, skipped = _resolve_color_mutation_labels(
            region_id, semantic_index
        )

        output, affected = apply_color_mutation(
            output, label_map, eligible, hue_shift, saturation_scale, strength
        )

        report.append({
            "semantic_region_id": region_id,
            "matching_step4_labels": [e["region_label"] for e in matching_entries],
            "eligible_labels": eligible,
            "skipped_labels": skipped,
            "strength": strength,
            "hue_shift": hue_shift,
            "saturation_scale": saturation_scale,
            "affected_pixel_count": affected,
        })

    return output, report


# ============================================================================
# SAVING
# ============================================================================

def save_image(image, path):
    Image.fromarray(image, mode="RGB").save(path)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":

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

    # ------------------------------------------------------------------
    # Synthetic fixture: small, fully controlled image/label map/regions.
    # ------------------------------------------------------------------
    #   labels layout (10x10):
    #     1 1 1 . . . . . . .
    #     1 1 1 . . . . . . .
    #     1 1 1 . . . . . . .
    #     . . . 2 2 . . . . .
    #     . . . 2 2 . . . . .
    #     . . . . . 3 3 . . .
    #     . . . . . 3 3 . . .
    #     . . . . . . . 4 4 .
    #     . . . . . . . 4 4 .
    #     . . . . . . . . . 5
    #
    #   1 = "sky"     color_safe
    #   2 = "roof"    exclude
    #   3 = "rain"    atmosphere_only
    #   4 = "windows" color+light_safe  (eligible instance of "windows")
    #   5 = "windows" shape_sensitive   (ineligible instance of "windows")

    def make_fixture():
        rng_img = np.zeros((10, 10, 3), dtype=np.uint8)
        # Give distinct, saturated colors so hue/sat shifts are visible.
        rng_img[:, :] = [40, 120, 200]          # base bluish color everywhere
        rng_img[0:3, 0:3] = [30, 100, 220]       # sky block
        rng_img[3:5, 3:5] = [60, 60, 60]         # roof block
        rng_img[5:7, 5:7] = [90, 200, 90]        # rain block
        rng_img[7:9, 7:9] = [220, 60, 60]        # windows-eligible block
        rng_img[9:10, 9:10] = [200, 200, 40]     # windows-ineligible block

        label_map = np.zeros((10, 10), dtype=np.uint16)
        label_map[0:3, 0:3] = 1
        label_map[3:5, 3:5] = 2
        label_map[5:7, 5:7] = 3
        label_map[7:9, 7:9] = 4
        label_map[9:10, 9:10] = 5

        region_data = [
            {"region_id": 1, "semantic_label": "sky", "mutation_safety": "color_safe", "region_label": 1},
            {"region_id": 2, "semantic_label": "roof", "mutation_safety": "exclude", "region_label": 2},
            {"region_id": 3, "semantic_label": "rain", "mutation_safety": "atmosphere_only", "region_label": 3},
            {"region_id": 4, "semantic_label": "windows", "mutation_safety": "color+light_safe", "region_label": 4},
            {"region_id": 5, "semantic_label": "windows", "mutation_safety": "shape_sensitive", "region_label": 5},
        ]
        return rng_img, label_map, region_data

    image, label_map, region_data = make_fixture()

    full_plan = {
        "mutations": [
            {"region_id": "sky", "mutation": "color", "strength": 0.8,
             "parameters": {"hue_shift": 90.0, "saturation_scale": 1.3}},
            {"region_id": "windows", "mutation": "color", "strength": 0.6,
             "parameters": {"hue_shift": -45.0, "saturation_scale": 0.6}},
        ]
    }

    # TEST 1 -- basic color mutation executes
    out1, report1 = apply_mutation_plan(image, label_map, region_data, full_plan)
    check("TEST1 basic color mutation executes", out1 is not None and len(report1) == 2)

    # TEST 2 -- output dimensions match input
    check("TEST2 output dimensions match input", out1.shape == image.shape)

    # TEST 3 -- determinism
    out1_again, _ = apply_mutation_plan(image, label_map, region_data, full_plan)
    check("TEST3 determinism (identical plan -> identical output)", np.array_equal(out1, out1_again))

    # TEST 4 -- zero strength -> output identical to input
    zero_plan = {"mutations": [
        {"region_id": "sky", "mutation": "color", "strength": 0.0,
         "parameters": {"hue_shift": 90.0, "saturation_scale": 1.5}},
    ]}
    out_zero, _ = apply_mutation_plan(image, label_map, region_data, zero_plan)
    check("TEST4 zero strength -> output byte-identical to input", np.array_equal(out_zero, image))

    # TEST 5 -- empty plan -> output identical to input
    empty_plan = {"mutations": []}
    out_empty, report_empty = apply_mutation_plan(image, label_map, region_data, empty_plan)
    check("TEST5 empty plan -> output byte-identical to input", np.array_equal(out_empty, image))
    check("TEST5b empty plan -> empty report", report_empty == [])

    # TEST 6 -- exact pixel isolation for a single-region mutation
    sky_only_plan = {"mutations": [
        {"region_id": "sky", "mutation": "color", "strength": 0.9,
         "parameters": {"hue_shift": 120.0, "saturation_scale": 1.4}},
    ]}
    out_sky, _ = apply_mutation_plan(image, label_map, region_data, sky_only_plan)
    changed = np.any(out_sky != image, axis=2)
    check(
        "TEST6 every changed pixel belongs to label==1 (sky)",
        bool(np.all(label_map[changed] == 1)) if changed.any() else True,
    )
    check(
        "TEST6b every pixel with label!=1 is unchanged",
        np.array_equal(out_sky[label_map != 1], image[label_map != 1]),
    )

    # TEST 7 -- excluded region ("roof") can never be mutated
    roof_plan = {"mutations": [
        {"region_id": "roof", "mutation": "color", "strength": 1.0,
         "parameters": {"hue_shift": 180.0, "saturation_scale": 2.0}},
    ]}
    out_roof, report_roof = apply_mutation_plan(image, label_map, region_data, roof_plan)
    check("TEST7 excluded region pixels are unchanged", np.array_equal(out_roof, image))
    check("TEST7b excluded region reported with 0 eligible labels",
          report_roof[0]["eligible_labels"] == [] and report_roof[0]["affected_pixel_count"] == 0)

    # TEST 8 -- atmosphere_only region ("rain") cannot receive color mutation
    rain_plan = {"mutations": [
        {"region_id": "rain", "mutation": "color", "strength": 1.0,
         "parameters": {"hue_shift": 180.0, "saturation_scale": 2.0}},
    ]}
    out_rain, report_rain = apply_mutation_plan(image, label_map, region_data, rain_plan)
    check("TEST8 atmosphere_only region pixels are unchanged", np.array_equal(out_rain, image))
    check("TEST8b atmosphere_only region reported with 0 eligible labels",
          report_rain[0]["eligible_labels"] == [])

    # TEST 9 -- color actually changes for a valid non-zero mutation
    check(
        "TEST9 hue/sat shift actually changes some sky pixels",
        not np.array_equal(out_sky[label_map == 1], image[label_map == 1]),
    )

    # TEST 10 -- outside-mask pixel preservation across a multi-region plan,
    # including a partially-eligible semantic category ("windows": one
    # eligible instance, one shape_sensitive instance that must be skipped).
    changed_full = np.any(out1 != image, axis=2)
    eligible_union = np.isin(label_map, [1, 4])  # sky (1) + windows-eligible (4) only
    check(
        "TEST10 no changed pixels outside the union of eligible labels",
        bool(np.all(eligible_union[changed_full])) if changed_full.any() else True,
    )
    check(
        "TEST10b windows-ineligible block (label 5, shape_sensitive) untouched",
        np.array_equal(out1[label_map == 5], image[label_map == 5]),
    )

    # Bonus: unresolvable semantic label raises a clear error rather than
    # silently doing nothing.
    bad_plan = {"mutations": [
        {"region_id": "nonexistent_category", "mutation": "color", "strength": 0.5,
         "parameters": {"hue_shift": 10.0, "saturation_scale": 1.0}},
    ]}
    try:
        apply_mutation_plan(image, label_map, region_data, bad_plan)
        check("BONUS unresolvable semantic label raises MutationEngineError", False)
    except MutationEngineError as e:
        check("BONUS unresolvable semantic label raises MutationEngineError",
              "nonexistent_category" in str(e))

    # Bonus: duplicate color mutation for the same region_id raises.
    dup_plan = {"mutations": [
        {"region_id": "sky", "mutation": "color", "strength": 0.5,
         "parameters": {"hue_shift": 10.0, "saturation_scale": 1.0}},
        {"region_id": "sky", "mutation": "color", "strength": 0.3,
         "parameters": {"hue_shift": 20.0, "saturation_scale": 1.1}},
    ]}
    try:
        apply_mutation_plan(image, label_map, region_data, dup_plan)
        check("BONUS duplicate region_id color mutation raises MutationEngineError", False)
    except MutationEngineError as e:
        check("BONUS duplicate region_id color mutation raises MutationEngineError",
              "Duplicate" in str(e))

    print(f"\nSynthetic fixture tests: {passed} passed, {failed} failed")

    # ------------------------------------------------------------------
    # VISUAL INTEGRATION TEST -- real Image 1 files
    # ------------------------------------------------------------------
    import os

    try:
        import parameter_mapper
        import region_mapper

        real_image = load_image("image_01_cropped.png")
        real_label_map = load_region_label_map("output_step4/image_01_region_labels.png")
        real_region_data = load_region_data("output_step4/image_01_regions.json")

        with open("knowledge/image_01.json") as f:
            knowledge = json.load(f)

        # Realistic scientific input -> artistic params (same style used in
        # earlier pipeline stages), producing a visible but moderate mutation.
        api_data = {
            "magnitude": 6.5, "depth": 12.0, "latitude": 34.0522, "longitude": -118.2437,
            "solar_wind_density": 25.0, "solar_wind_temperature": 255000.0,
            "bz": -12.5, "by": 10.0, "bt": 15.0, "kp": 4.5,
        }
        artistic_params = parameter_mapper.map_api_to_visuals(api_data)

        available_labels = {r["semantic_label"] for r in real_region_data}
        plan = region_mapper.build_mutation_plan(knowledge, artistic_params, available_labels)

        real_output, real_report = apply_mutation_plan(
            real_image, real_label_map, real_region_data, plan
        )

        os.makedirs("output_mutation_test", exist_ok=True)
        save_image(real_output, "output_mutation_test/image_01_L1_color.png")

        diff = np.abs(real_output.astype(np.int16) - real_image.astype(np.int16))
        diff_vis = np.clip(diff.sum(axis=2) * 3, 0, 255).astype(np.uint8)
        Image.fromarray(diff_vis, mode="L").save("output_mutation_test/image_01_L1_difference.png")

        print("\n--- Visual integration test (Image 1) ---")
        color_mutations_in_plan = [mm for mm in plan["mutations"] if mm["mutation"] == "color"]
        print(f"Color mutations in plan: {len(color_mutations_in_plan)}")
        for r in real_report:
            print(
                f"  region_id={r['semantic_region_id']!r} "
                f"matching={r['matching_step4_labels']} "
                f"eligible={r['eligible_labels']} "
                f"skipped={[s['region_label'] for s in r['skipped_labels']]} "
                f"strength={r['strength']} affected_pixels={r['affected_pixel_count']}"
            )

        sky_report = next((r for r in real_report if r["semantic_region_id"] == "sky_and_clouds"), None)
        windows_report = next((r for r in real_report if r["semantic_region_id"] == "city_windows"), None)

        check("VISUAL output dims match source (1200x547)", real_output.shape[:2] == (547, 1200))
        check("VISUAL sky_and_clouds mutation present and affected pixels > 0",
              sky_report is not None and sky_report["affected_pixel_count"] > 0)
        if windows_report is not None:
            check("VISUAL city_windows mutation present and affected pixels > 0",
                  windows_report["affected_pixel_count"] > 0)

        # Confirm exclude/atmosphere_only regions never appear as eligible anywhere.
        excluded_semantic_labels = {
            e["semantic_label"] for e in real_region_data
            if e["mutation_safety"] in ("exclude", "atmosphere_only")
        }
        exclude_violation = False
        for r in real_report:
            if r["semantic_region_id"] in excluded_semantic_labels and r["eligible_labels"]:
                exclude_violation = True
        check("VISUAL no exclude/atmosphere_only semantic category has eligible labels",
              not exclude_violation)

        print("\nSaved: output_mutation_test/image_01_L1_color.png")
        print("Saved: output_mutation_test/image_01_L1_difference.png")

    except FileNotFoundError as e:
        print(f"\n[SKIPPED] Visual integration test -- missing real project file: {e}")

    if failed:
        raise SystemExit(1)