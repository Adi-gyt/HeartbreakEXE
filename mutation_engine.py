"""
Heartbreak.exe - Mutation Engine (mutation_engine.py)

L1 -- SEMANTIC COLOR MUTATION.
L2 -- DIRECTIONAL LIGHTING MUTATION.

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
    - implement L3 (atmosphere) or L4 (shape)
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
# Per the L2 region contract, lighting is eligible for exactly the same
# Step 4 mutation_safety values as color: color_safe and color+light_safe.
LIGHT_ELIGIBLE_SAFETY = COLOR_ELIGIBLE_SAFETY
# Per the L3 region contract, atmosphere/texture is eligible ONLY for
# Step 4 regions whose own mutation_safety is atmosphere_only.
ATMOSPHERE_ELIGIBLE_SAFETY = {"atmosphere_only"}


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
    seen_light_region_ids = set()
    seen_atmosphere_region_ids = set()

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

        if mutation_type not in ("color", "light", "atmosphere"):
            # L4 -- out of scope for this engine.
            continue

        parameters = m.get("parameters")
        if not isinstance(parameters, dict):
            raise MutationEngineError(
                f"{mutation_type.capitalize()} mutation missing 'parameters' dict: {m!r}"
            )

        if mutation_type == "color":
            hue_shift = parameters.get("hue_shift")
            saturation_scale = parameters.get("saturation_scale")

            if isinstance(hue_shift, bool) or not isinstance(hue_shift, (int, float)):
                raise MutationEngineError(f"hue_shift must be numeric: {m!r}")
            if isinstance(saturation_scale, bool) or not isinstance(saturation_scale, (int, float)):
                raise MutationEngineError(f"saturation_scale must be numeric: {m!r}")

            if region_id in seen_color_region_ids:
                raise MutationEngineError(f"Duplicate color mutation for region_id {region_id!r}")
            seen_color_region_ids.add(region_id)

        elif mutation_type == "light":
            light_angle = parameters.get("light_angle")
            light_warmth = parameters.get("light_warmth")

            if isinstance(light_angle, bool) or not isinstance(light_angle, (int, float)):
                raise MutationEngineError(f"light_angle must be numeric: {m!r}")
            if isinstance(light_warmth, bool) or not isinstance(light_warmth, (int, float)):
                raise MutationEngineError(f"light_warmth must be numeric: {m!r}")

            if region_id in seen_light_region_ids:
                raise MutationEngineError(f"Duplicate light mutation for region_id {region_id!r}")
            seen_light_region_ids.add(region_id)

        else:  # "atmosphere"
            atmosphere_density = parameters.get("atmosphere_density")

            if isinstance(atmosphere_density, bool) or not isinstance(atmosphere_density, (int, float)):
                raise MutationEngineError(f"atmosphere_density must be numeric: {m!r}")

            if region_id in seen_atmosphere_region_ids:
                raise MutationEngineError(f"Duplicate atmosphere mutation for region_id {region_id!r}")
            seen_atmosphere_region_ids.add(region_id)

        if region_id not in semantic_index:
            raise MutationEngineError(
                f"{mutation_type.capitalize()} mutation refers to semantic label {region_id!r} "
                f"which has no corresponding Step 4 regions in the region JSON"
            )

    return semantic_index


# ============================================================================
# RESOLUTION: semantic region_id -> eligible / skipped Step 4 labels
# ============================================================================

def _resolve_mutation_labels(region_id, semantic_index, eligible_safety):
    """
    Resolve a mutation's semantic region_id into the Step 4 integer
    region_label values that are eligible for the given mutation, per
    each matching Step 4 entry's OWN mutation_safety (the Step 4 JSON is
    the runtime execution authority, not the knowledge-JSON-level safety
    already baked into the plan).
    """
    matching_entries = semantic_index[region_id]
    eligible = []
    skipped = []

    for entry in matching_entries:
        label = entry["region_label"]
        safety = entry["mutation_safety"]
        if safety in eligible_safety:
            eligible.append(label)
        else:
            skipped.append({"region_label": label, "reason": f"mutation_safety={safety}"})

    return matching_entries, eligible, skipped


def _resolve_color_mutation_labels(region_id, semantic_index):
    """Resolve eligible Step 4 labels for a color mutation. See
    _resolve_mutation_labels for details."""
    return _resolve_mutation_labels(region_id, semantic_index, COLOR_ELIGIBLE_SAFETY)


def _resolve_light_mutation_labels(region_id, semantic_index):
    """Resolve eligible Step 4 labels for a light mutation. Same
    eligibility rule as color, per the L2 region contract. See
    _resolve_mutation_labels for details."""
    return _resolve_mutation_labels(region_id, semantic_index, LIGHT_ELIGIBLE_SAFETY)


def _resolve_atmosphere_mutation_labels(region_id, semantic_index):
    """Resolve eligible Step 4 labels for an atmosphere mutation. Only
    Step 4 entries whose own mutation_safety is atmosphere_only are
    eligible, per the L3 region contract. See _resolve_mutation_labels
    for details."""
    return _resolve_mutation_labels(region_id, semantic_index, ATMOSPHERE_ELIGIBLE_SAFETY)


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
# L2 DIRECTIONAL LIGHTING TRANSFORM
# ============================================================================

# Conservative caps so the effect stays subtle and pixel-art friendly even
# at strength=1.0 / light_warmth=+-1.0.
_MAX_BRIGHTNESS_DELTA = 55.0
_MAX_WARMTH_DELTA = 40.0

# Warm/cool per-channel direction: positive light_warmth pushes toward a
# warm cast (more red, a little more green, less blue); negative light_warmth
# pushes the opposite (cool) direction. Purely an artistic constant.
_WARMTH_CHANNEL_VECTOR = np.array([1.0, 0.25, -0.85], dtype=np.float64)


def _directional_light_factor(shape, light_angle):
    """
    Deterministic per-pixel directional illumination factor in [0, 1],
    purely a function of pixel coordinates and light_angle (no neighbor
    access, no randomness, no resampling). 0 degrees points along +x
    (right); angles increase clockwise (toward +y, i.e. downward) to
    match image/array coordinates. 1.0 = fully facing the light,
    0.0 = fully turned away from it, varying linearly in between.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64)

    half_w = (w - 1) / 2.0 if w > 1 else 1.0
    half_h = (h - 1) / 2.0 if h > 1 else 1.0
    xs_norm = (xs - (w - 1) / 2.0) / half_w
    ys_norm = (ys - (h - 1) / 2.0) / half_h

    theta = np.deg2rad(light_angle)
    dx = np.cos(theta)
    dy = np.sin(theta)

    projection = xs_norm * dx + ys_norm * dy
    projection = np.clip(projection, -1.0, 1.0)
    return (projection + 1.0) / 2.0


def _directional_lighting_transform(image, light_angle, light_warmth):
    """
    Whole-image directional lighting: a spatially varying brightness ramp
    (direction set by light_angle) combined with a warm/cool color shift
    (light_warmth) that is strongest on the lit side of the ramp.
    Deterministic and purely per-pixel (fixed coordinates + fixed
    direction/warmth inputs); no blur, no interpolation, no resampling,
    so pixel-art edges stay hard.
    """
    light_factor = _directional_light_factor(image.shape[:2], light_angle)  # 0..1
    signed_factor = (light_factor * 2.0) - 1.0  # -1 (shadow side) .. +1 (lit side)

    brightness_delta = signed_factor * _MAX_BRIGHTNESS_DELTA  # (H, W)
    warmth_delta = (
        light_warmth * _MAX_WARMTH_DELTA
        * light_factor[..., np.newaxis]
        * _WARMTH_CHANNEL_VECTOR
    )  # (H, W, 3), warmth only shows up on the lit side

    transformed = (
        image.astype(np.float64)
        + brightness_delta[..., np.newaxis]
        + warmth_delta
    )
    return np.clip(np.round(transformed), 0, 255).astype(np.uint8)


def apply_light_mutation(image, label_map, eligible_labels, light_angle, light_warmth, strength):
    """
    Return (new_image, affected_pixel_count). Mirrors apply_color_mutation:
    new_image is a copy of `image` with the directional lighting transform
    blended in, restricted to exactly the pixels where label_map is one of
    eligible_labels. Every other pixel is byte-identical to the input.
    """
    output = image.copy()

    if not eligible_labels:
        return output, 0

    mask = np.isin(label_map, eligible_labels)
    affected = int(np.count_nonzero(mask))

    if affected == 0 or strength == 0:
        return output, affected

    transformed = _directional_lighting_transform(image, light_angle, light_warmth)

    blended = image.astype(np.float64) * (1.0 - strength) + transformed.astype(np.float64) * strength
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    output[mask] = blended[mask]
    return output, affected


# ============================================================================
# L3 ATMOSPHERE / TEXTURE TRANSFORM
# ============================================================================

# Conservative caps so the effect stays subtle even at atmosphere_density=1.0
# and strength=1.0 -- this is meant to read as "a bit more/less hazy", not a
# visible filter slapped over the scene.
_MAX_ATMOSPHERE_BRIGHTNESS = 26.0
_MAX_ATMOSPHERE_DESATURATION = 0.30

# Two fixed octaves of coherent value noise, both deliberately large-celled
# (low frequency) so the result reads as soft atmospheric patches rather
# than fine-grained/photographic noise. Odd, non-power-of-two cell sizes
# and distinct seeds keep the two octaves from lining up into an obvious
# repeating grid.
_ATMOSPHERE_OCTAVES = (
    {"cell_size": 71, "seed": 0xA17C, "weight": 0.7},
    {"cell_size": 29, "seed": 0x5EED, "weight": 0.3},
)
_ATMOSPHERE_BAND_LEVELS = 6  # discrete bands -> pixel-art-style stepped look


def _lattice_hash(ix, iy, seed):
    """
    Deterministic integer-lattice hash -> float in [0, 1). Pure bit
    mixing (same family of constants used elsewhere in this pipeline for
    stable hashing), no randomness, no global state, no timestamps.
    """
    ix64 = ix.astype(np.int64)
    iy64 = iy.astype(np.int64)
    seed64 = np.int64((seed * 2654435761) & 0x7FFFFFFF)

    h = (ix64 * np.int64(374761393)) ^ (iy64 * np.int64(668265263)) ^ seed64
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & np.int64(0xFFFFFFFF)).astype(np.float64) / 4294967295.0


def _bilinear_upsample(grid, out_h, out_w):
    """
    Upsample a small 2D lattice (grid_h, grid_w) to (out_h, out_w) via
    bilinear interpolation. This synthesizes the coherent noise field
    itself (not a blur of the photo); the image's own pixels are never
    resampled or interpolated.
    """
    grid_h, grid_w = grid.shape

    ys = np.linspace(0, grid_h - 1, out_h)
    xs = np.linspace(0, grid_w - 1, out_w)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, grid_h - 1)
    x1 = np.clip(x0 + 1, 0, grid_w - 1)

    wy = (ys - y0)[:, np.newaxis]
    wx = (xs - x0)[np.newaxis, :]

    top = grid[y0][:, x0] * (1.0 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1.0 - wx) + grid[y1][:, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _value_noise_octave(shape, cell_size, seed):
    """One coherent, low-frequency noise octave over the full image plane."""
    h, w = shape
    grid_h = (h // cell_size) + 2
    grid_w = (w // cell_size) + 2
    iy, ix = np.mgrid[0:grid_h, 0:grid_w]
    lattice_values = _lattice_hash(ix, iy, seed)
    return _bilinear_upsample(lattice_values, h, w)


def _quantize_field(field, levels):
    """Snap a [0,1] field to `levels` discrete steps -- gives the banded,
    stepped look already used elsewhere in this piece's clouds/fog rather
    than a smooth photographic gradient."""
    field = np.clip(field, 0.0, 1.0 - 1e-9)
    step = 1.0 / levels
    return np.floor(field / step) * step


def _procedural_atmosphere_field(shape):
    """
    Deterministic, coherent, low-frequency, quantized [0, 1) density
    field -- purely a function of pixel coordinates via fixed-seed
    lattice hashing (no randomness, no timestamps, no external state).
    0 = clear, higher = denser local atmospheric texture.
    """
    combined = np.zeros(shape, dtype=np.float64)
    for octave in _ATMOSPHERE_OCTAVES:
        combined += octave["weight"] * _value_noise_octave(
            shape, octave["cell_size"], octave["seed"]
        )
    return _quantize_field(combined, _ATMOSPHERE_BAND_LEVELS)


def _atmosphere_transform(image, atmosphere_density):
    """
    Whole-image atmospheric texture: a coherent, banded density field
    (fixed per image size) scaled by atmosphere_density, applied as a
    mild brightening + desaturation (haze reads as lighter and less
    saturated). Deterministic and purely per-pixel/per-coordinate; no
    blur, no resampling of the source image, no anti-aliased edges.
    """
    density_field = _procedural_atmosphere_field(image.shape[:2]) * atmosphere_density  # (H, W)

    brightness_delta = density_field * _MAX_ATMOSPHERE_BRIGHTNESS
    desat_amount = np.clip(density_field, 0.0, 1.0) * _MAX_ATMOSPHERE_DESATURATION

    image_f = image.astype(np.float64)
    luminance = image_f.mean(axis=2, keepdims=True)

    desaturated = image_f * (1.0 - desat_amount[..., np.newaxis]) + luminance * desat_amount[..., np.newaxis]
    transformed = desaturated + brightness_delta[..., np.newaxis]
    return np.clip(np.round(transformed), 0, 255).astype(np.uint8)


def apply_atmosphere_mutation(image, label_map, eligible_labels, atmosphere_density, strength):
    """
    Return (new_image, affected_pixel_count). Mirrors apply_color_mutation
    and apply_light_mutation: new_image is a copy of `image` with the
    atmosphere transform blended in, restricted to exactly the pixels
    where label_map is one of eligible_labels. Every other pixel is
    byte-identical to the input.
    """
    output = image.copy()

    if not eligible_labels:
        return output, 0

    mask = np.isin(label_map, eligible_labels)
    affected = int(np.count_nonzero(mask))

    if affected == 0 or strength == 0:
        return output, affected

    transformed = _atmosphere_transform(image, atmosphere_density)

    blended = image.astype(np.float64) * (1.0 - strength) + transformed.astype(np.float64) * strength
    blended = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    output[mask] = blended[mask]
    return output, affected


# ============================================================================
# ORCHESTRATION
# ============================================================================

def apply_mutation_plan(image, label_map, region_data, mutation_plan):
    """
    Validate inputs, then apply every L1 color, L2 light, and L3
    atmosphere mutation in the plan. L4 (shape) mutations are present in
    region_mapper.py's plan but are intentionally not executed here.

    Returns (output_image, report). report is a list of dicts, one per
    executed mutation, describing resolution, eligibility, and pixel counts.
    """
    semantic_index = validate_inputs(image, label_map, region_data, mutation_plan)

    output = image.copy()
    report = []

    for m in mutation_plan["mutations"]:
        mutation_type = m["mutation"]
        if mutation_type not in ("color", "light", "atmosphere"):
            # L4 (shape) -- not implemented by this engine yet.
            continue

        region_id = m["region_id"]
        strength = m["strength"]

        if mutation_type == "color":
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
                "mutation": "color",
                "matching_step4_labels": [e["region_label"] for e in matching_entries],
                "eligible_labels": eligible,
                "skipped_labels": skipped,
                "strength": strength,
                "hue_shift": hue_shift,
                "saturation_scale": saturation_scale,
                "affected_pixel_count": affected,
            })

        elif mutation_type == "light":
            light_angle = m["parameters"]["light_angle"]
            light_warmth = m["parameters"]["light_warmth"]

            matching_entries, eligible, skipped = _resolve_light_mutation_labels(
                region_id, semantic_index
            )

            output, affected = apply_light_mutation(
                output, label_map, eligible, light_angle, light_warmth, strength
            )

            report.append({
                "semantic_region_id": region_id,
                "mutation": "light",
                "matching_step4_labels": [e["region_label"] for e in matching_entries],
                "eligible_labels": eligible,
                "skipped_labels": skipped,
                "strength": strength,
                "light_angle": light_angle,
                "light_warmth": light_warmth,
                "affected_pixel_count": affected,
            })

        else:  # "atmosphere"
            atmosphere_density = m["parameters"]["atmosphere_density"]

            matching_entries, eligible, skipped = _resolve_atmosphere_mutation_labels(
                region_id, semantic_index
            )

            output, affected = apply_atmosphere_mutation(
                output, label_map, eligible, atmosphere_density, strength
            )

            report.append({
                "semantic_region_id": region_id,
                "mutation": "atmosphere",
                "matching_step4_labels": [e["region_label"] for e in matching_entries],
                "eligible_labels": eligible,
                "skipped_labels": skipped,
                "strength": strength,
                "atmosphere_density": atmosphere_density,
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

    # ------------------------------------------------------------------
    # L2 -- directional lighting tests (same synthetic fixture)
    # ------------------------------------------------------------------
    #   1 = "sky"     color_safe          (light-eligible, conservative)
    #   2 = "roof"    exclude             (never light-eligible)
    #   3 = "rain"    atmosphere_only     (never light-eligible)
    #   4 = "windows" color+light_safe    (light-eligible instance)
    #   5 = "windows" shape_sensitive     (light-ineligible instance)

    light_plan = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.8,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.6}},
        {"region_id": "windows", "mutation": "light", "strength": 0.7,
         "parameters": {"light_angle": 200.0, "light_warmth": -0.5}},
    ]}

    # L2-TEST1 -- basic light mutation executes
    out_l2, report_l2 = apply_mutation_plan(image, label_map, region_data, light_plan)
    check("L2-TEST1 basic light mutation executes", out_l2 is not None and len(report_l2) == 2)

    # L2-TEST2 -- output dimensions/mode correct
    check("L2-TEST2 light output dimensions match input", out_l2.shape == image.shape)
    check("L2-TEST2b light output dtype matches input", out_l2.dtype == image.dtype)

    # L2-TEST3 -- determinism
    out_l2_again, _ = apply_mutation_plan(image, label_map, region_data, light_plan)
    check("L2-TEST3 determinism (identical light plan -> identical output)",
          np.array_equal(out_l2, out_l2_again))

    # L2-TEST4 -- zero strength -> byte-identical to input
    zero_light_plan = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.0,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.6}},
    ]}
    out_zero_light, _ = apply_mutation_plan(image, label_map, region_data, zero_light_plan)
    check("L2-TEST4 zero-strength light -> output byte-identical to input",
          np.array_equal(out_zero_light, image))

    # L2-TEST5 -- empty plan already covered by TEST5/TEST5b above (shared code path).

    # L2-TEST6 -- exact pixel isolation: only eligible labels (1=sky, 4=windows) change
    changed_l2 = np.any(out_l2 != image, axis=2)
    eligible_union_l2 = np.isin(label_map, [1, 4])
    check(
        "L2-TEST6 every changed pixel belongs to an eligible label (1 or 4)",
        bool(np.all(eligible_union_l2[changed_l2])) if changed_l2.any() else True,
    )
    check(
        "L2-TEST6b every pixel outside eligible labels is byte-identical",
        np.array_equal(out_l2[~eligible_union_l2], image[~eligible_union_l2]),
    )

    # L2-TEST7 -- excluded region ("roof") never mutated by light
    roof_light_plan = {"mutations": [
        {"region_id": "roof", "mutation": "light", "strength": 1.0,
         "parameters": {"light_angle": 90.0, "light_warmth": 1.0}},
    ]}
    out_roof_light, report_roof_light = apply_mutation_plan(image, label_map, region_data, roof_light_plan)
    check("L2-TEST7 excluded region pixels unchanged by light", np.array_equal(out_roof_light, image))
    check("L2-TEST7b excluded region reported with 0 eligible labels (light)",
          report_roof_light[0]["eligible_labels"] == [] and report_roof_light[0]["affected_pixel_count"] == 0)

    # L2-TEST8 -- atmosphere_only region ("rain") never mutated by light
    rain_light_plan = {"mutations": [
        {"region_id": "rain", "mutation": "light", "strength": 1.0,
         "parameters": {"light_angle": 90.0, "light_warmth": 1.0}},
    ]}
    out_rain_light, report_rain_light = apply_mutation_plan(image, label_map, region_data, rain_light_plan)
    check("L2-TEST8 atmosphere_only region pixels unchanged by light", np.array_equal(out_rain_light, image))
    check("L2-TEST8b atmosphere_only region reported with 0 eligible labels (light)",
          report_rain_light[0]["eligible_labels"] == [])

    # L2-TEST9 -- shape_sensitive instance of "windows" (label 5) stays untouched
    # even though the sibling label 4 (color+light_safe) IS mutated.
    check(
        "L2-TEST9 shape_sensitive windows sub-label (5) untouched while label 4 changes",
        np.array_equal(out_l2[label_map == 5], image[label_map == 5])
        and not np.array_equal(out_l2[label_map == 4], image[label_map == 4]),
    )

    # L2-TEST10 -- outside-mask preservation restated at whole-plan level (redundant
    # with L2-TEST6b, kept for parity with the numbered requirement list).
    check("L2-TEST10 no changed pixels outside eligible union (whole plan)",
          bool(np.all(eligible_union_l2[changed_l2])) if changed_l2.any() else True)

    # L2-TEST12 -- changing light_angle changes the directional result
    plan_angle_a = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.9,
         "parameters": {"light_angle": 0.0, "light_warmth": 0.0}},
    ]}
    plan_angle_b = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.9,
         "parameters": {"light_angle": 180.0, "light_warmth": 0.0}},
    ]}
    out_angle_a, _ = apply_mutation_plan(image, label_map, region_data, plan_angle_a)
    out_angle_b, _ = apply_mutation_plan(image, label_map, region_data, plan_angle_b)
    check(
        "L2-TEST12 changing light_angle changes the lighting result",
        not np.array_equal(out_angle_a[label_map == 1], out_angle_b[label_map == 1]),
    )

    # L2-TEST13 -- changing light_warmth changes the lighting character
    plan_warm_a = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.9,
         "parameters": {"light_angle": 45.0, "light_warmth": 1.0}},
    ]}
    plan_warm_b = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.9,
         "parameters": {"light_angle": 45.0, "light_warmth": -1.0}},
    ]}
    out_warm_a, _ = apply_mutation_plan(image, label_map, region_data, plan_warm_a)
    out_warm_b, _ = apply_mutation_plan(image, label_map, region_data, plan_warm_b)
    check(
        "L2-TEST13 changing light_warmth changes the lighting character",
        not np.array_equal(out_warm_a[label_map == 1], out_warm_b[label_map == 1]),
    )

    # L2-TEST14 -- changing intensity (mutation strength) changes effect magnitude
    plan_strength_low = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.2,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.8}},
    ]}
    plan_strength_high = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.9,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.8}},
    ]}
    out_strength_low, _ = apply_mutation_plan(image, label_map, region_data, plan_strength_low)
    out_strength_high, _ = apply_mutation_plan(image, label_map, region_data, plan_strength_high)
    delta_low = np.abs(out_strength_low[label_map == 1].astype(np.int16) - image[label_map == 1].astype(np.int16)).sum()
    delta_high = np.abs(out_strength_high[label_map == 1].astype(np.int16) - image[label_map == 1].astype(np.int16)).sum()
    check(
        "L2-TEST14 higher intensity/strength produces larger effect magnitude",
        delta_high > delta_low,
    )

    # L2-TEST15 -- L1 (color) + L2 (light) execute together without breaking L1
    combined_plan = {"mutations": [
        {"region_id": "sky", "mutation": "color", "strength": 0.8,
         "parameters": {"hue_shift": 90.0, "saturation_scale": 1.3}},
        {"region_id": "sky", "mutation": "light", "strength": 0.5,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.4}},
        {"region_id": "windows", "mutation": "color", "strength": 0.6,
         "parameters": {"hue_shift": -45.0, "saturation_scale": 0.6}},
    ]}
    out_combined, report_combined = apply_mutation_plan(image, label_map, region_data, combined_plan)
    check(
        "L2-TEST15 combined color+light plan executes and reports both mutations",
        len(report_combined) == 3
        and {r["mutation"] for r in report_combined} == {"color", "light"},
    )
    check(
        "L2-TEST15b combined plan still isolates changes to eligible labels (1, 4)",
        np.array_equal(out_combined[label_map != 1][label_map[label_map != 1] != 4],
                        image[label_map != 1][label_map[label_map != 1] != 4]),
    )

    # L2-TEST16 -- no randomness/timestamps: re-running the exact same combined
    # plan on a fresh copy of the fixture gives byte-identical output.
    image2, label_map2, region_data2 = make_fixture()
    out_combined2, _ = apply_mutation_plan(image2, label_map2, region_data2, combined_plan)
    check(
        "L2-TEST16 identical combined plan on a fresh fixture is byte-identical",
        np.array_equal(out_combined, out_combined2),
    )

    # Bonus: duplicate light mutation for the same region_id raises.
    dup_light_plan = {"mutations": [
        {"region_id": "sky", "mutation": "light", "strength": 0.5,
         "parameters": {"light_angle": 10.0, "light_warmth": 0.2}},
        {"region_id": "sky", "mutation": "light", "strength": 0.3,
         "parameters": {"light_angle": 20.0, "light_warmth": 0.1}},
    ]}
    try:
        apply_mutation_plan(image, label_map, region_data, dup_light_plan)
        check("BONUS duplicate region_id light mutation raises MutationEngineError", False)
    except MutationEngineError as e:
        check("BONUS duplicate region_id light mutation raises MutationEngineError",
              "Duplicate" in str(e))

    # ------------------------------------------------------------------
    # L3 -- atmosphere/texture tests (same synthetic fixture)
    # ------------------------------------------------------------------
    #   1 = "sky"     color_safe          (never atmosphere-eligible)
    #   2 = "roof"    exclude             (never atmosphere-eligible)
    #   3 = "rain"    atmosphere_only     (atmosphere-eligible)
    #   4 = "windows" color+light_safe    (never atmosphere-eligible)
    #   5 = "windows" shape_sensitive     (never atmosphere-eligible)

    atmosphere_plan = {"mutations": [
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.8,
         "parameters": {"atmosphere_density": 0.9}},
    ]}

    # L3-TEST1 -- basic atmosphere mutation executes
    out_l3, report_l3 = apply_mutation_plan(image, label_map, region_data, atmosphere_plan)
    check("L3-TEST1 basic atmosphere mutation executes", out_l3 is not None and len(report_l3) == 1)

    # L3-TEST2 -- output dimensions correct
    check("L3-TEST2 atmosphere output dimensions match input", out_l3.shape == image.shape)

    # L3-TEST2b -- output remains RGB
    check("L3-TEST2b atmosphere output remains RGB (H,W,3) uint8",
          out_l3.ndim == 3 and out_l3.shape[2] == 3 and out_l3.dtype == np.uint8)

    # L3-TEST4 -- determinism
    out_l3_again, _ = apply_mutation_plan(image, label_map, region_data, atmosphere_plan)
    check("L3-TEST4 determinism (identical atmosphere plan -> identical output)",
          np.array_equal(out_l3, out_l3_again))

    # L3-TEST5 -- zero strength -> byte-identical
    zero_atmosphere_plan = {"mutations": [
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.0,
         "parameters": {"atmosphere_density": 0.9}},
    ]}
    out_zero_atm, _ = apply_mutation_plan(image, label_map, region_data, zero_atmosphere_plan)
    check("L3-TEST5 zero-strength atmosphere -> output byte-identical to input",
          np.array_equal(out_zero_atm, image))

    # L3-TEST6 -- empty plan already covered by TEST5/TEST5b above (shared code path).

    # L3-TEST7/8 -- exact pixel isolation: only label 3 (rain, atmosphere_only) changes
    changed_l3 = np.any(out_l3 != image, axis=2)
    eligible_atm = (label_map == 3)
    check(
        "L3-TEST7 every changed pixel belongs to the eligible atmosphere label (3)",
        bool(np.all(eligible_atm[changed_l3])) if changed_l3.any() else True,
    )
    check(
        "L3-TEST8 every pixel outside the eligible atmosphere label is byte-identical",
        np.array_equal(out_l3[~eligible_atm], image[~eligible_atm]),
    )

    # L3-TEST9 -- excluded region ("roof", label 2) unaffected
    check("L3-TEST9 exclude region (roof) unchanged by atmosphere",
          np.array_equal(out_l3[label_map == 2], image[label_map == 2]))

    # L3-TEST10 -- color_safe region ("sky", label 1) unaffected
    check("L3-TEST10 color_safe region (sky) unchanged by atmosphere",
          np.array_equal(out_l3[label_map == 1], image[label_map == 1]))

    # L3-TEST11 -- color+light_safe region ("windows", label 4) unaffected
    check("L3-TEST11 color+light_safe region (windows eligible) unchanged by atmosphere",
          np.array_equal(out_l3[label_map == 4], image[label_map == 4]))

    # L3-TEST12 -- shape_sensitive region (label 5) unaffected
    check("L3-TEST12 shape_sensitive region (windows ineligible) unchanged by atmosphere",
          np.array_equal(out_l3[label_map == 5], image[label_map == 5]))

    # An atmosphere mutation targeting a region with NO atmosphere_only Step 4
    # sub-labels (e.g. "sky", which is entirely color_safe) must be reported
    # as having zero eligible labels and change nothing.
    sky_atmosphere_plan = {"mutations": [
        {"region_id": "sky", "mutation": "atmosphere", "strength": 1.0,
         "parameters": {"atmosphere_density": 1.0}},
    ]}
    out_sky_atm, report_sky_atm = apply_mutation_plan(image, label_map, region_data, sky_atmosphere_plan)
    check("L3-BONUS color_safe-only semantic region has 0 eligible atmosphere labels",
          report_sky_atm[0]["eligible_labels"] == [] and report_sky_atm[0]["affected_pixel_count"] == 0)
    check("L3-BONUS color_safe-only semantic region unchanged by atmosphere mutation",
          np.array_equal(out_sky_atm, image))

    # L3-TEST13/14 -- atmosphere_density changes the effect, and different
    # density values are distinguishable from one another.
    plan_density_low = {"mutations": [
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.9,
         "parameters": {"atmosphere_density": 0.2}},
    ]}
    plan_density_high = {"mutations": [
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.9,
         "parameters": {"atmosphere_density": 1.0}},
    ]}
    out_density_low, _ = apply_mutation_plan(image, label_map, region_data, plan_density_low)
    out_density_high, _ = apply_mutation_plan(image, label_map, region_data, plan_density_high)
    delta_density_low = np.abs(
        out_density_low[label_map == 3].astype(np.int16) - image[label_map == 3].astype(np.int16)
    ).sum()
    delta_density_high = np.abs(
        out_density_high[label_map == 3].astype(np.int16) - image[label_map == 3].astype(np.int16)
    ).sum()
    check("L3-TEST13 atmosphere_density changes the effect magnitude",
          delta_density_high != delta_density_low)
    check("L3-TEST14 different atmosphere_density values are distinguishable",
          not np.array_equal(out_density_low[label_map == 3], out_density_high[label_map == 3]))

    # L3-TEST15 -- L1 + L2 + L3 execute together without breaking L1/L2
    combined_l123_plan = {"mutations": [
        {"region_id": "sky", "mutation": "color", "strength": 0.8,
         "parameters": {"hue_shift": 90.0, "saturation_scale": 1.3}},
        {"region_id": "sky", "mutation": "light", "strength": 0.5,
         "parameters": {"light_angle": 45.0, "light_warmth": 0.4}},
        {"region_id": "windows", "mutation": "color", "strength": 0.6,
         "parameters": {"hue_shift": -45.0, "saturation_scale": 0.6}},
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.7,
         "parameters": {"atmosphere_density": 0.6}},
    ]}
    out_l123, report_l123 = apply_mutation_plan(image, label_map, region_data, combined_l123_plan)
    check(
        "L3-TEST15 combined L1+L2+L3 plan executes and reports all three mutation types",
        len(report_l123) == 4
        and {r["mutation"] for r in report_l123} == {"color", "light", "atmosphere"},
    )
    # Every changed pixel must belong to one of the eligible labels touched by
    # this plan: sky(1), windows-eligible(4), rain(3). Roof(2) and
    # windows-ineligible(5) must be untouched.
    touched_union = np.isin(label_map, [1, 3, 4])
    changed_l123 = np.any(out_l123 != image, axis=2)
    check(
        "L3-TEST15b combined L1+L2+L3 plan isolates changes to the touched labels only",
        bool(np.all(touched_union[changed_l123])) if changed_l123.any() else True,
    )
    check(
        "L3-TEST15c combined L1+L2+L3 plan leaves roof/windows-ineligible untouched",
        np.array_equal(out_l123[label_map == 2], image[label_map == 2])
        and np.array_equal(out_l123[label_map == 5], image[label_map == 5]),
    )

    # L3-TEST16 -- no randomness/timestamps: re-running the exact same
    # combined plan on a fresh copy of the fixture gives byte-identical output.
    image3, label_map3, region_data3 = make_fixture()
    out_l123_again, _ = apply_mutation_plan(image3, label_map3, region_data3, combined_l123_plan)
    check(
        "L3-TEST16 identical L1+L2+L3 plan on a fresh fixture is byte-identical",
        np.array_equal(out_l123, out_l123_again),
    )

    # Bonus: duplicate atmosphere mutation for the same region_id raises.
    dup_atmosphere_plan = {"mutations": [
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.5,
         "parameters": {"atmosphere_density": 0.4}},
        {"region_id": "rain", "mutation": "atmosphere", "strength": 0.3,
         "parameters": {"atmosphere_density": 0.2}},
    ]}
    try:
        apply_mutation_plan(image, label_map, region_data, dup_atmosphere_plan)
        check("BONUS duplicate region_id atmosphere mutation raises MutationEngineError", False)
    except MutationEngineError as e:
        check("BONUS duplicate region_id atmosphere mutation raises MutationEngineError",
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

        # L3-only run (atmosphere mutations in isolation) for visual inspection.
        atmosphere_only_plan = {
            "image_id": plan["image_id"],
            "reference_bucket": plan["reference_bucket"],
            "mutations": [mm for mm in plan["mutations"] if mm["mutation"] == "atmosphere"],
        }
        real_output_l3_only, real_report_l3_only = apply_mutation_plan(
            real_image, real_label_map, real_region_data, atmosphere_only_plan
        )

        os.makedirs("output_mutation_test", exist_ok=True)
        save_image(real_output_l3_only, "output_mutation_test/image_01_L3_only.png")
        save_image(real_output, "output_mutation_test/image_01_L1_L2_L3_combined.png")

        diff = np.abs(real_output.astype(np.int16) - real_image.astype(np.int16))
        diff_vis = np.clip(diff.sum(axis=2) * 3, 0, 255).astype(np.uint8)
        Image.fromarray(diff_vis, mode="L").save("output_mutation_test/image_01_L1_L2_L3_difference.png")

        diff_l3_only = np.abs(real_output_l3_only.astype(np.int16) - real_image.astype(np.int16))
        diff_l3_only_vis = np.clip(diff_l3_only.sum(axis=2) * 3, 0, 255).astype(np.uint8)
        Image.fromarray(diff_l3_only_vis, mode="L").save("output_mutation_test/image_01_L3_only_difference.png")

        print("\n--- Visual integration test (Image 1) ---")
        color_mutations_in_plan = [mm for mm in plan["mutations"] if mm["mutation"] == "color"]
        light_mutations_in_plan = [mm for mm in plan["mutations"] if mm["mutation"] == "light"]
        atmosphere_mutations_in_plan = [mm for mm in plan["mutations"] if mm["mutation"] == "atmosphere"]
        print(f"Color mutations in plan: {len(color_mutations_in_plan)}")
        print(f"Light mutations in plan: {len(light_mutations_in_plan)}")
        print(f"Atmosphere mutations in plan: {len(atmosphere_mutations_in_plan)}")
        for r in real_report:
            print(
                f"  region_id={r['semantic_region_id']!r} "
                f"matching={r['matching_step4_labels']} "
                f"eligible={r['eligible_labels']} "
                f"skipped={[s['region_label'] for s in r['skipped_labels']]} "
                f"strength={r['strength']} affected_pixels={r['affected_pixel_count']}"
            )

        sky_reports = [r for r in real_report if r["semantic_region_id"] == "sky_and_clouds"]
        sky_color_report = next((r for r in sky_reports if r["mutation"] == "color"), None)
        sky_light_report = next((r for r in sky_reports if r["mutation"] == "light"), None)
        windows_reports = [r for r in real_report if r["semantic_region_id"] == "city_windows"]
        windows_color_report = next((r for r in windows_reports if r["mutation"] == "color"), None)
        windows_light_report = next((r for r in windows_reports if r["mutation"] == "light"), None)

        check("VISUAL output dims match source (1200x547)", real_output.shape[:2] == (547, 1200))
        check("VISUAL sky_and_clouds color mutation present and affected pixels > 0",
              sky_color_report is not None and sky_color_report["affected_pixel_count"] > 0)
        check("VISUAL sky_and_clouds light mutation present and affected pixels > 0",
              sky_light_report is not None and sky_light_report["affected_pixel_count"] > 0)
        if windows_color_report is not None:
            check("VISUAL city_windows color mutation present and affected pixels > 0",
                  windows_color_report["affected_pixel_count"] > 0)
        if windows_light_report is not None:
            check("VISUAL city_windows light mutation present and affected pixels > 0",
                  windows_light_report["affected_pixel_count"] > 0)

        rain_reports = [r for r in real_report if r["semantic_region_id"] == "rain_streaks"]
        rain_atmosphere_report = next((r for r in rain_reports if r["mutation"] == "atmosphere"), None)
        if rain_atmosphere_report is not None:
            check("VISUAL rain_streaks atmosphere mutation present and affected pixels > 0",
                  rain_atmosphere_report["affected_pixel_count"] > 0)
            rain_eligible_mask = np.isin(real_label_map, rain_atmosphere_report["eligible_labels"])
            check(
                "VISUAL L3-only run changes rain_streaks pixels and nothing else",
                not np.array_equal(real_output_l3_only[rain_eligible_mask], real_image[rain_eligible_mask])
                and np.array_equal(real_output_l3_only[~rain_eligible_mask], real_image[~rain_eligible_mask]),
            )

        # Confirm excluded regions never appear as eligible for ANY mutation
        # type, and atmosphere_only regions never appear as eligible for
        # color or light (they only become eligible for the new "atmosphere"
        # mutation type introduced in this step).
        excluded_semantic_labels = {
            e["semantic_label"] for e in real_region_data
            if e["mutation_safety"] == "exclude"
        }
        atmosphere_only_semantic_labels = {
            e["semantic_label"] for e in real_region_data
            if e["mutation_safety"] == "atmosphere_only"
        }
        exclude_violation = any(
            r["semantic_region_id"] in excluded_semantic_labels and r["eligible_labels"]
            for r in real_report
        )
        check("VISUAL excluded semantic category never has eligible labels (any mutation)",
              not exclude_violation)

        atmosphere_color_light_violation = any(
            r["semantic_region_id"] in atmosphere_only_semantic_labels
            and r["mutation"] in ("color", "light")
            and r["eligible_labels"]
            for r in real_report
        )
        check("VISUAL atmosphere_only semantic category never eligible for color/light",
              not atmosphere_color_light_violation)

        # shape_sensitive Step 4 sub-labels must never be eligible either, even
        # within an otherwise partially-eligible semantic category.
        shape_sensitive_labels = {
            e["region_label"] for e in real_region_data if e["mutation_safety"] == "shape_sensitive"
        }
        shape_violation = any(
            lbl in shape_sensitive_labels
            for r in real_report
            for lbl in r["eligible_labels"]
        )
        check("VISUAL no shape_sensitive Step 4 label ever appears eligible",
              not shape_violation)

        print("\nSaved: output_mutation_test/image_01_L3_only.png")
        print("Saved: output_mutation_test/image_01_L3_only_difference.png")
        print("Saved: output_mutation_test/image_01_L1_L2_L3_combined.png")
        print("Saved: output_mutation_test/image_01_L1_L2_L3_difference.png")

    except FileNotFoundError as e:
        print(f"\n[SKIPPED] Visual integration test -- missing real project file: {e}")

    if failed:
        raise SystemExit(1)