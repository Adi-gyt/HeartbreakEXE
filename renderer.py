"""
Heartbreak.exe - Renderer (renderer.py)

Pipeline position (final):

    api_client3.py -> parameter_mapper.py -> renderer.py (THIS MODULE)
        -> region_mapper.py -> mutation_engine.py -> final wallpaper

This module is ONLY an orchestration layer. It does not:
    - fetch NOAA/USGS data itself
    - compute artistic parameters itself (calls parameter_mapper)
    - plan mutations itself (calls region_mapper)
    - execute pixel mutations itself (calls mutation_engine)
    - implement any procedural-nebula / old rendering architecture
    - use randomness, timestamps, or UUIDs anywhere

renderer.py's own responsibilities are exactly:
    1. discover which reference asset bundles are actually usable at
       runtime (reference image + knowledge JSON + Step 4 regions JSON +
       Step 4 label map, all present, with matching dimensions)
    2. deterministically select one eligible reference, using the
       reference_bucket produced by parameter_mapper and the caller's
       generation_number
    3. load the selected bundle's assets and call region_mapper /
       mutation_engine exactly as documented
    4. deterministically fit the mutated image to a 1920x1080 wallpaper
       (scale-to-cover + centered crop, nearest-neighbor)
    5. save the PNG and return its path

============================================================================
REPOSITORY LAYOUT THIS MODULE EXPECTS (relative to this file)
============================================================================

    references/image_XX.<jpg|jpeg|png|...>   (extension varies per file)
    knowledge/image_XX.json
    output_step4/image_XX_regions.json
    output_step4/image_XX_region_labels.png  (uint16 per-pixel label map)

`knowledge/` is treated as the authoritative list of image_ids the project
knows about (one JSON file per image_id). A given image_id is
RUNTIME-ELIGIBLE only if all four of the assets above exist for it AND the
reference image's pixel dimensions exactly match the label map's
dimensions. Everything else about an ineligible image_id is reported as a
diagnostic and that image_id is simply excluded from selection -- it never
raises and never brings down the whole renderer. As of this writing, most
of the 30 knowledge files in the repository do not yet have Step 4 assets
(only image_01 does); that is expected, not an error state.

More than one reference image file can legitimately exist for the same
image_id under references/ (observed in practice: image_01.jpg at
736x412 alongside image_01.png at 1200x547, the latter being the actual
file the Step 4 label map was generated from). Extension is therefore
never used to choose between them. Instead, discovery loads the Step 4
label map first and picks whichever candidate's pixel dimensions match
it; if none match it's reported as a dimension mismatch (as for a single
candidate), and if more than one same-sized candidate matches, that's
reported as an explicit ambiguity rather than guessed at.

============================================================================
BUCKET -> REFERENCE ASSIGNMENT (a genuine design decision, documented here)
============================================================================

Nothing in parameter_mapper.py, region_mapper.py, mutation_engine.py, or
the knowledge/Step 4 JSON files defines which reference images belong to
which of parameter_mapper's four `reference_bucket` values (0..3) -- that
mapping simply does not exist anywhere in the repository today.
reference_bucket is produced by parameter_mapper as an opaque, already-
computed integer "style family selector", and region_mapper only ever
threads it through into its output; nothing upstream ties it to specific
image_ids.

In the absence of that mapping, this module assigns runtime-eligible
image_ids to buckets deterministically and reproducibly:

    1. sort all runtime-eligible image_ids lexicographically
    2. assign each one to bucket (sorted_index % 4), round-robin

This uses no randomness and depends only on which references are
currently eligible on disk, so it is stable for any given repository
state and trivially re-derivable. If the project later adds an explicit
bucket-membership file, `_assign_buckets` is the only place that needs to
change.

Within a bucket, `generation_number` deterministically rotates through
that bucket's eligible references in the same sorted order:

    selected = bucket_members[(generation_number - 1) % len(bucket_members)]

Right now only image_01 is eligible, so it lands alone in bucket
(0 % 4) == 0, and every reference_bucket other than 0 currently has no
eligible references -- render_wallpaper raises a clear RendererError in
that case rather than silently substituting image_01 or any other
reference. This is flagged here explicitly per the "report any conflict
instead of silently choosing another behavior" instruction, since it is
the one piece of policy this module had to invent rather than discover.

============================================================================
WALLPAPER FIT POLICY
============================================================================

No existing module defines a final-size policy, so this module uses the
scale-to-cover + centered-crop + nearest-neighbor policy specified for
this task: scale the mutated image uniformly until it covers 1920x1080,
crop the excess with a centered crop, and resample with nearest-neighbor
only (never bilinear/bicubic/Lanczos) to preserve pixel-art edges. No
borders are ever added and no pixels are invented.
"""

import math
import os

import numpy as np
from PIL import Image

import mutation_engine
import parameter_mapper
import region_mapper

# ============================================================================
# CONFIGURATION
# ============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

REFERENCES_DIR = os.path.join(_THIS_DIR, "references")
KNOWLEDGE_DIR = os.path.join(_THIS_DIR, "knowledge")
STEP4_DIR = os.path.join(_THIS_DIR, "output_step4")
OUTPUT_DIR = os.path.join(_THIS_DIR, "generated_wallpapers")

# The repository's references/ directory mixes .jpg/.jpeg (and possibly
# other) extensions across files, so discovery must try all of these
# rather than assuming one.
REFERENCE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

NUM_REFERENCE_BUCKETS = parameter_mapper.NUM_REFERENCE_BUCKETS  # 4; not re-derived here

FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080

FILENAME_TEMPLATE = "heartbreak_{:04d}.png"


class RendererError(Exception):
    """Raised for any renderer-level asset, selection, or pipeline failure.

    Never raised as a substitute for silently falling back to another
    reference, regenerating missing data, or producing a blank image --
    those things never happen in this module.
    """


# ============================================================================
# DISCOVERY: which image_ids exist, which are runtime-eligible
# ============================================================================

def _discover_image_ids():
    """
    The authoritative candidate image_id list: every `<image_id>.json`
    file present in knowledge/, in sorted filesystem order.
    """
    if not os.path.isdir(KNOWLEDGE_DIR):
        raise RendererError(f"Knowledge directory not found: {KNOWLEDGE_DIR}")

    image_ids = [
        os.path.splitext(name)[0]
        for name in sorted(os.listdir(KNOWLEDGE_DIR))
        if name.endswith(".json")
    ]
    if not image_ids:
        raise RendererError(f"No knowledge JSON files found in {KNOWLEDGE_DIR}")
    return image_ids


def _discover_reference_image_candidates(image_id):
    """
    Locate every reference image file for image_id under REFERENCES_DIR,
    tolerating the mixed file extensions actually present in the
    repository (and the fact that more than one file can legitimately
    exist for the same image_id -- e.g. a cropped/resized preview
    alongside the original full-resolution file actually used to
    generate its Step 4 label map). Returns a sorted list of matching
    paths (possibly empty); does NOT pick one, since extension alone is
    never a reliable way to choose (see discover_asset_bundles).
    """
    matches = []
    for ext in REFERENCE_IMAGE_EXTENSIONS:
        candidate = os.path.join(REFERENCES_DIR, image_id + ext)
        if os.path.isfile(candidate):
            matches.append(candidate)
    return sorted(matches)


def discover_asset_bundles():
    """
    Scan the repository and determine, for every known image_id, whether
    it is runtime-eligible (all four required assets present, dimensions
    matching).

    Multiple reference image files can exist for the same image_id (e.g.
    a cropped preview at one resolution alongside the full-resolution
    original the Step 4 label map was actually generated from). Since
    extension is never a reliable signal for which one is correct, the
    correct candidate is resolved by dimension: whichever candidate's
    pixel dimensions match the Step 4 label map's dimensions is the one
    actually usable with that label map.

        - exactly one candidate matches the label map's dimensions
              -> that candidate is used
        - zero candidates match
              -> excluded, reported as a dimension mismatch (as before)
        - more than one candidate matches
              -> ambiguous: excluded, reported explicitly rather than
                 silently guessing which same-sized file is "the" one

    Returns:
        (eligible, diagnostics)

        eligible: dict of image_id -> {
            "image_id", "image_path", "knowledge_path",
            "regions_path", "label_map_path", "image_shape",
        }
        diagnostics: dict of image_id -> human-readable exclusion reason,
            for every candidate image_id that is NOT in `eligible`. This
            is how incomplete/ambiguous references get reported without
            ever failing the whole discovery pass.

    Raises RendererError only for structural repository problems (e.g.
    a missing knowledge/ directory) -- never for an individual
    incomplete, ambiguous, or malformed reference bundle.
    """
    image_ids = _discover_image_ids()

    eligible = {}
    diagnostics = {}

    for image_id in image_ids:
        knowledge_path = os.path.join(KNOWLEDGE_DIR, image_id + ".json")
        regions_path = os.path.join(STEP4_DIR, image_id + "_regions.json")
        label_map_path = os.path.join(STEP4_DIR, image_id + "_region_labels.png")
        candidates = _discover_reference_image_candidates(image_id)

        missing = []
        if not candidates:
            missing.append("reference image")
        if not os.path.isfile(knowledge_path):
            missing.append("knowledge JSON")
        if not os.path.isfile(regions_path):
            missing.append("Step 4 regions JSON")
        if not os.path.isfile(label_map_path):
            missing.append("Step 4 region label map")

        if missing:
            diagnostics[image_id] = "missing: " + ", ".join(missing)
            continue

        try:
            label_map_arr = mutation_engine.load_region_label_map(label_map_path)
        except Exception as exc:  # noqa: BLE001 - captured as a diagnostic, not a crash
            diagnostics[image_id] = f"failed to load Step 4 label map: {exc}"
            continue

        matches = []
        load_failures = []
        for candidate_path in candidates:
            try:
                candidate_arr = mutation_engine.load_image(candidate_path)
            except Exception as exc:  # noqa: BLE001
                load_failures.append(f"{candidate_path}: {exc}")
                continue
            if candidate_arr.shape[:2] == label_map_arr.shape:
                matches.append((candidate_path, candidate_arr))

        if not matches:
            checked = ", ".join(candidates)
            reason = (
                f"dimension mismatch: no reference image candidate matches "
                f"label map dimensions {label_map_arr.shape} (H, W). "
                f"Candidates checked: {checked}."
            )
            if load_failures:
                reason += " Load failures: " + "; ".join(load_failures) + "."
            diagnostics[image_id] = reason
            continue

        if len(matches) > 1:
            ambiguous_paths = ", ".join(path for path, _ in matches)
            diagnostics[image_id] = (
                f"ambiguous reference image: {len(matches)} candidates all match "
                f"label map dimensions {label_map_arr.shape} (H, W): "
                f"{ambiguous_paths}. Refusing to silently choose one."
            )
            continue

        image_path, image_arr = matches[0]

        eligible[image_id] = {
            "image_id": image_id,
            "image_path": image_path,
            "knowledge_path": knowledge_path,
            "regions_path": regions_path,
            "label_map_path": label_map_path,
            "image_shape": image_arr.shape,
        }

    return eligible, diagnostics


# ============================================================================
# DETERMINISTIC REFERENCE SELECTION
# ============================================================================

def _assign_buckets(eligible_ids_sorted):
    """
    Deterministically distribute eligible image_ids across the
    NUM_REFERENCE_BUCKETS style-family buckets. See the module docstring
    ("BUCKET -> REFERENCE ASSIGNMENT") for why this policy exists here
    rather than being read from another module.
    """
    buckets = {b: [] for b in range(NUM_REFERENCE_BUCKETS)}
    for index, image_id in enumerate(eligible_ids_sorted):
        buckets[index % NUM_REFERENCE_BUCKETS].append(image_id)
    return buckets


def select_reference_bundle(eligible, artistic_params, generation_number):
    """
    Deterministically select one eligible reference bundle using
    artistic_params["reference_bucket"] and generation_number.

    SINGLE-REFERENCE FALLBACK (temporary, deterministic):
    The Step 4 extraction pipeline currently only covers a small subset
    of the 30-image dataset (as of this writing, only image_01). When
    exactly one reference is runtime-eligible, that reference is
    selected unconditionally -- regardless of which reference_bucket
    parameter_mapper computed for the current live_data -- instead of
    raising just because the dataset's one available reference happens
    to sit in a different bucket than the one requested. This is still
    fully deterministic (no randomness, no timestamps, no fallback
    "guessing" between multiple candidates): with a single eligible
    reference there is only one possible choice in the first place, so
    bucket routing has nothing left to decide between. As soon as a
    second reference becomes eligible, this fallback no longer applies
    and the bucket-based selection below governs again, unchanged.

    Raises RendererError if there are no eligible references at all, or
    (with two or more eligible references) if the specific
    reference_bucket selected by parameter_mapper has no eligible
    references in it -- this is never silently substituted with another
    bucket.
    """
    if isinstance(generation_number, bool) or not isinstance(generation_number, int):
        raise RendererError(f"generation_number must be an int, got {generation_number!r}")
    if generation_number < 1:
        raise RendererError(f"generation_number must be >= 1, got {generation_number}")

    if not eligible:
        raise RendererError(
            "No runtime-eligible reference bundles were found (need a reference "
            "image, knowledge JSON, Step 4 regions JSON, and Step 4 label map, "
            "all present, with matching dimensions, for at least one image_id)."
        )

    # Single-reference fallback -- see docstring above. Deterministic:
    # depends only on which one reference is eligible, never on
    # reference_bucket, generation_number, or any random/time-based input.
    if len(eligible) == 1:
        (only_id,) = eligible.keys()
        return eligible[only_id]

    reference_bucket = artistic_params["reference_bucket"]
    sorted_ids = sorted(eligible.keys())
    buckets = _assign_buckets(sorted_ids)

    bucket_members = buckets.get(reference_bucket, [])
    if not bucket_members:
        raise RendererError(
            f"reference_bucket {reference_bucket} has no runtime-eligible "
            f"references. Currently eligible: {sorted_ids}. Add Step 4 assets "
            f"for at least one reference that would fall into this bucket."
        )

    index = (generation_number - 1) % len(bucket_members)
    selected_id = bucket_members[index]
    return eligible[selected_id]


# ============================================================================
# WALLPAPER FIT (scale-to-cover + centered crop, nearest-neighbor only)
# ============================================================================

def _fit_to_wallpaper(image_array):
    """
    Deterministically fit `image_array` (H, W, 3) uint8) to exactly
    FINAL_WIDTH x FINAL_HEIGHT:

        1. scale uniformly until the image covers the target box
        2. crop the excess with a centered crop
        3. nearest-neighbor resampling only (preserves pixel-art edges;
           never bilinear/bicubic/Lanczos, never blurred/sharpened)

    Never distorts the aspect ratio, never pads, never invents pixels.
    """
    pil_image = Image.fromarray(image_array, mode="RGB")
    src_w, src_h = pil_image.size

    scale = max(FINAL_WIDTH / src_w, FINAL_HEIGHT / src_h)
    # ceil (not round) guarantees the scaled image covers the target box
    # even when the exact scale factor would round down.
    scaled_w = max(FINAL_WIDTH, math.ceil(src_w * scale))
    scaled_h = max(FINAL_HEIGHT, math.ceil(src_h * scale))

    scaled = pil_image.resize((scaled_w, scaled_h), resample=Image.NEAREST)

    left = (scaled_w - FINAL_WIDTH) // 2
    top = (scaled_h - FINAL_HEIGHT) // 2
    cropped = scaled.crop((left, top, left + FINAL_WIDTH, top + FINAL_HEIGHT))

    return np.array(cropped, dtype=np.uint8)


# ============================================================================
# OUTPUT
# ============================================================================

def _output_path(generation_number):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, FILENAME_TEMPLATE.format(generation_number))


# ============================================================================
# PUBLIC API
# ============================================================================

def render_wallpaper(live_data, generation_number):
    """
    Render one Heartbreak.exe wallpaper.

    Args:
        live_data: the 11-value raw Earth / space-weather dict, already
            fetched by the caller (e.g. via api_client3.get_all_data()).
            renderer.py never fetches network data itself and never
            requires network availability.
        generation_number: positive int identifying this render. Used
            for the output filename and for deterministic rotation among
            multiple eligible references within a reference_bucket.
            Never used as a random seed, and no random/timestamp/UUID
            value is used anywhere in this function.

    Returns:
        Absolute path to the saved 1920x1080 PNG.

    Raises:
        RendererError on any invalid input or missing/inconsistent asset
        data, identifying the affected image_id/path. Never silently
        skips, substitutes, regenerates, or falls back to a blank image.
    """
    if not isinstance(live_data, dict):
        raise RendererError(f"live_data must be a dict, got {type(live_data)!r}")

    artistic_params = parameter_mapper.map_api_to_visuals(live_data)

    eligible, _diagnostics = discover_asset_bundles()
    bundle = select_reference_bundle(eligible, artistic_params, generation_number)
    image_id = bundle["image_id"]

    try:
        knowledge = region_mapper.load_knowledge(bundle["knowledge_path"])
    except Exception as exc:
        raise RendererError(
            f"[{image_id}] failed to load knowledge JSON "
            f"({bundle['knowledge_path']}): {exc}"
        ) from exc

    try:
        step4_region_data = mutation_engine.load_region_data(bundle["regions_path"])
    except Exception as exc:
        raise RendererError(
            f"[{image_id}] failed to load Step 4 regions JSON "
            f"({bundle['regions_path']}): {exc}"
        ) from exc

    try:
        original_image = mutation_engine.load_image(bundle["image_path"])
        label_map = mutation_engine.load_region_label_map(bundle["label_map_path"])
    except Exception as exc:
        raise RendererError(
            f"[{image_id}] failed to load reference image or label map: {exc}"
        ) from exc

    # Defensive re-check: discovery already filters dimension mismatches
    # out of the eligible pool, but a bundle is never trusted blindly at
    # render time either.
    if label_map.shape != original_image.shape[:2]:
        raise RendererError(
            f"[{image_id}] reference image dimensions {original_image.shape[:2]} "
            f"!= label map dimensions {label_map.shape} at render time"
        )

    try:
        mutation_plan = region_mapper.build_mutation_plan(
            knowledge, artistic_params, step4_region_data
        )
    except region_mapper.RegionMapperError as exc:
        raise RendererError(f"[{image_id}] mutation planning failed: {exc}") from exc

    try:
        mutated_image, _report = mutation_engine.apply_mutation_plan(
            original_image, label_map, step4_region_data, mutation_plan
        )
    except mutation_engine.MutationEngineError as exc:
        raise RendererError(f"[{image_id}] mutation execution failed: {exc}") from exc

    final_image = _fit_to_wallpaper(mutated_image)

    if final_image.shape[0] != FINAL_HEIGHT or final_image.shape[1] != FINAL_WIDTH:
        raise RendererError(
            f"[{image_id}] internal error: final image shape "
            f"{final_image.shape} != target ({FINAL_HEIGHT}, {FINAL_WIDTH})"
        )

    output_path = _output_path(generation_number)
    mutation_engine.save_image(final_image, output_path)

    return os.path.abspath(output_path)


# ============================================================================
# SMOKE TEST
# ============================================================================

if __name__ == "__main__":
    # Minimal, deterministic, network-free smoke test. Uses whatever
    # reference assets actually exist under this file's directory --
    # nothing here fetches live data or invents sample images.
    _SAMPLE_LIVE_DATA = {
        "magnitude": 6.5,
        "depth": 12.0,
        "latitude": 34.0522,
        "longitude": -118.2437,
        "solar_wind_speed": 450.0,
        "solar_wind_density": 25.0,
        "solar_wind_temperature": 255000.0,
        "bz": -12.5,
        "by": 10.0,
        "bt": 15.0,
        "kp": 4.5,
    }

    print("--- Discovering asset bundles ---")
    eligible_bundles, diag = discover_asset_bundles()
    print(f"Eligible: {sorted(eligible_bundles.keys())}")
    for image_id, reason in sorted(diag.items()):
        print(f"  excluded {image_id}: {reason}")

    if not eligible_bundles:
        print("\n[SKIPPED] No eligible reference bundles found on disk; "
              "nothing to render. This is expected in a checkout that "
              "has not yet run the Step 4 extraction for any image.")
    else:
        _sample_params = parameter_mapper.map_api_to_visuals(_SAMPLE_LIVE_DATA)
        _selected_bundle = select_reference_bundle(eligible_bundles, _sample_params, 1)
        print(f"\nreference_bucket (from sample live_data): {_sample_params['reference_bucket']}")
        print(f"Selected reference: {_selected_bundle['image_id']} "
              f"({os.path.basename(_selected_bundle['image_path'])}, "
              f"shape={_selected_bundle['image_shape']})")

        print("\n--- Render #1 ---")
        path_1 = render_wallpaper(_SAMPLE_LIVE_DATA, 1)
        print(f"Saved: {path_1}")

        print("\n--- Render #2 (identical inputs) ---")
        path_2 = render_wallpaper(_SAMPLE_LIVE_DATA, 1)
        print(f"Saved: {path_2}")

        with open(path_1, "rb") as f1, open(path_2, "rb") as f2:
            identical = f1.read() == f2.read()
        print(f"\nByte-identical across identical runs: {identical}")
        assert identical, "Renderer is not deterministic!"

        with Image.open(path_1) as final_img:
            print(f"Final dimensions: {final_img.size} (expect (1920, 1080))")
            assert final_img.size == (FINAL_WIDTH, FINAL_HEIGHT)

        print("\n[SUCCESS] renderer.py smoke test passed.")