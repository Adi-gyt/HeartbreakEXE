# Heartbreak.exe

> **The universe makes art. We delete it.**

Heartbreak.exe is an intentionally over-engineered useless project that turns **live Earth and space-weather data into procedural pixel art**, sets that artwork as the desktop wallpaper, keeps it for **300 seconds**, and then permanently deletes it.

There is no practical reason to do this.

That is the point.

---

## The Idea

Heartbreak.exe takes real scientific data from the current state of the Earth and near-Earth space and feeds it through a procedural visual pipeline.

The system reads things such as:
- Earthquake magnitude, depth, latitude, and longitude
- Solar-wind speed, density, and temperature
- IMF Bz, By, Bt
- Geomagnetic Kp index

These values are **not scientifically related to the appearance of the generated artwork**. Instead, they are deliberately mapped to arbitrary visual parameters.

For example:
```text
Solar-wind temperature → hue shift
Solar-wind density     → saturation
IMF Bz                 → lighting direction
IMF By                 → lighting warmth
IMF Bt + Kp            → visual intensity
Kp                     → atmospheric density
Earthquake magnitude   → structural deformation
Earthquake location    → reference/style selection
```

The result is a pixel-art wallpaper whose appearance is determined by the current state of the universe. 

Then:
```text
WAIT 300 SECONDS
        ↓
DELETE THE WALLPAPER
        ↓
GENERATE ANOTHER ONE
```

The artwork exists for five minutes. Then it is gone.

**Why?** Because this is a Useless Project. We deliberately took something that could have been a simple wallpaper generator and made it unnecessarily complicated:

```text
LIVE SCIENTIFIC DATA
        ↓
DATA VALIDATION
        ↓
ARTISTIC PARAMETER MAPPING
        ↓
REFERENCE PIXEL ART
        ↓
OFFLINE AI VISUAL ANALYSIS
        ↓
COMPUTER-VISION REGION EXTRACTION
        ↓
SEMANTIC REGION MAPPING
        ↓
MUTATION PLANNING
        ↓
COLOR MUTATION
        ↓
DIRECTIONAL LIGHTING
        ↓
ATMOSPHERIC TEXTURE
        ↓
WALLPAPER RENDERING
        ↓
WINDOWS DESKTOP
        ↓
WAIT 300 SECONDS
        ↓
DELETE
```

It is serious engineering applied to a completely unnecessary problem.

---

## Core Design Principle

The project does not ask an AI to generate the final artwork. Instead:

**AI studies the reference. Code mutates the actual pixels.**

This distinction is important. The visual style is learned during an offline preprocessing stage. At runtime, the system does not need an AI model to generate an image. The runtime pipeline is deterministic Python/OpenCV/Pillow processing driven by live data.

This makes the system:
- Reproducible
- Inspectable
- Deterministic
- Much faster
- Easier to debug
- Genuinely procedural

---

## Architecture

The project is divided into several layers rather than one giant script.

```text
                    ┌──────────────────────┐
                    │     LIVE DATA        │
                    │                      │
                    │ USGS + NOAA sources  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   parameter_mapper   │
                    │                      │
                    │ scientific → artistic│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    region_mapper     │
                    │                      │
                    │ semantic regions →   │
                    │ mutation plan        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   mutation_engine    │
                    │                      │
                    │ actual pixel changes │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      renderer        │
                    │                      │
                    │ final 1920 × 1080    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ wallpaper_controller │
                    │                      │
                    │ Windows desktop      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        main.py       │
                    │                      │
                    │ 300-second lifecycle │
                    └──────────────────────┘
```

---

## 1. Live Scientific Data
`api_client3.py`

The data layer is responsible for retrieving the current external data. It currently handles:

**USGS (Earthquake data):**
- Magnitude, depth, latitude, longitude

**NOAA (Space weather):**
- Solar-wind: speed, density, temperature
- Magnetic-field: Bz, By, Bt
- Geomagnetic activity: Kp

The API client was intentionally designed with independent caches:
- `cache_usgs.json`
- `cache_noaa_wind.json`
- `cache_noaa_mag.json`
- `cache_noaa_kp.json`

This means failure of one external source does not automatically destroy the entire pipeline. The client also performs timeout handling, finite-number validation, independent fallback behaviour, and separate source status handling.

A live pipeline run successfully produced values such as:
```text
Magnitude: 6.6
Depth: 358.597 km
Latitude: -5.0932
Longitude: 106.903

Speed: 420.5 km/s
Density: 6.18 p/cm³
Temperature: 148655.0 K

Bz: -1.36 nT
By: 4.95 nT
Bt: 5.15 nT

Kp: 0.0
```
These values have also been observed changing between live requests, confirming that the pipeline is actually consuming live data rather than a permanently hard-coded dataset.

---

## 2. Scientific Data → Artistic Parameters
`parameter_mapper.py`

The parameter mapper converts the raw scientific values into a standardized artistic vector.

**Public API:** `map_api_to_visuals(api_data)`

The current artistic vector is:
- `hue_shift`
- `saturation_scale`
- `light_angle`
- `light_warmth`
- `intensity`
- `atmosphere_density`
- `reference_bucket`
- `deform_strength`

**Current mappings:**
- `solar_wind_temperature` → `hue_shift`
- `solar_wind_density` → `saturation_scale`
- `IMF Bz` → `light_angle`
- `IMF By` → `light_warmth`
- `IMF Bt + Kp` → `intensity`
- `Kp` → `atmosphere_density`
- `earthquake magnitude` → `deform_strength`
- `latitude + longitude` → `reference_bucket`

The earthquake coordinates are converted into a deterministic hash. This means the same location always produces the same reference bucket. There is deliberately no similarity-search system; the reference selection is deterministic.

---

## 3. Reference Pixel Art

The project uses reference pixel-art images as the visual foundation. The reference is not simply passed through a global filter. Instead, we want the system to understand that different parts of the image are different things.

For example: *sky, clouds, buildings, trees, road, traffic, lights, character, rain, windows, power lines*. 

These regions can then receive different types of mutations. This is what allows the output to remain visually coherent instead of becoming "Python applied a random color filter to a PNG."

---

## 4. Offline AI Visual Analysis

The project uses vision AI during the offline preprocessing stage. The AI does not generate the final wallpaper. Instead, it studies each reference image and produces structured technical knowledge.

The analysis includes style family, image dimensions, semantic regions, normalized bounding boxes, palette, lighting, atmosphere, pixel-art characteristics, region relationships, composition constraints, mutation safety, visual importance, mutation priority, and generator recommendations.

Each image gets a standardized JSON file:
```text
references/image_01.png
        ↓
Gemini vision analysis
        ↓
knowledge/image_01.json
```

The analysis prompt explicitly prevents the model from inventing objects that are not actually present. It also classifies regions according to mutation safety (`color_safe`, `color+light_safe`, `shape_sensitive`, `atmosphere_only`, `exclude`). This allows later stages to know what they are allowed to change.

---

## 5. Computer Vision Region Extraction (Step 3)

The first computer-vision prototype tested how the image could be segmented using Gaussian blur, Lab-space color quantization, k-means clustering, connected components, and component filtering. 

It successfully identified hundreds of meaningful connected components but produced too many fragmented regions. Instead of pretending the output was good enough, the pipeline was redesigned around semantic knowledge from the AI analysis.

---

## 6. Semantic Region Mapping (Step 4)
`step4_region_mapping.py`

The Step 4 system combines **AI semantic knowledge + computer-vision segmentation**.

- **AI says:** "This area is the sky."
- **Computer vision says:** "These actual pixels belong to this visual structure."

The mapper combines the two. The resulting output includes `region_id`, `semantic_label`, `area_pixels`, `area_fraction`, `bbox`, `centroid`, `dominant_colors`, `confidence`, and `mutation_safety`.

For Image 01, the current processed dataset contains 179 regions (Coverage: 64.38%). Semantic labels include things like `sky_and_clouds`, `highway_traffic`, `character_silhouette`, etc. 

Special pattern regions (rain, windows, power lines) were handled separately. Power lines are explicitly classified as `exclude` to prevent accidental deformation.

---

## 7. Region Label Map

Step 4 also generates a precise label map (`output_step4/image_01_region_labels.png`). The label map is stored as a 16-bit image. Each pixel contains the numeric ID of its region, creating a bridge between:

```text
semantic JSON → numeric pixel labels → actual image pixels
```

**Final validation confirmed:**
- Label map: 1200 × 547 (Format: I;16)
- Regions: 179 | Labels: 179

---

## 8. Mutation Planning
`region_mapper.py`

The region mapper converts the semantic knowledge and artistic vector into a concrete mutation plan.

**Public API:** `build_mutation_plan(knowledge, artistic_params, regions=None)`

The mutation plan is based on semantic regions rather than arbitrary numeric region IDs:
```json
{
  "region_id": "sky_and_clouds",
  "mutation": "color",
  "strength": 0.065,
  "parameters": {}
}
```

Safety rules are enforced (e.g., `color_safe` → color only; `exclude` → no mutation). The module passed 13/13 tests.

---

## 9. Pixel Mutation Engine
`mutation_engine.py`

This is where the actual pixels change. The engine consumes the original RGB image, uint16 region label map, Step 4 region JSON, and mutation plan. It currently provides the core mutation layers:

- **L1 — Color:** Changes regional palette properties (hue, saturation, tonal relationships).
- **L2 — Directional Lighting:** Applies lighting based on mapped light direction and warmth to semantic regions, not the whole image.
- **L3 — Atmosphere:** Adds deterministic low-frequency atmospheric/texture variation (haze, environmental density) without destroying the pixel-art structure.
- **L4 — Shape Deformation:** Exists as an optional/conservative layer, not yet treated as part of the frozen core visual path.

---

## 10. Mutation Engine Testing

Tested extensively using 57 synthetic tests and 10 visual integration tests. Verified byte-identical behaviour for identical deterministic inputs.

**Diagnostic test (Low vs High mutation):**
- **LOW:** 0 pixels changed (Mean diff: 0.00)
- **HIGH:** 195,507 pixels changed (29.78% of image. Mean diff: 63.74)

Proved the mutation pipeline was not silently producing the same image for every parameter set.

---

## 11. Live Pipeline Diagnostic

A complete live-data diagnostic was verified:
```text
REAL LIVE DATA → ARTISTIC VECTOR → SEMANTIC MUTATION PLAN
```
The system generated 13 mutations successfully targeting regions like `sky_and_clouds`, `distant_cityscape`, etc.

---

## 12. Renderer
`renderer.py`

The orchestration layer. It discovers valid reference bundles, verifies required assets, loads the correct reference and semantic data, builds the mutation plan, runs the mutation engine, converts the result to 1920 × 1080, and saves the final wallpaper.

**Public API:** `render_wallpaper(live_data, generation_number)`

---

## 13. Reference Selection

Deterministic selection. Earthquake latitude/longitude are converted into a stable hash that selects a reference bucket (`sorted_index % 4`). Has a single-reference fallback for incomplete datasets.

---

## 14. Dimension Safety

The renderer uses the Step 4 label-map dimensions to determine which reference asset is valid. If 0 match, it throws an error. If multiple match, it reports an ambiguity rather than silently guessing, preventing misaligned outputs.

---

## 15. Final Wallpaper Rendering

Processed reference is converted to 1920 × 1080 using a scale-to-cover strategy. It is centrally cropped and resized using nearest-neighbour behaviour where appropriate to preserve pixel-art character.

---

## 16. Visual Result

The first full generated wallpaper successfully preserved rain, city depth, window lights, palette relationships, and pixel-art character. The mutation looks like an intentional variation of the original artwork, not a generic image filter. 

**Goal:** Make the same world feel different, not just make the image different.

---

## 17. Windows Wallpaper Controller
`wallpaper_controller.py`

Handles the Windows desktop via `SystemParametersInfoW` (ctypes).
Responsibilities: Windows wallpaper API + generated wallpaper file lifecycle.

Before Heartbreak.exe changes the desktop, the original wallpaper path is saved to `.wallpaper_original_path` (ignored by Git).

---

## 18. Wallpaper Safety

The controller keeps the active wallpaper alive until the next wallpaper has been successfully selected. The active wallpaper must never be deleted while Windows is still using it.

---

## 19. Main Lifecycle
`main.py`

The top-level orchestrator.

**Production lifecycle:**
START → capture original wallpaper → fetch live data → render wallpaper → set wallpaper → delete previous generated wallpaper → wait 300 seconds → repeat.

On interruption (`Ctrl+C`): restore original wallpaper → exit cleanly.

Dependency injection allows the lifecycle to be tested without actually modifying the user's desktop.

---

## 20. Lifecycle Safety Test

Simulated tests confirmed that if setting the next wallpaper fails, the system does not blindly delete the currently active wallpaper. The test completed with the original wallpaper successfully restored.

---

## 21. Real Production Run

Successfully launched. The real pipeline can fetch live data, generate a wallpaper, set it on Windows, and enter the 300-second wait.

---

## 22. Why Earlier Test Images Looked Identical

The `main.py --test` mode intentionally uses fixed sample data. Because the system is deterministic (same data + same reference + same parameters = same output), identical test wallpapers were expected. This proves reproducibility.

---

## 23. Current File Structure

```text
HeartbreakEXE/
│
├── api_client3.py
├── parameter_mapper.py
├── region_mapper.py
├── mutation_engine.py
├── renderer.py
├── wallpaper_controller.py
├── main.py
│
├── step4_region_mapping.py
│
├── references/
│   └── image_01.png
│
├── knowledge/
│   └── image_01.json
│
├── output_step4/
│   ├── image_01_regions.json
│   └── image_01_region_labels.png
│
├── generated_wallpapers/
│
├── cache_usgs.json
├── cache_noaa_wind.json
├── cache_noaa_mag.json
└── cache_noaa_kp.json
```

---

## 24. GitHub / Version Control
Repository: [https://github.com/Adi-gyt/HeartbreakEXE](https://github.com/Adi-gyt/HeartbreakEXE)

Runtime files and caches are ignored by Git.

---

## 25. What Has Been Debugged

- **Region fragmentation:** Fixed with semantic AI knowledge + spatial constraints.
- **Incorrect region label map:** Fixed by ensuring exact numeric region IDs write to the pixel mask.
- **Nested bounding boxes:** Smaller/specific regions win overlapping pixels.
- **Pattern regions:** Rain/windows behave differently now.
- **Power-line deformation:** Explicitly marked `exclude`.
- **Unsafe deletion:** Controller keeps active wallpaper alive until swap is successful.
- **Wallpaper restoration:** Handled in lifecycle cleanup.
- **Reference dimension mismatch:** Renderer enforces strict dimension checking.

---

## 26. Current Status

- **Live data (USGS + NOAA):** DONE
- **Scientific → artistic mapping:** DONE
- **Image 01 offline analysis:** DONE
- **Image 01 CV processing:** DONE
- **Mutation planning:** DONE
- **L1/L2/L3 (Color, Light, Atmosphere) Mutations:** DONE
- **Renderer:** DONE
- **Windows wallpaper control:** DONE
- **Wallpaper restoration:** DONE
- **300-second lifecycle:** DONE
- **Lifecycle safety testing:** DONE

---

## 27. What Is NOT Finished Yet

Processing the rest of the reference dataset (approx. 30 references). The current system safely runs with Image 01 while the remaining references are prepared.

---

## 28. Remaining Engineering Work

1. **Process remaining references:** Target 30 references with consistent bundles.
2. **Validate each reference visually:** Check dimensions, semantic regions, and safety rules.
3. **Improve reference family coverage:** Expand beyond the current deterministic bucket mechanism.
4. **Tune mutation intensity:** Decouple certain variables from global intensity so calm space weather doesn't make mutations too subtle.
5. **Solar-wind speed:** Currently fetched but unused. Plan to map to atmospheric movement.
6. **L4 shape deformation:** Keep conservative to preserve artwork identity.

---

## 29. Design Philosophy

1. **Preserve the artwork:** Should still look like the reference world.
2. **Mutate semantically:** No giant global filters. Treat sky, buildings, road differently.
3. **Scientific data controls art, not truth:** The mapping relationship is intentionally arbitrary.
4. **Deterministic:** Same inputs = same output.
5. **Fail safely:** Don't destroy the user's desktop.
6. **The final image must actually be good:** The joke only works if it looks beautiful.

---

## 30. The Uselessness

The complete system solves a problem nobody has.

> "What does your project do?"
> "It watches earthquakes and solar weather, converts them into artistic parameters, semantically analyzes pixel art, performs computer-vision region mapping, generates a procedural wallpaper, sets it as your desktop background, waits five minutes, and deletes it."
> "Why?"
> "There is no reason."

---

## 31. Current Pipeline

```text
                 ┌───────────────┐
                 │     USGS      │
                 │  Earthquakes  │
                 └───────┬───────┘
                         │
                 ┌───────▼───────┐
                 │     NOAA      │
                 │ Space Weather │
                 └───────┬───────┘
                         │
                         ▼
                ┌─────────────────┐
                │ api_client3.py  │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ parameter_mapper.py  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   reference image    │
              │ + semantic knowledge │
              │ + CV region labels   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   region_mapper.py   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  mutation_engine.py  │
              │                      │
              │ L1 Color             │
              │ L2 Lighting          │
              │ L3 Atmosphere        │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      renderer.py     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ generated wallpaper  │
              │     1920 × 1080      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ wallpaper_controller │
              └──────────┬───────────┘
                         │
                         ▼
                    WINDOWS
                    DESKTOP
                         │
                         ▼
                    300 SECONDS
                         │
                         ▼
                      DELETE
                         │
                         ▼
                      REPEAT
```

---

## 32. Running the Project

**Test lifecycle:**
```bash
python main.py --test
```
*(Uses simulated functions and does not intentionally modify/delete production wallpaper files.)*

**Production mode:**
```bash
python main.py
```
*(Starts the real lifecycle. Press `Ctrl+C` to stop and restore the original wallpaper.)*

---

## 33. Project Philosophy in One Sentence

**Heartbreak.exe takes the state of the universe, turns it into something beautiful, gives you exactly five minutes to care about it, and then deletes it.**

---

## 34. Future Ideas

- Processing all 30 references
- More style families & stronger reference rotation
- More sophisticated atmosphere simulation
- Better use of solar-wind speed
- More controlled shape deformation
- Richer semantic region relationships
- Additional scientific inputs
- Mobile/device variants
- A visualization of the scientific values driving each generation

*(These are deliberately secondary. The core useless machine comes first.)*

---

## 35. Final Goal

The finished experience should feel like this:

```text
THE UNIVERSE CHANGES
        ↓
HEARTBREAK.EXE NOTICES
        ↓
A NEW PIXEL WORLD APPEARS
        ↓
IT BECOMES YOUR WALLPAPER
        ↓
YOU GET ATTACHED
        ↓
300 SECONDS PASS
        ↓
IT IS DELETED
        ↓
THE UNIVERSE HAS ALREADY MOVED ON
        ↓
NEW WALLPAPER
```

**Heartbreak.exe**
*The universe makes art. We delete it.*