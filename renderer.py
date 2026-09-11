"""
Heartbreak.exe - Procedural Nebula Renderer
============================================

Turns live Earth + space-weather data into a temporary piece of
procedural cosmic artwork.

This module does NOT draw objects (houses, roads, trees...). It builds
an organic nebula out of interacting mathematical fields:

    domain-warped fractal noise  -> large irregular cloud masses
    blob composition             -> asymmetric focal structure
    void carving                 -> dark cavities / negative space
    flow-field particle tracing  -> curved, branching filaments
    threshold-based emission     -> glowing cores + bloom
    depth blending               -> near (sharp) vs far (hazy) layers
    sparse weighted star fields  -> stars suppressed by dense cloud
    ordered dithering            -> deliberate limited-palette pixel art

No machine learning. No external image-generation API. No external
image assets. Python + Pillow + standard library only.

Public interface (do not rename - wallpaper.py depends on this):

    render_wallpaper(live_data, generation_number) -> str
"""

import hashlib
import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageChops


# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "generated_wallpapers")

INTERNAL_WIDTH = 320
INTERNAL_HEIGHT = 180

FINAL_WIDTH = 1920
FINAL_HEIGHT = 1080


# ============================================================================
# SMALL MATH HELPERS
# ============================================================================

def _clamp(value, low=0.0, high=1.0):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _lerp(a, b, t):
    return a + (b - a) * t


def _smoothstep(t):
    t = _clamp(t)
    return t * t * (3.0 - 2.0 * t)


def _smooth_falloff(edge0, edge1, x):
    """
    1.0 when x <= edge0, 0.0 when x >= edge1, smooth in between.
    """
    if edge0 == edge1:
        return 0.0 if x >= edge1 else 1.0
    t = _clamp((x - edge0) / (edge1 - edge0))
    return 1.0 - _smoothstep(t)


def _lerp_color(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )


# ============================================================================
# VALUE-NOISE FIELDS
#
# A lattice of random values, smoothly interpolated (bilinear + smoothstep).
# Several lattices at increasing frequency, summed with decaying amplitude,
# form a fractal (fBm) field. This is the mathematical substrate for every
# organic shape in the piece - never hand-placed geometry.
# ============================================================================

def _build_lattice(rng, cells):
    size = cells + 1
    return [[rng.uniform(-1.0, 1.0) for _ in range(size + 1)] for _ in range(size + 1)]


def _sample_lattice(lattice, cells, x, y):
    """x, y are in normalized [0, 1] space."""
    gx = _clamp(x) * cells
    gy = _clamp(y) * cells
    x0 = int(gx)
    y0 = int(gy)
    if x0 >= cells:
        x0 = cells - 1
    if y0 >= cells:
        y0 = cells - 1
    x1 = x0 + 1
    y1 = y0 + 1
    fx = gx - x0
    fy = gy - y0
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    row0 = lattice[y0]
    row1 = lattice[y1]
    top = row0[x0] + (row0[x1] - row0[x0]) * ux
    bottom = row1[x0] + (row1[x1] - row1[x0]) * ux
    return top + (bottom - top) * uy


def _build_fbm_octaves(rng, base_cells, octaves, persistence, lacunarity):
    """
    Returns a list of (lattice, cells, amplitude) ready for summation.
    Amplitudes are pre-normalized so the fbm result stays in [-1, 1].
    """
    layers = []
    amplitude = 1.0
    cells = base_cells
    total_amp = 0.0
    for _ in range(octaves):
        cells_i = max(1, int(round(cells)))
        layers.append([_build_lattice(rng, cells_i), cells_i, amplitude])
        total_amp += amplitude
        amplitude *= persistence
        cells *= lacunarity

    if total_amp > 0:
        for layer in layers:
            layer[2] /= total_amp

    return layers


def _fbm(layers, x, y):
    total = 0.0
    for lattice, cells, amplitude in layers:
        total += amplitude * _sample_lattice(lattice, cells, x, y)
    return total


def _pixel_hash01(x, y, seed):
    """
    Deterministic, non-periodic pseudo-random value in [0, 1) for a pixel.
    Used for dithering instead of a tiled Bayer matrix, which produced a
    visible repeating grid artifact.
    """
    h = (x * 374761393 + y * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h ^= (h >> 16)
    return (h & 0xFFFFFFFF) / 0xFFFFFFFF


# ============================================================================
# DATA -> ARTISTIC PARAMETERS
#
# Live scientific values are never printed or literally traced. Each one
# is mapped to a role inside the generative system.
# ============================================================================

def _derive_params(live_data, generation_number):
    magnitude = float(live_data.get("magnitude", 2.5))
    depth = float(live_data.get("depth", 10.0))
    latitude = float(live_data.get("latitude", 0.0))
    longitude = float(live_data.get("longitude", 0.0))

    wind_speed = float(live_data.get("solar_wind_speed", 400.0))
    wind_density = float(live_data.get("solar_wind_density", 5.0))
    wind_temp = float(live_data.get("solar_wind_temperature", 100000.0))

    bz = float(live_data.get("bz", 0.0))
    by = float(live_data.get("by", 0.0))
    bt = float(live_data.get("bt", 0.0))

    kp = float(live_data.get("kp", 0.0))

    params = {
        # earthquake magnitude -> structural intensity / contrast
        "mag_n": _clamp((magnitude - 1.0) / 8.0),
        # earthquake depth -> foreground / background emphasis
        "depth_n": _clamp(depth / 700.0),
        # solar wind speed -> turbulence / flow energy
        "speed_n": _clamp((wind_speed - 250.0) / 750.0),
        # solar wind density -> cloud density
        "dens_n": _clamp((wind_density - 0.5) / 30.0),
        # solar wind temperature -> emission energy / brightness
        "temp_n": _clamp((wind_temp - 10000.0) / 900000.0),
        # Bz -> colour temperature bias (south = warm crimson, north = cool blue)
        "bz_n": _clamp(bz / 20.0, -1.0, 1.0),
        # By -> filament base orientation
        "by_n": _clamp(by / 20.0, -1.0, 1.0),
        # Bt -> magnetic coherence / filament straightness
        "bt_n": _clamp(bt / 30.0),
        # Kp -> global activity: emission intensity + star activity
        "kp_n": _clamp(kp / 9.0),
        # composition bias from the quake's position on Earth
        "lon_bias": (((longitude + 180.0) % 360.0) / 360.0),
        "lat_bias": (((latitude + 90.0) % 180.0) / 180.0),
    }
    params["generation_number"] = generation_number
    return params


def _derive_seed(live_data, generation_number):
    parts = [f"gen:{generation_number}"]
    for key in sorted(live_data.keys()):
        parts.append(f"{key}:{float(live_data[key]):.5f}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


# ============================================================================
# COMPOSITION: BLOB PLACEMENT
#
# Composition is generated BEFORE fine detail. A dominant mass, one or
# two secondary masses, placed asymmetrically. Negative space is whatever
# is left uncovered.
# ============================================================================

def _build_blobs(rng, params):
    lon_bias = params["lon_bias"]
    lat_bias = params["lat_bias"]
    dens_n = params["dens_n"]

    dom_cx = _clamp(0.5 + (lon_bias - 0.5) * 0.55 + rng.uniform(-0.12, 0.12), 0.16, 0.84)
    dom_cy = _clamp(0.5 + (lat_bias - 0.5) * 0.5 + rng.uniform(-0.12, 0.12), 0.16, 0.84)
    dom_rx = rng.uniform(0.30, 0.44) + 0.08 * dens_n
    dom_ry = rng.uniform(0.22, 0.34) + 0.06 * dens_n
    dom_rot = rng.uniform(0.0, math.pi)

    blobs = [
        {
            "cx": dom_cx, "cy": dom_cy,
            "rx": dom_rx, "ry": dom_ry,
            "rot": dom_rot, "weight": 1.0,
            "dominant": True,
        }
    ]

    secondary_count = 1 if rng.random() < 0.65 else 2
    for _ in range(secondary_count):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        dist = rng.uniform(0.38, 0.62)
        cx = _clamp(dom_cx + math.cos(angle) * dist, 0.05, 0.95)
        cy = _clamp(dom_cy + math.sin(angle) * dist * 0.75, 0.05, 0.95)
        rx = rng.uniform(0.12, 0.22)
        ry = rng.uniform(0.10, 0.18)
        rot = rng.uniform(0.0, math.pi)
        blobs.append({
            "cx": cx, "cy": cy, "rx": rx, "ry": ry, "rot": rot,
            "weight": rng.uniform(0.5, 0.8), "dominant": False,
        })

    return blobs


def _blob_influence(blobs, x, y):
    """
    Union of soft elliptical falloffs. Also returns the dominant blob's
    own influence separately (used later for near/far depth blending).
    """
    union_gap = 1.0
    dominant_val = 0.0

    for blob in blobs:
        dx = x - blob["cx"]
        dy = y - blob["cy"]
        cos_r = math.cos(-blob["rot"])
        sin_r = math.sin(-blob["rot"])
        rdx = dx * cos_r - dy * sin_r
        rdy = dx * sin_r + dy * cos_r
        d = math.sqrt((rdx / blob["rx"]) ** 2 + (rdy / blob["ry"]) ** 2)
        val = _smooth_falloff(0.55, 1.35, d) * blob["weight"]
        union_gap *= (1.0 - val)
        if blob["dominant"]:
            dominant_val = val

    return 1.0 - union_gap, dominant_val


# ============================================================================
# COLOUR: PALETTE LUT
#
# A curated crimson / burgundy / magenta / violet / deep-blue family,
# shifted warm or cool by the interplanetary magnetic field's Bz value,
# with brightness / emission reach driven by solar wind temperature and
# geomagnetic Kp. Built once as a 256-entry lookup table.
# ============================================================================

_COOL_STOPS = [
    (0.00, (4, 4, 11)),
    (0.14, (12, 10, 34)),
    (0.32, (28, 18, 66)),
    (0.50, (58, 26, 104)),
    (0.68, (110, 42, 150)),
    (0.83, (190, 96, 190)),
    (0.94, (235, 170, 215)),
    (1.00, (255, 244, 250)),
]

_WARM_STOPS = [
    (0.00, (6, 3, 6)),
    (0.14, (36, 8, 22)),
    (0.32, (82, 14, 34)),
    (0.50, (150, 24, 42)),
    (0.68, (206, 56, 52)),
    (0.83, (245, 122, 70)),
    (0.94, (255, 190, 130)),
    (1.00, (255, 248, 224)),
]


def _build_palette_lut(params):
    warm_bias = _clamp((1.0 - params["bz_n"]) / 2.0)
    emission_reach = _clamp(0.35 * params["temp_n"] + 0.25 * params["kp_n"])

    stops = []
    for (pos, cool_c), (_, warm_c) in zip(_COOL_STOPS, _WARM_STOPS):
        color = _lerp_color(cool_c, warm_c, warm_bias)
        stops.append((pos, color))

    r_table = [0] * 256
    g_table = [0] * 256
    b_table = [0] * 256

    for i in range(256):
        # push brighter with emission_reach so hot data pushes more of the
        # range into the glowing top end of the palette
        d = i / 255.0
        d = d ** (1.0 - 0.35 * emission_reach)

        for s in range(len(stops) - 1):
            p0, c0 = stops[s]
            p1, c1 = stops[s + 1]
            if p0 <= d <= p1 or s == len(stops) - 2:
                span = (p1 - p0) if p1 != p0 else 1.0
                t = _clamp((d - p0) / span)
                color = _lerp_color(c0, c1, t)
                break

        r_table[i] = int(_clamp(color[0], 0, 255))
        g_table[i] = int(_clamp(color[1], 0, 255))
        b_table[i] = int(_clamp(color[2], 0, 255))

    return r_table, g_table, b_table


# ============================================================================
# STEP 1: DENSITY FIELD
#
# Domain-warped fractal noise, sculpted by the blob composition, carved
# by a void field. This single pass also records the dominant-blob
# influence (used for depth blending) in the same loop.
# ============================================================================

def _generate_density_field(seed, params, blobs):
    # Stage-1 warp: broad, defines the large irregular cloud silhouette.
    warp1_oct_x = _build_fbm_octaves(random.Random(seed + 11), base_cells=3, octaves=3,
                                      persistence=0.55, lacunarity=2.1)
    warp1_oct_y = _build_fbm_octaves(random.Random(seed + 23), base_cells=3, octaves=3,
                                      persistence=0.55, lacunarity=2.1)
    # Stage-2 warp: applied on top of stage-1's result, at a smaller scale
    # and smaller amplitude. This is what breaks a clean ellipse edge into
    # a genuinely ragged, non-parallel boundary (fixes the "banding" look).
    warp2_oct_x = _build_fbm_octaves(random.Random(seed + 131), base_cells=6, octaves=2,
                                      persistence=0.5, lacunarity=2.0)
    warp2_oct_y = _build_fbm_octaves(random.Random(seed + 149), base_cells=6, octaves=2,
                                      persistence=0.5, lacunarity=2.0)

    macro_oct = _build_fbm_octaves(random.Random(seed + 37), base_cells=4, octaves=3,
                                    persistence=0.5, lacunarity=2.2)
    detail_oct = _build_fbm_octaves(random.Random(seed + 59), base_cells=10, octaves=4,
                                     persistence=0.5, lacunarity=2.0)
    void_oct = _build_fbm_octaves(random.Random(seed + 71), base_cells=5, octaves=2,
                                   persistence=0.5, lacunarity=2.3)

    warp1_amount = 0.16 + 0.10 * params["speed_n"]
    warp2_amount = 0.045 + 0.03 * params["speed_n"]
    gamma = 1.55 - 0.75 * params["mag_n"]
    void_thresh = 0.62 - 0.10 * params["dens_n"]
    void_strength = 0.85

    density_buf = bytearray(INTERNAL_WIDTH * INTERNAL_HEIGHT)
    far_weight_buf = bytearray(INTERNAL_WIDTH * INTERNAL_HEIGHT)

    for y in range(INTERNAL_HEIGHT):
        ny = (y + 0.5) / INTERNAL_HEIGHT
        row_offset = y * INTERNAL_WIDTH
        for x in range(INTERNAL_WIDTH):
            nx = (x + 0.5) / INTERNAL_WIDTH

            wx1 = _fbm(warp1_oct_x, nx, ny) * warp1_amount
            wy1 = _fbm(warp1_oct_y, nx, ny) * warp1_amount
            mnx = _clamp(nx + wx1)
            mny = _clamp(ny + wy1)

            wx2 = _fbm(warp2_oct_x, mnx, mny) * warp2_amount
            wy2 = _fbm(warp2_oct_y, mnx, mny) * warp2_amount
            wnx = _clamp(mnx + wx2)
            wny = _clamp(mny + wy2)

            macro = _fbm(macro_oct, wnx, wny)
            detail = _fbm(detail_oct, wnx, wny)
            base = macro * 0.62 + detail * 0.38
            base01 = _clamp(base * 0.5 + 0.5)

            influence, dominant_influence = _blob_influence(blobs, wnx, wny)
            density = base01 * influence

            voidn = _fbm(void_oct, nx, ny) * 0.5 + 0.5
            if voidn > void_thresh:
                t = (voidn - void_thresh) / max(1e-6, (1.0 - void_thresh))
                density *= max(0.0, 1.0 - t * void_strength)

            density = _clamp(density) ** gamma
            density = _clamp(density)

            density_buf[row_offset + x] = int(density * 255)
            far_weight_buf[row_offset + x] = int((1.0 - dominant_influence) * 255)

    return density_buf, far_weight_buf


# ============================================================================
# STEP 2: DEPTH LAYERING
#
# A blurred, lower-contrast copy stands in for distant / atmospheric
# cloud. It is blended back under the sharp field, weighted by how far
# each pixel is from the dominant (near) mass, and by earthquake depth.
# ============================================================================

def _apply_depth_layers(density_img, far_weight_img, params):
    blur_radius = 2.2 + 2.0 * params["depth_n"]
    far_img = density_img.filter(ImageFilter.GaussianBlur(blur_radius))
    far_img = far_img.point(lambda v: int(v * (0.55 + 0.2 * (1.0 - params["depth_n"]))))

    depth_strength = 0.35 + 0.45 * params["depth_n"]
    far_mask = far_weight_img.point(lambda v: int(v * depth_strength))

    return Image.composite(far_img, density_img, far_mask)


# ============================================================================
# STEP 3: FILAMENT SYSTEM
#
# Curved filaments traced through a noise-driven flow field, biased by
# By (orientation) and Bt (coherence), turbulence from solar wind speed.
# They brighten in dense regions and dissolve into the voids.
# ============================================================================

def _sample_density(density_buf, x, y):
    px = int(_clamp(x) * (INTERNAL_WIDTH - 1))
    py = int(_clamp(y) * (INTERNAL_HEIGHT - 1))
    return density_buf[py * INTERNAL_WIDTH + px] / 255.0


def _curl_direction(coarse_oct, fine_oct, x, y, eps, coherence, rotation):
    """
    Direction of flow at (x, y), derived from the rotated gradient (curl)
    of two potential fields at different scales. This gives every point
    on the canvas its own locally-varying sweep direction, instead of the
    filament system following one global constant angle (which is what
    produced the "combed / parallel bands" artifact previously).
    """
    x0 = _clamp(x - eps)
    x1 = _clamp(x + eps)
    y0 = _clamp(y - eps)
    y1 = _clamp(y + eps)
    denom = max(1e-6, (x1 - x0))
    denom_y = max(1e-6, (y1 - y0))

    dxc = (_fbm(coarse_oct, x1, y) - _fbm(coarse_oct, x0, y)) / denom
    dyc = (_fbm(coarse_oct, x, y1) - _fbm(coarse_oct, x, y0)) / denom_y
    dxf = (_fbm(fine_oct, x1, y) - _fbm(fine_oct, x0, y)) / denom
    dyf = (_fbm(fine_oct, x, y1) - _fbm(fine_oct, x, y0)) / denom_y

    # rotate gradient 90 degrees -> divergence-free-looking flow vector
    vxc, vyc = dyc, -dxc
    vxf, vyf = dyf, -dxf

    vx = vxc * coherence + vxf * (1.0 - coherence)
    vy = vyc * coherence + vyf * (1.0 - coherence)

    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    rvx = vx * cos_r - vy * sin_r
    rvy = vx * sin_r + vy * cos_r

    return math.atan2(rvy, rvx)


def _trace_filaments(seed, params, density_buf, r_table, g_table, b_table):
    rng = random.Random(seed + 4242)
    coarse_flow_oct = _build_fbm_octaves(random.Random(seed + 4343), base_cells=3, octaves=2,
                                          persistence=0.5, lacunarity=2.0)
    fine_flow_oct = _build_fbm_octaves(random.Random(seed + 4444), base_cells=9, octaves=2,
                                        persistence=0.5, lacunarity=2.0)

    filament_rgb = Image.new("RGB", (INTERNAL_WIDTH, INTERNAL_HEIGHT), (0, 0, 0))
    filament_alpha = Image.new("L", (INTERNAL_WIDTH, INTERNAL_HEIGHT), 0)
    draw_rgb = ImageDraw.Draw(filament_rgb)
    draw_alpha = ImageDraw.Draw(filament_alpha)

    rotation_bias = params["by_n"] * math.pi * 0.6
    turbulence = 0.5 + 1.1 * params["speed_n"]
    coherence = 0.30 + 0.55 * params["bt_n"]
    flow_eps = 1.5 / INTERNAL_WIDTH

    num_filaments = int(70 + 170 * params["mag_n"] + 90 * params["dens_n"])
    num_filaments = max(50, min(num_filaments, 320))

    step_len = (1.1 / INTERNAL_WIDTH) * (1.0 + 0.4 * params["speed_n"])

    queue = []
    for _ in range(num_filaments):
        start = None
        for _try in range(6):
            cand = (rng.random(), rng.random())
            d = _sample_density(density_buf, *cand)
            if 0.12 <= d <= 0.8:
                start = cand
                break
        if start is None:
            start = (rng.random(), rng.random())
        length = rng.randint(14, 46)
        queue.append((start, length, 0))

    branches_spawned = 0
    max_branches = int(30 + 60 * params["bt_n"])

    i = 0
    while i < len(queue):
        (sx, sy), length, depth_level = queue[i]
        i += 1

        pos_x, pos_y = sx, sy
        for step in range(length):
            d_here = _sample_density(density_buf, pos_x, pos_y)
            fade_in = _smoothstep(step / 5.0)
            fade_out = _smoothstep((length - step) / 6.0)
            progress_fade = min(fade_in, fade_out)

            if d_here < 0.05 and rng.random() < 0.35:
                break

            flow_angle = _curl_direction(
                coarse_flow_oct, fine_flow_oct, pos_x, pos_y, flow_eps, coherence, rotation_bias
            )
            jitter = rng.uniform(-1.0, 1.0) * (1.0 - coherence) * 0.35 * turbulence
            angle = flow_angle + jitter

            new_x = pos_x + math.cos(angle) * step_len
            new_y = pos_y + math.sin(angle) * step_len

            if not (0.0 <= new_x <= 1.0 and 0.0 <= new_y <= 1.0):
                break

            thickness = 1.0 + 2.4 * d_here
            alpha = _clamp(d_here * 1.3) * progress_fade

            if alpha > 0.02:
                brightness = _clamp(d_here * 1.35 + 0.15)
                idx = int(brightness * 255)
                color = (r_table[idx], g_table[idx], b_table[idx])

                x0 = pos_x * INTERNAL_WIDTH
                y0 = pos_y * INTERNAL_HEIGHT
                x1 = new_x * INTERNAL_WIDTH
                y1 = new_y * INTERNAL_HEIGHT

                width = max(1, int(round(thickness)))
                draw_rgb.line([(x0, y0), (x1, y1)], fill=color, width=width)
                draw_alpha.line([(x0, y0), (x1, y1)], fill=int(alpha * 255), width=width)

            if (
                depth_level == 0
                and branches_spawned < max_branches
                and step > 6
                and rng.random() < 0.02 + 0.015 * params["bt_n"]
            ):
                queue.append(((pos_x, pos_y), rng.randint(8, 20), 1))
                branches_spawned += 1

            pos_x, pos_y = new_x, new_y

    return filament_rgb, filament_alpha


# ============================================================================
# STEP 4: EMISSION BLOOM
#
# The brightest cores softly illuminate the cloud material around them.
# ============================================================================

def _apply_bloom(base_img, density_img, params):
    threshold = 178 - int(30 * params["kp_n"])
    hot = density_img.point(lambda v: max(0, min(255, int((v - threshold) * (255.0 / max(1, 255 - threshold))))))
    glow_radius = 3.0 + 3.0 * params["kp_n"]
    hot = hot.filter(ImageFilter.GaussianBlur(glow_radius))

    tint = (255, 214, 198) if params["bz_n"] < 0 else (214, 226, 255)
    glow_rgb = Image.merge("RGB", (
        hot.point(lambda v: int(v * tint[0] / 255)),
        hot.point(lambda v: int(v * tint[1] / 255)),
        hot.point(lambda v: int(v * tint[2] / 255)),
    ))

    return ImageChops.screen(base_img, glow_rgb)


# ============================================================================
# STEP 5: STARS
#
# Sparse, non-uniform. Dense nebula regions suppress stars. Kp increases
# how many stars are bright / active enough to punch through the cloud.
# ============================================================================

def _scatter_stars(img, seed, params, density_buf):
    rng = random.Random(seed + 909090)
    pixels = img.load()

    num_dim = int(220 + 160 * (1.0 - params["dens_n"]))
    num_medium = int(14 + 24 * params["kp_n"])
    num_bright = int(1 + 5 * params["kp_n"])

    def place(count, brightness_range, glow):
        placed = 0
        attempts = 0
        while placed < count and attempts < count * 8:
            attempts += 1
            x = rng.randrange(INTERNAL_WIDTH)
            y = rng.randrange(INTERNAL_HEIGHT)
            d = density_buf[y * INTERNAL_WIDTH + x] / 255.0
            suppress = d * (0.92 - 0.45 * params["kp_n"])
            if rng.random() < suppress:
                continue

            v = rng.randint(*brightness_range)
            cool = rng.random() < 0.5
            color = (v, v, min(255, v + 12)) if cool else (min(255, v + 10), v, v)
            pixels[x, y] = color

            if glow and 0 < x < INTERNAL_WIDTH - 1 and 0 < y < INTERNAL_HEIGHT - 1:
                dim = max(0, v // 3)
                dim_color = (dim, dim, dim)
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx_, ny_ = x + ox, y + oy
                    existing = pixels[nx_, ny_]
                    pixels[nx_, ny_] = tuple(max(existing[c], dim_color[c]) for c in range(3))

            placed += 1

    place(num_dim, (40, 95), glow=False)
    place(num_medium, (120, 195), glow=False)
    place(num_bright, (225, 255), glow=True)


# ============================================================================
# STEP 6: PIXEL-ART FINISHING
#
# Quantizing to a limited number of levels per channel with a *tiled*
# dither matrix produced a visible repeating grid on top of the nebula.
# Instead: a non-periodic, per-pixel pseudo-random threshold (deterministic
# hash, not a tiled matrix), with its strength modulated by how much
# natural detail already exists at that pixel - strong dither in flat
# blob interiors where banding would otherwise show, weak-to-none in
# already-detailed/filament-rich areas.
# ============================================================================

def _quantize_with_dither(img, density_img, seed, levels=40):
    blurred = density_img.filter(ImageFilter.GaussianBlur(1.5))
    edge_img = ImageChops.difference(density_img, blurred)

    pixels = img.load()
    edge_px = edge_img.load()
    step = 255.0 / (levels - 1)

    for y in range(INTERNAL_HEIGHT):
        for x in range(INTERNAL_WIDTH):
            flatness = 1.0 - _clamp(edge_px[x, y] / 40.0)
            h = _pixel_hash01(x, y, seed)
            threshold = (h - 0.5) * step * flatness

            r, g, b = pixels[x, y]
            r = round(_clamp(r + threshold, 0, 255) / step) * step
            g = round(_clamp(g + threshold, 0, 255) / step) * step
            b = round(_clamp(b + threshold, 0, 255) / step) * step
            pixels[x, y] = (
                int(_clamp(r, 0, 255)),
                int(_clamp(g, 0, 255)),
                int(_clamp(b, 0, 255)),
            )


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

def render_wallpaper(live_data, generation_number):
    """
    Generate one procedural nebula wallpaper from live scientific data.

    Args:
        live_data: dict as returned by api_client2.get_all_data() - expects
            magnitude, depth, latitude, longitude, solar_wind_speed,
            solar_wind_density, solar_wind_temperature, bz, by, bt, kp.
        generation_number: int, mixed into the seed so repeated calls with
            the same live_data still produce distinct, related artwork.

    Returns:
        str: filesystem path to the rendered 1920x1080 PNG.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seed = _derive_seed(live_data, generation_number)
    params = _derive_params(live_data, generation_number)
    comp_rng = random.Random(seed + 1)

    blobs = _build_blobs(comp_rng, params)
    r_table, g_table, b_table = _build_palette_lut(params)

    density_buf, far_weight_buf = _generate_density_field(seed, params, blobs)
    density_img = Image.frombytes("L", (INTERNAL_WIDTH, INTERNAL_HEIGHT), bytes(density_buf))
    far_weight_img = Image.frombytes("L", (INTERNAL_WIDTH, INTERNAL_HEIGHT), bytes(far_weight_buf))

    density_img = _apply_depth_layers(density_img, far_weight_img, params)
    # refresh density_buf so filaments/stars read the depth-blended field
    density_buf = bytearray(density_img.tobytes())

    base_img = density_img.convert("RGB")
    base_img = Image.merge("RGB", (
        density_img.point(r_table),
        density_img.point(g_table),
        density_img.point(b_table),
    ))

    filament_rgb, filament_alpha = _trace_filaments(seed, params, density_buf, r_table, g_table, b_table)
    composed = Image.composite(filament_rgb, base_img, filament_alpha)

    composed = _apply_bloom(composed, density_img, params)

    _scatter_stars(composed, seed, params, density_buf)

    _quantize_with_dither(composed, density_img, seed, levels=40)

    final_img = composed.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

    filename = f"nebula_gen{generation_number:05d}_{seed:012x}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    final_img.save(output_path, "PNG")

    return output_path


# ============================================================================
# MANUAL SANITY CHECK
# ============================================================================

if __name__ == "__main__":
    sample_live_data = {
        "magnitude": 5.8,
        "depth": 32.0,
        "latitude": 21.4,
        "longitude": -158.0,
        "solar_wind_speed": 610.0,
        "solar_wind_density": 8.4,
        "solar_wind_temperature": 240000.0,
        "bz": -6.2,
        "by": 3.1,
        "bt": 9.0,
        "kp": 5.0,
    }

    path = render_wallpaper(sample_live_data, 1)
    print(f"Wallpaper written to: {path}")

    with Image.open(path) as check_img:
        print(f"Size: {check_img.size}")
        assert check_img.size == (FINAL_WIDTH, FINAL_HEIGHT)
    print("Sanity check passed.")