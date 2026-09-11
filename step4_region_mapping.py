import json
import numpy as np
import cv2
import pandas as pd
import scipy.spatial
from PIL import Image
from pathlib import Path

# Setup Output Directory
OUT = Path('output_step4')
OUT.mkdir(exist_ok=True, parents=True)

# ==========================================
# 1. LOAD DATA
# ==========================================
img = np.array(Image.open('image_01_cropped.png').convert('RGB'))
h, w = img.shape[:2]
TOTAL_PIXELS = h * w

# STANDARDIZED KNOWLEDGE SOURCE (image_01.json). gemini_analysis.json is no
# longer read anywhere in this pipeline -- it is not the semantic authority.
with open('knowledge/image_01.json', 'r') as f:
    dataset = json.load(f)

images = dataset.get('images')
if not images:
    raise ValueError("image_01.json contains no 'images' entries")

json_data = images[0]  # single-image knowledge file, same assumption region_mapper.py makes

expected_w = json_data.get('dimensions', {}).get('width')
expected_h = json_data.get('dimensions', {}).get('height')
if expected_w is not None and expected_h is not None:
    if (w, h) != (expected_w, expected_h):
        raise ValueError(
            f"image_01_cropped.png is {w}x{h} but image_01.json declares "
            f"{expected_w}x{expected_h} -- refusing to run against a mismatched image "
            f"(Non-negotiable rule 6: preserve image dimensions)."
        )

# Standardized regions carry no per-region dominant_color/secondary_colors, so
# semantic assignment can no longer be color-driven (see Section 4 below).
# `bbox` normalized coords are converted to absolute pixel coords once, up front.
targets = []
for sr in json_data.get('semantic_regions', []):
    bx, by, bxx, byy = sr['bbox']
    targets.append({
        'label': sr['region_name'],
        'bbox_abs': [bx * w, by * h, bxx * w, byy * h],
        'type': 'semantic',
        'safety': sr['mutation_safety'],
        'confidence': sr.get('confidence', 0.85),
        'visual_importance': sr.get('visual_importance', 0.5),
        'mutation_priority': sr.get('mutation_priority', 0.5),
    })

# Pre-calculate absolute bounding boxes for Pattern Overlap checks
pattern_bboxes = {}
pattern_confidence = {}
pattern_safety = {}
for pr in json_data.get('pattern_regions', []):
    bx, by, bxx, byy = pr['approximate_bbox']
    pattern_bboxes[pr['pattern_name']] = [bx * w, by * h, bxx * w, byy * h]
    pattern_confidence[pr['pattern_name']] = pr.get('confidence', 0.85)
    pattern_safety[pr['pattern_name']] = pr['mutation_safety']
    targets.append({
        'label': pr['pattern_name'],
        'bbox_abs': [bx * w, by * h, bxx * w, byy * h],
        'type': 'pattern',
        'safety': pr['mutation_safety'],
        'confidence': pr.get('confidence', 0.85),
        'visual_importance': 0.5,
        'mutation_priority': 0.5,
    })

# ==========================================
# 2. COLOR QUANTIZATION
# ==========================================
blur = cv2.GaussianBlur(img, (3, 3), 0)
lab = cv2.cvtColor(blur, cv2.COLOR_RGB2LAB).astype(np.float32)
pix = lab.reshape(-1, 3)
K = 12
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
_, labels, centers = cv2.kmeans(pix, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
labels = labels.reshape(h, w)
rgb_centers = cv2.cvtColor(centers.astype(np.uint8)[None,...], cv2.COLOR_LAB2RGB)[0]

# ==========================================
# 3. RAW COMPONENTS
# ==========================================
raw_components = []
cc_map = np.zeros((h, w), dtype=np.int32)
cc_id_counter = 1

for k in range(K):
    mask = (labels == k).astype(np.uint8)
    n, cc, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 5: 
            continue # Drop microscopic noise immediately
            
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        ww, hh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = cents[i]
        
        cc_mask = (cc == i).astype(np.uint8)
        cc_map[cc_mask > 0] = cc_id_counter
        
        raw_components.append({
            'id': cc_id_counter, 'area': area, 'k': k, 'x': x, 'y': y, 'w': ww, 'h': hh,
            'cx': cx, 'cy': cy, 'mean_rgb': rgb_centers[k]
        })
        cc_id_counter += 1

# ==========================================
# 4. HIGH-PRECISION SEMANTIC ASSIGNMENT
# ==========================================

# A component must have at least this fraction of its own area contained
# within a knowledge region's bbox to be claimed by that region. This is
# what turns bbox from a "weak penalty" into a real spatial constraint.
MIN_SEMANTIC_OVERLAP = 0.5


def get_intersection_ratio(comp_rect, bbox):
    cx, cy, cw, ch = comp_rect
    px, py, pxx, pyy = bbox
    ix, iy = max(cx, px), max(cy, py)
    ixx, iyy = min(cx+cw, pxx), min(cy+ch, pyy)
    
    if ix < ixx and iy < iyy:
        return ((ixx - ix) * (iyy - iy)) / (cw * ch)
    return 0.0

def get_semantic_assignment(comp):
    cx, cy, mean_color = comp['cx'], comp['cy'], comp['mean_rgb']
    cw, ch, area = comp['w'], comp['h'], comp['area']
    comp_rect = [comp['x'], comp['y'], cw, ch]
    mean_lum = np.mean(mean_color)
    
    # 1. STRICT PATTERN FILTERS
    #
    # Safety and confidence for each pattern are pulled live from
    # pattern_safety/pattern_confidence (sourced from image_01.json) rather
    # than hardcoded, so this never silently reproduces gemini_analysis.json's
    # stale judgment (e.g. gemini said power_lines was shape_sensitive;
    # image_01.json says exclude -- image_01.json wins).

    # Rain Streaks (Vertical, sparse, extremely small)
    intersect_rain = get_intersection_ratio(comp_rect, pattern_bboxes.get('rain_streaks', [0,0,w,h]))
    if intersect_rain > 0.5 and 'rain_streaks' in pattern_safety:
        if cw < 15 and ch > 10 and (cw / max(1, ch)) < 0.4 and area < 500 and mean_lum > 80:
            return 'rain_streaks', pattern_safety['rain_streaks'], True, pattern_confidence['rain_streaks']

    # Power Lines (Horizontal/diagonal, thin thickness)
    intersect_wire = get_intersection_ratio(comp_rect, pattern_bboxes.get('power_lines', [0,0,w,h]))
    if intersect_wire > 0.3 and 'power_lines' in pattern_safety:
        length = max(cw, ch)
        thickness = area / max(1, length) # Calculated exact thickness
        aspect = cw / max(1, ch)
        if length > 30 and thickness < 8 and (aspect > 3.0 or aspect < 0.3) and mean_lum < 120 and area < 4000:
            return 'power_lines', pattern_safety['power_lines'], True, pattern_confidence['power_lines']

    # City Windows (Small, compact, bright)
    intersect_window = get_intersection_ratio(comp_rect, pattern_bboxes.get('city_windows', [0,0,w,h]))
    if intersect_window > 0.8 and 'city_windows' in pattern_safety:
        if area < 150 and cw < 25 and ch < 25 and mean_lum > 140:
            return 'city_windows', pattern_safety['city_windows'], True, pattern_confidence['city_windows']
            
    # 2. BBOX-PRIORITY SEMANTIC ASSIGNMENT
    #
    # image_01.json carries no per-region dominant_color/secondary_colors, and
    # the visually-similar dark foreground regions (overhang_roof,
    # bus_shelter_left, horizontal_railing, foreground_foliage,
    # character_silhouette) cannot be told apart by color regardless. Spatial
    # evidence -- how much of this component actually falls inside a given
    # knowledge region's bbox -- is now the sole assignment criterion.
    #
    # `targets` preserves image_01.json's semantic_regions list order, so
    # ties (equal overlap ratio) deterministically resolve to whichever
    # region appears first in the knowledge file.
    best_target, best_overlap, best_area = None, 0.0, None
    for t in targets:
        if t['type'] == 'pattern':
            continue

        overlap_ratio = get_intersection_ratio(comp_rect, t['bbox_abs'])
        t_area = (t['bbox_abs'][2] - t['bbox_abs'][0]) * (t['bbox_abs'][3] - t['bbox_abs'][1])
        # On a tie (e.g. a component fully inside a smaller region that is
        # itself nested inside a larger region, such as character_silhouette
        # inside bus_shelter_left), prefer the smaller/more specific region
        # instead of silently keeping whichever appeared first in the list.
        if overlap_ratio > best_overlap or (overlap_ratio == best_overlap and best_area is not None and t_area < best_area):
            best_overlap = overlap_ratio
            best_target = t
            best_area = t_area

    if best_target is not None and best_overlap >= MIN_SEMANTIC_OVERLAP:
        return best_target['label'], best_target['safety'], False, best_target['confidence']

    # No knowledge bbox contains enough of this component to justify a claim.
    # Per spec: do not force it into a foreground (or any) semantic region.
    return None, None, False, None

for comp in raw_components:
    lbl, safety, is_pattern, confidence = get_semantic_assignment(comp)
    comp['sem_label'], comp['sem_safety'], comp['is_pattern'], comp['sem_confidence'] = lbl, safety, is_pattern, confidence

# Components with no sufficiently-overlapping knowledge bbox were returned as
# (None, None, False) above -- drop them now rather than forcing them into a
# semantic region or letting a `None` label leak into grouping/output.
raw_components = [c for c in raw_components if c['sem_label'] is not None]


# ==========================================
# 5. SPATIAL & COLOR MERGING (KDTree)
# ==========================================
class DSU:
    def __init__(self, nodes): self.parent = {n: n for n in nodes}
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j: self.parent[root_i] = root_j

comp_dict = {c['id']: c for c in raw_components}
dsu = DSU(comp_dict.keys())

for label in set(c['sem_label'] for c in raw_components):
    label_comps = [c for c in raw_components if c['sem_label'] == label]
    
    # We enforce strict spatial isolation for patterns so they don't merge into one ID.
    is_pattern = label in ['rain_streaks', 'city_windows', 'power_lines']
    dist_t = 10 if is_pattern else 80 
    color_t = 60
        
    pts = np.array([[c['cx'], c['cy']] for c in label_comps])
    if len(pts) > 1:
        tree = scipy.spatial.cKDTree(pts)
        pairs = tree.query_pairs(dist_t)
        for i, j in pairs:
            u, v = label_comps[i]['id'], label_comps[j]['id']
            dist_c = np.linalg.norm(comp_dict[u]['mean_rgb'] - comp_dict[v]['mean_rgb'])
            if dist_c < color_t: 
                dsu.union(u, v)

groups = {}
for n in comp_dict.keys():
    root = dsu.find(n)
    if root not in groups: groups[root] = []
    groups[root].append(n)


# ==========================================
# 6. EXACT MASK EXTRACTION & VALIDATION
# ==========================================
sem_map = np.zeros((h, w, 3), dtype=np.uint8)
merged_map = np.zeros((h, w, 3), dtype=np.uint8)
overlay = img.copy()
# Exact per-pixel region-label map (0 = unassigned). Reuses `region_mask`,
# already computed below for bbox/centroid/contours -- no new segmentation.
region_label_map = np.zeros((h, w), dtype=np.uint16)

rng = np.random.default_rng(42)

# Visualization colors, one per label in the standardized image_01.json taxonomy.
sem_colors = {
    'sky_and_clouds': [100, 150, 250],
    'distant_cityscape': [150, 150, 150],
    'midground_trees': [50, 200, 50],
    'highway_traffic': [250, 50, 50],
    'overhang_roof': [90, 60, 30],
    'bus_shelter_left': [50, 50, 100],
    'horizontal_railing': [180, 120, 60],
    'foreground_foliage': [30, 120, 30],
    'character_silhouette': [200, 200, 0],
    'rain_streaks': [200, 255, 255],
    'city_windows': [255, 255, 100],
    'power_lines': [200, 100, 255],
}

report_data = []
region_counter = 1

for root, members in groups.items():
    area = sum(comp_dict[u]['area'] for u in members)
    label = comp_dict[members[0]]['sem_label']
    is_pattern = comp_dict[members[0]]['is_pattern']
    
    if (is_pattern and area < 5) or (not is_pattern and area < 100):
        continue
        
    region_mask = np.zeros((h, w), dtype=np.uint8)
    for u in members: 
        region_mask |= (cc_map == u).astype(np.uint8)
        
    ys, xs = np.where(region_mask > 0)
    
    merged_map[region_mask > 0] = rng.integers(50, 255, size=3, dtype=np.uint8)
    sem_map[region_mask > 0] = sem_colors.get(label, [255, 255, 255])
    region_label_map[region_mask > 0] = region_counter
    
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    
    report_data.append({
        'region_id': region_counter,
        'semantic_label': label,
        'area_pixels': area,
        'area_fraction': round(area / TOTAL_PIXELS, 5),
        'bbox': [round(xs.min()/w, 4), round(ys.min()/h, 4), round(xs.max()/w, 4), round(ys.max()/h, 4)],
        'centroid': [round(np.mean(xs)/w, 4), round(np.mean(ys)/h, 4)],
        'dominant_colors': [f"#{int(comp_dict[members[0]]['mean_rgb'][0]):02x}{int(comp_dict[members[0]]['mean_rgb'][1]):02x}{int(comp_dict[members[0]]['mean_rgb'][2]):02x}"],
        'confidence': comp_dict[members[0]]['sem_confidence'],
        'mutation_safety': comp_dict[members[0]]['sem_safety'],
        'region_label': region_counter,
    })
    region_counter += 1

image_id = json_data.get('image_id', 'image_01')

Image.fromarray(merged_map).save(OUT / f'{image_id}_merged_regions.png')
Image.fromarray(sem_map).save(OUT / f'{image_id}_semantic_regions.png')
Image.fromarray(overlay).save(OUT / f'{image_id}_region_overlay.png')

# Exact per-pixel region-label map, uint16 grayscale PNG. 0 = unassigned;
# positive integers correspond 1:1 to region_id/region_label in the JSON.
assert region_label_map.shape == (h, w), "label map dimensions must match source image"
cv2.imwrite(str(OUT / f'{image_id}_region_labels.png'), region_label_map)

with open(OUT / f'{image_id}_regions.json', 'w') as f:
    json.dump(report_data, f, indent=2)

pd.DataFrame(report_data).to_csv(OUT / f'{image_id}_regions.csv', index=False)
print(f"Step 4 Fix Complete. Files saved to {OUT}/")