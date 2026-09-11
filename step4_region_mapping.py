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

with open('gemini_analysis.json', 'r') as f:
    json_data = json.load(f)

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)

targets = []
for sr in json_data.get('semantic_regions', []):
    colors = [hex_to_rgb(sr['dominant_color'])]
    for sc in sr.get('secondary_colors', []):
        colors.append(hex_to_rgb(sc))
    targets.append({
        'id': sr['id'], 'label': sr['label'], 'bbox': sr['approx_bbox'],
        'colors': colors, 'type': 'semantic', 'safety': sr['mutation_safety']
    })

# Pre-calculate absolute bounding boxes for Pattern Overlap checks
pattern_bboxes = {}
for pr in json_data.get('pattern_regions', []):
    bx, by, bxx, byy = pr['approx_bbox']
    pattern_bboxes[pr['label']] = [bx*w, by*h, bxx*w, byy*h]
    targets.append({
        'id': pr['id'] + 100, 'label': pr['label'], 'bbox': pr['approx_bbox'],
        'colors': [], 'type': 'pattern', 'safety': pr['mutation_safety']
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
    
    # Rain Streaks (Vertical, sparse, extremely small)
    intersect_rain = get_intersection_ratio(comp_rect, pattern_bboxes.get('rain_streaks', [0,0,w,h]))
    if intersect_rain > 0.5:
        if cw < 15 and ch > 10 and (cw / max(1, ch)) < 0.4 and area < 500 and mean_lum > 80:
            return 'rain_streaks', 'atmosphere_only', True
            
    # Power Lines (Horizontal/diagonal, thin thickness)
    intersect_wire = get_intersection_ratio(comp_rect, pattern_bboxes.get('power_lines', [0,0,w,h]))
    if intersect_wire > 0.3:
        length = max(cw, ch)
        thickness = area / max(1, length) # Calculated exact thickness
        aspect = cw / max(1, ch)
        if length > 30 and thickness < 8 and (aspect > 3.0 or aspect < 0.3) and mean_lum < 120 and area < 4000:
            return 'power_lines', 'shape_sensitive', True
            
    # City Windows (Small, compact, bright)
    intersect_window = get_intersection_ratio(comp_rect, pattern_bboxes.get('city_windows', [0,0,w,h]))
    if intersect_window > 0.8:
        if area < 150 and cw < 25 and ch < 25 and mean_lum > 140:
            return 'city_windows', 'color+light_safe', True
            
    # 2. SEMANTIC FALLBACK
    best_target, best_score = None, float('inf')
    for t in targets:
        if t['type'] == 'pattern': 
            continue
            
        bx, by, bxx, byy = t['bbox']
        px, py, pxx, pyy = bx*w, by*h, bxx*w, byy*h
        
        # Determine if component centroid is inside the target bbox with slight padding
        is_inside = (px - 20 <= cx <= pxx + 20) and (py - 20 <= cy <= pyy + 20)
        
        min_color_dist = min([np.linalg.norm(mean_color - tc) for tc in t['colors']])
        
        # Heavy penalty if outside bounding box to prevent overlapping regions stealing pixels
        penalty = 0 if is_inside else 500
        score = min_color_dist + penalty
        
        if score < best_score:
            best_score = score
            best_target = t
            
    return best_target['label'], best_target['safety'], False

for comp in raw_components:
    lbl, safety, is_pattern = get_semantic_assignment(comp)
    comp['sem_label'], comp['sem_safety'], comp['is_pattern'] = lbl, safety, is_pattern


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

rng = np.random.default_rng(42)

sem_colors = {
    'sky_and_clouds': [100, 150, 250], 'city_skyline': [150, 150, 150], 'midground_foliage': [50, 200, 50],
    'road_and_traffic': [250, 50, 50], 'foreground_structure': [50, 50, 100], 'rain_streaks': [200, 255, 255],
    'city_windows': [255, 255, 100], 'power_lines': [200, 100, 255]
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
        'confidence': 0.85,
        'mutation_safety': comp_dict[members[0]]['sem_safety'],
    })
    region_counter += 1

Image.fromarray(merged_map).save(OUT / 'image_01_merged_regions.png')
Image.fromarray(sem_map).save(OUT / 'image_01_semantic_regions.png')
Image.fromarray(overlay).save(OUT / 'image_01_region_overlay.png')

with open(OUT / 'image_01_regions.json', 'w') as f:
    json.dump(report_data, f, indent=2)

pd.DataFrame(report_data).to_csv(OUT / 'image_01_regions.csv', index=False)
print(f"Step 4 Fix Complete. Files saved to {OUT}/")