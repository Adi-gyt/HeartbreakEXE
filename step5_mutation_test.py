import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Setup Output Directory
OUT = Path('output_step5')
OUT.mkdir(exist_ok=True, parents=True)

# 1. Load Original Image and Semantic Mask
# We use the semantic map from Step 4, matching exactly the cropped dimensions.
img = np.array(Image.open('image_01_cropped.png').convert('RGB'))
sem_map = np.array(Image.open('output_step4/image_01_semantic_regions.png').convert('RGB'))
h, w, c = img.shape

# Semantic reference colors matching Step 4 export
sem_colors = {
    'sky_and_clouds': [100, 150, 250],
    'road_and_traffic': [250, 50, 50],
    'rain_streaks': [200, 255, 255],
}

def get_mask(label):
    color = sem_colors[label]
    # Create a boolean mask identifying exactly the pixels assigned to this label
    mask = (sem_map[:, :, 0] == color[0]) & (sem_map[:, :, 1] == color[1]) & (sem_map[:, :, 2] == color[2])
    return mask

# ==========================================
# TEST 1: Sky & Clouds (Subtle Purple/Blue Hue Shift)
# ==========================================
sky_mask = get_mask('sky_and_clouds')
img_sky = img.copy()

# Convert to HSV to easily shift the hue
hsv_sky = cv2.cvtColor(img_sky, cv2.COLOR_RGB2HSV).astype(np.int32)
# OpenCV Hue is 0-179. Shift slightly towards purple.
hsv_sky[sky_mask, 0] = (hsv_sky[sky_mask, 0] + 15) % 180 
# Increase saturation slightly to ensure the color shift is visible
hsv_sky[sky_mask, 1] = np.clip(hsv_sky[sky_mask, 1] + 20, 0, 255)

img_sky_mod = cv2.cvtColor(hsv_sky.astype(np.uint8), cv2.COLOR_HSV2RGB)
img_sky = np.where(sky_mask[:, :, None], img_sky_mod, img)


# ==========================================
# TEST 2: Road & Traffic (Boost Red/Pink Lighting)
# ==========================================
road_mask = get_mask('road_and_traffic')
img_road = img.copy()
img_road_float = img_road.astype(np.float32)

# Boost the Red channel and slightly boost the Blue channel to emulate a pinkish neon glow
img_road_float[road_mask, 0] = np.clip(img_road_float[road_mask, 0] * 1.2 + 30, 0, 255) # R channel
img_road_float[road_mask, 2] = np.clip(img_road_float[road_mask, 2] * 1.1 + 15, 0, 255) # B channel
img_road = np.where(road_mask[:, :, None], img_road_float.astype(np.uint8), img)


# ==========================================
# TEST 3: Rain Streaks (Increase Brightness)
# ==========================================
rain_mask = get_mask('rain_streaks')
img_rain = img.copy()
img_rain_float = img_rain.astype(np.float32)

# Substantially multiply brightness of just the rain pixels
img_rain_float[rain_mask] = np.clip(img_rain_float[rain_mask] * 1.5 + 40, 0, 255)
img_rain = np.where(rain_mask[:, :, None], img_rain_float.astype(np.uint8), img)


# ==========================================
# TEST 4: Combined Mutations
# ==========================================
img_combined = img.copy()
img_combined = np.where(sky_mask[:, :, None], img_sky_mod, img_combined)
img_combined = np.where(road_mask[:, :, None], img_road_float.astype(np.uint8), img_combined)
img_combined = np.where(rain_mask[:, :, None], img_rain_float.astype(np.uint8), img_combined)


# ==========================================
# VERIFICATION & OUTPUT
# ==========================================
def verify(mod_img, mask, name):
    mod_pixels = np.sum(mask)
    total_pixels = h * w
    percent = mod_pixels / total_pixels
    
    # Assert pixels outside the mask are completely untouched
    outside_mask = ~mask
    diff = np.sum(img[outside_mask] != mod_img[outside_mask])
    
    print(f"[{name}]")
    print(f" - Pixels modified: {mod_pixels}")
    print(f" - Percentage of Image: {percent:.2%}")
    print(f" - Outside pixels unchanged: {'Yes' if diff == 0 else 'NO (' + str(diff) + ' differences found)'}")
    print()

verify(img_sky, sky_mask, "Test 1: Sky")
verify(img_road, road_mask, "Test 2: Traffic")
verify(img_rain, rain_mask, "Test 3: Rain")

Image.fromarray(img_sky).save(OUT / 'test_sky_mutation.png')
Image.fromarray(img_road).save(OUT / 'test_traffic_mutation.png')
Image.fromarray(img_rain).save(OUT / 'test_rain_mutation.png')
Image.fromarray(img_combined).save(OUT / 'test_combined_mutation.png')

print(f"Outputs successfully generated and saved to {OUT}/")