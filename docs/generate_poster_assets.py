import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure vlfm is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlfm.vlm.yolov7 import YOLOv7
from vlfm.object_centric.sam_segmenter import SAMSegmenter

# ══════════════════════════════════════════════════════════════════
# Helper Methods (Exact parity with object_segmentation.py)
# ══════════════════════════════════════════════════════════════════

def increase_bbox_by_margin(bbox, margin):
    """Exact logic from object_segmentation.py Line 292"""
    x, y, w, h = bbox
    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    return (x, y, w, h)

def _create_bbox_crop(rgb: np.ndarray, bbox: tuple, bbox_margin=0) -> np.ndarray:
    """Exact logic from object_segmentation.py Line 316"""
    x, y, w, h = increase_bbox_by_margin(bbox, bbox_margin)
    x, y, w, h = int(x), int(y), int(w), int(h)
    # Ensure bounds (not in original but prevents crashes if bounding box exceeds image)
    crop = rgb[y : min(y + h, rgb.shape[0]), x : min(x + w, rgb.shape[1])]
    return crop

def _create_masked_crop(rgb: np.ndarray, mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """Exact logic from object_segmentation.py Line 329"""
    x, y, w, h = bbox
    # Apply the mask first
    masked = rgb * np.expand_dims(mask, axis = -1)
    x, y, w, h = int(x), int(y), int(w), int(h)
    # Get the cropped mask
    crop = masked[y : min(y + h, rgb.shape[0]), x : min(x + w, rgb.shape[1])]
    return crop

def _create_transparent_crop(rgb: np.ndarray, mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """Creates a transparent PNG crop specifically for diagram aesthetics"""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # Create RGBA image
    rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    # Set alpha channel to 0 where mask is False (transparent background)
    rgba[~mask, 3] = 0
    # Crop to bounding box
    crop = rgba[y : min(y + h, rgb.shape[0]), x : min(x + w, rgb.shape[1])]
    return crop

# ══════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════

def generate_assets(image_path: str):
    """Generates poster visuals using VLFM's native YOLOv7 and MobileSAM models."""
    os.makedirs("poster_assets", exist_ok=True)

    print(f"Loading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Please provide a valid local image path!")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # The original image is intentionally not saved twice, it is saved under 3c_global_scene.png in the pipeline.

    # 2. Load Models
    print("Loading YOLOv7 (this might take a few seconds)...")
    try:
        yolo = YOLOv7(weights="data/yolov7-e6e.pt", image_size=640)
    except Exception as e:
        print(f"Failed to load YOLO. Do you have data/yolov7-e6e.pt downloaded? Error: {e}")
        return

    # 3. Predict Boxes
    print("Running YOLO inference...")
    detections = yolo.predict(image_rgb, conf_thres=0.4)
    
    print("Loading SAMSegmenter...")
    try:
        sam = SAMSegmenter(sam_checkpoint="data/mobile_sam.pt")
    except Exception as e:
        print(f"Failed to load SAM. Do you have data/mobile_sam.pt downloaded? Error: {e}")
        return
    
    # 4. Draw Boxes & Masks for Visualization
    annotated_img = image_rgb.copy()
    
    target_idx = 0 # We'll just grab the first object to generate exhaustive poster crops
    all_masks = []
    
    # Generate some distinct colors for masks
    np.random.seed(42)
    
    height, width = image_rgb.shape[:2]
    
    for idx, box in enumerate(detections.boxes):
        # YOLO returns NORMALIZED boxes (0.0 to 1.0)
        # We must denormalize them to raw image pixels before converting to integer!
        if hasattr(box, 'cpu'):
            box_np = box.cpu().numpy()
        else:
            box_np = box.numpy() if hasattr(box, 'numpy') else np.array(box)
            
        box_denorm = box_np * np.array([width, height, width, height])
        box_int = box_denorm.astype(int)
        x1, y1, x2, y2 = box_int
        label = detections.phrases[idx]
        
        # A. Segment the object using MobileSAM
        mask, score = sam.segment_bbox(image_rgb, box_int.tolist())
        all_masks.append(mask)
        
        # Create a colored overlay for the mask
        color = np.random.randint(50, 255, (3,), dtype=np.uint8)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        colored_mask[mask] = color
        
        # Blend mask with image
        annotated_img = cv2.addWeighted(annotated_img, 1.0, colored_mask, 0.4, 0)
        
        # B. Draw Bounding Box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color.tolist(), 3)
        
        # C. Draw Label
        cv2.putText(annotated_img, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    plt.imsave("poster_assets/2_yolo_sam_segmentation.png", annotated_img)
    print("Saved 2_yolo_sam_segmentation.png")

    # 5. Extract Crops for the HOV-SG Diagram (Using the exact methods from GLOSMNav)
    if len(detections.boxes) > 0:
        box = detections.boxes[target_idx]
        if hasattr(box, 'cpu'):
            box_np = box.cpu().numpy()
        else:
            box_np = box.numpy() if hasattr(box, 'numpy') else np.array(box)
            
        box_denorm = box_np * np.array([width, height, width, height])
        box_int = box_denorm.astype(int)
        x1, y1, x2, y2 = box_int
        label = detections.phrases[target_idx]
        print(f"Generating exact feature crops for: {label}")

        # Convert YOLO xyxy to xywh for object_segmentation logic parity
        w, h = x2 - x1, y2 - y1
        xywh_bbox = (x1, y1, w, h)
        bbox_margin = 50 # exactly matches object_segmentation.py

        target_mask = all_masks[target_idx] if len(all_masks) > target_idx else np.ones(image_rgb.shape[:2], dtype=bool)

        # Crop 1: Context BBox (RGB) - exact parity
        crop_bbox = _create_bbox_crop(image_rgb, xywh_bbox, bbox_margin=bbox_margin)
        
        # Crop 2: Masked BBox (Tight BBox on the Masked RGB image) - exact parity
        crop_masked_bbox = _create_masked_crop(image_rgb, target_mask, xywh_bbox)
        
        # Crop 3: Transparent BBox (For Canva aesthetics)
        crop_transparent = _create_transparent_crop(image_rgb, target_mask, xywh_bbox)
        
        if crop_bbox.size > 0:
            crop_bbox_resize = cv2.resize(crop_bbox, (512, 512)) # Parity with Line 179
            plt.imsave("poster_assets/3a_hero_crop_bbox.png", crop_bbox_resize)
        if crop_masked_bbox.size > 0:
            crop_masked_bbox_resize = cv2.resize(crop_masked_bbox, (512, 512)) # Parity with Line 180
            plt.imsave("poster_assets/3b_hero_crop_masked_bbox.png", crop_masked_bbox_resize)
        if crop_transparent.size > 0:
            crop_transparent_resize = cv2.resize(crop_transparent, (512, 512))
            plt.imsave("poster_assets/3c_hero_crop_transparent.png", crop_transparent_resize)
            
        print("Saved 3a_hero_crop_bbox.png, 3b_hero_crop_masked_bbox.png, and 3c_hero_crop_transparent.png")
            
        # D) Global Scene (Extracted ONCE by the system)
        h_orig, w_orig = image_rgb.shape[:2]
        global_resized = cv2.resize(image_rgb, (300, int(300 * (h_orig/w_orig))))
        plt.imsave("poster_assets/3d_global_scene.png", global_resized)
        print("Saved 3d_global_scene.png")
        
    print(f"\nExtracting all crop variations for all other {len(detections.boxes)-1} objects...")
    for idx, box in enumerate(detections.boxes):
        if idx == target_idx:
            continue
            
        if hasattr(box, 'cpu'):
            box_np = box.cpu().numpy()
        else:
            box_np = box.numpy() if hasattr(box, 'numpy') else np.array(box)
            
        box_denorm = box_np * np.array([width, height, width, height])
        box_int = box_denorm.astype(int)
        x1, y1, x2, y2 = box_int
        w, h = x2 - x1, y2 - y1
        xywh_bbox = (x1, y1, w, h)
        label = detections.phrases[idx].replace(" ", "_").replace("/", "_")
        
        mask = all_masks[idx] if len(all_masks) > idx else np.ones(image_rgb.shape[:2], dtype=bool)
        
        obj_crop_bbox = _create_bbox_crop(image_rgb, xywh_bbox, bbox_margin=50)
        obj_crop_masked = _create_masked_crop(image_rgb, mask, xywh_bbox)
        obj_crop_transparent = _create_transparent_crop(image_rgb, mask, xywh_bbox)
        
        if obj_crop_bbox.size > 0:
            obj_crop_bbox_resize = cv2.resize(obj_crop_bbox, (512, 512))
            plt.imsave(f"poster_assets/obj_{idx}_{label}_crop_bbox.png", obj_crop_bbox_resize)
            
        if obj_crop_masked.size > 0:
            obj_crop_masked_resize = cv2.resize(obj_crop_masked, (512, 512))
            plt.imsave(f"poster_assets/obj_{idx}_{label}_crop_masked_bbox.png", obj_crop_masked_resize)
            
        if obj_crop_transparent.size > 0:
            obj_crop_transparent_resize = cv2.resize(obj_crop_transparent, (512, 512))
            plt.imsave(f"poster_assets/obj_{idx}_{label}_transparent.png", obj_crop_transparent_resize)
            
        print(f"Saved all 3 crop variants for secondary object: {label}")

    print("\nSuccess! All assets generated in the /poster_assets folder.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (e.g., sample_room.jpg)")
    args = parser.parse_args()
    
    generate_assets(args.image)
