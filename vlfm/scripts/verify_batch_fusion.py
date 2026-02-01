
import os
import sys
import numpy as np
import torch
import cv2
from unittest.mock import MagicMock

# Add codebase to path
sys.path.append("c:/Johns Hopkins/Capstone Project/codebase/vlfm")

from vlfm.object_centric.object_detection import ObjectSegmenter
from vlfm.object_centric.clip_encoder import CLIPClient
from vlfm.object_centric.sam_detector import MobileSAMClient

def test_pipeline():
    print("=== Testing Batch Pipeline (Real CLIP) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Real CLIP Model
    # This verifies clip_encoder.py imports and model loading
    print("Loading CLIP (ViT-H-14)... This may take a moment.")
    try:
        # Initialize lighter model for quick testing if possible, but user has ViT-H-14
        encoder = CLIPClient()
        print("CLIP Loaded successfully.")
    except Exception as e:
        print(f"FAILED to load CLIP: {e}")
        return

    # 2. Test Encoder Batch Processing directly
    print("\n--- Testing Encoder Batch Input ---")
    BATCH_SIZE = 4
    # Create fake batch images (N, H, W, 3)
    fake_batch = np.random.randint(0, 255, (BATCH_SIZE, 512, 512, 3), dtype=np.uint8)
    
    try:
        obs_features = encoder.encode_image(fake_batch)
        print(f"Batch Input Shape: {fake_batch.shape}")
        print(f"Output Feature Shape: {obs_features.shape}")
        
        if obs_features.shape == (BATCH_SIZE, 1024):
            print("SUCCESS: Encoder handled batch correctly.")
        else:
            print(f"FAIL: Expected ({BATCH_SIZE}, 1024), got {obs_features.shape}")
    except Exception as e:
        print(f"FAIL: Encoder crashed on batch input: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test ObjectSegmenter Integration
    print("\n--- Testing ObjectSegmenter Integration ---")
    
    # Mock inputs
    H, W = 480, 640
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32) * 2.0
    camera_intrinsics = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])
    camera_pose = np.eye(4)
    
    # Mock SAM (we don't need real SAM for this test)
    mock_sam = MagicMock(spec=MobileSAMClient)
    # Create 3 valid masks -> 3 objects
    masks = []
    for i in range(3):
        m = np.zeros((H, W), dtype=bool)
        m[i*50:(i+1)*50, i*50:(i+1)*50] = True # Non-overlapping boxes
        masks.append({
            'segmentation': m,
            'bbox': (i*50, i*50, 50, 50),
            'predicted_iou': 0.95,
            'stability_score': 0.95
        })
    mock_sam.segment_image.return_value = masks
    
    # Instantiate Segmenter with REAL CLIP
    segmenter = ObjectSegmenter(
        sam_detector=mock_sam,
        encoder=encoder, # Pass the real instance
        camera_intrinsics=camera_intrinsics,
        min_points=1,
        bbox_margin=10
    )
    
    # Run Detection
    print("Running detect_objects()...")
    try:
        detections = segmenter.detect_objects(rgb, depth, camera_pose)
        print(f"Detected {len(detections)} objects.")
        
        if len(detections) > 0:
            print(f"Detection 0 Feature Shape: {detections[0].features.shape}")
            if detections[0].features.shape[0] == 1024:
                 print("SUCCESS: Fusion pipeline produced correct feature dimensions.")
            else:
                 print(f"FAIL: Dimension mismatch. Got {detections[0].features.shape[0]}")
        else:
            print("FAIL: No detections returned (check mask/crop filters).")
            
    except Exception as e:
        print(f"CRASHED during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
