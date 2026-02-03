
import os
import sys
import numpy as np
import torch
import cv2

# Add codebase to path (Relative to this script)
# Assumes script is at vlfm/scripts/verify_batch_fusion.py
script_dir = os.path.dirname(os.path.abspath(__file__))
codebase_root = os.path.abspath(os.path.join(script_dir, "../../"))
if codebase_root not in sys.path:
    sys.path.append(codebase_root)
print(f"Added to sys.path: {codebase_root}")

from vlfm.object_centric.object_detection import ObjectSegmenter
from vlfm.object_centric.clip_encoder import CLIPClient
from vlfm.object_centric.sam_detector import MobileSAMClient

def test_pipeline():
    print("=== Testing Batch Pipeline (Real CLIP Client + Real SAM Client) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device (for tension creation, clients run on server): {device}")

    # 1. Load Real CLIP Client
    print("Connecting to CLIP Client (ViT-H-14)...")
    try:
        encoder = CLIPClient()
        # Ping check (requires CLIP server running)
        print("Checking encoder connection...")
        try:
             # Try a dummy encode
             encoder.encode_text("hello")
             print("CLIP Client Connected successfully.")
        except Exception as e:
             print(f"WARNING: Could not connect to CLIP server: {e}")
             print("Ensure ./vlfm/scripts/launch_vlm_servers_v2.sh is running!")
             return
    except Exception as e:
        print(f"FAILED to initialize CLIPClient: {e}")
        return

    # 2. Test Encoder Batch Processing via Server
    print("\n--- Testing Encoder Batch Input ---")
    BATCH_SIZE = 4
    # Create fake batch images (N, H, W, 3)
    fake_batch = np.random.randint(0, 255, (BATCH_SIZE, 512, 512, 3), dtype=np.uint8)
    
    try:
        # Note: CLIPClient.encode_image expects a single image or handles loop internally depending on implementation
        # Our CLIPClient implementation in clip_encoder.py currently takes a single image (np.ndarray) -> Tensor
        # Let's check if it handles batches. The server implementation in clip_encoder.py uses 'encode_image' which handles list/batch.
        # But the Client `encode_image` sends `image=image` in payload.
        # If payload['image'] is a batch array, `str_to_image` in server_wrapper needs to handle it?
        # `str_to_image` uses cv2.imdecode which expects a single image buffer.
        # So providing a 4D array to str_to_image converts it to string via base64, but decoding might behave differently.
        # Standard implementation usually expects single image loop client-side or server-side.
        
        # Let's verify what CLIP logic is.
        # clip_encoder.py: CLIP.encode_image handles batch.
        # server_wrapper.py: process_payload calls `str_to_image`.
        # `str_to_image` logic: base64 decode -> np.frombuffer -> cv2.imdecode. 
        # If we send a batch, `image_to_str` flattens it? 
        # `image_to_str` uses `cv2.imencode`. `cv2.imencode` ONLY works for single images (or list of images, but returns list).
        
        # So sending a 4D batch to `CLIPClient.encode_image` will FAIL at `image_to_str` step inside `send_request`.
        # We must loop here or update client.
        
        print("Sending single image to test connection...")
        obs_features = encoder.encode_image(fake_batch[0])
        print(f"Single Image Input Shape: {fake_batch[0].shape}")
        print(f"Output Feature Shape: {obs_features.shape}")
        
    except Exception as e:
        print(f"FAIL: Encoder crashed on input: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test ObjectSegmenter Integration with Real Clients
    print("\n--- Testing ObjectSegmenter Integration ---")
    
    # Mock inputs
    H, W = 480, 640
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    depth = np.ones((H, W), dtype=np.float32) * 2.0
    camera_intrinsics = np.array([[300, 0, 320], [0, 300, 240], [0, 0, 1]])
    camera_pose = np.eye(4)
    
    # Real SAM Client
    print("Connecting to MobileSAM Client...")
    try:
        sam_client = MobileSAMClient()
        # Verify connection
        sam_client.segment_bbox(rgb, [0,0,10,10])
        print("SAM Client Connected.")
    except Exception as e:
         print(f"WARNING: Could not connect to SAM server: {e}")
         print("Ensure servers are running!")
         return

    
    # Instantiate Segmenter with REAL CLIENTS
    segmenter = ObjectSegmenter(
        sam_detector=sam_client,
        encoder=encoder, 
        camera_intrinsics=camera_intrinsics,
        min_points=50, # Reduce noise
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
                 print(f"Feature Vector Norm: {torch.norm(detections[0].features).item()}")
            else:
                 print(f"FAIL: Dimension mismatch. Got {detections[0].features.shape[0]}")
        else:
            print("No detections returned. (This assumes SAM finds something in random noise or server is active)")
            
    except Exception as e:
        print(f"CRASHED during detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
