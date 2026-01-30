
import numpy as np
import torch
# Use DIRECT model class, not Client
from vlfm.object_centric.siglip2 import SigLIP
import os

def test_siglip():
    # port = int(os.environ.get("SIGLIP2_PORT", "12185"))
    # print(f"Connecting to SigLIP on port {port}...")
    # client = SigLIPClient(port=port)
    
    print("Loading SigLIP model locally...")
    client = SigLIP() # Instantiating the model directly

    # 1. Test Text Encoding
    text = "a photo of a red square"
    print(f"\nEncoding text: '{text}'")
    text_feats = client.encode_text(text)

    # 2. Test Positive Match (Red Square)
    
    # Create a Red Square (Red in RGB is [255, 0, 0])
    img_red_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    img_red_rgb[:, :, 0] = 255 

    # Create a Blue Square (Red in BGR [0, 0, 255] which creates Blue if interpreted as RGB)
    # OR: If cv2 swaps channels, maybe what we think is Red is being seen as Blue?
    
    print("\n--- Testing Positive Match ---")
    print("Sending RGB Red Square [255, 0, 0]...")
    feats_red = client.encode_image(img_red_rgb)
    sim_red = client.compute_similarity(torch.from_numpy(feats_red), torch.from_numpy(text_feats))
    print(f"Similarity ('{text}' vs RGB Red Square detection): {sim_red:.4f}")

    # Test BGR hypothesis:
    img_blue_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    img_blue_rgb[:, :, 2] = 255 # Blue channel

    print("Sending RGB Blue Square [0, 0, 255]...")
    feats_blue = client.encode_image(img_blue_rgb)
    sim_blue = client.compute_similarity(torch.from_numpy(feats_blue), torch.from_numpy(text_feats))
    print(f"Similarity ('{text}' vs RGB Blue Square detection): {sim_blue:.4f}")

    if sim_red > 0.15:
        print("✅ SUCCESS: Model correctly identifies Red Square!")
    elif sim_blue > 0.15:
        print("⚠️ WARNING: BGR/RGB Channel Swap detected! Blue image matched 'red square' text.")
    else:
        print("❌ CRITICAL: Model fails to match 'red square' to EITHER Red or Blue image. Model/Weights likely broken.")


if __name__ == "__main__":
    test_siglip()
