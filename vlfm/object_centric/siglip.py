"""
siglip.py

SigLIP vision-language model wrapper for feature extraction.
Follows VLFM's pattern from blip2.py and sam.py.

This file depends on:
    - transformers library (HuggingFace)
"""

import numpy as np
import torch
from typing import Any, Optional, Union
from PIL import Image

try:
    from transformers import AutoProcessor, AutoModel
except ModuleNotFoundError:
    print("Could not import transformers. Install with: pip install transformers")


class SigLIP:
    """
    SigLIP vision model for image feature extraction.

    Usage:
        siglip = SigLIP(model_name="google/siglip-base-patch16-224")
        image_features = siglip.encode_image(image)  # Returns (1, D) normalized features
        text_features = siglip.encode_text("a photo of a cat")
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[Any] = None,
    ) -> None:
        """
        Initialize SigLIP model.

        Args:
            model_name: HuggingFace model identifier
                Options:
                - "google/siglip-base-patch16-224" (D=768)
                - "google/siglip-large-patch16-256" (D=1024)
            device: Device for inference (cuda/cpu)
        """
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        print(f"Loading SigLIP model: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        print("SigLIP loaded successfully!")

    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract and return L2-normalized image features.

        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image

        Returns:
            np.ndarray: L2-normalized features (1, D) where D=768 or 1024
        """
        # TODO: Implement image encoding
        # Steps:
        # 1. Convert numpy to PIL if needed:
        #    if isinstance(image, np.ndarray):
        #        image = Image.fromarray(image)
        #
        # 2. Preprocess image:
        #    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        #
        # 3. Extract features:
        #    with torch.inference_mode():
        #        outputs = self.model.get_image_features(**inputs)
        #
        # 4. L2 normalize:
        #    features = outputs / outputs.norm(dim=-1, keepdim=True)
        #
        # 5. Return as numpy:
        #    return features.cpu().numpy()

        raise NotImplementedError("Implement image encoding with SigLIP")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Extract and return L2-normalized text features.

        Args:
            text: Text string to encode

        Returns:
            np.ndarray: L2-normalized features (1, D)
        """
        # TODO: Implement text encoding
        # Steps:
        # 1. Preprocess text:
        #    inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        #
        # 2. Extract features:
        #    with torch.inference_mode():
        #        outputs = self.model.get_text_features(**inputs)
        #
        # 3. L2 normalize:
        #    features = outputs / outputs.norm(dim=-1, keepdim=True)
        #
        # 4. Return as numpy:
        #    return features.cpu().numpy()

        raise NotImplementedError("Implement text encoding with SigLIP")

    def compute_similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between image and text features.

        Args:
            image_features: (1, D) normalized image features
            text_features: (1, D) normalized text features

        Returns:
            float: Cosine similarity (dot product since both are normalized)
        """
        # TODO: Implement similarity computation
        # Since features are L2 normalized, cosine similarity = dot product
        # Hint: similarity = (image_features @ text_features.T).item()

        raise NotImplementedError("Implement similarity computation")


# ══════════════════════════════════════════════════════════════════════
# Usage Example
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    siglip = SigLIP(model_name="google/siglip-base-patch16-224")

    # Dummy image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Encode image
    image_features = siglip.encode_image(image)
    print(f"Image features shape: {image_features.shape}")  # (1, 768)

    # Encode text
    text_features = siglip.encode_text("a photo of a cat")
    print(f"Text features shape: {text_features.shape}")  # (1, 768)

    # Compute similarity
    similarity = siglip.compute_similarity(image_features, text_features)
    print(f"Similarity: {similarity:.4f}")
