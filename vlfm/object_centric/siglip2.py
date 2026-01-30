import os
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.nn.functional import normalize
from PIL import Image

from .server_wrapper import (
    ServerMixin,
    host_model,
    send_request,
    str_to_image,
)

try:
    from transformers import AutoProcessor, AutoModel, AutoTokenizer
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")


class SigLIP:
    """SigLIP Vision-Language Model for feature extraction and similarity computation."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype, device_map=self.device, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Capture SigLIP's learned scale and bias for probability calibration
        if hasattr(self.model, "logit_scale") and hasattr(self.model, "logit_bias"):
            self.logit_scale = self.model.logit_scale.item()
            self.logit_bias = self.model.logit_bias.item()
        else:
            # Fallback for models that might not have these specific attributes exposed similarly
            print("WARNING: Could not find logit_scale/bias. Using raw cosine similarity.")
            self.logit_scale = None
            self.logit_bias = None

    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract and return normalized image features.

        Args:
            image: Input image as a numpy array or PIL Image.

        Returns:
            image_features: np.ndarray of shape (1, output_dim)
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = normalize(image_features, p = 2.0, dim = -1)
            image_features = np.float32(image_features.cpu())

        return image_features


    def encode_text(self, text: str) -> np.ndarray:
        """
        Extract and return normalized text features.

        Args:
            text: Text string to encode

        Returns:
            text_features: np.ndarray of shape (1, output_dim)
        """
        # SigLIP2 requires lowercasing and fixed length padding
        text = text.lower()
        inputs = self.tokenizer([text], padding="max_length", max_length=64, truncation=True, return_tensors = "pt").to(self.device)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = normalize(text_features, p = 2.0, dim = -1)
            text_features = np.float32(text_features.cpu())
        
        return text_features

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> float:
        """
        Compute probability score between image and text features.
        
        Uses SigLIP's sigmoid transformation: sigmoid( (cosine * exp(scale)) + bias )

        Args:
            image_features: torch.Tensor of shape (1, output_dim) normalized image features
            text_features: torch.Tensor of shape (1, output_dim) normalized text features

        Returns:
            float: Probability score [0, 1]
        """

        # 1. Compute Raw Cosine Similarity
        cosine_sim = torch.nn.functional.cosine_similarity(image_features, text_features, dim = -1)

        # 2. Apply SigLIP Calibration (if available) -> Probability
        if self.logit_scale is not None and self.logit_bias is not None:
             # logits = (cosine * exp(log_scale)) + bias
             logits = (cosine_sim * np.exp(self.logit_scale)) + self.logit_bias
             probability = torch.sigmoid(logits)
             return float(probability.squeeze())
        
        return float(cosine_sim.squeeze())
        

class SigLIPClient:
    def __init__(self, port: int = None, device = None):
        if port is None:
            port = int(os.environ.get("SIGLIP2_PORT", "12185"))
        self.url = f"http://localhost:{port}/siglip"
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract image features via server.

        Returns: (1, D) feature vector as torch.Tensor
        """

        response = send_request(self.url,
                                request_type = "encode_image",
                                image = image)

        # return np.array(response["image_features"], dtype=np.float32)
        return torch.Tensor(response["image_features"]).float().to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Extract text features via server.

        Returns: (1, D) feature vector as torch.Tensor
        """
        response = send_request(self.url,
                                request_type = "encode_text",
                                text = text)

        # return np.array(response["text_features"], dtype=np.float32)
        return torch.Tensor(response["text_features"]).float().to(self.device)

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> float:
        """
        Compute similarity via server.

        Returns: float similarity score
        """
        response = send_request(self.url,
                                request_type = "compute_similarity",
                                image_features = image_features.tolist(),
                                text_features = text_features.tolist())
        
        return float(response["similarity"])

    def get_model_params(self) -> dict:
        """
        Get model parameters from server.
        """
        response = send_request(self.url, request_type="get_model_params")
        return response

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    parser.add_argument(
        "--model",
        type=str,
        default="google/siglip2-base-patch16-224",
        help="SigLIP2 model variant to use"
    )
    args = parser.parse_args()

    print("Loading model...")

    class SigLIPServer(ServerMixin, SigLIP):
        def process_payload(self, payload: dict) -> dict:

            request_type = payload.get("request_type")

            if request_type == "encode_image":
                
                image = str_to_image(payload["image"])
                image_features = self.encode_image(image)
                response = {
                    "image_features": image_features.tolist()
                }

                return response
            
            elif request_type == "encode_text":
               
                text = payload["text"]
                text_features = self.encode_text(text)
                response = {
                    "text_features": text_features.tolist()
                }

                return response
        
            elif request_type == "compute_similarity":
                
                # image_features = np.array(payload["image_features"])
                # text_features = np.array(payload["text_features"])
                image_features = torch.Tensor(payload["image_features"]).float().to(self.device)
                text_features = torch.Tensor(payload["text_features"]).float().to(self.device)
                
                similarity = self.compute_similarity(image_features, text_features)
                response = {
                    "similarity": similarity
                }

                return response
            
            elif request_type == "get_model_params":
                return {
                    "logit_scale": self.logit_scale,
                    "logit_bias": self.logit_bias
                }
            
            else:
                raise ValueError(f"Unknown request type: {request_type}")


    siglip = SigLIPServer(model_name=args.model)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(siglip, name="siglip", port=args.port)