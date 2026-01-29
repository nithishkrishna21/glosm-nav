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
        inputs = self.tokenizer([text], padding="max_length", return_tensors = "pt").to(self.device)
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
        Compute cosine similarity between image and text features.

        Args:
            image_features: torch.Tensor of shape (1, output_dim) normalized image features
            text_features: torch.Tensor of shape (1, output_dim) normalized text features

        Returns:
            float: Cosine similarity
        """

        # dot_product = np.dot(image_features, text_features)
        # norm_img = np.linalg.norm(image_features)
        # norm_text = np.linalg.norm(text_features)

        # similarity = dot_product / (norm_img * norm_text + 1e-8)

        similarity = torch.nn.functional.cosine_similarity(image_features, text_features, dim = -1)

        return float(similarity.squeeze())
        

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
            
            else:
                raise ValueError(f"Unknown request type: {request_type}")


    siglip = SigLIPServer(model_name=args.model)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(siglip, name="siglip", port=args.port)