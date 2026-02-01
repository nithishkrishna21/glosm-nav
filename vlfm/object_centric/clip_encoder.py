import os
from typing import Any, Optional, Union, List

import numpy as np
import torch
from torch.nn.functional import normalize
from PIL import Image
import open_clip

from .server_wrapper import (
    ServerMixin,
    host_model,
    send_request,
    str_to_image,
)

try:
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
except ModuleNotFoundError:
    print("Could not import transformers. This is OK if you are only using the client.")


class CLIP:

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[Any] = None
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = torch.float32

        self.model, _, self.image_processor = open_clip.create_model_and_transforms(model_name, 
                                            pretrained=pretrained)

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()

    
    def encode_image(self, image: Union[np.ndarray, Image.Image, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """
        Extract and return normalized image features.
        Supports single image or batch of images.

        Args:
            image: Input image as numpy array (H,W,3) or (N,H,W,3), PIL Image, or List of them.

        Returns:
            image_features: np.ndarray of shape (N, output_dim)
        """
        # Convert inputs to List[PIL.Image]
        pil_images = []
        
        if isinstance(image, np.ndarray):
            if image.ndim == 4: # Batch (N, H, W, 3)
                pil_images = [Image.fromarray(img) for img in image]
            elif image.ndim == 3: # Single (H, W, 3)
                pil_images = [Image.fromarray(image)]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        
        elif isinstance(image, list):
            # List of arrays or List of PIL
            for img in image:
                if isinstance(img, np.ndarray):
                    pil_images.append(Image.fromarray(img))
                else:
                    pil_images.append(img)
        else:
             # Assume single PIL
             pil_images = [image]

        # Process images
        # self.image_processor returns (C, H, W). We stack to get (N, C, H, W)
        image_tensors = [self.image_processor(img) for img in pil_images]
        image_batch = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features = normalize(image_features, p=2, dim=-1)
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

        with torch.no_grad():
            text = self.tokenizer([text])
            text_features = self.model.encode_text(text)
            text_features = normalize(text_features, p=2, dim=-1)
            text_features = np.float32(text_features.cpu())

        return text_features

    def compute_similarity(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> float:
        """
        Compute probability score between image and text features.

        Args:
            image_features: torch.Tensor of shape (1, output_dim) normalized image features
            text_features: torch.Tensor of shape (1, output_dim) normalized text features

        Returns:
            float: Probability score [0, 1]
        """

        # 1. Compute Raw Cosine Similarity
        cosine_sim = torch.nn.functional.cosine_similarity(image_features, text_features, dim = -1)

        return float(cosine_sim.squeeze())


    class CLIPClient:
        def __init__(self, port: int = None, device = None):
            if port is None:
                port = int(os.environ.get("CLIP_PORT", "12186"))
            self.url = f"http://localhost:{port}/clip"
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12186)
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-H-14",
        help="CLIP model variant to use"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="pretrained weights to use"
    )
    args = parser.parse_args()

    print("Loading model...")

    class CLIPServer(ServerMixin, CLIP):
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


    clip = CLIPServer(model_name=args.model_name, pretrained=args.pretrained)
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(clip, name="clip", port=args.port)