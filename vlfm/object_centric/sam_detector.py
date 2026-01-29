import os
from typing import Any, List, Optional, Dict

import numpy as np
import torch

from .server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    send_request,
    str_to_bool_arr,
    str_to_image,
)

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
except ModuleNotFoundError:
    print("Could not import mobile_sam. This is OK if you are only using the client.")


class SAMDetector:
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_t",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        # self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=mobile_sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.8,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=50
        )

    def segment_image(self, image: np.ndarray) -> List[Dict]:
        """Segments the objects in the given image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
        Returns:
            List[Dict]: The segmented objects as a list of dictionaries.
        """

        with torch.inference_mode():
            masks = self.mask_generator.generate(image)
        
        return masks


class MobileSAMClient:
    def __init__(self, port: int = None):
        if port is None:
            port = int(os.environ.get("SAM_PORT", "12183"))
        self.url = f"http://localhost:{port}/mobile_sam"

    def segment_image(self, image: np.ndarray) -> List[Dict]:
        response = send_request(self.url, image=image)
        masks = response["masks"]
        shape = tuple(response["shape"])

        # Deserialize each mask's segmentation string back to numpy array
        for mask in masks:
            mask['segmentation'] = str_to_bool_arr(mask['segmentation'], shape=shape)

        return masks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)
    args = parser.parse_args()

    print("Loading model...")

    class MobileSAMServer(ServerMixin, SAMDetector):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            masks = self.segment_image(image)

            # Serialize each mask's segmentation array for JSON transmission
            serialized_masks = []
            for mask in masks:
                serialized_mask = {
                    'segmentation': bool_arr_to_str(mask['segmentation']),
                    'bbox': mask['bbox'].tolist() if isinstance(mask['bbox'], np.ndarray) else mask['bbox'],
                    'area': int(mask['area']),
                    'predicted_iou': float(mask['predicted_iou']),
                    'stability_score': float(mask['stability_score']),
                    'crop_box': mask['crop_box'].tolist() if isinstance(mask['crop_box'], np.ndarray) else mask['crop_box'],
                }
                serialized_masks.append(serialized_mask)

            return {"masks": serialized_masks, "shape": image.shape[:2]}

    mobile_sam = MobileSAMServer(sam_checkpoint=os.environ.get("MOBILE_SAM_CHECKPOINT", "data/mobile_sam.pt"))
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(mobile_sam, name="mobile_sam", port=args.port)
