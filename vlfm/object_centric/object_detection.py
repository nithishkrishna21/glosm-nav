"""
object_detection.py

Purpose: Stage 1 - Object Detection, Feature Extraction, and Point Cloud Generation
Handles: SAM segmentation → HOV-SG 3-crop fusion → Depth to point cloud

This file will be used by: object_policy.py (the main orchestrator)
This file depends on:
    - MobileSAM (for segmentation)
    - SigLIP (for feature extraction)
    - ConceptGraphs utils (for depth → point cloud conversion)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

# Custom Wrappers
from .sam_detector import SAMDetector
from .siglip import SigLIP

# TODO: Import ConceptGraphs utilities when available
# from conceptgraph.slam.utils import create_object_pcd


class Detection:
    """
    Represents a single detected object from one frame.

    Attributes:
        mask: (H, W) binary mask
        bbox: (x1, y1, x2, y2) bounding box
        features: (D,) fused feature vector (after HOV-SG 3-crop fusion)
        point_cloud: (N, 3) 3D points in WORLD frame
        confidence: Detection confidence from SAM
    """
    def __init__(
        self,
        mask: np.ndarray,
        bbox: np.ndarray,
        features: np.ndarray,
        point_cloud: np.ndarray,
        confidence: float
    ):
        self.mask = mask
        self.bbox = bbox
        self.features = features
        self.point_cloud = point_cloud
        self.confidence = confidence


class ObjectDetector:
    """
    Handles object detection pipeline for a single frame.

    Pipeline:
        1. Segment image with SAM → get masks
        2. For each mask:
            a. Extract 3 crops (global, masked bbox, cropped masked)
            b. Extract features for each crop with SigLIP
            c. Perform HOV-SG weighted fusion
            d. Project mask to 3D point cloud
        3. Return list of Detection objects
    """

    def __init__(
        self,
        sam_detector: SAMDetector,      # SAMDetector instance
        siglip: SigLIP,                 # SigLIP instance
        camera_intrinsics: np.ndarray,  # 3×3 intrinsics matrix K
        min_points: int = 50,           # Minimum 3D points for valid object
    ):
        """
        Initialize the object detector.

        Args:
            sam_detector: SAMDetector instance for mask generation
            siglip: SigLIP instance for feature extraction
            camera_intrinsics: 3×3 camera intrinsics matrix K
            min_mask_area: Filter out masks smaller than this
            min_points: Filter out objects with fewer 3D points
        """
        self.sam_detector = sam_detector
        self.siglip = siglip
        self.camera_intrinsics = camera_intrinsics
        self.min_points = min_points


    def detect_objects(
        self,
        rgb: np.ndarray,           # (H, W, 3) RGB image
        depth: np.ndarray,         # (H, W) depth image
        camera_pose: np.ndarray    # (4, 4) camera-to-world transform
    ) -> List[Detection]:
        """
        Detect objects in a single frame and extract their features + point clouds.

        Args:
            rgb: RGB image (H, W, 3)
            depth: Depth image (H, W)
            camera_pose: 4×4 transformation matrix (camera → world)

        Returns:
            List of Detection objects, each with mask, features, and point cloud
        """
        # ══════════════════════════════════════════════════════════════
        # STEP 1: Segment image with SAM
        # ══════════════════════════════════════════════════════════════
        # TODO: Use SAM to get masks
        # Expected output: List of binary masks, each (H, W)
        # MobileSAM returns list of dicts with keys: 'segmentation', 'bbox', 'stability_score'

        masks = self._segment_with_sam(rgb)

        # Discard low-confidence masks 
        masks = [m for m in masks if m['predicted_iou'] > 0.88
                 and m['stability_score'] > 0.95
                 and m['area'] < (0.5 * rgb.shape[0] * rgb.shape[1])]


        # ══════════════════════════════════════════════════════════════
        # STEP 2: Extract global feature ONCE (reused for all objects)
        # ══════════════════════════════════════════════════════════════
        # TODO: Encode full RGB image with SigLIP
        # Expected output: (1, D) feature vector where D = 768 or 1024 depending on model
        # Hint:
        #   inputs = siglip_processor(images=rgb, return_tensors="pt").to(device)
        #   outputs = siglip_model.get_image_features(**inputs)
        #   features = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 normalize

        global_features = self._extract_global_features(rgb)


        # ══════════════════════════════════════════════════════════════
        # STEP 3: Process each detected mask
        # ══════════════════════════════════════════════════════════════
        detections = []

        for mask_data in masks:
            mask = mask_data['segmentation']  # (H, W) binary
            bbox = mask_data['bbox']          # (x, y, w, h)
            confidence = mask_data.get('stability_score', 1.0)

            # ──────────────────────────────────────────────────────────
            # STEP 3a: Extract 3 crops for HOV-SG fusion
            # ──────────────────────────────────────────────────────────
            # TODO: Create 2 local crops from mask
            # 1. Masked bbox crop (object + background in bbox) - "unmasked" in HOV-SG terms
            # 2. Cropped masked crop (only object, background blacked out) - "masked" in HOV-SG terms
            # Hint:
            #   - Use bbox (x, y, w, h) to crop
            #   - For masked crop: set pixels where mask==0 to black [0, 0, 0]

            crop_masked_bbox = self._create_masked_bbox_crop(rgb, mask, bbox)
            crop_masked_only = self._create_masked_crop(rgb, mask, bbox)


            # ──────────────────────────────────────────────────────────
            # STEP 3b: Extract features for local crops using SigLIP
            # ──────────────────────────────────────────────────────────
            # TODO: Encode both crops with SigLIP
            # Expected output: (1, D) feature vectors
            # Same process as global features but with cropped images

            features_masked_bbox = self._extract_features(crop_masked_bbox)
            features_masked_only = self._extract_features(crop_masked_only)


            # ──────────────────────────────────────────────────────────
            # STEP 3c: HOV-SG weighted fusion
            # ──────────────────────────────────────────────────────────
            # TODO: Implement 2-stage fusion
            # Stage 1: Combine masked_only and masked_bbox (local fusion)
            #   F_l = 0.4418 * F_masked_only + 0.5582 * F_masked_bbox
            #   F_l = F_l / ||F_l||  (L2 normalize)
            #
            # Stage 2: Combine F_l with global using similarity weighting
            #   similarity = cosine_similarity(F_l, F_g)  # dot product since normalized
            #   w = softmax([similarity])  # Convert to weight
            #   F_final = w * F_g + (1 - w) * F_l
            #   F_final = F_final / ||F_final||  (L2 normalize)

            fused_features = self._fuse_features(
                global_features,
                features_masked_bbox,
                features_masked_only
            )


            # ──────────────────────────────────────────────────────────
            # STEP 3d: Project mask to 3D point cloud
            # ──────────────────────────────────────────────────────────
            # TODO: Use ConceptGraphs' depth → point cloud conversion
            # Input: depth, mask, camera_intrinsics, rgb (for colors - optional)
            # Process:
            #   1. Get masked depth pixels: depth[mask]
            #   2. Get pixel coordinates: u, v where mask == True
            #   3. Unproject using intrinsics:
            #      x_cam = (u - cx) * depth / fx
            #      y_cam = (v - cy) * depth / fy
            #      z_cam = depth
            # Output: (N, 3) point cloud in CAMERA frame
            #
            # Hint: You can reuse create_object_pcd() from ConceptGraphs

            point_cloud_camera = self._depth_to_pointcloud(
                depth, mask, self.camera_intrinsics, rgb
            )

            # Transform to world frame
            point_cloud_world = self._transform_to_world(
                point_cloud_camera, camera_pose
            )

            # Filter: minimum points threshold
            if len(point_cloud_world) < self.min_points:
                continue


            # ──────────────────────────────────────────────────────────
            # STEP 3e: Create Detection object
            # ──────────────────────────────────────────────────────────
            detection = Detection(
                mask=mask,
                bbox=bbox,
                features=fused_features,
                point_cloud=point_cloud_world,
                confidence=confidence
            )
            detections.append(detection)

        return detections


    # ══════════════════════════════════════════════════════════════════
    # Helper Methods to Implement
    # ══════════════════════════════════════════════════════════════════

    def _segment_with_sam(self, rgb: np.ndarray) -> List[Dict]:
        """
        Run SAM segmentation on RGB image.

        Returns: List of dicts with keys: 'segmentation', 'bbox', 'stability_score', 'predicted_iou
                 - 'segmentation': (H, W) binary mask
                 - 'bbox': (x, y, w, h) bounding box
                 - 'stability_score': float confidence score
                 - 'predicted_iou': float IoU score
        """
        # raise NotImplementedError("Implement SAM segmentation")
        masks = self.sam_detector.segment_image(rgb)
        return masks


    def _extract_global_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        Extract features from full RGB image using SigLIP.

        TODO: Implement SigLIP encoding
        Steps:
            1. Preprocess image: inputs = siglip_processor(images=rgb, return_tensors="pt")
            2. Extract features: outputs = siglip_model.get_image_features(**inputs)
            3. L2 normalize: features = outputs / outputs.norm(dim=-1, keepdim=True)

        Returns: (1, D) feature vector (D = 768 or 1024 depending on SigLIP variant)
        """
        raise NotImplementedError("Implement global feature extraction with SigLIP")


    def _create_masked_bbox_crop(
        self, rgb: np.ndarray, mask: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """
        Create crop of bounding box region (object + background).
        This is the "unmasked" crop in HOV-SG terminology.

        TODO: Crop RGB using bbox coordinates
        Given bbox = (x, y, w, h):
            crop = rgb[y:y+h, x:x+w]

        Returns: Cropped RGB image (h, w, 3)
        """
        raise NotImplementedError("Implement masked bbox crop")


    def _create_masked_crop(
        self, rgb: np.ndarray, mask: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """
        Create crop with only object visible (background blacked out).
        This is the "masked" crop in HOV-SG terminology.

        TODO:
        1. Crop RGB using bbox: crop = rgb[y:y+h, x:x+w]
        2. Crop mask using bbox: mask_crop = mask[y:y+h, x:x+w]
        3. Apply mask (set background pixels to 0):
           crop[~mask_crop] = 0  # or [0, 0, 0]

        Returns: Cropped and masked RGB image (h, w, 3)
        """
        raise NotImplementedError("Implement masked crop")


    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract features from a crop using SigLIP.

        TODO: Same as _extract_global_features but for a crop
        Steps:
            1. Preprocess: inputs = siglip_processor(images=crop, return_tensors="pt")
            2. Extract: outputs = siglip_model.get_image_features(**inputs)
            3. Normalize: features = outputs / outputs.norm(dim=-1, keepdim=True)

        Returns: (1, D) feature vector
        """
        raise NotImplementedError("Implement feature extraction for crops")


    def _fuse_features(
        self,
        global_feat: np.ndarray,      # (1, D)
        masked_bbox_feat: np.ndarray, # (1, D) - "unmasked" in HOV-SG
        masked_only_feat: np.ndarray  # (1, D) - "masked" in HOV-SG
    ) -> np.ndarray:
        """
        Perform HOV-SG 2-stage weighted fusion.

        TODO: Implement fusion logic from HOV-SG paper:

        Stage 1: Local feature fusion
            F_l = 0.4418 * masked_only_feat + 0.5582 * masked_bbox_feat
            F_l = F_l / ||F_l||_2  (L2 normalize)

        Stage 2: Global-local fusion with similarity weighting
            similarity = F_l · F_g  (dot product, since both are normalized)
            w = softmax([similarity])[0]  (convert to probability)
            F_final = w * F_g + (1 - w) * F_l
            F_final = F_final / ||F_final||_2  (L2 normalize)

        Note: Since features are already L2 normalized, cosine similarity = dot product

        Returns: (1, D) fused feature vector
        """
        raise NotImplementedError("Implement HOV-SG fusion")


    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,           # (H, W)
        mask: np.ndarray,            # (H, W) binary
        camera_intrinsics: np.ndarray, # (3, 3)
        rgb: np.ndarray              # (H, W, 3) - optional for colors
    ) -> np.ndarray:
        """
        Convert masked depth pixels to 3D point cloud in camera frame.

        TODO: Implement depth unprojection using camera intrinsics

        Steps:
        1. Extract camera intrinsics:
           fx, fy = K[0, 0], K[1, 1]  (focal lengths)
           cx, cy = K[0, 2], K[1, 2]  (principal point)

        2. Get valid pixels:
           valid_mask = (mask == True) & (depth > 0)
           u, v = np.where(valid_mask)  # pixel coordinates
           depths = depth[valid_mask]

        3. Unproject to 3D (camera frame):
           x_cam = (u - cx) * depths / fx
           y_cam = (v - cy) * depths / fy
           z_cam = depths

        4. Stack into point cloud:
           points = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

        Hint: You can also use ConceptGraphs' create_object_pcd() function

        Returns: (N, 3) point cloud in camera frame
        """
        raise NotImplementedError("Implement depth to point cloud conversion")


    def _transform_to_world(
        self, point_cloud_camera: np.ndarray, camera_pose: np.ndarray
    ) -> np.ndarray:
        """
        Transform point cloud from camera frame to world frame.

        TODO: Apply 4×4 transformation matrix

        Steps:
        1. Convert points to homogeneous coordinates:
           points_homo = np.hstack([point_cloud_camera, np.ones((N, 1))])  # (N, 4)

        2. Apply transformation:
           points_world_homo = (camera_pose @ points_homo.T).T  # (N, 4)

        3. Convert back to 3D:
           points_world = points_world_homo[:, :3]  # (N, 3)

        Returns: (N, 3) point cloud in world frame
        """
        raise NotImplementedError("Implement camera to world transformation")


# ══════════════════════════════════════════════════════════════════════
# Usage Example (for reference)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This is how Stage 1 will be used in the main pipeline

    # TODO: Load models
    # from mobile_sam import sam_model_registry, SamPredictor
    # from transformers import AutoProcessor, AutoModel
    #
    # sam_model = sam_model_registry["vit_t"](checkpoint="path/to/mobile_sam.pth")
    # siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
    # siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    # Initialize detector
    detector = ObjectDetector(
        sam_model=None,  # TODO: Load MobileSAM
        siglip_model=None,  # TODO: Load SigLIP
        siglip_processor=None,  # TODO: Load SigLIP processor
        camera_intrinsics=np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])  # 3×3 K matrix
    )

    # Process a frame
    rgb = np.zeros((480, 640, 3))  # Dummy RGB image
    depth = np.zeros((480, 640))   # Dummy depth image
    camera_pose = np.eye(4)        # Dummy camera pose

    detections = detector.detect_objects(rgb, depth, camera_pose)

    # Each detection has:
    for det in detections:
        print(f"Mask shape: {det.mask.shape}")           # (H, W)
        print(f"Features shape: {det.features.shape}")   # (1, D)
        print(f"Point cloud shape: {det.point_cloud.shape}")  # (N, 3)
        print(f"Confidence: {det.confidence}")
