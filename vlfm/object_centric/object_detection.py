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

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN

# Custom Wrappers (Client classes for HTTP communication with model servers)
from .sam_detector import MobileSAMClient
from .siglip2 import SigLIPClient


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
        sam_detector: MobileSAMClient,  # MobileSAMClient instance (HTTP client)
        siglip: SigLIPClient,           # SigLIPClient instance (HTTP client)
        camera_intrinsics: np.ndarray,  # 3×3 intrinsics matrix K
        min_points: int = 50,           # Minimum 3D points for valid object
        bbox_margin: int = 50,         # Margin to increase bounding box by
        masked_weight: float = 0.75    # Weight for masked background in fusion
    ):
        """
        Initialize the object detector.

        Args:
            sam_detector: MobileSAMClient instance for mask generation (connects to SAM server)
            siglip: SigLIPClient instance for feature extraction (connects to SigLIP server)
            camera_intrinsics: 3×3 camera intrinsics matrix K
            min_points: Filter out objects with fewer 3D points
        """
        self.sam_detector = sam_detector
        self.siglip = siglip
        self.camera_intrinsics = camera_intrinsics
        self.min_points = min_points
        self.bbox_margin = bbox_margin
        self.masked_weight = masked_weight


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

        masks = self._segment_with_sam(rgb)

        # Discard low-confidence masks 
        masks = [m for m in masks if m['predicted_iou'] > 0.88
                 and m['stability_score'] > 0.95
                 and m['area'] < (0.5 * rgb.shape[0] * rgb.shape[1])]


        # ══════════════════════════════════════════════════════════════
        # STEP 2: Extract global feature ONCE (reused for all objects)
        # ══════════════════════════════════════════════════════════════

        global_features = self._extract_global_features(rgb)


        # ══════════════════════════════════════════════════════════════
        # STEP 3: Process each detected mask
        # ══════════════════════════════════════════════════════════════
        detections = []

        for mask_data in masks:
            mask = mask_data['segmentation']  # (H, W) binary
            bbox = mask_data['bbox']          # (x, y, w, h)
            confidence = mask_data['predicted_iou'] # float score

            # ──────────────────────────────────────────────────────────
            # STEP 3a: Extract 3 crops for HOV-SG fusion
            # ──────────────────────────────────────────────────────────

            crop_bbox = self._create_bbox_crop(rgb, bbox, bbox_margin=self.bbox_margin)
            crop_bbox = cv2.resize(crop_bbox, (512, 512))

            crop_masked_bbox = self._create_masked_crop(rgb, mask, bbox)
            crop_masked_bbox = cv2.resize(crop_masked_bbox, (512, 512))


            # ──────────────────────────────────────────────────────────
            # STEP 3b: Extract features for local crops using SigLIP
            # ──────────────────────────────────────────────────────────

            cropped_feats = self._extract_features(crop_bbox)
            cropped_masked_feats = self._extract_features(crop_masked_bbox)


            # ──────────────────────────────────────────────────────────
            # STEP 3c: HOV-SG weighted fusion
            # ──────────────────────────────────────────────────────────

            fused_features = self._fuse_features(
                global_features,
                cropped_feats,
                cropped_masked_feats
            )


            # ──────────────────────────────────────────────────────────
            # STEP 3d: Project mask to 3D point cloud
            # ──────────────────────────────────────────────────────────

            point_cloud_camera = self._depth_to_pointcloud(
                depth, mask, self.camera_intrinsics
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
    # Helper Methods
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

        masks = self.sam_detector.segment_image(rgb)
        return masks


    def _extract_global_features(self, rgb: np.ndarray) -> np.ndarray:
        """
        Extract features from full RGB image using SigLIP.

        Returns: (1, D) feature vector 
        """
        return self.siglip.encode_image(rgb)
    
    def increase_bbox_by_margin(self, bbox, margin):
        """
        Increases the size of a bounding box by the given margin.

        :param bbox: The bounding box coordinates in XYWH format as a tuple of (x, y, w, h).
        :param margin: The margin to increase the bounding box size by in pixels.
        :return: The increased bounding box coordinates as a tuple of (x, y, w, h).
        """
        x, y, w, h = bbox
        x -= margin
        y -= margin
        w += margin * 2
        h += margin * 2
        # Check if x is negative
        if x < 0:
            w += x
            x = 0

        # Check if y is negative
        if y < 0:
            h += y
            y = 0
        return (x, y, w, h)

    def _create_bbox_crop(
        self, rgb: np.ndarray, bbox: Tuple, bbox_margin=0
    ) -> np.ndarray:
        """
        This is the image crop of the mask based on its bounding box

        Returns: Cropped RGB image (h, w, 3)
        """
        x, y, w, h = self.increase_bbox_by_margin(bbox, bbox_margin)
        x, y, w, h = int(x), int(y), int(w), int(h)
        return rgb[y : y + h, x : x +w]


    def _create_masked_crop(
        self, rgb: np.ndarray, mask: np.ndarray, bbox: Tuple
    ) -> np.ndarray:
        """
        Create an image of the isolated mask without background

        Returns: Cropped and masked RGB image (h, w, 3)
        """

        x, y, w, h = bbox
        
        # Apply the mask first
        masked = rgb * np.expand_dims(mask, axis = -1)
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Get the cropped mask
        crop = masked[y : y + h, x : x + w]
        return crop



    def _extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract features from a crop using SigLIP.

        Returns: (1, D) feature vector
        """
        return self.siglip.encode_image(crop)


    def _fuse_features(
        self,
        F_g: np.ndarray,      # (1, D)
        cropped_feats: np.ndarray,        # (1, D) - "unmasked" in HOV-SG
        cropped_masked_feats: np.ndarray  # (1, D) - "masked" in HOV-SG
    ) -> np.ndarray:
        """
        Perform HOV-SG 2-stage weighted fusion.

        Returns: (1, D) fused feature vector
        """

        fused_crop_feats = torch.from_numpy(
            self.masked_weight * cropped_masked_feats +
            (1- self.masked_weight) * cropped_feats    
        )

        F_l = np.float32(torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu())

        # 1. Compute the cosine similarity between the local feature F_l and global feature F_g
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
        w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)

        # 2. Compute the final fused feature F_fused
        F_fused = w_i * torch.from_numpy(F_g) + (1 - w_i) * torch.from_numpy(F_l)
        F_fused = torch.nn.functional.normalize(F_fused, p=2, dim=-1)
        F_fused = np.float32(F_fused.cpu())

        return F_fused

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,           # (H, W)
        mask: np.ndarray,            # (H, W) binary
        camera_intrinsics: np.ndarray, # (3, 3)
    ) -> np.ndarray:
        """
        Convert masked depth pixels to 3D point cloud in camera frame.

        Steps:
        1. Extract camera intrinsics from K matrix
        2. Get valid pixels where mask is True and depth > 0
        3. Unproject to 3D using pinhole camera model
        4. Return point cloud in camera frame

        Returns: (N, 3) point cloud in camera frame
        """
        # Extract camera intrinsics
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        # Remove points with invalid depth values
        mask = np.logical_and(mask, depth > 0)

        # Handle empty mask case
        if mask.sum() == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # Get pixel coordinates
        height, width = depth.shape
        x = np.arange(0, width, 1.0)
        y = np.arange(0, height, 1.0)
        u, v = np.meshgrid(x, y)

        masked_depth = depth[mask]
        u = u[mask]
        v = v[mask]

        x = (u - cx) * masked_depth / fx
        y = (v - cy) * masked_depth / fy
        z = masked_depth

        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)

        # Perturb the points a bit to avoid colinearity
        points += np.random.normal(0, 4e-3, points.shape)

        # Denoise with DBSCAN to remove outliers
        points = self._denoise_point_cloud_dbscan(points)

        # Downsample to voxel grid to reduce point count
        points = self._downsample_voxel(points)

        return points

    def _denoise_point_cloud_dbscan(
        self,
        points: np.ndarray,  # (N, 3)
        eps: float = 0.02,   # Max distance between points in same cluster (2cm)
        min_samples: int = 10  # Min points to form a dense cluster
    ) -> np.ndarray:
        """
        Remove outlier points using DBSCAN clustering.

        Keeps only the largest cluster (the main object), discarding:
        - Isolated noise points (depth sensor errors)
        - Small clusters (edge artifacts, reflections)

        Args:
            points: (N, 3) point cloud in camera frame
            eps: Maximum distance between two points to be in same cluster (meters)
            min_samples: Minimum points required to form a dense region

        Returns:
            (M, 3) denoised point cloud where M <= N
        """
        

        # Handle empty or very small point clouds
        if len(points) < min_samples:
            return points

        # Run DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # labels[i] = -1 means noise (outlier)
        # labels[i] >= 0 means cluster ID

        valid_labels = labels[labels >= 0]  # Exclude noise (-1)

        if(len(valid_labels) > 0):

            # Find the most common label
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            largest_cluster_id = unique_labels[np.argmax(counts)]

            largest_cluster_points = points[clustering.labels_ == largest_cluster_id]

            if(len(largest_cluster_points) < 5):
                return points

            points = largest_cluster_points
        
        return points

    def _downsample_voxel(
        self,
        points: np.ndarray,
        voxel_size: float = 0.025  # 2.5cm voxel grid (matches ConceptGraphs)
    ) -> np.ndarray:
        """
        Downsample point cloud using voxel grid filtering.

        Points within the same voxel are merged into one point.
        This reduces point count while preserving geometric structure.

        Args:
            points: (N, 3) point cloud
            voxel_size: Size of voxel grid in meters (default: 2.5cm)

        Returns: (M, 3) downsampled point cloud where M <= N
        """
        if len(points) == 0:
            return points

        # Quantize points to voxel grid
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Find unique voxels (removes duplicate points in same voxel)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

        # Return one point per voxel
        return points[unique_indices]

    def _transform_to_world(
        self, point_cloud_camera: np.ndarray, camera_pose: np.ndarray
    ) -> np.ndarray:
        """
        Transform point cloud from camera frame to world frame.
        """
        N = point_cloud_camera.shape[0]

        points_homogeneous = np.hstack([point_cloud_camera, np.ones((N, 1))]) # (N, 4)
        points_transformed = np.dot(camera_pose, points_homogeneous.T) # (4, N)

        points_transformed = points_transformed[:3, :] / points_transformed[3:4, :] # (3, N)

        points_transformed = points_transformed.T # (N, 3)

        return points_transformed
