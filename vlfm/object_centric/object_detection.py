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
# from sklearn.cluster import DBSCAN
import open3d as o3d

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
        features: torch.Tensor,
        point_cloud: o3d.geometry.PointCloud,
        confidence: float
    ):
        self.mask = mask
        self.bbox = bbox
        self.features = features
        self.point_cloud = point_cloud
        self.confidence = confidence


class ObjectSegmenter:
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        print(f"DEBUG SAM: Raw masks from SAM: {len(masks)}")

        # # Discard low-confidence masks 
        # masks = [m for m in masks if m['predicted_iou'] > 0.70
        #          and m['stability_score'] > 0.75
        #          and m['area'] < (0.5 * rgb.shape[0] * rgb.shape[1])]
        # print(f"DEBUG SAM: After filtering (iou>0.7, stability>0.75): {len(masks)} masks")


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
            num_points = len(point_cloud_world.points)
            if num_points < self.min_points:
                # print(f"DEBUG: Rejecting mask (only {num_points} points, need {self.min_points})")
                continue
            # print(f"DEBUG: Accepting mask with {num_points} points")


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
        F_g: torch.Tensor,      # (1, D)
        cropped_feats: torch.Tensor,        # (1, D) - "unmasked" in HOV-SG
        cropped_masked_feats: torch.Tensor  # (1, D) - "masked" in HOV-SG
    ) -> torch.Tensor:
        """
        Perform HOV-SG 2-stage weighted fusion.

        Returns: (1, D) fused feature vector
        """

        # F_g = torch.from_numpy(F_g).float()
        # cropped_feats = torch.from_numpy(cropped_feats).float()
        # cropped_masked_feats = torch.from_numpy(cropped_masked_feats).float()

        fused_crop_feats = self.masked_weight * cropped_masked_feats + (1- self.masked_weight) * cropped_feats    

        F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1)

        # 1. Compute the cosine similarity between the local feature F_l and global feature F_g
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        phi_l_G = cos(F_l, F_g)
        w_i = torch.sigmoid(phi_l_G).reshape(-1, 1)

        # 2. Compute the final fused feature F_fused
        F_fused = w_i * F_g + (1 - w_i) * F_l
        F_fused = torch.nn.functional.normalize(F_fused, p=2, dim=-1)
        # F_fused = F_fused.cpu().numpy()

        return F_fused.float().to(self.device)

    def _depth_to_pointcloud(
        self,
        depth: np.ndarray,           # (H, W)
        mask: np.ndarray,            # (H, W) binary
        camera_intrinsics: np.ndarray, # (3, 3)
    ) -> o3d.geometry.PointCloud:
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
            # return np.zeros((0, 3), dtype=np.float32)
            return o3d.geometry.PointCloud()

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

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd
    

    def _denoise_point_cloud_dbscan(self,
        pcd: o3d.geometry.PointCloud,
        eps = 0.2,
        min_points = 10
    ) -> o3d.geometry.PointCloud:
        
        pcd_clusters = pcd.cluster_dbscan(eps=eps, min_points=min_points)

        # convert to numpy arrays
        obj_points = np.asarray(pcd.points)
        pcd_clusters = np.asarray(pcd_clusters)

        # count all the labels in the clusters
        valid_labels = pcd_clusters[pcd_clusters >= 0]

        if(len(valid_labels) > 0):

            unique_labels, counts = np.unique(valid_labels, return_counts = True)
            largest_cluster_id = unique_labels[np.argmax(counts)]

            largest_cluster_points = obj_points[pcd_clusters == largest_cluster_id]

            if(len(largest_cluster_points) < 5):
                return pcd

            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            pcd = largest_cluster_pcd 
        
        return pcd
    
    def _downsample_voxel(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.25,
    ) -> o3d.geometry.PointCloud:
        
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd
    
    def _transform_to_world(
        self,
        camera_object_pcd: o3d.geometry.PointCloud,
        camera_pose: np.ndarray
    ) -> o3d.geometry.PointCloud:
        
        """
        Transform point cloud from camera frame to world frame.
        """

        if camera_pose is not None:
            world_object_pcd = camera_object_pcd.transform(camera_pose)
        else:
            world_object_pcd = camera_object_pcd

        # downsample pcd
        world_object_pcd = self._downsample_voxel(pcd=world_object_pcd)

        # denoise pcd
        world_object_pcd = self._denoise_point_cloud_dbscan(pcd=world_object_pcd)

        return world_object_pcd