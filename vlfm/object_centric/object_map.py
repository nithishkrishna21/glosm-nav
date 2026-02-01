"""
object_map.py

Purpose: Stage 2 - Object Association & Mapping
Maintains a persistent map of objects across frames by matching new detections
to existing objects using geometric similarity (IoU or nnratio) and semantic similarity (cosine).

This file will be used by: object_policy.py (the main orchestrator)
This file depends on: object_detection.py (Detection class)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from .object_detection import Detection
import open3d as o3d


class SemanticMapObject:
    """
    Represents a persistent object in the map, accumulated across multiple frames.

    Attributes:
        point_cloud: Accumulated 3D points in world frame
        features: Averaged feature vector
        bbox_3d: 3D bounding box [min_xyz, max_xyz]
        num_detections: Number of times observed
        confidence: Average confidence score
        is_visible: Whether object was seen in current frame
    """
    def __init__(
        self,
        point_cloud: o3d.geometry.PointCloud,
        features: torch.Tensor,
        confidence: float,
        object_id: int
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.object_id = object_id
        self.point_cloud = point_cloud
        self.features = features
        self.confidence = confidence
        self.num_detections = 1
        self.is_visible = False

        # Compute 3D bounding box
        self.bbox_3d = self._compute_bbox_3d(point_cloud)
        

    def _compute_bbox_3d(self, pcd: o3d.geometry.PointCloud) -> torch.Tensor:
        """
        Compute 3D axis-aligned bounding box from point cloud.

        Args:
            pcd: o3d.geometry.PointCloud

        Returns: 8 points that define the bounding box
        """
        # create the bounding box
        v = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
        # get the box points
        v = np.asarray(v.get_box_points())
        # return as tensor
        return torch.from_numpy(v).float().to(self.device)
    
    def _downsample_voxel(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.025,
    ) -> o3d.geometry.PointCloud:
        
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd

    def update(self, detection: Detection):
        """
        Merge a new detection into this object.
        """

        # append the new point cloud to the existing point cloud
        merged_point_cloud = self.point_cloud + detection.point_cloud

        # downsample to remove redundant points
        self.point_cloud = self._downsample_voxel(merged_point_cloud)

        # recompute the bounding box only if we have enough points
        if len(self.point_cloud.points) >= 4:
            self.bbox_3d = self._compute_bbox_3d(self.point_cloud)
        # else: keep the old bbox

        # update the semantic features
        self.features = (self.num_detections * self.features + detection.features) / (self.num_detections + 1)
        # normalize the features
        self.features = F.normalize(self.features, dim=-1).float().to(self.device)

        # update the confidence
        self.confidence = (self.num_detections * self.confidence + detection.confidence) / (self.num_detections + 1)

        # update the number of detections
        self.num_detections += 1

        # Make the object visible
        self.is_visible = True

class SemanticMap:
    """
    Maintains a persistent map of all detected objects.

    Uses batch/vectorized similarity computation for efficiency (M×N matrix).
    Supports both IoU and nnratio for geometric similarity.
    """

    def __init__(
        self,
        similarity_threshold: float = 1.0,      # Minimum combined similarity for matching
        geometric_sim_type: str = "iou",        # "iou" or "overlap" (nnratio)
        nn_distance_threshold: float = 0.02,    # Distance threshold for nnratio (meters)
        max_objects: int = 100                  # Maximum objects to track
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.objects: List[SemanticMapObject] = []
        self.similarity_threshold = similarity_threshold
        self.geometric_sim_type = geometric_sim_type
        self.nn_distance_threshold = nn_distance_threshold
        self.max_objects = max_objects
        self.next_object_id = 0

    def reset(self):
        """
        Reset the object map to empty.
        """
        self.objects = []
        self.next_object_id = 0

    def update(self, detections: List[Detection]):
        """
        Main update function: match detections to existing objects or create new ones.
        """
        
        # Mark all objects as not visible
        for obj in self.objects:
            obj.is_visible = False

        # if there are no detection, then return
        if(len(detections) == 0):
            return

        # else if there are no objects in the map, then just create a mapobject for each detection
        elif(len(self.objects) == 0):
            for det in detections:
                new_obj = self._create_new_object(det)
                new_obj.is_visible = True
                self.objects.append(new_obj)
                self.next_object_id += 1
            return
        
        else:
        
            # Compute the MxN geometric similarity matrix
            geometric_sim = self._compute_geometric_similarities(detections)

            # Compute the MxN semantic similarity matrix
            semantic_sim = self._compute_semantic_similarities(detections)

            # Aggregate the similarities
            total_sim = self._aggregate_similarities(geometric_sim, semantic_sim)

            # Perform greedy assignment
            self._greedy_assignment(detections, total_sim)

    def _compute_geometric_similarities(
        self,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Compute (M×N) geometric similarity matrix.

        Dispatches to either IoU or nnratio based on self.geometric_sim_type.

        Args:
            detections: List of M detections

        Returns: (M, N) numpy array of geometric similarities
        """
        # print(f"[DEBUG] Geometric Sim Type: {self.geometric_sim_type}")
        if self.geometric_sim_type == "iou":
            return self._compute_iou_similarities(detections)
        elif self.geometric_sim_type == "overlap":
            return self._compute_nnratio_similarities(detections)
        else:
            raise ValueError(f"Unknown geometric_sim_type: {self.geometric_sim_type}")

    def _compute_iou_similarities(
        self,
        detections: List[Detection]
    ) -> torch.Tensor:
        """
        Compute (M×N) geometric similarity matrix using 3D bbox IoU.

        Args:
            detections: List of M detections

        Returns: (M, N) torch tensor of IoU scores [0, 1]
        """

        # Compute bboxes for detections
        dec_bbox = [compute_3d_bbox_from_points(d.point_cloud) for d in detections]
        dec_bbox = torch.stack(dec_bbox, dim=0).to(self.device)  # (M, 8, 3)

        obj_bbox = torch.stack([obj.bbox_3d for obj in self.objects], dim=0).to(self.device)  # (N, 8, 3)

        # Compute IoU matrix
        iou_matrix = compute_3d_iou_batch(dec_bbox, obj_bbox) # (M, N)
        iou_matrix = iou_matrix.float().to(self.device)

        return iou_matrix
    

    def _compute_nnratio_similarities(
        self,
        detections: List[Detection]
    ) -> torch.Tensor:
        """
        Compute (M×N) geometric similarity matrix using nnratio (nearest neighbor ratio).

        nnratio[i,j] = proportion of points in detection[i] that have nearest neighbors
                       in object[j] within distance threshold

        Args:
            detections: List of M detections

        Returns: (M, N) torch tensor of nnratio scores [0, 1]
        """
        M = len(detections)
        N = len(self.objects)
        overlap_matrix = np.zeros((M, N))

        # Build KDTrees for all map objects (for efficient nearest neighbor search)
        kdtrees = []
        for obj in self.objects:
            kdtree = o3d.geometry.KDTreeFlann(obj.point_cloud)
            kdtrees.append(kdtree)

        # Get detection point clouds as numpy arrays
        det_points_list = [np.asarray(det.point_cloud.points) for det in detections]

        # Compute pairwise overlaps
        for i in range(M):
            det_points = det_points_list[i]
            num_det_points = len(det_points)

            if num_det_points == 0:
                continue

            for j in range(N):
                # Optional: skip if bounding boxes don't overlap (optimization)
                # This saves computation for objects that are far apart
                iou = compute_3d_iou_batch(
                    self.objects[j].bbox_3d.unsqueeze(0),  # (1, 8, 3)
                    compute_3d_bbox_from_points(detections[i].point_cloud).unsqueeze(0)  # (1, 8, 3)
                )
                if iou.item() < 1e-6:
                    continue

                # Count points within distance threshold
                overlap_count = 0
                for point in det_points:
                    # Search for nearest neighbor
                    [_, _, dist_sq] = kdtrees[j].search_knn_vector_3d(point, 1)
                    # dist_sq is squared distance, compare with squared threshold
                    if dist_sq[0] < self.nn_distance_threshold ** 2:
                        overlap_count += 1

                # Compute ratio
                overlap_matrix[i, j] = overlap_count / num_det_points

        return torch.from_numpy(overlap_matrix).float().to(self.device)
    

    def _compute_semantic_similarities(
        self,
        detections: List[Detection]
    ) -> torch.Tensor:
        """
        Compute (M×N) semantic similarity matrix using cosine similarity.

        M = Number of detections
        N = Number of map objects

        Args:
            detections: List of M detections

        Returns: (M, N) torch tensor of cosine similarities [-1, 1]
        """
        # Stack detection features into (M, D) tensor
        det_features = torch.stack([d.features.squeeze() for d in detections], dim=0).to(self.device)  # (M, D)
        # print(f"DEBUG SHAPE: det_features after stack: {det_features.shape}")

        # Stack object features into (N, D) tensor
        obj_features = torch.stack([obj.features.squeeze() for obj in self.objects], dim=0).to(self.device)  # (N, D)
        # print(f"DEBUG SHAPE: obj_features after stack: {obj_features.shape}")

        # Reshape for broadcasting: (M, 1, D) and (1, N, D)
        det_features = det_features.unsqueeze(1)  # (M, 1, D)
        obj_features = obj_features.unsqueeze(0)  # (1, N, D)
        # print(f"DEBUG SHAPE: det_features after unsqueeze: {det_features.shape}")
        # print(f"DEBUG SHAPE: obj_features after unsqueeze: {obj_features.shape}")

        visual_sim = F.cosine_similarity(det_features, obj_features, dim=2)  # (M, N)
        visual_sim = visual_sim.float().to(self.device)

        return visual_sim

    def _aggregate_similarities(
        self,
        geometric_sim: torch.Tensor,
        semantic_sim: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine geometric and semantic similarities.

        Args:
            geometric_sim: (M, N) torch Tensor of geometric similarities
            semantic_sim: (M, N) torch tensor of semantic similarities

        Returns: (M, N) torch tensor of combined similarities
        """

        sims = geometric_sim + semantic_sim  # Element-wise addition
        sims = sims.float().to(self.device)

        return sims

    def _greedy_assignment(
        self,
        detections: List[Detection],
        similarity_matrix: torch.Tensor
    ):
        """
        Perform greedy assignment of detections to objects.

        Args:
            detections: List of M detections
            similarity_matrix: (M, N) combined similarity scores
        """
        # for each detection
        for i, det in enumerate(detections):
            
            max_sim, best_obj_idx = torch.max(similarity_matrix[i], dim=0)

            if(max_sim.item() < self.similarity_threshold):
                # create new object
                new_obj = self._create_new_object(det)
                new_obj.is_visible = True
                self.objects.append(new_obj)
                self.next_object_id += 1
            
            else:
                # merge into the best_obj_idx object
                self.objects[best_obj_idx.item()].update(det)
                self.objects[best_obj_idx.item()].is_visible = True

    def _create_new_object(self, detection: Detection) -> SemanticMapObject:
        """
        Create a new SemanticMapObject from a detection.

        Returns: New SemanticMapObject with unique ID
        """
        obj = SemanticMapObject(
            point_cloud = detection.point_cloud,
            features = detection.features,
            confidence = detection.confidence,
            object_id = self.next_object_id
        )

        return obj

    def get_visible_objects(self) -> List[SemanticMapObject]:
        """
        Get all objects marked as visible in current frame.

        Returns: List of visible SemanticMapObjects
        """
        return [obj for obj in self.objects if obj.is_visible]

    def get_all_objects(self) -> List[SemanticMapObject]:
        """
        Get all objects in the map (visible or not).

        Returns: List of all SemanticMapObjects
        """
        return self.objects


# ══════════════════════════════════════════════════════════════════════
# Helper Functions for 3D Geometry
# ══════════════════════════════════════════════════════════════════════

def compute_3d_bbox_from_points(pcd: o3d.geometry.PointCloud) -> torch.Tensor:
    """
    Compute 3D axis-aligned bounding box from point cloud.

    Args:
        pcd: o3d.geometry.PointCloud

    Returns: 8 points that define the bounding box
    """
    # create the bounding box
    v = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points)
    # get the box points
    v = np.asarray(v.get_box_points())
    # return as tensor
    return torch.from_numpy(v).float()


def compute_3d_iou(bbox1: np.ndarray, bbox2: np.ndarray, use_iou=True) -> float:
    """
    Compute IoU between two 3D axis-aligned bounding boxes.

    Args:
        bbox1: (8, 3) [min_xyz, max_xyz]
        bbox2: (8, 3) [min_xyz, max_xyz]

    Returns: IoU score [0, 1]
    """
    # Get the coordinates of the first bounding box
    bbox1_min = bbox1[0]
    bbox1_max = bbox1[1]

    # Get the cooridnates of the second bounding box
    bbox2_min = bbox2[0]
    bbox2_max = bbox2[1]

    # Compute the overlap between the two bounding boxes
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)

    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)

    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)

    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume + 1e-6)

    if use_iou:
        return iou
    else:
        return max_overlap

def compute_3d_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of 3D bounding boxes (vectorized).

    Args:
        bbox1: (M, 8, 3) - M bounding boxes
        bbox2: (N, 8, 3) - N bounding boxes

    Returns: (M, N) IoU matrix
    """

    # Compute min and max for each box
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)

    # Expand dimensions for broadcasting
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

    # Compute max of min values and min of max values
    # to obtain the coordinates of intersection box.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

    # Compute volume of intersection box
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

    # Compute volumes of the two sets of boxes
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

    # Compute IoU, handling the special case where there is no intersection
    # by setting the intersection volume to 0.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

    return iou