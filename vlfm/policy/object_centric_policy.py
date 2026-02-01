"""
object_centric_policy.py

Purpose: Stage 3 - Object-Centric Navigation Policy
Orchestrates object detection (Stage 1) and mapping (Stage 2) for frontier-based navigation.

Key difference from ITMPolicy:
    - ITMPolicy: Scores the ENTIRE image with BLIP2ITM → uniform score for whole FOV
    - ObjectCentricPolicy: Scores INDIVIDUAL objects → different scores per object location

Pipeline per timestep:
    1. Detect objects (SAM + SigLIP fusion) → List[Detection]
    2. Update object map (association) → persistent SemanticMapObjects
    3. Score visible objects against target text
    4. Project object scores onto value map
    5. Select best frontier using value map scores
"""

import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from vlfm.policy.itm_policy import BaseITMPolicy, ITMPolicyV2
from vlfm.policy.habitat_policies import HabitatMixin
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer

# Our object-centric modules (Stage 1 & 2)
from vlfm.object_centric.object_detection import ObjectSegmenter
from vlfm.object_centric.object_map import SemanticMap, SemanticMapObject
from vlfm.object_centric.sam_detector import MobileSAMClient
from vlfm.object_centric.siglip2 import SigLIPClient
from vlfm.object_centric.clip_encoder import CLIPClient
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_policy
class ObjectCentricPolicy(HabitatMixin, ITMPolicyV2):
    """
    Object-centric navigation policy.

    Inherits from BaseITMPolicy to reuse:
        - Frontier selection logic (_get_best_frontier)
        - Value map infrastructure
        - Obstacle map and navigation
    """

    def __init__(
        self,
        text_prompt: str,
        camera_intrinsics: np.ndarray,
        encoder = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(text_prompt=text_prompt, *args, **kwargs)

        self.mobile_sam_client = MobileSAMClient()

        if encoder is None:
            self.encoder = CLIPClient()
        else:
            self.encoder = encoder
        
        self.object_segmenter = ObjectSegmenter(
            sam_detector=self.mobile_sam_client,
            encoder=self.encoder,
            camera_intrinsics=camera_intrinsics,
            min_points=16,  # Match ConceptGraphs min_points_threshold
        )

        self.semantic_map = SemanticMap(
            similarity_threshold=0.7,
            geometric_sim_type="iou"
        )

        # Store the text prompt template (e.g., "Seems like there is a target_object ahead.")
        # The actual target will be set in _pre_step() when we know the objectgoal
        self._text_prompt = text_prompt
        self.target_text_features = None  # Will be set after we know the target object
        
        self.cos = torch.nn.CosineSimilarity(dim = -1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._value_map = ValueMap(
            value_channels=1,
            use_max_confidence=True,
            obstacle_map=self._obstacle_map
        )
        self._acyclic_enforcer = AcyclicEnforcer()
        
        # Frontier tracking for _get_best_frontier() persistence
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _reset(self) -> None:
        """Reset policy state for new episode."""
        super()._reset()
        self.semantic_map.reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self.target_text_features = None  # Reset for new episode
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _pre_step(self, observations: Dict, masks: Tensor) -> None:
        """Pre-step processing to encode target text features."""
        super()._pre_step(observations, masks)
        
        # Encode target text features only once per episode when target is known
        if self.target_text_features is None and self._target_object:
            # Official SigLIP2 format from HuggingFace examples:
            # texts = ["a photo of 2 cats", "a photo of 2 dogs"]
            prompt = f"a photo of a {self._target_object.lower()}"
            self.target_text_features = self.encoder.encode_text(prompt).squeeze(0)  # [1, 768] -> [768]
            print(f"[Encoder] Encoded target: '{prompt}'")
            # print(f"[DEBUG] Text features shape: {self.target_text_features.shape}")
            # print(f"[DEBUG] Text features norm: {torch.norm(self.target_text_features):.4f}")
            # print(f"[DEBUG] Text features sample: {self.target_text_features[:5]}\n")

    def _initialize(self) -> Tensor:
        return super()._initialize()

    def _get_policy_info(self, detections) -> Dict[str, Any]:
        return super()._get_policy_info(detections)

    # _cache_observations the super method is used

    # for _act, the implementation of ITMPolicyV2 is used here

    def _update_value_map(self) -> None:
        """
        Update value map with object-centric scores.

        """
        # Skip if target features not yet encoded
        if self.target_text_features is None:
            return
        
        # Step 1: Get observations
        rgb, depth, camera_pose, min_depth, max_depth, fov = self._observations_cache["value_map_rgbd"][0]

        # Step 2: Detect objects in current frame
        detections = self.object_segmenter.detect_objects(rgb, depth, camera_pose)
        print(f"DEBUG: Detected {len(detections)} objects this frame")

        # Step 3: Update object map
        self.semantic_map.update(detections)

        # Step 4: Get visible objects and compute scores
        visible_objects = self.semantic_map.get_visible_objects()
        scores = self.compute_object_target_similarity(visible_objects, self.target_text_features)
        # print(f"DEBUG: {len(visible_objects)} visible objects in map, scores: {scores}\n")

        # Step 5: Update value map
        # OLD WAY: Paint individual objects (Sparse)
        # self._value_map.update_map_object_wise(visible_objects, scores, depth, 
        # camera_pose, min_depth, max_depth, fov)

        # NEW WAY: Paint entire FOV with MAX score (Dense)
        # matches original VLFM/BLIP2 behavior but with SigLIP object scores
        if len(scores) > 0:
            max_score = float(scores.max())
        else:
            max_score = 0.0
            
        self._value_map.update_map(
            np.array([max_score]), 
            depth, 
            camera_pose, 
            min_depth, 
            max_depth, 
            fov
        )
        
        # Debug: Check value map statistics
        vm_min = self._value_map._value_map.min()
        vm_max = self._value_map._value_map.max()
        vm_nonzero = self._value_map._value_map[self._value_map._value_map != 0]
        if len(vm_nonzero) > 0:
            print(f"DEBUG: Value map - min: {vm_min:.4f}, max: {vm_max:.4f}, nonzero_mean: {vm_nonzero.mean():.4f}, nonzero_count: {len(vm_nonzero)}")
        else:
            print(f"DEBUG: Value map - ALL ZEROS (no objects painted)")

        # Step 6: Update agent trajectory
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"]
        )

    # Same implementation as ITMPolicyV2, but with debug statements
    def _sort_frontiers_by_value(
        self,
        observations: Any,
        frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Sort frontiers by value map scores.

        Same as ITMPolicyV2 - uses value map's sort_waypoints().
        """
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        print(f"[DEBUG] Frontiers: {len(frontiers)}, Top 3 values: {sorted_values[:3] if len(sorted_values) > 0 else 'none'}")
        return sorted_frontiers, sorted_values

    # new logic
    def compute_object_target_similarity(
        self, 
        visible_objects: List[SemanticMapObject],
        target_text_feats: torch.Tensor
    ) -> np.ndarray:
        """
        Compute similarity between visible objects and target text.
        """

        if not visible_objects:
            return np.array([])
        
        object_feats = torch.stack([obj.features for obj in visible_objects])
        
        # Ensure proper shape: remove any extra dimensions
        object_feats = object_feats.squeeze()  # [N, 1, 768] -> [N, 768]
        if object_feats.dim() == 1:  # Handle case of single object
            object_feats = object_feats.unsqueeze(0)
        
        # compute cosine similarity        
        raw_cosine = self.cos(object_feats, target_text_feats)
        scores = raw_cosine.clone()
             
        # DEBUG: Print stats
        if len(scores) > 0:
            print(f"[DEBUG] Scores - Max Cosine: {raw_cosine.max():.4f}")
            if scores.max() > 0.1:
                print(f"[DEBUG] FOUND CANDIDATE! Score: {scores.max():.4f}")

        return np.float32(scores.squeeze(-1).cpu())