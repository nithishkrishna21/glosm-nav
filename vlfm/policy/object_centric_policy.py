"""
object_centric_policy.py

Purpose: Stage 3 - Object-Centric Navigation Policy
Orchestrates object detection (Stage 1) and mapping (Stage 2) for frontier-based navigation.

Key difference from ITMPolicy:
    - ITMPolicy: Scores the ENTIRE image with BLIP2ITM → uniform score for whole FOV
    - ObjectCentricPolicy: Scores INDIVIDUAL objects → different scores per object location

Pipeline per timestep:
    1. Detect objects (SAM + SigLIP fusion) → List[Detection]
    2. Update object map (association) → persistent MapObjects
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

from vlfm.policy.itm_policy import BaseITMPolicy
from vlfm.policy.habitat_policies import HabitatMixin
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer

# Our object-centric modules (Stage 1 & 2)
from vlfm.object_centric.object_detection import ObjectDetector
from vlfm.object_centric.object_map import ObjectMap, MapObject
from vlfm.object_centric.sam_detector import MobileSAMClient
from vlfm.object_centric.siglip2 import SigLIPClient
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_policy
class ObjectCentricPolicy(HabitatMixin, BaseITMPolicy):
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
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(text_prompt=text_prompt, *args, **kwargs)

        self.mobile_sam_client = MobileSAMClient()
        self.siglip_client = SigLIPClient()

        self.object_detector = ObjectDetector(
            sam_detector=self.mobile_sam_client,
            siglip=self.siglip_client,
            camera_intrinsics=camera_intrinsics,
        )

        self.object_map = ObjectMap(
            similarity_threshold=0.7,
            geometric_sim_type="iou"
        )

        self.target_text_features = self.siglip_client.encode_text(text_prompt)

        self.cos = torch.nn.CosineSimilarity(dim = -1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ValueMap (reusing VLFM's implementation)
        # Note: We use 1 channel per target object (usually 1)
        self._value_map = ValueMap(
            value_channels=1,
            use_max_confidence=True,
            obstacle_map=self._obstacle_map
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    def _reset(self) -> None:
        """Reset policy state for new episode."""
        super()._reset()
        self.object_map.reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()

    def _initialize(self) -> Tensor:
        return super()._initialize()

    def _get_policy_info(self) -> Dict[str, Any]:
        return super()._get_policy_info()

    # _cache_observations the super method is used

    # for _act, the implementation of ITMPolicyV2 is used here
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:

        self._pre_step(observations, masks)
        
        # if self._visualize:
        #     self._update_value_map()

        self._update_value_map()
            
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    # custome logic, needs to be overwritten
    def _update_value_map(self) -> None:
        """
        Update value map with object-centric scores.

        """
        # Step 1: Get observations
        rgb, depth, camera_pose, min_depth, max_depth, fov = self._observations_cache["value_map_rgbd"][0]

        # Step 2: Detect objects in current frame
        detections = self._object_detector.detect_objects(rgb, depth, camera_pose)

        # Step 3: Update object map
        self._object_map.update(detections)

        # Step 4: Get visible objects and compute scores
        visible_objects = self._object_map.get_visible_objects()
        scores = self.compute_object_target_similarity(visible_objects, self.target_text_features)

        # Step 5: Update value map
        self._value_map.update_map_object_wise(visible_objects, scores, depth, 
        camera_pose, min_depth, max_depth, fov)

        # Step 6: Update agent trajectory
        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"]
        )

    # originally the logic of ITMPolicyV2 is used here, 
    # but since we inherit BaseITMPolicy, we need to overwrite this method
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
        return sorted_frontiers, sorted_values

    # new logic
    def compute_object_target_similarity(
        self, 
        visible_objects: List[MapObject], 
        target_text_feats: torch.Tensor
    ) -> np.ndarray:
        """
        Compute similarity between visible objects and target text.
        """

        if not visible_objects:
            return np.array([])
        
        object_feats = torch.stack([obj.features for obj in visible_objects])
        
        # compute cosine similarity        
        scores = self.cos(object_feats, target_text_feats)

        return np.float32(scores.squeeze(-1).cpu())