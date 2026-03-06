"""
object_centric_policy.py

Purpose: Stage 3 - Object-Centric Navigation Policy
Orchestrates object detection (Stage 1) and mapping (Stage 2) for frontier-based navigation.

Key difference from ITMPolicy:
    - ITMPolicy: Scores the ENTIRE image with BLIP2ITM → uniform score for whole FOV
    - ObjectCentricPolicy: Scores INDIVIDUAL objects → different scores per object location

Pipeline per timestep:
    1. Detect objects (SAM + CLIP fusion) → List[Segmentation]
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
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from vlfm.policy.itm_policy import BaseITMPolicy, ITMPolicyV2
from vlfm.policy.habitat_policies import (
    HM3D_ID_TO_NAME,
    MP3D_ID_TO_NAME,
    HabitatMixin,
)
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta

# Our object-centric modules (Stage 1 & 2)
from vlfm.object_centric.object_segmentation import ObjectSegmenter
from vlfm.object_centric.semantic_map import SemanticMap, SemanticMapObject
from vlfm.object_centric.sam_segmenter import MobileSAMClient
from vlfm.object_centric.clip_encoder import CLIPClient
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import PolicyActionData

DEBUG = True


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
            sam_segmenter=self.mobile_sam_client,
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

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:

        if ObjectGoalSensor.cls_uuid in observations:
            obj_goal = observations[ObjectGoalSensor.cls_uuid]
            if isinstance(obj_goal, torch.Tensor) and obj_goal.dtype in [torch.int, torch.long]:
                object_id = obj_goal[0].item()
                if hasattr(observations, "to_tree"):
                    observations = observations.to_tree()
                
                if self._dataset_type == "hm3d":
                    observations[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
                elif self._dataset_type == "mp3d":
                    observations[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
                    # Update non-coco caption for MP3D (critical for Yolo World)
                    self._non_coco_caption = " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
        
        try:
            """
            Starts the episode by 'initializing' and allowing robot to get its bearings
            (e.g., spinning in place to get a good view of the scene).
            Then, explores the scene until it finds the target object.
            Once the target object is found, it navigates to the object.
            """
            print(f"\n==================================================")
            self._pre_step(observations, masks)

            object_map_rgbd = self._observations_cache["object_map_rgbd"]
            detections = [
                self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
                for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
            ]

            # _current_detections no longer needed: _update_value_map runs its own detection
            # on value_map_rgbd to guarantee bbox alignment with the segmented image
            # self._current_detections = detections[0]
            # if self._current_detections.num_detections > 0:
            #     print(f"[Det] Found {self._current_detections.num_detections} objects")

            # Update Maps (Semantic & Value) — detections run internally on value_map_rgbd
            self._update_value_map()

            robot_xy = self._observations_cache["robot_xy"]
            goal = self._get_target_object_location(robot_xy)

            if not self._done_initializing:  # Initialize
                mode = "initialize"
                pointnav_action = self._initialize()
            elif goal is None:  # Haven't found target object yet
                mode = "explore"
                pointnav_action = self._explore(observations)
            else:
                mode = "navigate"
                pointnav_action = self._pointnav(goal[:2], stop=True)

            action_numpy = pointnav_action.detach().cpu().numpy()[0]
            if len(action_numpy) == 1:
                action_numpy = action_numpy[0]
            print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
            print(f"==================================================")
            self._policy_info.update(self._get_policy_info(detections[0]))
            self._num_steps += 1

            self._observations_cache = {}
            self._did_reset = False

            # return pointnav_action, rnn_hidden_states
            action = pointnav_action

        except StopIteration:
            action = self._stop_action
            
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        # Return all the object detections for object-centric policy
        # detections.filter_by_class(target_classes)
        # det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        # detections.filter_by_conf(det_conf_threshold)
        detections.filter_by_conf(0.4)
    
        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            # Return all the object detections for object-centric policy
            # detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        # Filter background classes — adapted from ConceptGraphs (streamlined_detections.py):
        bg_classes = ["wall", "floor", "ceiling", "door", "window"]
        keep = torch.tensor(
            [p not in bg_classes for p in detections.phrases], dtype=torch.bool
        )
        detections._filter(keep)

        return detections

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        target_classes = self._target_object.split("|")
        has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        # ObjectMap uses strict baseline thresholds (0.8 COCO / 0.4 non-COCO)
        # SemanticMap (in _update_value_map) uses its own broader 0.4 detections
        objectmap_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold

        for idx in range(len(detections.logits)):

            # Strict ObjectMap filtering: target class + baseline confidence
            if detections.phrases[idx] != self._target_object:
                continue
            if detections.logits[idx] < objectmap_conf_threshold:
                continue
        
            bbox_denorm = detections.boxes[idx].cpu().numpy() * np.array([width, height, width, height])
            x1, y1, x2, y2 = bbox_denorm.astype(int)

            if self._use_vqa:
                if self.target_text_features is not None:
                    crop = rgb[y1:y2, x1:x2]
                    crop_features = self.encoder.encode_image(crop).squeeze(0)
                    verify_score = self.cos(crop_features, self.target_text_features).item()
                    print(f"[Verify] '{detections.phrases[idx]}' CLIP score={verify_score:.3f}")
                    if verify_score < 0.30:
                        continue

            object_mask, _ = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _update_value_map(self) -> None:
        """
        Update value map with object-centric scores.

        """
        # Skip if target features not yet encoded
        if self.target_text_features is None:
            return
        
        # Step 1: Get observations
        rgb, depth, camera_pose, min_depth, max_depth, fov = self._observations_cache["value_map_rgbd"][0]

        # Step 2: Run detection on value_map_rgbd image (guarantees bbox alignment with rgb)
        value_map_detections = self._get_object_detections(rgb)
        if value_map_detections.num_detections > 0:
            print(f"[Map] Detected {value_map_detections.num_detections} objects in value_map frame")

        # Step 3: Segment objects in current frame using aligned detections
        # segmentations = self.object_segmenter.segment_objects(rgb, depth,
        #                                                 camera_pose, value_map_detections)
        globalFallback, segmentations, global_features = self.object_segmenter.segment_objects(rgb, depth,
                                                                                        camera_pose, value_map_detections)
        print(f"[Map] Segmented {len(segmentations)} objects in this frame")

        # Step 4: Update semantic map
        self.semantic_map.update(segmentations)

        # Step 5: Get visible objects
        visible_objects = self.semantic_map.get_visible_objects()
        
        # Step 6: Compute score to update the value map
        max_score = None
        if globalFallback or not visible_objects:
            max_score = self.compute_cosine_similarity(global_features, self.target_text_features)
        else:
            scores = self.compute_object_target_similarities(visible_objects, self.target_text_features)
            # print(f"[Map] Visible: {len(visible_objects)}, Scores: {scores}\n")
            max_score = float(scores.max())
           
        # Step 7: Update value map
        # Paint entire FOV with MAX score (Dense)
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
            print(f"[Map] ValueMap: min={vm_min:.2f}, max={vm_max:.2f}, count={len(vm_nonzero)}")
        else:
            print(f"[Map] ValueMap: Empty")

        # Step 8: Update agent trajectory
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
        if len(frontiers) > 0:
            print(f"[Nav] Frontiers: {len(frontiers)}, Top 3 Vals: {sorted_values[:3] if len(sorted_values) > 0 else 'none'}")
        return sorted_frontiers, sorted_values


    def compute_cosine_similarity(
        self,
        global_features: torch.Tensor,
        target_text_feats: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between global features and target text features.
        """
        global_features = global_features.squeeze(0)
        score = self.cos(global_features, target_text_feats).item()
        print(f"[Map] Scores - Global Features Cosine: {score:.4f}")
        return score

    def compute_object_target_similarities(
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
             
        if len(scores) > 0:
            print(f"[Map] Scores - Max Cosine: {raw_cosine.max():.4f}")

        return np.atleast_1d(scores.squeeze(-1).float().cpu().numpy())