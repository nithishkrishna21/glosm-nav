# Vision-Language Frontier Maps: Plan and Ideas

## Section 1: Proposed Methodology (Model-Agnostic)

### Overview

Our approach builds upon VLFM by replacing frontier-based scoring with object-centric semantic scoring. We combine ConceptGraphs-inspired 3D object mapping with vision-language models to create persistent object representations and score them against navigation goals.

### Pipeline Steps

#### Step 1: Object Detection & Feature Extraction

1. **Segmentation**: Use Mobile-SAM to segment RGB images into 2D object masks
2. **Multi-view Feature Fusion**: For each detected object, extract features from 3 different views:
   - Full RGB image (provides context)
   - Cropped region with background (object + surroundings)
   - Cropped region without background (masked object only)
3. **Weighted Fusion**: Combine the three feature vectors using learned or fixed weights
4. **3D Point Cloud**: Project masked depth to create 3D point cloud for each object
5. **Result**: Each detection has `{fused_features, 3D_points}`

#### Step 2: Object Association & Mapping

1. **Geometric Matching**: Compare new detections with existing objects using point cloud overlap
2. **Semantic Matching**: Compare feature vectors between new and existing objects
3. **Decision**: If match found → merge detection with existing object, else → create new object
4. **Result**: Persistent 3D object map with associated features

#### Step 3: Semantic Scoring

1. **Episode Initialization**: Encode target text once at start (e.g., "bed")
2. **Per-Timestep Scoring**: For every visible object in the map:
   - Compute cosine similarity between object's fused features and target features
   - Assign relevance score (0-1) indicating likelihood of being the target
3. **Result**: Each mapped object has a semantic score

#### Step 4: Project Scores to Value Map

1. **Spatial Projection**: For objects in current field of view, project their scores onto a 2D value map
2. **Implementation**: Leverage VLFM's existing value map projection code
3. **Result**: 2D grid where each cell contains aggregated object scores

#### Step 5: Confidence Weighting (VLFM - Unchanged)

1. **Cone-shaped Confidence**: Apply distance-based confidence mask
2. **Temporal Fusion**: Weighted averaging with previous timesteps
   - `v_new = (c_curr * v_curr + c_prev * v_prev) / (c_curr + c_prev)`
3. **Result**: Refined value map accounting for observation uncertainty

#### Step 6: Frontier Selection & Navigation (VLFM - Unchanged)

1. **Frontier Extraction**: Identify frontiers (boundaries between explored/unexplored)
2. **Value-based Selection**: Pick frontier with highest value on the value map
3. **Navigation**: Use pre-trained PointNav policy to reach selected frontier
4. **Termination**: Stop when target object score exceeds threshold and robot is close

### Key Innovations

1. **Object-centric vs Frontier-centric**: Score persistent objects rather than ephemeral frontier views
2. **Multi-view Fusion**: Robust feature extraction using HOV-SG-inspired weighted fusion
3. **Persistent Mapping**: ConceptGraphs-style object association maintains object identity across timesteps
4. **Efficient Scoring**: Encode target text once, reuse for all objects throughout episode

---

## Section 2: Understanding BLIP2-ITM (Original VLFM Approach)

### Overview

VLFM uses BLIP2-ITM (Image-Text Matching) to score how well different frontier viewpoints match the navigation goal. This section explains how it works.

### Key Files

- **[vlfm/vlm/blip2itm.py](vlfm/vlm/blip2itm.py)** - BLIP2-ITM implementation
- **[vlfm/vlm/blip2.py](vlfm/vlm/blip2.py)** - BLIP2 for VQA (Visual Question Answering)
- **[vlfm/policy/base_objectnav_policy.py](vlfm/policy/base_objectnav_policy.py)** - Uses both BLIP2 models

### How BLIP2-ITM Works

#### Architecture Components

BLIP2-ITM has **two separate encoders** that output embeddings in the **same aligned space**:

1. **Vision Encoder (ViT - Vision Transformer)**
   - Encodes images into visual embeddings
   - Pre-trained on large-scale vision datasets

2. **Text Encoder (Q-Former + Language Model)**
   - Encodes text into text embeddings
   - Q-Former is BLIP-2's innovation that bridges vision and language
   - Works with a language model backbone (e.g., T5, OPT)

3. **Alignment**
   - Both encoders are trained together using contrastive learning
   - Outputs exist in the same embedding space (can be directly compared)
   - Uses Image-Text Contrastive (ITC) matching head

#### The `cosine()` Method

Located at [blip2itm.py:37-54](vlfm/vlm/blip2itm.py#L37)

```python
def cosine(self, image: np.ndarray, txt: str) -> float:
    pil_img = Image.fromarray(image)
    img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
    txt = self.text_processors["eval"](txt)
    with torch.inference_mode():
        cosine = self.model({"image": img, "text_input": txt}, match_head="itc").item()
    return cosine
```

**Step-by-step:**
1. Convert numpy array to PIL Image
2. Preprocess image (resize, normalize, tensorize, add batch dimension)
3. Preprocess text (tokenize, encode)
4. Run model with `match_head="itc"` (Image-Text Contrastive)
5. Return cosine similarity score (float, typically -1 to 1, higher = better match)

#### What Gets Loaded

From `__init__` method at [blip2itm.py:20-35](vlfm/vlm/blip2itm.py#L20):

```python
self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
    name="blip2_image_text_matching",
    model_type="pretrain",
    is_eval=True,
    device=device,
)
```

- **`self.model`**: The BLIP2-ITM neural network (vision encoder + text encoder + matching head)
- **`self.vis_processors`**: Image preprocessing pipeline (transforms for eval mode)
- **`self.text_processors`**: Text preprocessing pipeline (tokenization for eval mode)

### Usage in VLFM Paper

#### Inputs

1. **Image**: Cropped RGB images from different frontier viewpoints
   - Candidate views the robot might navigate to
   - Each frontier region gets an image representation

2. **Text**: Target object description
   - Examples: "bed", "chair", "toilet"
   - The navigation goal (what the robot is searching for)

#### Scoring Process

```python
score = blip2itm.cosine(frontier_image, "bed")
```

This computes:
```
cosine_similarity(vision_encoder(frontier_image), text_encoder("bed"))
```

- **High score**: Frontier likely contains or leads to target object
- **Low score**: Frontier probably doesn't contain target

#### Navigation Decision

The robot uses these scores to:
1. Score all available frontiers against the target object
2. Select the frontier with the highest score
3. Navigate to that frontier using PointNav policy
4. Repeat until target is found

### BLIP2-ITM vs CLIP

#### Similarities
- Both use contrastive learning to align vision and text
- Both have separate vision and text encoders
- Both compute cosine similarity in aligned embedding space

#### Differences
- **CLIP**: Simple vision encoder + text encoder
- **BLIP2**: Vision encoder + **Q-Former** + LLM (more sophisticated)
- BLIP2 uses additional training objectives beyond contrastive loss
- BLIP2-ITM is specifically fine-tuned for image-text matching

#### Important Note
BLIP2-ITM does NOT use CLIP's encoders - it has its own separate architecture and weights. The approach is similar (contrastive learning), but the implementation is different.

### Server-Client Architecture

BLIP2-ITM runs as a server (see [blip2itm.py:76-84](vlfm/vlm/blip2itm.py#L76)):

```python
class BLIP2ITMServer(ServerMixin, BLIP2ITM):
    def process_payload(self, payload: dict) -> dict:
        image = str_to_image(payload["image"])
        return {"response": self.cosine(image, payload["txt"])}
```

Policies use the client (see [blip2itm.py:57-64](vlfm/vlm/blip2itm.py#L57)):

```python
class BLIP2ITMClient:
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/blip2itm"

    def cosine(self, image: np.ndarray, txt: str) -> float:
        response = send_request(self.url, image=image, txt=txt)
        return float(response["response"])
```

This separation allows:
- Heavy model runs on GPU server
- Lightweight client makes requests from navigation policy
- Multiple policies can share the same model server

### Other BLIP2 Usage in VLFM

BLIP2 (not ITM) is used for VQA at [base_objectnav_policy.py:326-335](vlfm/policy/base_objectnav_policy.py#L326):

```python
if self._use_vqa:
    question = f"Question: {self._vqa_prompt}"
    if not detections.phrases[idx].endswith("ing"):
        question += "a "
    question += detections.phrases[idx] + "? Answer:"
    answer = self._vqa.ask(annotated_rgb, question)
    if not answer.lower().startswith("yes"):
        continue
```

This verifies object detections by asking "Is this a [object]?" to reduce false positives.

### Summary

**Bottom Line:**
1. BLIP2-ITM has separate vision and text encoders
2. Both encoders output embeddings in the same aligned space
3. Cosine similarity measures how well image matches text
1.  BLIP2-ITM has separate vision and text encoders
2.  Both encoders output embeddings in the same aligned space
3.  Cosine similarity measures how well image matches text
4.  Higher scores indicate better match between frontier view and target object
5.  VLFM uses these scores to guide exploration toward the goal

---

## Section 3: Implementation Options

### Option 1: OpenCLIP (Current Implementation)

#### Overview

We have standardized on the `open_clip` library, which serves as a unified interface for accessing a vast array of state-of-the-art vision-language models. This is our **active implementation**.

#### Key Advantages

1.  **Massive Model Selection**: Access to OpenAI CLIP, LAION-2B models, DataComp, EVA-CLIP, and DFN.
2.  **Unified API**: Switch between architectures (e.g., `ViT-B-32` vs `ViT-H-14`) by changing a single string argument.
3.  **Optimization**: Supports Flash Attention and other speedups natively.
4.  **Community Standard**: The `laion2b_s34b_b79k` weights are the gold standard for Zero-Shot retrieval.

#### Implementation Status

-   **Implemented**: `vlfm/object_centric/clip_encoder.py`
-   **Deployed**: `vlfm/scripts/launch_vlm_servers_v2.sh` (Port 12186)
-   **Model**: Defaulting to `ViT-H-14` (Huge) for maximum accuracy.

---

### Option 2: MetaCLIP (Recommended Upgrade)

#### Overview

MetaCLIP (by Meta AI) uses the standard CLIP architecture but is trained on a rigorously curated dataset to remove noise and biases found in LAION/OpenAI data. Matches or beats OpenAI weights while being fully compatible with the OpenCLIP ecosystem.

#### Which Version?
-   **MetaCLIP v1 (Recommended)**: Released late 2023. English-focused, cleaner data. Perfect for this project.
-   **MetaCLIP v2**: Released mid-2025. Focused on **Worldwide/Multilingual** scaling. Unless you need to navigate in non-English environments, this is overkill.

#### How to Switch (MetaCLIP v1)

In `launch_vlm_servers_v2.sh`:
```bash
# Old
--model_name ViT-H-14 --pretrained laion2b_s34b_b79k

# New (MetaCLIP v1 - High Performance)
--model_name ViT-H-14-quickgelu --pretrained metaclip_fullcc
```

---

### Option 3: SigLIP 2 (Google's Alternative)

#### Overview

Google's generic Vision-Language Model using Sigmoid Loss for Image-Text Pre-training. Released Feb 2025. It is superior at handling dense images with multiple concepts because naturally "multi-label" (Sigmoid) rather than "one-of-many" (Softmax).

#### Status

-   **Implemented**: `vlfm/object_centric/siglip2.py`
-   **Shelved**: We moved to OpenCLIP for easier alignment with HOV-SG fusion weights, but this remains a strong backup if we need better dense scene understanding.

#### When to Use
-   If the agent struggles to distinguish multiple objects in a single crop.
-   If we want to leverage Google's latest multilingual capabilities.

---

### Option 4: DINOv3 + DINOv3-CLIP Adapter (Structure-First)

#### Overview

Use actual DINOv3 (Visual Foundation Model) for vision features, piped through a lightweight MLP adapter (3.15M parameters) to align them with CLIP's text space.

#### How It Works

1.  **Frozen DINOv3**: Extract rich, structure-aware visual features (great for geometry/parts).
2.  **Adapter**: Maps DINOv3 $\to$ CLIP Space.
3.  **CLIP Text**: Use standard CLIP to encode text.

#### Pros & Cons
-   **Pros**: Access to DINO's superior segmentation/part-awareness.
-   **Cons**: The adapter is a bottleneck. It was trained on images, so "Zero-Shot" text retrieval is less robust than end-to-end trained models like OpenCLIP or SigLIP.

#### When to Use
-   If we find that CLIP/SigLIP can *find* the object but fails to *segment* it accurately.
-   For experimental comparison.

---

## Section 4: HOV-SG Weighted Feature Fusion - Detailed Implementation

This section documents the exact weighted feature fusion implementation from HOV-SG that we'll adapt for our approach.

### Overview

HOV-SG extracts and fuses features from **three different views/crops** for each detected object to create robust, context-aware object representations.

### The 3 Views/Crops

For each detected object (segmented by SAM):

1. **F_g (Global Feature)** - Full image context
   - Input: The entire RGB image
   - Purpose: Captures scene-level context
   - Extraction: `F_g = CLIP_encoder(full_image)`

2. **F_l_unmasked (Local Unmasked)** - Cropped bounding box without background masking
   - Input: Bounding box crop with background intact
   - Purpose: Object + surrounding context
   - Extraction: `F_l_unmasked = CLIP_encoder(crop_bbox(image, mask, block_background=False))`

3. **F_l_masked (Local Masked)** - Cropped bounding box WITH background masking
   - Input: Bounding box crop with background masked out
   - Purpose: Pure object features without background interference
   - Extraction: `F_l_masked = CLIP_encoder(crop_bbox(image, mask, block_background=True))`

### Two-Stage Fusion Process

#### Stage 1: Fuse the Two Local Crops

**File:** `HOV-SG/hovsg/models/sam_clip_feats_extractor.py` (lines 52, 117)

```python
# Combine masked and unmasked local features
maskedd_weight = 0.4418  # From config: clip_masked_weight
F_l = maskedd_weight * F_l_masked + (1 - maskedd_weight) * F_l_unmasked
F_l = F_l / F_l.norm(dim=-1, keepdim=True)  # L2 normalize
```

**Weights:**
- Masked crop: 44.18%
- Unmasked crop: 55.82%

**Result:** Balanced local feature capturing both pure object and contextual information.

#### Stage 2: Fuse Local with Global (Similarity-Weighted)

**File:** `HOV-SG/hovsg/models/sam_clip_feats_extractor.py` (lines 122-128)

```python
# Compute cosine similarity between local and global features
cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
phi_l_G = cos(F_l, F_g)  # Similarity for each object

# Convert similarities to weights via softmax
w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)

# Weighted combination
F_p = w_i * F_g + (1 - w_i) * F_l

# Final normalization
F_p = F_p / F_p.norm(dim=-1, keepdim=True)
```

**Key Insight:** Weights are **dynamic per object**:
- Objects similar to global context → higher weight on F_g
- Objects dissimilar to context → higher weight on F_l (local details)

### Complete Fusion Pipeline

```
Input: RGB Image + SAM Masks
  ↓
[Step 1] Extract Global Feature
  └─→ F_g = CLIP(full_image)
  ↓
[Step 2] For each detected object:
  ├─→ crop_unmasked = bbox_crop(image, mask, block_background=False)
  ├─→ crop_masked = bbox_crop(image, mask, block_background=True)
  ├─→ F_l_unmasked = CLIP(crop_unmasked)
  └─→ F_l_masked = CLIP(crop_masked)
  ↓
[Step 3] Stage 1 Fusion - Combine local crops
  └─→ F_l = 0.4418 * F_l_masked + 0.5582 * F_l_unmasked
  └─→ F_l = normalize(F_l)
  ↓
[Step 4] Stage 2 Fusion - Combine with global
  ├─→ similarity = cosine_sim(F_l, F_g)
  ├─→ w_i = softmax(similarity)
  └─→ F_final = w_i * F_g + (1 - w_i) * F_l
  ↓
[Step 5] Final Normalization
  └─→ F_final = normalize(F_final)
  ↓
Output: Fused feature vector per object
```

### Configuration Parameters

From `HOV-SG/config/create_graph.yaml`:

```yaml
pipeline:
  clip_masked_weight: 0.4418      # Stage 1: masked vs unmasked weight
  clip_bbox_margin: 50            # Bounding box crop margin (pixels)
```

### Key Implementation Files

| Component | File | Lines |
|-----------|------|-------|
| Core fusion logic | `hovsg/models/sam_clip_feats_extractor.py` | 18-139 |
| Bounding box cropping | `hovsg/utils/sam_utils.py` | - |
| CLIP feature extraction | `hovsg/utils/clip_utils.py` | - |
| Graph construction & usage | `hovsg/graph/graph.py` | 170-210 |
| Configuration | `config/create_graph.yaml` | - |

### Adaptation for Our Approach

We'll use this exact fusion strategy but replace CLIP with our chosen VLM (ImageBind/SigLIP2):

```python
# For each detected object:
# 1. Extract 3 crops (full, bbox, masked_bbox)
F_g = VLM.encode_vision(full_image)
F_l_unmasked = VLM.encode_vision(crop_unmasked)
F_l_masked = VLM.encode_vision(crop_masked)

# 2. Stage 1 fusion
F_l = 0.4418 * F_l_masked + 0.5582 * F_l_unmasked
F_l = normalize(F_l)

# 3. Stage 2 fusion
similarity = cosine_sim(F_l, F_g)
w = softmax(similarity)
F_final = w * F_g + (1 - w) * F_l
F_final = normalize(F_final)

# 4. Store in object map with 3D points
object.fused_features = F_final
object.point_cloud = project_depth_to_3d(mask, depth)
```

---

## Section 5: Complete Implementation - Value Map, Confidence, and Frontier Selection

This section documents the complete implementation workflow for our object-centric approach, including how VLFM's value map and frontier selection work (which we reuse), and how to adapt them for object-centric scoring.

### Overview

**Key Concept:** VLFM uses a **value map** as a bridge between vision-language scoring and frontier selection. We keep this entire infrastructure but change **what gets scored** (objects instead of views) and **where scores are projected** (object point clouds instead of FOV cones).

### Key Files (VLFM - We Reuse These)

| Component | File | Key Functions |
|-----------|------|---------------|
| Value map creation/update | `vlfm/mapping/value_map.py` | `ValueMap.__init__()`, `update_map()` |
| VLM score computation | `vlfm/vlm/blip2itm.py` | `BLIP2ITM.cosine()` |
| Score projection | `vlfm/policy/itm_policy.py` | `_update_value_map()` (lines 191-211) |
| Frontier extraction | `vlfm/mapping/obstacle_map.py` | `_get_frontiers()` (lines 155-169) |
| Frontier scoring | `vlfm/mapping/value_map.py` | `sort_waypoints()` (lines 146-187) |
| Frontier selection | `vlfm/policy/itm_policy.py` | `_get_best_frontier()` (lines 76-152) |

---

## Part A: How VLFM Currently Works (Baseline Understanding)

### VLFM's Approach: Full-View Scoring

#### Step 1: Compute VLM Scores

**File:** `vlfm/policy/itm_policy.py` (lines 191-206)

```python
def _update_value_map(self) -> None:
    all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]

    # Score each full RGB image
    cosines = [
        [
            self._itm.cosine(
                rgb,  # Entire RGB frame (not cropped)
                p.replace("target_object", self._target_object)
            )
            for p in self._text_prompt.split(PROMPT_SEPARATOR)
        ]
        for rgb in all_rgb
    ]

    # Project scores to value map
    for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
        cosines, self._observations_cache["value_map_rgbd"]
    ):
        self._value_map.update_map(
            np.array(cosine), depth, tf, min_depth, max_depth, fov
        )
```

**What gets scored:** Single full RGB image per frame
**Text prompt:** `"Seems like there is a {target_object} ahead."`
**Output:** Single cosine score per frame (range: -1 to 1, typically 0-1)

#### Step 2: Project Scores to Value Map (Two-Channel System)

**File:** `vlfm/mapping/value_map.py` - `update_map()` method

VLFM uses a **two-channel value map**:
1. **Semantic Value Channel**: Contains VLM cosine similarity scores
2. **Confidence Channel**: Contains confidence weights for temporal fusion

**Why Two Channels? The Temporal Fusion Problem**

The confidence channel solves a critical problem: **how to properly fuse observations from different viewpoints over time**.

*Example scenario:*
```
Time 1: Robot at position A sees location X from edge of FOV
  - BLIP2 score: 0.3 (unreliable - bad angle, edge distortion)
  - Without confidence: Store 0.3

Time 2: Robot at position B sees same location X from center of FOV
  - BLIP2 score: 0.9 (reliable - good angle, center of view)
  - Without confidence: Average (0.3 + 0.9) / 2 = 0.6 ❌

Problem: The bad observation (0.3) corrupts the good one (0.9)!
```

*With confidence weighting:*
```
Time 1: Edge view
  - Semantic value: 0.3
  - Confidence: 0.1 (low - edge of FOV has low reliability)

Time 2: Center view
  - Semantic value: 0.9
  - Confidence: 1.0 (high - center of FOV is most reliable)

Weighted fusion:
  final_value = (0.1 * 0.3 + 1.0 * 0.9) / (0.1 + 1.0)
              = (0.03 + 0.9) / 1.1
              = 0.845 ✅

Result: High-confidence observation dominates, preserving quality!
```

**Key benefits of separate confidence channel:**
- **Quality preservation**: Good observations aren't degraded by bad ones
- **View-dependent weighting**: Center-of-FOV observations weighted more than edges
- **Temporal stability**: Once you get a good view, it persists even if later views are worse
- **Occlusion handling**: Partial occlusions don't destroy previously clear observations

**The confidence update formula is biased toward higher confidence:**
```python
c_new = (c_curr² + c_prev²) / (c_curr + c_prev)

Example:
  High replaces low: c_prev=0.2, c_curr=1.0 → c_new=0.867 (increases!)
  Low "updates" high: c_prev=1.0, c_curr=0.2 → c_new=0.867 (stays high!)
```

This means once you get a reliable observation, it's hard to erase - exactly what you want for stable navigation!

**Inputs:**
- `values`: Cosine score(s) from VLM (1D array)
- `depth`: Depth image (normalized 0-1)
- `tf_camera_to_episodic`: Camera pose (4x4 transform)
- `min_depth`, `max_depth`: Depth range
- `fov`: Field of view in radians

**Process:**

**Step 2a: Create Cone-Shaped Confidence Mask**

The confidence of a pixel within the FOV depends on its angular distance from the optical axis:

```python
# Confidence formula from VLFM paper
confidence(θ) = cos²(θ/(θ_fov/2) * π/2)

where:
  θ = angle between pixel and optical axis
  θ_fov = horizontal field of view
```

**Confidence distribution:**
- Pixels along optical axis (center): confidence = 1.0
- Pixels at FOV edges: confidence = 0.0
- Smooth falloff in between following cos² curve

**Step 2b: Apply Depth-Based Occlusion Masking**

Using the depth image, exclude areas of the FOV that are obstructed by obstacles (behind objects).

**Step 2c: Update Semantic Value Channel**

For pixels within the (non-occluded) FOV mask:

```python
# If pixel was seen before (has previous value):
v_new[i,j] = (c_curr[i,j] * v_curr[i,j] + c_prev[i,j] * v_prev[i,j]) /
             (c_curr[i,j] + c_prev[i,j])

# If pixel is seen for first time:
v_new[i,j] = v_curr[i,j]  # Just use current value
```

**Step 2d: Update Confidence Channel**

Confidence is updated using a weighted average biased toward higher confidence:

```python
c_new[i,j] = (c_curr[i,j]² + c_prev[i,j]²) / (c_curr[i,j] + c_prev[i,j])
```

**Key Insight:** This biases toward the more confident observation when fusing temporal information.

**Result:**
- 2D semantic value grid with confidence-weighted temporal fusion
- Center of FOV (high confidence) has more influence on updates
- Edge of FOV (low confidence) has less influence

**VLFM's Projection Pattern with Confidence:**
```
        Robot 🤖
        Camera FOV
          /   \
         / 0.5 \        ← Edge: low confidence
        /0.8 1.0\       ← Center: high confidence
       /__0.5____\      ← Edge: low confidence

Semantic Value Map (entire FOV gets same VLM score 0.8):
    ...........
   .............
  ....0.8.0.8....
 .....0.8.0.8.....
..................

Confidence Map (cone-shaped, peaks at center):
    ...........
   .....0.3.....
  ....0.7.1.0....  ← Center has confidence 1.0
 .....0.7.0.3.....
..................

When revisiting areas:
- High-confidence observations (center) dominate fusion
- Low-confidence observations (edges) contribute less
```

**Update Summary:**
1. Create cone-shaped confidence mask (cos² falloff from center)
2. Apply depth-based occlusion masking
3. Update semantic values with confidence-weighted averaging
4. Update confidence scores (biased toward higher confidence)

#### Step 3: Extract Frontiers

**File:** `vlfm/mapping/obstacle_map.py` (lines 155-169)

```python
def _get_frontiers(self) -> np.ndarray:
    # Dilate explored area to prevent small gaps
    explored_area = cv2.dilate(
        self.explored_area.astype(np.uint8),
        np.ones((5, 5), np.uint8),
        iterations=1,
    )

    # Detect frontier waypoints
    frontiers = detect_frontier_waypoints(
        self._navigable_map.astype(np.uint8),  # Navigable terrain
        explored_area,                          # Where robot has seen
        self._area_thresh_in_pixels,           # Min frontier size
    )
    return frontiers
```

**Frontier Definition:** Points on the boundary between explored and unexplored areas, on navigable terrain.

#### Step 4: Score Frontiers Using Value Map

**File:** `vlfm/mapping/value_map.py` (lines 146-187)

**CRITICAL: Frontiers are scored using SEMANTIC VALUES, not confidence scores!**

```python
def sort_waypoints(
    self,
    waypoints: np.ndarray,       # Frontier locations (Nx2)
    radius: float,                # Aggregation radius in meters (default: 0.5)
    reduce_fn: Optional[Callable] = None  # Multi-channel reduction
) -> Tuple[np.ndarray, List[float]]:

    def get_value(point: np.ndarray):
        # Convert to pixel coordinates
        px, py = world_to_pixel(point)

        # Extract values from SEMANTIC VALUE MAP (self._value_map)
        # NOT from confidence map (self._map)!
        all_values = [
            pixel_value_within_radius(self._value_map[..., c], (px, py), radius_px)
            for c in range(self._value_channels)
        ]
        return all_values

    values = [get_value(point) for point in waypoints]
    sorted_inds = np.argsort([-v for v in values])
    return waypoints[sorted_inds], sorted_values
```

**Process:**
- Convert frontier (x, y) meters → (px, py) pixels
- Lookup `pixel_value_within_radius(self._value_map[..., c], (px, py), radius_px)`
  - **Uses `self._value_map`** = semantic value channel (VLM scores)
  - **Does NOT use `self._map`** = confidence channel
- Returns maximum **semantic value** in circular region around frontier

**Key Data Structures in ValueMap:**
```python
# Line 66: Semantic value map (multi-channel for different prompts)
self._value_map = np.zeros((size, size, value_channels), np.float32)

# BaseMap line 22: Confidence map (single channel, inherited from BaseMap)
self._map = np.zeros((size, size), dtype=np.float32)
```

**What Each Stores:**
- `self._value_map[x, y, c]` = VLM cosine similarity score at location (x,y) for channel c
- `self._map[x, y]` = Confidence weight at location (x,y)

**ANSWER TO YOUR QUESTION:**

**Q: What scores are used to select the best frontiers - semantic scores or confidence scores?**

**A: SEMANTIC SCORES from `self._value_map`**

The confidence channel (`self._map`) is **ONLY used during temporal fusion** to weight how much current vs. previous observations matter. It's used in the fusion formula:

```python
# Fusion (lines 414-424 in value_map.py)
v_new = (c_curr * v_curr + c_prev * v_prev) / (c_curr + c_prev)  # Semantic values
c_new = (c_curr² + c_prev²) / (c_curr + c_prev)                   # Confidence values
```

But when **scoring frontiers** (line 169 in `sort_waypoints()`):
```python
all_values = [
    pixel_value_within_radius(self._value_map[..., c], ...)  # Uses semantic values!
    for c in range(self._value_channels)
]
```

**The workflow:**
1. **Confidence weights semantic fusion** → produces better semantic values over time
2. **Frontiers are scored** using the final (fused) **semantic values**
3. **Confidence is never directly used** for frontier selection

**Why this matters for your approach:**
- You must maintain **both** `value_map` (semantic scores) and `confidence_map`
- Confidence helps create better semantic values through temporal fusion
- But the final frontier selection uses **only semantic values**
- Your object scores will go into `value_map`, and you can use FOV-based or object-based confidence

**Visual Summary:**
```
┌─────────────────────────────────────────────────────────────────┐
│              How Confidence and Semantic Values Work            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Time 1: First observation                                      │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ VLM Score: 0.3          Confidence: 0.1 (edge view)  │     │
│  └───────────────────────────────────────────────────────┘     │
│           ↓                           ↓                          │
│    value_map[x,y] = 0.3        confidence_map[x,y] = 0.1       │
│                                                                  │
│  Time 2: Same location, better view                             │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ VLM Score: 0.9          Confidence: 1.0 (center view)│     │
│  └───────────────────────────────────────────────────────┘     │
│           ↓                           ↓                          │
│  FUSION STEP (uses BOTH channels):                              │
│  v_new = (0.1*0.3 + 1.0*0.9)/(0.1+1.0) = 0.845                │
│  c_new = (0.1² + 1.0²)/(0.1+1.0) = 0.867                       │
│           ↓                           ↓                          │
│    value_map[x,y] = 0.845      confidence_map[x,y] = 0.867     │
│                                                                  │
│  Frontier Selection:                                             │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  frontier_score = value_map[frontier_location]        │     │
│  │                 = 0.845  ← Uses SEMANTIC value!       │     │
│  │                                                        │     │
│  │  Confidence NOT used here!                            │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                  │
│  Summary:                                                        │
│  • Confidence → Used in FUSION to weight observations          │
│  • Semantic Values → Used in FRONTIER SELECTION                │
└─────────────────────────────────────────────────────────────────┘
```

#### Step 5: Select Best Frontier

**File:** `vlfm/policy/itm_policy.py` (lines 76-152)

```python
def _get_best_frontier(
    self,
    observations: Dict,
    frontiers: np.ndarray,
) -> Tuple[np.ndarray, float]:

    # 1. Sort frontiers by value
    sorted_frontiers, sorted_values = self._sort_frontiers_by_value(
        observations, frontiers
    )

    # 2. Apply stickiness (prefer previously selected frontier)
    if self._last_frontier is not None:
        if curr_value + 0.01 > self._last_value:  # 0.01 hysteresis
            return self._last_frontier, self._last_value

    # 3. Check for cyclic behavior
    for frontier, value in zip(sorted_frontiers, sorted_values):
        cyclic = self._acyclic_enforcer.check_cyclic(
            robot_xy, frontier, top_two_values
        )
        if not cyclic:
            return frontier, value  # Return first non-cyclic

    # 4. Fallback: return closest frontier if all cyclic
    return closest_frontier, closest_value
```

**Selection Strategy:**
1. Sort by value (highest first)
2. Prefer "sticky" frontier (avoid oscillation)
3. Avoid cyclic behavior
4. Fallback to closest if stuck

Detailed comparison with our approach is in Part D.

---

## Part B: Our Object-Centric Approach (Complete Implementation)

### Coordinate Transformation Pipeline (Verified from VLFM Codebase)

**How VLFM Projects Depth to Value Map:**

VLFM uses a simplified projection approach since it broadcasts a uniform score:

```python
# value_map.py:_process_local_data() (lines 221-286)
# 1. Squash depth image to 1D boundary (max per column)
depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

# 2. Convert to 2D camera-frame coordinates using angles
angles = np.linspace(-fov / 2, fov / 2, len(depth_row))
x = depth_row  # Forward distance
y = depth_row * np.tan(angles)  # Horizontal offset

# 3. Create filled cone contour (not per-pixel projection)
# - Draws filled contour from depth boundary
# - Applies cos² confidence weighting

# 4. Rotate mask to camera yaw (value_map.py:_localize_new_data, lines 288-319)
yaw = extract_yaw(tf_camera_to_episodic)
curr_data = rotate_image(curr_data, -yaw)

# 5. Overlay at camera position on global value map
cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]
```

**For Our Object-Centric Approach:**

We need **full per-pixel 3D projection** because we have object-specific scores at specific locations:

```python
# Complete transformation pipeline for each depth pixel in object mask:

# 1. Depth pixel (u, v) in camera image
depth_value = depth[u, v]

# 2. Convert to 3D camera frame coordinates
# Using camera intrinsics K (focal length, principal point)
x_cam = (u - cx) * depth_value / fx
y_cam = (v - cy) * depth_value / fy
z_cam = depth_value
point_camera = [x_cam, y_cam, z_cam, 1]

# 3. Transform to 3D world frame
point_world = camera_pose @ point_camera  # 4x4 transformation
x_world, y_world, z_world = point_world[:3]

# 4. Project to 2D value map grid (top-down view)
# From base_map.py:_xy_to_px() logic
grid_x = int(np.round(
    self._episode_pixel_origin[0] - y_world * self.pixels_per_meter
))
grid_y = int(np.round(
    self._episode_pixel_origin[1] + x_world * self.pixels_per_meter
))

# Note: z_world is ignored for top-down projection
```

**Key Differences:**
| Aspect | VLFM | Our Approach |
|--------|------|--------------|
| **What gets projected** | Filled cone contour (uniform score) | Per-object 3D point clouds |
| **Projection method** | Depth boundary → camera frame → rotate → overlay | Each pixel → 3D camera → 3D world → 2D grid |
| **Transformation** | Simplified (squash + angle-based) | Full 4x4 transformation matrix |
| **Why approach works** | Uniform score broadcast | Object-specific scores at specific locations |

---

### Summary: Coordinate Transformations - VLFM vs Our Approach

**VLFM's Coordinate Pipeline (Simplified):**
```
Depth Image (480×640)
  ↓ [Squash to max per column]
Depth Boundary (640,) - 1D array
  ↓ [Convert using angles: x = depth, y = depth * tan(θ)]
2D Camera-Frame Coordinates (x, y) in meters
  ↓ [Convert to pixels: multiply by pixels_per_meter]
Local Cone Pixels (relative to cone center)
  ↓ [Rotate entire cone image to camera yaw]
Rotated Cone Image
  ↓ [Extract camera position from pose matrix]
Camera Grid Position (px, py)
  ↓ [Overlay rotated cone at camera position]
Global Value Map (1000×1000) - filled with uniform score
```

**VLFM's World → Grid Conversion (Camera Position):**
```python
# From value_map.py:_localize_new_data() (lines 309-313)
cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]
```

**VLFM's General World → Grid Conversion Function:**
```python
# From base_map.py:_xy_to_px() (lines 35-46)
def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
    """
    Convert (x, y) world coordinates to (row, col) grid coordinates.

    Args: points (N, 2) - world coordinates in meters
    Returns: (N, 2) - grid pixel coordinates
    """
    # Step 1: Swap x/y order (world to grid axis swap)
    px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin

    # Step 2: Flip row coordinate (Y-axis flip)
    px[:, 0] = self._map.shape[0] - px[:, 0]

    return px.astype(int)

# Breakdown:
# Input:  (x, y) world coordinates
# Step 1: Reverse to (y, x), scale to pixels, add origin
# Step 2: Flip row = map.shape[0] - row
# Output: (row, col) grid coordinates
```

**Coordinate System Differences:**
```
World Frame (meters):              Grid Frame (pixels):
    Y ↑                                row 0 ┌─────→ col
      |                                      │
──────┼──────→ X                             │
      |                                      ↓
   (0,0)                                  row 999

X: Forward/backward              Row ≈ -Y (flipped)
Y: Left/right                    Col ≈ +X
Origin: (0, 0)                   Origin: (500, 500) center
```

---

**Our Approach's Coordinate Pipeline (Full 3D):**
```
2D Mask + Depth Image
  ↓ [ConceptGraphs: create_object_pcd() - unproject using camera intrinsics]
3D Point Cloud in Camera Frame (x_cam, y_cam, z_cam)
  ↓ [Transform using 4×4 camera_pose matrix]
3D Point Cloud in World Frame (x_world, y_world, z_world)
  ↓ [Convert using VLFM's _xy_to_px() function]
2D Grid Coordinates (row, col)
  ↓ [Look up confidence, apply per-object score]
Value Map Updated (1000×1000) - different scores per object
```

**What We Reuse from VLFM:**

| Component | VLFM Function | How We Use It |
|-----------|---------------|---------------|
| **World → Grid conversion** | `value_map._xy_to_px(points)` | Pass our 3D point cloud's (x, y) coordinates |
| **Confidence mask** | `value_map._localize_new_data()` | Call once per timestep, look up values |
| **Temporal fusion** | Formulas from `_fuse_new_data()` | Copy the fusion logic |

**Implementation Example:**
```python
# ═══════════════════════════════════════════════════════════════
# STEP 1: Create Confidence Cone (Call Once Per Timestep)
# ═══════════════════════════════════════════════════════════════
confidence_mask = value_map._localize_new_data(
    depth=depth,
    tf_camera_to_episodic=camera_pose,
    min_depth=0.5,
    max_depth=10.0,
    fov=79 * np.pi / 180
)
# Returns: (1000, 1000) confidence mask
# - Automatically rotated to camera direction
# - Automatically placed at camera position
# - Confidence: 1.0 at FOV center, 0.0 at edges/outside

# ═══════════════════════════════════════════════════════════════
# STEP 2: Project Objects and Update Both Channels
# ═══════════════════════════════════════════════════════════════
for obj in visible_objects:
    # Get 3D point cloud from ConceptGraphs (already in world frame)
    world_points = obj.point_cloud  # (N, 3) - (x, y, z)

    # Convert to grid using VLFM's function
    world_points_2d = world_points[:, :2]  # (N, 2) - ignore z for top-down
    grid_coords = value_map._xy_to_px(world_points_2d)  # (N, 2) - (row, col)

    # For each grid coordinate
    for (grid_row, grid_col) in grid_coords:
        # Bounds check
        if not (0 <= grid_row < 1000 and 0 <= grid_col < 1000):
            continue

        # Look up confidence from cone mask
        c_curr = confidence_mask[grid_row, grid_col]
        if c_curr <= 0:  # Outside FOV or occluded
            continue

        # Get previous values
        v_curr = obj.score
        v_prev = value_map._value_map[grid_row, grid_col, 0]
        c_prev = value_map._map[grid_row, grid_col]

        # Update Channel 1: Semantic Value
        if c_prev > 0:  # Temporal fusion
            value_map._value_map[grid_row, grid_col, 0] = (
                (c_curr * v_curr + c_prev * v_prev) / (c_curr + c_prev)
            )
        else:  # First observation
            value_map._value_map[grid_row, grid_col, 0] = v_curr

        # Update Channel 2: Confidence
        if c_prev > 0:  # Temporal fusion
            value_map._map[grid_row, grid_col] = (
                (c_curr**2 + c_prev**2) / (c_curr + c_prev)
            )
        else:  # First observation
            value_map._map[grid_row, grid_col] = c_curr
```

**Key Takeaway:**
- Use `_localize_new_data()` to get confidence cone (automatically rotated/positioned)
- Use `_xy_to_px()` to convert object point clouds to grid coordinates
- Look up confidence from cone, update both semantic and confidence channels
- The only difference from VLFM: we have object-specific scores, VLFM has uniform score

---

**CRITICAL: Confidence Mask and Value Map Share the Same Coordinate Space**

The confidence mask returned by `_localize_new_data()` is in the **exact same (1000×1000) grid space** as the value map:

```python
# Proof from value_map.py:_localize_new_data() (lines 316-317)
curr_map = np.zeros_like(self._map)  # ← Creates array with SAME shape as value map
curr_map = place_img_in_img(curr_map, curr_data, px, py)
return curr_map  # Returns (1000, 1000) aligned with value map
```

**What this means:**
- ✅ Both use same size: (1000, 1000)
- ✅ Both use same origin: `_episode_pixel_origin = [500, 500]`
- ✅ Both use same scale: `pixels_per_meter = 20`
- ✅ Index `[i, j]` refers to the **same world location** in both grids

**Direct correspondence:**
```python
# After projecting a point to grid coordinates (480, 550):
confidence_from_cone = confidence_mask[480, 550]      # Current FOV confidence
semantic_value      = value_map._value_map[480, 550, 0]  # Accumulated semantic score
prev_confidence     = value_map._map[480, 550]        # Accumulated confidence

# All three refer to the EXACT SAME world location!
# Think of them as aligned layers:
#   Layer 1: confidence_mask[i, j]       - Current observation confidence
#   Layer 2: value_map._map[i, j]        - Fused confidence (over time)
#   Layer 3: value_map._value_map[i, j, 0] - Fused semantic value (over time)
```

This perfect alignment is why you can directly use `grid_coords` to index into both the confidence mask and value map channels.

---

### Critical Clarifications

#### Q1: How do objects get their confidence scores?

**A:** Objects **don't have intrinsic confidence scores**. Instead, confidence is **spatial** - it depends on where the object's pixels fall in the camera's FOV:

- Pixels at **center of FOV** → confidence = 1.0 (most reliable)
- Pixels at **edge of FOV** → confidence = 0.0 (unreliable)
- Smooth falloff following: `confidence(θ) = cos²(θ/(θ_fov/2) * π/2)`

When we project an object to the value map, we look up the confidence at each pixel location based on the current camera FOV.

#### Q2: How do we project objects to the value map?

**A:** For each visible object:
1. Get its 3D point cloud (Nx3 array of world coordinates)
2. Project each 3D point to 2D value map grid coordinates
3. Look up FOV-based confidence at each 2D pixel location
4. Update both semantic value and confidence channels using VLFM's fusion formulas

---

## Part C: Pipeline Summary & Worked Example

### Pipeline Summary (5 Steps)

**Step 1: Segmentation & Feature Extraction**
- Use MobileSAM to segment objects → get masks
- For each object: Extract 3 crops → HOV-SG weighted fusion → fused features
- Project mask to 3D point cloud using depth

**Step 2: Object Association & Map Update**
- Compute semantic + geometric similarity with existing objects
- Greedy matching: Merge if match found, create new object if not

**Step 3: Score Visible Objects**
- Get ALL visible objects in current FOV (not just new/updated!)
- Score each: `object.score = cosine_similarity(object.features, target_embedding)`

**Step 4: Project to Value Map**
- Project each object's 3D point cloud → 2D grid coordinates
- Look up FOV-based confidence at each location
- Apply temporal fusion formulas (confidence-weighted averaging)

**Step 5: Frontier Selection (Reuse VLFM)**
- Extract frontiers → Score using semantic values → Select best

---

### Worked Example: Two Timesteps

Let's trace through a concrete example to see how temporal fusion works.

#### Timestep t=0: First observation

```
Robot Position: (0, 0)
Camera FOV: Points forward (0°)
Objects detected: Bed at (2.5m, 0m)

Step 1-4: Detection & Association
  - SAM detects bed
  - No existing objects → Create new bed object
  - bed.features = fused features (HOV-SG)
  - bed.point_cloud = 500 3D points

Step 5: Visible Objects
  - visible_objects = [bed]  (newly created)

Step 6: Score Objects
  - bed.current_score = cosine_sim(bed.features, target_features) = 0.85

Step 7: Create Confidence Mask
  - Bed is at edge of FOV
  - confidence_mask[bed_location] ≈ 0.3 (low confidence - edge view)

Step 8: Project to Value Map
  - For each of bed's 500 3D points:
      - Project to 2D grid: (x=2.5, y=0) → (grid_x=480, grid_y=550)
      - c_curr = 0.3 (from confidence mask)
      - v_curr = 0.85 (bed score)
      - c_prev = 0 (first observation)

      - UPDATE:
          value_map._value_map[480, 550, 0] = 0.85
          value_map._map[480, 550] = 0.3

Result:
  value_map[bed_location] = 0.85 (semantic)
  confidence_map[bed_location] = 0.3 (low confidence)
```

#### Timestep t=1: Robot rotates toward bed

```
Robot Position: (0, 0)
Camera FOV: Rotated 20° toward bed
Objects detected: Bed at (2.5m, 0m) - now at CENTER of FOV

Step 1-4: Detection & Association
  - SAM detects bed again
  - Object association: MATCHED with existing bed object
  - Fuse features: bed.features = (1*old_features + new_features) / 2

Step 5: Visible Objects
  - visible_objects = [bed]  (existing object, just updated)

Step 6: Score Objects
  - bed.current_score = 0.88 (slightly different from new view)

Step 7: Create Confidence Mask
  - Bed is NOW at CENTER of FOV
  - confidence_mask[bed_location] ≈ 1.0 (high confidence!)

Step 8: Project to Value Map
  - For each of bed's 500 3D points:
      - Project to 2D grid: (grid_x=480, grid_y=550) [same location]
      - c_curr = 1.0 (CENTER of FOV!)
      - v_curr = 0.88 (bed score)
      - c_prev = 0.3 (from previous timestep)
      - v_prev = 0.85 (from previous timestep)

      - TEMPORAL FUSION:
          v_new = (1.0 * 0.88 + 0.3 * 0.85) / (1.0 + 0.3)
                = (0.88 + 0.255) / 1.3
                = 0.873  ← Close to high-confidence score!

          c_new = (1.0² + 0.3²) / (1.0 + 0.3)
                = (1.0 + 0.09) / 1.3
                = 0.838  ← Confidence increases!

Result:
  value_map[bed_location] = 0.873 (semantic)
  confidence_map[bed_location] = 0.838 (high confidence)

Key Insight: The high-confidence center view (1.0) DOMINATED the
             low-confidence edge view (0.3) in the fusion!
```

---

## Part D: Comparison & Analysis

### 1. High-Level Comparison

| Aspect | VLFM | Our Approach |
|--------|------|--------------|
| **What's scored** | Entire RGB frame | Individual objects |
| **Scoring method** | BLIP2-ITM(full_image, text) | VLM(object_features, text) |
| **Result** | Single score → broadcast to entire FOV | Per-object scores → project to specific locations |
| **Representation** | Ephemeral (per-frame) | Persistent (3D object map) |

### 2. Deep Dive: The Coordinate System Challenge

**Why We Can't Use VLFM's "Rotation Method"**

*   **VLFM's Shortcut**: Creates a 2D cone image, fills it with a *single* score (e.g., 0.8), and rotates the entire image. This works because the score is uniform across the FOV—rotating a uniform blob preserves the information.
*   **Our Challenge (Granularity)**: We have multiple objects in the same FOV with *different* scores (e.g., Target=0.9, Obstacle=0.2). Rotating the image as a whole would lose the pixel-to-object correspondence.
*   **Our Solution (Unprojection)**: We must track points through the full 3D pipeline:
    1.  **Pixel (u,v)** → **Camera Frame** (Unproject using depth & intrinsics)
    2.  **Camera Frame** → **World Frame** (Transform using camera pose)
    3.  **World Frame** → **Grid** (Project to map)

This ensures that the score for a specific object lands exactly on that object's map cells, regardless of the camera's orientation.

### 3. Coordinate Systems Summary

| Frame | Coordinates | Origin | Purpose |
|-------|-------------|--------|---------|
| **Pixel** | $(u, v)$ | Top-left of image | Original SAM Detection |
| **Camera** | $(x_c, y_c, z_c)$ | Camera center | 3D shape relative to robot |
| **World** | $(x_w, y_w, z_w)$ | Map origin (0,0) | Persistent global location |
| **Grid** | $(row, col)$ | Map center (500,500) | Value Map storage |

---

## Section 6: Implementation Guide

### What You Need to Implement

#### 1. New Code to Write

**a) Depth-to-Point Cloud Projection**
```python
def depth_to_pointcloud(depth, mask, camera_pose, camera_intrinsics):
    """Convert masked depth pixels to 3D world coordinates."""
    points_3d_world = []
    for u, v in mask_pixels:
        # Camera frame
        depth_value = depth[u, v]
        x_cam = (u - cx) * depth_value / fx
        y_cam = (v - cy) * depth_value / fy
        z_cam = depth_value

        # World frame
        point_world = camera_pose @ [x_cam, y_cam, z_cam, 1]
        points_3d_world.append(point_world[:3])

    return np.array(points_3d_world)
```

**b) Object Scoring**
```python
def score_objects(visible_objects, target_features):
    """Score each object against target."""
    for obj in visible_objects:
        obj.score = (obj.features @ target_features.T).item()
```

**c) Value Map Update (The Only Custom Function)**
```python
def update_value_map_with_objects(value_map, visible_objects, depth, camera_pose, fov):
    """Project per-object scores to value map."""

    # ✅ REUSE: Get FOV confidence mask
    confidence_mask = value_map._localize_new_data(depth, camera_pose, 0.5, 10.0, fov)

    # ❌ NEW: Project each object's point cloud
    for obj in visible_objects:
        for (x_w, y_w, z_w) in obj.point_cloud:
            # World → Grid conversion (from base_map.py:_xy_to_px logic)
            grid_x = int(np.round(
                value_map._episode_pixel_origin[0] - y_w * value_map.pixels_per_meter
            ))
            grid_y = int(np.round(
                value_map._episode_pixel_origin[1] + x_w * value_map.pixels_per_meter
            ))

            if not (0 <= grid_x < value_map.size and 0 <= grid_y < value_map.size):
                continue

            c_curr = confidence_mask[grid_x, grid_y]
            if c_curr <= 0:
                continue

            v_curr = obj.score
            v_prev = value_map._value_map[grid_x, grid_y, 0]
            c_prev = value_map._map[grid_x, grid_y]

            # ✅ REUSE: Temporal fusion formulas
            if c_prev > 0:
                value_map._value_map[grid_x, grid_y, 0] = (
                    (c_curr * v_curr + c_prev * v_prev) / (c_curr + c_prev)
                )
                value_map._map[grid_x, grid_y] = (
                    (c_curr**2 + c_prev**2) / (c_curr + c_prev)
                )
            else:
                value_map._value_map[grid_x, grid_y, 0] = v_curr
                value_map._map[grid_x, grid_y] = c_curr
```

#### 2. VLFM Components to Reuse (No Changes)

| Component | File | What It Does |
|-----------|------|--------------|
| Confidence mask creation | `value_map._localize_new_data()` | FOV cone with cos² falloff |
| Frontier extraction | `obstacle_map.get_frontiers()` | Boundary detection |
| Frontier scoring | `value_map.sort_waypoints()` | Rank by semantic values |
| Frontier selection | `policy._get_best_frontier()` | Stickiness + acyclicity |

### Implementation Checklist

**Before Implementation:**
- [ ] Understand: Only replacing projection step, everything else reuses VLFM
- [ ] Verify: `value_map._localize_new_data()` returns (1000, 1000) confidence mask
- [ ] Review: Complete timestep workflow in Part B above

**Core Implementation:**
- [ ] Implement `depth_to_pointcloud()` with full 4x4 transformation
- [ ] Implement `update_value_map_with_objects()` with point cloud projection
- [ ] Score ALL visible objects (not just new/updated!)
- [ ] Call `value_map._localize_new_data()` once per timestep
- [ ] Apply VLFM's temporal fusion formulas unchanged

**Integration:**
- [ ] Replace `itm_policy.py:_update_value_map()` with your custom function
- [ ] Keep `value_map.sort_waypoints()` unchanged
- [ ] Keep `policy._get_best_frontier()` unchanged
- [ ] Keep frontier extraction unchanged

**Testing:**
- [ ] Objects at FOV center get higher confidence than edges
- [ ] Different objects get different semantic scores on value map
- [ ] Temporal fusion improves scores when revisiting from better viewpoints
- [ ] Frontiers scored using semantic values (verify with visualizations)

### Critical Implementation Notes

**1. Always score ALL visible objects:**
- Not just newly created/updated ones
- Enables temporal fusion from better viewpoints
- Example: Object at edge (c=0.3) → rotate to center (c=1.0) → fusion updates value map

**2. Confidence is spatial, not object-specific:**
- Determined by FOV location (center=1.0, edges=0.0)
- Look up from `confidence_mask[grid_x, grid_y]` after projection
- NOT based on object properties

**3. Coordinate transformation pipeline:**
- Depth pixel (u, v) → 3D camera frame → 3D world frame → 2D grid (grid_x, grid_y)
- Use full 4x4 transformation matrix (VLFM uses shortcut because uniform score)

### Files to Create/Modify

**New:**
- Object map (ConceptGraphs-style association)
- Object scoring module (ImageBind/SigLIP wrapper)
- `update_value_map_with_objects()` function

**Modify:**
- `itm_policy.py:_update_value_map()` → call your custom function

**Reuse (unchanged):**
- `vlfm/mapping/value_map.py`
- `vlfm/mapping/obstacle_map.py`
- `vlfm/policy/itm_policy.py:_get_best_frontier()`

---

**END OF IMPLEMENTATION GUIDE**



## Appendix B: ConceptGraphs 2D-to-3D Projection Pipeline

This section documents how ConceptGraphs projects 2D masked regions to 3D point clouds, based on analysis of the actual codebase.

### Overview

ConceptGraphs uses a 4-stage pipeline to convert 2D semantic masks into 3D point clouds suitable for object mapping:

```
2D Mask + Depth → Camera Frame PCD → Denoising (DBSCAN) → Map Frame PCD → Object
```

---

### Stage 1: Depth Pixel to 3D Point Conversion

**Function:** `create_object_pcd()`
**Location:** `concept-graphs/conceptgraph/slam/utils.py` (lines 61-108)

**Process:**
```python
def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None):
    """Convert masked depth pixels to 3D camera frame coordinates"""

    # Extract camera intrinsics from 3x3 matrix K
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    # fx, fy = focal lengths
    # cx, cy = principal point (optical center)

    # Step 1: Apply mask to depth and get pixel coordinates
    mask = np.logical_and(mask, depth_array > 0)  # Filter invalid depths
    masked_depth = depth_array[mask]
    u = u[mask]  # Pixel x-coordinates
    v = v[mask]  # Pixel y-coordinates

    # Step 2: Unproject to 3D using pinhole camera model
    x = (u - cx) * masked_depth / fx  # X in camera frame
    y = (v - cy) * masked_depth / fy  # Y in camera frame
    z = masked_depth                   # Z in camera frame (depth)

    # Step 3: Stack into (N, 3) point cloud
    points = np.stack((x, y, z), axis=-1)

    # Step 4: Add small noise to avoid numerical issues
    points += np.random.normal(0, 4e-3, points.shape)

    # Step 5: Create Open3D PointCloud with RGB colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Extract colors from RGB image at mask pixels
    colors = image[mask]  # (N, 3) RGB values
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd  # Point cloud in camera frame
```

**Camera Intrinsics Extraction:**
```python
# Location: concept-graphs/conceptgraph/dataset/datasets_common.py (lines 46-56)
def from_intrinsics_matrix(K: torch.Tensor) -> tuple[float, float, float, float]:
    """Extract focal lengths and principal point from 3x3 intrinsics matrix"""
    fx = K[0, 0]  # Focal length x (pixels)
    fy = K[1, 1]  # Focal length y (pixels)
    cx = K[0, 2]  # Principal point x (image center offset)
    cy = K[1, 2]  # Principal point y (image center offset)
    return fx, fy, cx, cy
```

**Key Formula (Inverse Camera Projection):**
```
Given:
  - Pixel coordinates: (u, v)
  - Depth value: z
  - Camera intrinsics: fx, fy, cx, cy

Convert to 3D camera frame:
  X = (u - cx) * z / fx
  Y = (v - cy) * z / fy
  Z = z
```

---

### Stage 2: DBSCAN Clustering & Denoising

**Function:** `pcd_denoise_dbscan()`
**Location:** `concept-graphs/conceptgraph/slam/utils.py` (lines 110-151)

**Purpose:** Remove noise points and keep only the largest coherent cluster.

**Algorithm:**
```python
def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10):
    """
    Apply DBSCAN clustering to remove outlier noise points.

    Parameters:
    - eps: 0.02m - Maximum distance between points in a cluster
    - min_points: 10 - Minimum points to form a dense region
    """

    # Step 1: Run DBSCAN clustering
    pcd_clusters = pcd.cluster_dbscan(eps=eps, min_points=min_points)
    # Returns: array of cluster labels (-1 for noise, 0+ for clusters)

    # Step 2: Count points per cluster
    pcd_clusters = np.array(pcd_clusters)
    counter = Counter(pcd_clusters)

    # Step 3: Remove noise label (-1)
    if -1 in counter:
        del counter[-1]

    # Step 4: Find largest cluster
    if counter:
        most_common_label, _ = counter.most_common(1)[0]
        largest_mask = pcd_clusters == most_common_label

        # Step 5: Extract points and colors from largest cluster
        obj_points = np.asarray(pcd.points)
        obj_colors = np.asarray(pcd.colors)

        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # Step 6: Create new PCD if cluster is large enough
        if len(largest_cluster_points) >= 5:
            largest_cluster_pcd = o3d.geometry.PointCloud()
            largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
            largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
            return largest_cluster_pcd

    # Fallback: return original if denoising fails
    return pcd
```

**Typical Configuration:**
- `eps = 0.02m` - Points within 2cm are neighbors
- `min_points = 10` - Need at least 10 points for a cluster
- Noise points (isolated points) are discarded
- Only the largest cluster is retained

---

### Stage 3: Voxel Downsampling

**Function:** `process_pcd()`
**Location:** `concept-graphs/conceptgraph/slam/utils.py` (lines 153-166)

**Purpose:** Reduce point cloud density while preserving shape.

```python
def process_pcd(pcd, cfg, run_dbscan=True):
    """
    Apply voxel downsampling and optional DBSCAN denoising.

    Typical voxel_size: 0.01m (1cm grid)
    """

    # Step 1: Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
    # Replaces all points in each voxel cell with their centroid

    # Step 2: Conditional DBSCAN denoising
    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd,
            eps=cfg.dbscan_eps,
            min_points=cfg.dbscan_min_points
        )

    return pcd
```

**Effect:**
- Input: Dense point cloud (e.g., 5000 points)
- Output: Sparse point cloud (e.g., 500 points)
- Benefit: Faster processing, reduced memory

---

### Stage 4: Transformation to Map Frame

**Function:** `transform_detection_list()` / Direct `.transform()`
**Location:** `concept-graphs/conceptgraph/slam/utils.py` (lines 574-600)

**Purpose:** Convert from camera coordinate system to global map coordinate system.

```python
# Within main pipeline (gobs_to_detection_list):

# Step 1: Create PCD in camera frame
camera_object_pcd = create_object_pcd(depth_array, mask, cam_K, image)

# Step 2: Transform to map/world frame
if trans_pose is not None:  # trans_pose is 4x4 transformation matrix
    global_object_pcd = camera_object_pcd.transform(trans_pose)
else:
    global_object_pcd = camera_object_pcd

# Step 3: Denoise and downsample
global_object_pcd = process_pcd(global_object_pcd, cfg)
```

**Transformation Matrix Structure:**
```
trans_pose = [R | t]  # 4x4 matrix
             [0 | 1]

Where:
  R = 3x3 rotation matrix (camera orientation)
  t = 3x1 translation vector (camera position)

For each point p_camera = [x, y, z, 1]:
  p_world = trans_pose @ p_camera
```

---

### Complete Pipeline Integration

**Function:** `gobs_to_detection_list()`
**Location:** `concept-graphs/conceptgraph/slam/utils.py` (lines 478-572)

### Complete Pipeline Integration

**Function:** `gobs_to_detection_list()`

This functionality will be replaced by our custom `ObjectDetector` class which orchestrates:
1. SAM Segmentation
2. Feature Extraction (SigLIP/ImageBind)
3. 3D Point Cloud Creation (Camera Frame)
4. Transformation to Map Frame
5. Point Cloud Downsampling & Denoising (DBSCAN)

---

### Summary Table

| Stage | Function | Usage |
|-------|----------|-------|
| 1 | `create_object_pcd()` | Depth unprojection to Camera Frame PCD |
| 2 | `transform` | Convert Camera Frame PCD to World/Map Frame |
| 3 | `pcd_denoise_dbscan()` | Remove outliers and noise |
| 4 | `voxel_down_sample()` | Reduce point density for efficiency |
| 5 | Object Creation | Combine PCD, features, and score into Object |

---

### Key Files Reference

1. **Main Pipeline:**
   - `concept-graphs/conceptgraph/slam/utils.py` (lines 478-600)

2. **Intrinsics Utilities:**
   - `concept-graphs/conceptgraph/dataset/datasets_common.py` (lines 46-56)

3. **Geometry Math:**
   - `concept-graphs/conceptgraph/utils/geometry.py`

---

This pipeline provides a robust method for converting 2D semantic segmentation masks into accurate 3D point clouds suitable for persistent object mapping and scene graph construction.

---

## Appendix C: Directory Structure and Architecture

### `vlfm/policy/` Directory
**Purpose:** Contains the navigation policies (the "brains" that decide where to move)

**Key files:**
- `base_policy.py` - Abstract base class for all policies
- `base_objectnav_policy.py` - Base class for ObjectNav policies
  - This is where VLM clients are instantiated (GroundingDINO, SAM, YOLOv7, BLIP2)
  - Handles object detection, mapping, and navigation orchestration
- `itm_policy.py` - Image-Text Matching policy (uses BLIP2ITM for semantic scoring)
- `habitat_policies.py` - Policies for Habitat simulator
- `reality_policies.py` - Policies for real robot deployment

**What they do:** These integrate all the components (VLMs, mapping, navigation) and make high-level decisions about where the robot should move next based on semantic information.

### `vlfm/reality/` Directory
**Purpose:** Real-world robot deployment code (Boston Dynamics Spot robot)

**Key files:**
- `objectnav_env.py` - ObjectNav environment wrapper for real Spot robot
- `pointnav_env.py` - PointNav environment wrapper for real robot
- `robots/` - Robot-specific drivers and hardware interfaces

**What they do:** Handle interfacing with real robots - camera streams, motor commands, sensor fusion, coordinate transformations from robot odometry to global frames.

### Implementation Plan for Object-Centric Policy

**Do NOT create new directories.** Instead:

1. **Create a new policy file** in the existing `vlfm/policy/` directory:
   - Name: `object_centric_policy.py` or `hovsig_policy.py`
   - Inherit from `BaseObjectNavPolicy` (following the pattern from `itm_policy.py`)
   - Replace BLIP2ITM scoring with SigLIP + HOV-SG fusion scoring

2. **Reuse existing infrastructure:**
   - Use existing `reality/` directory as-is (for future real robot deployment)
   - Use existing mapping classes: `ValueMap`, `ObjectPointCloudMap`, `ObstacleMap`, `FrontierMap`
   - Use existing `BaseObjectNavPolicy` client instantiation pattern

3. **Client instantiation pattern** (from `base_objectnav_policy.py`):
   ```python
   self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
   self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
   self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
   ```

4. **For the new object-centric policy:**
   ```python
   # vlfm/policy/object_centric_policy.py

   from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
   from vlfm.object_centric.object_detection import ObjectDetector
   from vlfm.object_centric.sam_detector import MobileSAMClient
   from vlfm.object_centric.siglip2 import SigLIPClient

   class ObjectCentricPolicy(BaseObjectNavPolicy):
       def __init__(self, text_prompt: str, *args, **kwargs):
           super().__init__(*args, **kwargs)

           # Instantiate clients (connect to running model servers)
           sam_client = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
           siglip_client = SigLIPClient(port=int(os.environ.get("SIGLIP2_PORT", "12185")))

           # Create Stage 1 detector
           self.object_detector = ObjectDetector(
               sam_detector=sam_client,
               siglip=siglip_client,
               camera_intrinsics=...,  # Extracted from observations
               min_points=50
           )

           self._text_prompt = text_prompt
           # Stages 2-4 will be implemented in this policy class
   ```

**Key insight:** The policy class orchestrates all 4 stages, using `ObjectDetector` for Stage 1.

---

## Appendix D: Understanding Coordinate Frames

### Why Camera Frame and World Frame?

**Camera Frame**: Coordinates relative to the camera
- Example: "Chair is 2m forward, 0.5m right from camera"
- Problem: Changes when the robot moves or turns

**World Frame**: Fixed global coordinates in the room
- Example: "Chair is at position (5.2, 3.1, 0.5) in the room"
- Advantage: Never changes regardless of robot position

### Example: Same Chair, Different Frames

```
Timestep 1 (robot facing north):
  Camera frame: chair at (2.0, 0.5, 0.0)
  World frame:  chair at (5.2, 3.1, 0.5)

Timestep 2 (robot turned 90° east):
  Camera frame: chair at (0.5, -2.0, 0.0)  ← Different!
  World frame:  chair at (5.2, 3.1, 0.5)   ← Same!
```

Without world frame → map thinks there are 2 chairs
With world frame → correctly recognizes it's the same chair

This is why we transform from camera → world using the `camera_pose` matrix.

---

## Appendix E: VLFM Directory Analysis

### What We Use vs. Skip vs. Modify

| Directory | Purpose | Usage Plan | Notes |
|-----------|---------|------------|-------|
| **mapping/** | Value map, obstacle map, frontier detection | **USE AS-IS** | Core infrastructure. We use `ValueMap` to score frontiers and `ObstacleMap` to extract frontiers. |
| **policy/** | Navigation policies | **MODIFY** | We will create a new `ObjectCentricPolicy` here that inherits from `BaseObjectNavPolicy`. |
| **vlm/** | Vision-language models | **KEEP RELEVANT** | We retain `grounding_dino.py`, `yolo_world.py`, and `blip2.py` as they are used by the base infrastructure for initialization and termination conditions. We REPLACE the scoring mechanism (BLIP2-ITM) with our new object-centric scoring. |
| **utils/** | Geometry & utilities | **USE AS-IS** | Robust geometry utilities. |
| **obs_transformers/** | Preprocessing | **USE AS-IS** | Handles RGB-D resizing/normalization. |

### How GroundingDINO is Used (We Keep This)

In the original VLFM and our implementation, GroundingDINO is used by `BaseObjectNavPolicy` to:
1.  **Initialize the detector**: `_get_object_detections()` uses GroundingDINO to check if the target object is visible (for stopping logic).
2.  **Termination Condition**: The policy stops when the target object is confidently detected and the robot is close enough.

**Crucial Distinction**:
*   **VLFM**: Uses GroundingDINO for target detection (stopping) AND BLIP2-ITM for frontier scoring (exploration).
*   **Our Approach**: We **KEEP** GroundingDINO for target detection (stopping) but **REPLACE** BLIP2-ITM with our Object-Centric Scoring (SigLIP/ImageBind) for frontier scoring (exploration).

Therefore, we do NOT empty the `vlm/` folder. We keep the files required by `BaseObjectNavPolicy` but implement our own scoring module in `vlfm/object_centric/`.

### Directory Structure Plan

```
vlfm/
├── policy/
│   ├── base_objectnav_policy.py  # Keep (we inherit from this)
│   ├── itm_policy.py             # Reference (we replace this)
│   └── object_centric_policy.py  # [NEW] Our policy
├── object_centric/               # [NEW] Our modules
│   ├── object_detection.py       # Stage 1: Detection pipeline
│   ├── object_map.py             # Stage 2: Association & Mapping
│   ├── siglip2.py                # VLM Client
│   └── sam_detector.py           # SAM Client
├── mapping/                      # Use as-is
├── vlm/                          # Keep GroundingDINO/YOLO/BLIP2
└── utils/                        # Use as-is
```

