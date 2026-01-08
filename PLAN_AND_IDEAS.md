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
2. **Per-Timestep Scoring**: For each object in the map:
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
4. Higher scores indicate better match between frontier view and target object
5. VLFM uses these scores to guide exploration toward the goal

---

## Section 3: Implementation Options for Proposed Methodology

### Option 1: ImageBind (Recommended)

#### Why ImageBind?

**ImageBind** is Meta's multimodal alignment model that:
- Uses **DINOv2** as its vision backbone (similar quality to DINOv3)
- Has pre-trained **vision-text alignment** (no training needed)
- Supports multi-view feature fusion workflow
- Provides separate `encode_vision()` and `encode_text()` methods

#### Key Advantages

1. **Meta Ecosystem**: Same research group as DINOv3/SAM
2. **Strong Visual Features**: DINOv2-based encoder provides rich object representations
3. **Pre-aligned Embeddings**: Vision and text features already in same space
4. **Flexible API**: Can extract features separately for fusion

#### Implementation Workflow

```python
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# Load model (one-time setup)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# 1. Extract features from 3 crops per object
# Note: load_and_transform_vision_data expects file paths or PIL images
inputs_full = {
    ModalityType.VISION: data.load_and_transform_vision_data([full_image], device)
}
inputs_crop_bg = {
    ModalityType.VISION: data.load_and_transform_vision_data([crop_with_bg], device)
}
inputs_crop_no_bg = {
    ModalityType.VISION: data.load_and_transform_vision_data([crop_no_bg], device)
}

with torch.no_grad():
    feat_full = model(inputs_full)[ModalityType.VISION]          # Shape: [1, D]
    feat_crop_bg = model(inputs_crop_bg)[ModalityType.VISION]     # Shape: [1, D]
    feat_crop_no_bg = model(inputs_crop_no_bg)[ModalityType.VISION]  # Shape: [1, D]

# 2. Weighted fusion (HOV-SG style)
weights = [w1, w2, w3]  # e.g., [0.3, 0.3, 0.4]
object_features = (weights[0] * feat_full +
                   weights[1] * feat_crop_bg +
                   weights[2] * feat_crop_no_bg)

# 3. Normalize fused features
object_features = object_features / object_features.norm(dim=-1, keepdim=True)

# 4. Encode target once at episode start
text_input = {
    ModalityType.TEXT: data.load_and_transform_text(["bed"], device)
}
with torch.no_grad():
    target_features = model(text_input)[ModalityType.TEXT]  # Shape: [1, D]

# 5. Score objects each timestep
score = (object_features @ target_features.T).item()
```

#### Wrapper Class Design (Optional)

For easier integration with VLFM's server-client pattern, you can wrap ImageBind:

```python
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy as np
from typing import Optional, Any

class ImageBindWrapper:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

    def encode_vision(self, image: np.ndarray) -> torch.Tensor:
        """Extract normalized image features"""
        # Convert numpy array to format expected by ImageBind
        # Note: May need to save temporarily or convert to PIL
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([image], self.device)
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.VISION]  # Shape: [1, D]

    def encode_text(self, text: str) -> torch.Tensor:
        """Extract normalized text features"""
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], self.device)
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings[ModalityType.TEXT]  # Shape: [1, D]

    def cosine(self, image: np.ndarray, text: str) -> float:
        """Convenience method for single-image scoring"""
        img_feat = self.encode_vision(image)
        txt_feat = self.encode_text(text)
        return (img_feat @ txt_feat.T).item()
```

**Note**: ImageBind's `load_and_transform_vision_data()` expects file paths or PIL Images. You may need to convert numpy arrays or save temporarily.

#### Integration with VLFM

1. **Replace BLIP2ITMClient** with ImageBindClient
2. **Modify object detection loop** to extract 3 crops per object
3. **Add fusion step** before storing object features
4. **Keep everything else unchanged**: value map projection, confidence weighting, frontier selection

#### Why Not Just Use CLIP?

While CLIP also has vision-text alignment:
- ImageBind's vision encoder (DINOv2-based) has better object-centric features
- ImageBind is trained on more diverse data modalities
- Stays within Meta's ecosystem (DINOv2, SAM, ImageBind)

### Option 2: SigLIP 2 (Latest from Google - February 2025)

#### Overview

Google's newest vision-language encoder, released February 2025. SigLIP 2 extends the original SigLIP with improved semantic understanding, localization, and dense features.

#### Key Advantages

1. **State-of-the-art Performance**: Outperforms SigLIP and CLIP at all model scales
2. **Recent Release**: Literally cutting-edge (Feb 2025)
3. **Better Capabilities**: Improved zero-shot classification, image-text retrieval, and VLM transfer
4. **Multilingual Support**: Enhanced multilingual vision-language understanding
5. **Easy Integration**: Available on HuggingFace with simple API

#### Implementation Workflow

```python
# Similar to ImageBind workflow
from transformers import AutoModel, AutoProcessor

# Load SigLIP 2
processor = AutoProcessor.from_pretrained("google/siglip2-...")
model = AutoModel.from_pretrained("google/siglip2-...")

# 1. Extract features from 3 crops per object
feat_full = model.get_image_features(processor(full_image))
feat_crop_bg = model.get_image_features(processor(crop_with_bg))
feat_crop_no_bg = model.get_image_features(processor(crop_no_bg))

# 2. Weighted fusion
object_features = (w1*feat_full + w2*feat_crop_bg + w3*feat_crop_no_bg)
object_features = object_features / object_features.norm(dim=-1, keepdim=True)

# 3. Encode target
target_features = model.get_text_features(processor(text="bed"))

# 4. Score
score = (object_features @ target_features.T).item()
```

#### Pros
- Latest model (Feb 2025), most recent research
- Better than CLIP and original SigLIP
- Production-ready, well-supported
- HuggingFace integration

#### Cons
- Very new (less community testing than CLIP)
- Not DINOv2/v3 based (standard vision encoder)

#### When to Use
- If you want the latest and best vision-language model
- If you prefer proven Google models over Meta
- For maximum performance without DINOv2 requirement

#### References
- [SigLIP 2 ArXiv Paper](https://arxiv.org/pdf/2502.14786)
- [HuggingFace Blog](https://huggingface.co/blog/siglip2)

### Option 3: CLIP (Baseline)

#### Overview

The original vision-language model from OpenAI. Well-established baseline, but superseded by newer models.

#### Pros
- Most well-known and tested
- Extensive documentation and community support
- Many pre-trained variants (OpenCLIP)
- Lightest weight option

#### Cons
- Older (2021)
- Outperformed by SigLIP 2, ImageBind
- Vision features not as strong as DINOv2-based models

#### When to Use
- As a baseline for comparison
- If you need maximum stability and community support
- For quick prototyping before switching to better models

### Option 4: DINOv3 + DINOv3-CLIP Adapter

#### Overview

Use actual DINOv3 for vision features with a lightweight MLP adapter (3.15M parameters) that maps DINOv3 embeddings into CLIP's image space, enabling text alignment.

#### How It Works

1. **Frozen DINOv3**: Extract rich visual features
2. **Lightweight Adapter**: Maps DINOv3 → CLIP image space (pre-trained, 3.15M params)
3. **CLIP Text Encoder**: Standard CLIP for text features
4. **Compute Similarity**: Cosine similarity in CLIP's aligned space

#### Implementation Workflow

```python
# Using dinov3clip package
from dinov3clip import load_model

# Load DINOv3 + adapter
dinov3_model = load_dinov3()
adapter = load_adapter("path/to/checkpoint.pt")
clip_text = load_clip_text_encoder()

# 1. Extract DINOv3 features from 3 crops
feat_full = dinov3_model(full_image)
feat_crop_bg = dinov3_model(crop_with_bg)
feat_crop_no_bg = dinov3_model(crop_no_bg)

# 2. Map to CLIP space via adapter
feat_full_clip = adapter(feat_full)
feat_crop_bg_clip = adapter(feat_crop_bg)
feat_crop_no_bg_clip = adapter(feat_crop_no_bg)

# 3. Weighted fusion
object_features = (w1*feat_full_clip + w2*feat_crop_bg_clip + w3*feat_crop_no_bg_clip)
object_features = object_features / object_features.norm(dim=-1, keepdim=True)

# 4. Encode target with CLIP text encoder
target_features = clip_text("bed")

# 5. Score
score = (object_features @ target_features.T).item()
```

#### Pros
- **Actual DINOv3**: Get the DINOv3 features you originally wanted
- **Pre-trained adapter**: No training required
- **Lightweight**: Only 3.15M adapter parameters
- **Proven approach**: Published work with available code

#### Cons
- **Performance gap**: Not as good as end-to-end trained models (SigLIP 2, ImageBind)
- **Domain sensitivity**: Works best on in-domain images (but indoor robotics likely is in-domain)
- **Open vocab limitations**: Out-of-domain labels can get inflated scores
- **Two-stage**: DINOv3 → adapter → CLIP space (more complexity)

#### Important Caveat

⚠️ The adapter was trained on images only (never saw text). Rankings are usually sensible for in-domain images, but out-of-domain text labels may score unexpectedly high.

**For your use case**: Indoor navigation with specific objects ("bed", "chair") should be in-domain, so rankings should be reliable.

#### When to Use
- If you specifically need DINOv3 features (not DINOv2)
- If DINOv3's visual quality is critical for your research
- For experimental comparison vs other models

#### References
- [DINOv3-CLIP GitHub](https://github.com/duriantaco/dinov3clip)

### Recommendation Summary

| Option | Difficulty | Performance | Speed | Vision Backbone | Best For |
|--------|-----------|-------------|-------|----------------|----------|
| **ImageBind** | Easy | High | Fast | DINOv2 | **Recommended: Best balance** |
| **SigLIP 2** | Easy | Very High | Fast | Standard ViT | Latest SOTA, Google ecosystem |
| **CLIP** | Very Easy | Medium | Very Fast | Standard ViT | Baseline, prototyping |
| **DINOv3-CLIP** | Medium | Medium-High* | Fast | DINOv3 | If you must have DINOv3 |

*Performance depends on domain match

### Our Recommendation

**Start with ImageBind (Option 1)** for these reasons:
1. Best balance of performance, ease of implementation, and alignment with goals
2. DINOv2-quality features (very close to DINOv3)
3. No training required, pre-aligned embeddings
4. Clear path to implementation
5. Meta ecosystem (pairs well with SAM, DINOv2)

**Consider SigLIP 2 (Option 2)** if:
- You want the absolute latest model (Feb 2025)
- You prefer Google's ecosystem
- DINOv2-specific features aren't critical

**Fall back to CLIP (Option 3)** only for:
- Quick baseline comparisons
- Maximum stability requirements

**Try DINOv3-CLIP (Option 4)** only if:
- You specifically need DINOv3 (not DINOv2)
- You're willing to accept potential performance gaps
- You want to experiment with actual DINOv3

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

---

### Visual Comparison: VLFM vs Our Approach

```
┌─────────────────────────────────────────────────────────────┐
│                      VLFM (Original)                        │
├─────────────────────────────────────────────────────────────┤
│ RGB Frame → BLIP2-ITM(full_image, "bed") → Score: 0.8      │
│      ↓                                                       │
│ Project to FOV cone on value map                            │
│      ↓                                                       │
│ Value Map: [Entire FOV cone = 0.8]                         │
│      ↓                                                       │
│ Extract frontiers → Score using value map → Pick best      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Our Approach (Object-Centric)             │
├─────────────────────────────────────────────────────────────┤
│ RGB Frame → SAM segment → For each object:                 │
│   ├─ Extract 3 crops → Weighted fusion → Object features   │
│   └─ VLM(object_features, "bed") → Object scores           │
│                 ↓                                            │
│ Build/Update persistent 3D object map                      │
│                 ↓                                            │
│ Project object point clouds + scores to value map          │
│                 ↓                                            │
│ Value Map: [Only at object locations with their scores]    │
│                 ↓                                            │
│ Extract frontiers → Score using value map → Pick best      │
│                     (Same as VLFM)                          │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Aspect | VLFM (Original) | Our Approach |
|--------|-----------------|---------------|
| **What's scored** | Full RGB frames (single view) | Individual objects (3-crop fusion) |
| **Score computation** | `BLIP2-ITM(image, text)` | `VLM(fused_object_features, text)` |
| **Scores per frame** | 1 (entire view) | N (one per detected object) |
| **Persistence** | None (ephemeral per frame) | Yes (persistent 3D object map) |
| **Value map projection** | Cone-shaped FOV region | Object point cloud locations |
| **Projection area** | Large (entire visible cone) | Sparse (only object locations) |
| **Confidence mechanism** | FOV-based (cos² falloff) | FOV-based (reuse VLFM's cos² falloff) |
| **Semantic value fusion** | ✅ v = (c·v_curr + c_prev·v_prev)/(c + c_prev) | ✅ Same formula |
| **Confidence fusion** | ✅ c = (c_curr² + c_prev²)/(c_curr + c_prev) | ✅ Same formula |
| **Frontier extraction** | ✅ Same | ✅ Same (reuse VLFM) |
| **Frontier scoring** | ✅ Same | ✅ Same (reuse VLFM) |
| **Frontier selection** | ✅ Same | ✅ Same (reuse VLFM) |

### Key Insights

**VLFM provides the complete infrastructure:**
- Value map with two-channel structure (semantic values + confidence)
- Confidence-weighted temporal fusion formulas
- Frontier extraction, scoring, and selection mechanisms
- All of this works perfectly for object-centric scoring

**Our contribution is the scoring mechanism:**
- VLFM: Single score per view → broadcast across FOV cone
- Us: Per-object scores → project to specific point cloud locations
- Everything downstream (frontiers, navigation) stays identical

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

**Why VLFM Can Use This Shortcut:**
- Single uniform score across entire FOV
- Creates filled cone contour (not per-pixel projection)
- Rotates and overlays the mask as a whole

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

Before diving into implementation, let's resolve common points of confusion:

#### Q1: How do objects get their confidence scores?

**A:** Objects **don't have intrinsic confidence scores**. Instead, confidence is **spatial** - it depends on where the object's pixels fall in the camera's FOV:

- Pixels at **center of FOV** → confidence = 1.0 (most reliable)
- Pixels at **edge of FOV** → confidence = 0.0 (unreliable)
- Smooth falloff following: `confidence(θ) = cos²(θ/(θ_fov/2) * π/2)`

When we project an object to the value map, we look up the confidence at each pixel location based on the current camera FOV.

#### Q2: Do we score only newly created/updated objects?

**A:** **NO!** We score **ALL visible objects** in the current FOV, not just newly created/updated ones.

**Why this is critical:**
- Object association tracks which objects were matched/created this timestep
- But for value map projection, we need **all objects currently in FOV**
- This allows temporal fusion to work correctly - better views update previous observations

**Example showing why:**
```
Timestep t-1: Bed detected at edge of FOV → low confidence (0.3)
Timestep t: Robot rotates, bed now at center of FOV
  - Object association: MATCHED with existing bed (not "newly created")
  - But we MUST re-project bed with NEW high confidence (1.0)!
  - Temporal fusion formula ensures center view dominates
  - Result: value_map updates from 0.85 → 0.87 (improved by better view)
```

If we only projected newly created/updated objects, we'd miss the opportunity to improve the value map with better viewpoints!

#### Q3: How do we project objects to the value map?

**A:** For each visible object:
1. Get its 3D point cloud (Nx3 array of world coordinates)
2. Project each 3D point to 2D value map grid coordinates
3. Look up FOV-based confidence at each 2D pixel location
4. Update both semantic value and confidence channels using VLFM's fusion formulas

---

### Complete Timestep Workflow

```python
# ============================================================================
# TIMESTEP t: Complete Pipeline
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────
# STEP 1: Get Observations
# ──────────────────────────────────────────────────────────────────────────
rgb, depth, camera_pose = get_observation()
# rgb: (H, W, 3) RGB image
# depth: (H, W) depth image
# camera_pose: 4x4 transformation matrix (camera → world)

# ──────────────────────────────────────────────────────────────────────────
# STEP 2: Segment Objects
# ──────────────────────────────────────────────────────────────────────────
masks = SAM.segment(rgb)  # List of binary masks, one per detected object

# ──────────────────────────────────────────────────────────────────────────
# STEP 3: Extract Features & Create Detections
# ──────────────────────────────────────────────────────────────────────────
detections = []

# Encode full image ONCE (global feature)
F_g = VLM.encode_vision(rgb)  # Shape: (1, D)

for mask in masks:
    # 3a. Extract 3 crops (HOV-SG style)
    crop_unmasked = crop_bbox(rgb, mask, block_background=False)
    crop_masked = crop_bbox(rgb, mask, block_background=True)

    F_l_unmasked = VLM.encode_vision(crop_unmasked)  # Shape: (1, D)
    F_l_masked = VLM.encode_vision(crop_masked)      # Shape: (1, D)

    # 3b. Stage 1 fusion: Combine masked/unmasked local features
    F_l = 0.4418 * F_l_masked + 0.5582 * F_l_unmasked
    F_l = F_l / F_l.norm(dim=-1, keepdim=True)  # L2 normalize

    # 3c. Stage 2 fusion: Combine with global feature
    similarity = torch.nn.functional.cosine_similarity(F_l, F_g, dim=-1)
    w = torch.nn.functional.softmax(similarity, dim=0).reshape(-1, 1)
    fused_features = w * F_g + (1 - w) * F_l
    fused_features = fused_features / fused_features.norm(dim=-1, keepdim=True)

    # 3d. Project mask to 3D point cloud
    point_cloud_3d = depth_to_pointcloud(depth, mask, camera_pose)

    detections.append({
        'features': fused_features,     # Shape: (1, D)
        'point_cloud': point_cloud_3d,  # Shape: (N, 3) - N 3D points
        'mask': mask                     # Shape: (H, W) - binary mask
    })

# ──────────────────────────────────────────────────────────────────────────
# STEP 4: Object Association (ConceptGraphs approach)
# ──────────────────────────────────────────────────────────────────────────
newly_created_objects = []
updated_objects = []

for detection in detections:
    # 4a. Find existing objects with geometric overlap
    overlapping_objects = find_overlapping_objects(
        detection.point_cloud,
        object_map
    )

    if len(overlapping_objects) == 0:
        # No overlap → Create new object
        new_obj = Object(
            features=detection.features,
            point_cloud=detection.point_cloud,
            num_observations=1
        )
        object_map.add(new_obj)
        newly_created_objects.append(new_obj)
    else:
        # 4b. Compute similarity with all overlapping objects
        similarities = []
        for obj in overlapping_objects:
            # Geometric similarity (point cloud overlap)
            phi_geo = nnratio(detection.point_cloud, obj.point_cloud)

            # Semantic similarity (feature cosine distance)
            phi_sem = (detection.features @ obj.features.T / 2 + 0.5).item()

            # Overall similarity
            phi_total = phi_sem + phi_geo
            similarities.append((obj, phi_total))

        # 4c. Greedy assignment: pick best match
        best_match, best_sim = max(similarities, key=lambda x: x[1])

        if best_sim > delta_sim:  # Threshold (e.g., 1.0)
            # 4d. Fuse detection with existing object
            n = best_match.num_observations

            # Update features (running average)
            best_match.features = (n * best_match.features + detection.features) / (n + 1)

            # Update point cloud (merge + downsample)
            best_match.point_cloud = merge_pointclouds(
                best_match.point_cloud,
                detection.point_cloud
            )

            best_match.num_observations += 1
            updated_objects.append(best_match)
        else:
            # No good match → Create new object
            new_obj = Object(
                features=detection.features,
                point_cloud=detection.point_cloud,
                num_observations=1
            )
            object_map.add(new_obj)
            newly_created_objects.append(new_obj)

# ──────────────────────────────────────────────────────────────────────────
# STEP 5: Get ALL Visible Objects (Not Just New/Updated!)
# ──────────────────────────────────────────────────────────────────────────
# CRITICAL: We need ALL objects in current FOV, not just newly created/updated
visible_objects = get_visible_objects_in_fov(
    object_map,
    camera_pose,
    fov=79 * np.pi / 180,  # Field of view in radians
    max_depth=10.0          # Maximum visible distance (meters)
)

# This includes:
# - Newly created objects (from this timestep)
# - Updated objects (matched and fused this timestep)
# - Existing objects that are still in FOV but weren't detected this timestep

# ──────────────────────────────────────────────────────────────────────────
# STEP 6: Score ALL Visible Objects
# ──────────────────────────────────────────────────────────────────────────
# Encode target text ONCE at episode start (cache for all timesteps)
if not hasattr(self, '_target_features'):
    self._target_features = VLM.encode_text("bed")  # Shape: (1, D)

for obj in visible_objects:
    # Compute cosine similarity between object features and target
    obj.current_score = (obj.features @ self._target_features.T).item()
    # Result: scalar in range [-1, 1], typically [0, 1]

# ──────────────────────────────────────────────────────────────────────────
# STEP 7: Create FOV-Based Confidence Mask (REUSE VLFM!)
# ──────────────────────────────────────────────────────────────────────────
confidence_mask_2d = value_map._localize_new_data(
    depth=depth,
    tf_camera_to_episodic=camera_pose,
    min_depth=0.5,
    max_depth=10.0,
    fov=79 * np.pi / 180
)
# Returns: (1000, 1000) array with confidence values [0, 1]
# - 1.0 at center of FOV (optical axis)
# - 0.0 at edges/outside FOV
# - Smooth cos² falloff in between
# - Already rotated to camera yaw and overlaid at camera position!

# ──────────────────────────────────────────────────────────────────────────
# STEP 8: Project ALL Visible Objects to Value Map
# ──────────────────────────────────────────────────────────────────────────
for obj in visible_objects:
    # 8a. Project 3D point cloud to 2D grid coordinates
    points_2d = []
    for point_3d in obj.point_cloud:  # point_3d: (x_world, y_world, z_world)
        x_world, y_world, z_world = point_3d

        # Convert world coordinates to grid pixel coordinates
        # Top-down projection (ignore z, project x,y to grid)
        grid_x = int(np.round(
            episode_pixel_origin[0] - y_world * pixels_per_meter
        ))
        grid_y = int(np.round(
            episode_pixel_origin[1] + x_world * pixels_per_meter
        ))

        # Check bounds
        if 0 <= grid_x < value_map_size and 0 <= grid_y < value_map_size:
            points_2d.append((grid_x, grid_y))

    # 8b. Update value map at each pixel the object occupies
    for (grid_x, grid_y) in points_2d:
        # Look up FOV-based confidence at this grid location
        c_curr = confidence_mask_2d[grid_x, grid_y]

        if c_curr <= 0:  # No confidence (outside FOV or occluded)
            continue

        # Get object's semantic score
        v_curr = obj.current_score

        # Get previous values at this location
        v_prev = value_map._value_map[grid_x, grid_y, 0]  # Semantic channel
        c_prev = value_map._map[grid_x, grid_y]           # Confidence channel

        # Apply temporal fusion (VLFM formulas)
        if c_prev > 0:  # Location was observed before
            # Fuse semantic values (confidence-weighted average)
            value_map._value_map[grid_x, grid_y, 0] = (
                (c_curr * v_curr + c_prev * v_prev) / (c_curr + c_prev)
            )

            # Fuse confidence scores (biased toward higher confidence)
            value_map._map[grid_x, grid_y] = (
                (c_curr**2 + c_prev**2) / (c_curr + c_prev)
            )
        else:  # First time observing this location
            value_map._value_map[grid_x, grid_y, 0] = v_curr
            value_map._map[grid_x, grid_y] = c_curr

# ──────────────────────────────────────────────────────────────────────────
# STEP 9: Extract Frontiers (Reuse VLFM)
# ──────────────────────────────────────────────────────────────────────────
frontiers = obstacle_map.get_frontiers()
# Returns: Nx2 array of (x, y) frontier locations in meters
# Frontiers = boundary between explored and unexplored areas

# ──────────────────────────────────────────────────────────────────────────
# STEP 10: Score Frontiers Using Value Map (Reuse VLFM)
# ──────────────────────────────────────────────────────────────────────────
sorted_frontiers, sorted_scores = value_map.sort_waypoints(
    frontiers,
    radius=0.5  # Aggregate values within 0.5m radius
)
# This looks up value_map._value_map (SEMANTIC channel) at frontier locations
# Returns frontiers sorted by descending semantic value

# ──────────────────────────────────────────────────────────────────────────
# STEP 11: Select Best Frontier (Reuse VLFM)
# ──────────────────────────────────────────────────────────────────────────
best_frontier = policy.get_best_frontier(
    sorted_frontiers,
    sorted_scores,
    robot_xy=current_robot_position
)
# Applies:
# - Stickiness (prefer previous frontier if score similar)
# - Acyclicity (avoid oscillating between frontiers)
# Returns: (x, y) coordinates of selected frontier

# ──────────────────────────────────────────────────────────────────────────
# STEP 12: Navigate to Frontier
# ──────────────────────────────────────────────────────────────────────────
navigate_to(best_frontier)
```


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

## Pipeline Summary (High-Level Overview)

### Complete Pipeline (5 Steps)

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

## Section 6: Implementation Guide

### Core Difference: VLFM vs Our Approach

| Aspect | VLFM | Our Approach |
|--------|------|--------------|
| **What's scored** | Entire RGB frame | Individual objects |
| **Scoring method** | BLIP2-ITM(full_image, text) | VLM(object_features, text) |
| **Result** | Single score → broadcast to entire FOV | Per-object scores → project to specific locations |
| **Representation** | Ephemeral (per-frame) | Persistent (3D object map) |

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

---

## Appendix A: Common Confusion Points - Q&A

### Q: "Why does VLFM do rotation/overlay differently than our approach? What's going on with the coordinates?"

**A: The Simple Explanation**

**VLFM has ONE score (e.g., 0.8) that applies to EVERYTHING in the FOV.**

**You have DIFFERENT scores for DIFFERENT objects (bed=0.9, chair=0.2).**

This fundamental difference means they can take shortcuts you can't.

---

#### VLFM's Approach (Shortcut Works Because Uniform Score)

```
Step 1: Create a cone-shaped image (like a cone of vision)
  - This cone is just a 2D image/mask
  - It has confidence values (center=1.0, edges=0.0)
  - It's like drawing a cone on paper

Step 2: Rotate this ENTIRE cone image to match camera angle
  - The whole cone rotates as one piece
  - Like rotating a photo in Photoshop

Step 3: Stamp/overlay this rotated cone onto the global map
  - Place it at the camera's position
  - Like placing a sticker on a poster

Step 4: Fill the ENTIRE cone with score 0.8
  - Every pixel inside the cone gets the same value
```

**Why this works for VLFM:** They only have ONE score, so they can treat the whole FOV as a single blob and move it around like a sticker.

---

#### Your Approach (Must Track Individual Pixels)

You **cannot** do the rotation trick because:
- Bed pixels need score 0.9
- Chair pixels need score 0.2
- They're in DIFFERENT locations within the same FOV!

So you must do this:

```
Step 1: For EACH pixel in the bed's mask:
  - Take pixel (100, 200) with depth=3.5m
  - Convert to 3D: "This pixel is 3.5m forward, 0.2m left in camera space"
  - Transform to world: "In the world, this point is at (5.2, 3.1, 0.5)"
  - Convert to grid: "On my top-down map, this is at grid position (480, 550)"
  - Look up confidence at grid (480, 550) from VLFM's cone mask
  - Assign bed's score 0.9 to grid (480, 550)

Step 2: For EACH pixel in the chair's mask:
  - Same process but assign chair's score 0.2
```

---

#### Why VLFM Can Rotate But You Can't

**VLFM's situation:**
```
Cone image before rotation:
    [0.5]
  [0.8 1.0 0.8]
    [0.5]

Rotate entire image by 45°:
    [0.5]
  [0.8 1.0 0.8]  ← Whole thing rotates together
    [0.5]

Every pixel gets filled with: 0.8 (same score everywhere)
```

**Your situation:**
```
FOV contains:
  - Bed pixels → need score 0.9
  - Chair pixels → need score 0.2

If you rotate the whole FOV as an image:
  - Which pixels should get 0.9?
  - Which should get 0.2?
  - You CAN'T tell anymore after rotation!
```

**The problem:** After rotating an image, you lose track of which pixel belonged to which object. So you need to:
1. First convert each object's pixels to 3D world coordinates (where rotation is already baked into the transformation)
2. Then project those specific world coordinates to the grid
3. Then assign that object's specific score

---

#### The Confidence Mask Part (This Is What's Confusing)

**You DO still use VLFM's rotated confidence cone!** Here's how:

```python
# VLFM creates and rotates the confidence cone for you
confidence_mask = value_map._localize_new_data(depth, camera_pose, ...)
# This gives you a (1000, 1000) grid where:
# - 1.0 at center of current FOV
# - 0.0 at edges/outside FOV
# - Already rotated and positioned correctly!

# Then YOU do your own projection:
for bed_pixel in bed_mask:
    # Convert bed pixel to world coordinates
    world_point = camera_pose @ camera_pixel_to_3d(bed_pixel)

    # Convert world to grid
    grid_x, grid_y = world_to_grid(world_point)

    # Look up confidence from VLFM's cone
    confidence = confidence_mask[grid_x, grid_y]  # ← Use VLFM's rotated cone!

    # Apply bed's score with this confidence
    update_value_map(grid_x, grid_y, score=0.9, confidence=confidence)
```

---

#### Summary Table

**VLFM:**
- "I have one score for everything"
- "I'll create a cone, rotate it, place it, fill it all with one score"
- **Works because: uniform score everywhere**

**You:**
- "I have different scores for different objects"
- "I can't rotate everything together - I'll lose track of what's what"
- "I'll convert each object's pixels to world coordinates (rotation handled by math)"
- "Then project to grid and assign that object's specific score"
- "But I'll still REUSE VLFM's confidence cone to look up confidence values"

---

#### What You Reuse vs What You Implement

**Reuse from VLFM:**
- ✅ Confidence cone creation (they rotate it for you, you just look up values from it)

**Implement yourself:**
- ❌ Per-pixel projection: depth pixel → 3D camera → 3D world → 2D grid
- ❌ Assigning different scores to different grid locations

**The rotation is still there!** It's just:
- VLFM: Rotates the cone as an image, then fills with uniform score
- You: Rotation is baked into the `camera_pose` transformation matrix, applied per-pixel

---

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

**Full Workflow:**
```python
def gobs_to_detection_list(cfg, image, depth_array, cam_K, idx, gobs, trans_pose=None):
    """
    Convert 2D segmentation masks to 3D detection objects.

    Flow: Masks → Camera PCDs → Map PCDs → Filtered Objects
    """

    # Prepare: Filter and resize masks
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)

    fg_detection_list = DetectionList()
    n_masks = len(gobs['xyxy'])

    for mask_idx in range(n_masks):
        # Extract current mask and metadata
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][gobs['class_id'][mask_idx]]

        # ── Stage 1: Depth to 3D (Camera Frame) ──
        camera_object_pcd = create_object_pcd(
            depth_array, mask, cam_K, image, obj_color=None
        )

        # Filter: Minimum point threshold
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue

        # ── Stage 4: Transform to Map Frame ──
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd

        # ── Stage 2 & 3: Denoise + Downsample ──
        global_object_pcd = process_pcd(global_object_pcd, cfg)

        # Compute bounding box
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0, 1, 0]

        # Filter: Bounding box volume threshold
        if pcd_bbox.volume() < 1e-6:
            continue

        # Create detection object with all metadata
        detected_object = {
            'image_idx': [idx],
            'mask_idx': [mask_idx],
            'class_name': [class_name],
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'conf': [gobs['confidence'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],

            # 3D Geometry
            'pcd': global_object_pcd,      # Point cloud in map frame
            'bbox': pcd_bbox,              # 3D bounding box

            # Features (from CLIP/DINO)
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }

        fg_detection_list.append(detected_object)

    return fg_detection_list
```

---

### Configuration Parameters

**Typical ConceptGraphs Config:**
```python
# Mask filtering
mask_area_threshold: 100          # Minimum pixel area
max_bbox_area_ratio: 0.5          # Maximum bbox area vs image

# Point cloud thresholds
min_points_threshold: 50          # Minimum 3D points required

# Downsampling
downsample_voxel_size: 0.01       # 1cm voxel grid

# DBSCAN denoising
dbscan_remove_noise: true         # Enable DBSCAN
dbscan_eps: 0.02                  # 2cm neighborhood radius
dbscan_min_points: 10             # Minimum cluster size
```

---

### HOV-SG Alternative Implementation

**Location:** `HOV-SG/hovsg/dataloader/generic.py`

**Key Difference:** Uses KD-Tree approach for refinement.

```python
def create_3d_masks(self, masks, depth, full_pcd, full_pcd_tree, camera_pose):
    """
    HOV-SG variant: Match projected points to full scene PCD using KD-Tree.
    """
    pcd_list = []
    pcd_points = np.asarray(full_pcd.points)

    for mask in masks:
        # Step 1: Standard depth unprojection
        pcd_masked = self.create_pcd(mask, depth, camera_pose, mask_img=True)
        pcd_masked_pts = np.asarray(pcd_masked.points)

        # Step 2: Refine using KD-Tree nearest neighbor search
        dist, indices = full_pcd_tree.query(pcd_masked_pts, k=1)
        pcd_masked_pts = pcd_points[indices]

        # Step 3: Create refined PCD
        pcd_mask = o3d.geometry.PointCloud()
        pcd_mask.points = o3d.utility.Vector3dVector(pcd_masked_pts)
        pcd_mask.colors = o3d.utility.Vector3dVector(colors[indices])

        # Step 4: Downsample
        pcd_mask = pcd_mask.voxel_down_sample(voxel_size=0.02)

        pcd_list.append(pcd_mask)

    return pcd_list
```

**Advantage:** Ensures masked points align exactly with reconstructed scene geometry.

---

### Summary Table

| Stage | Function | Input | Output | Purpose |
|-------|----------|-------|--------|---------|
| 1 | `create_object_pcd()` | Mask + Depth + K | Camera frame PCD | Depth unprojection |
| 2 | `pcd_denoise_dbscan()` | Noisy PCD | Denoised PCD | Remove outliers (DBSCAN) |
| 3 | `voxel_down_sample()` | Dense PCD | Sparse PCD | Reduce density |
| 4 | `.transform()` | Camera PCD + Pose | Map frame PCD | Coordinate transformation |
| Final | `gobs_to_detection_list()` | Masks + Depth + Pose | Detection objects | Full integration |

---

### Key Files Reference

1. **Main Pipeline:**
   - `concept-graphs/conceptgraph/slam/utils.py` (lines 478-600)

2. **Intrinsics Utilities:**
   - `concept-graphs/conceptgraph/dataset/datasets_common.py` (lines 46-56)

3. **Geometry Math:**
   - `concept-graphs/conceptgraph/utils/geometry.py`

4. **HOV-SG Alternative:**
   - `HOV-SG/hovsg/dataloader/generic.py` (lines 86-168)
   - `HOV-SG/hovsg/utils/graph_utils.py` (lines 429-477)

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

