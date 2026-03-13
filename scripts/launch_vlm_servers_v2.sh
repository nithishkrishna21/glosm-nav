#!/usr/bin/env bash

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLFM_ROOT="$(dirname "$SCRIPT_DIR")"

# export VLFM_PYTHON=${VLFM_PYTHON:-`which python`}
export VLFM_PYTHON=${VLFM_PYTHON:-/data/nshreen1/anaconda3/envs/vlfm_v2/bin/python}
export VLFM_DATA=${VLFM_DATA:-${VLFM_ROOT}/data}
export MOBILE_SAM_CHECKPOINT=${MOBILE_SAM_CHECKPOINT:-data/mobile_sam.pt}
export GROUNDING_DINO_CONFIG=${GROUNDING_DINO_CONFIG:-GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}
export GROUNDING_DINO_WEIGHTS=${GROUNDING_DINO_WEIGHTS:-data/groundingdino_swint_ogc.pth}
export CLASSES_PATH=${CLASSES_PATH:-vlfm/vlm/classes.txt}
export GROUNDING_DINO_PORT=${GROUNDING_DINO_PORT:-12181}
# export BLIP2ITM_PORT=${BLIP2ITM_PORT:-12182}  # Commented out - using SigLIP2 instead
export SAM_PORT=${SAM_PORT:-12183}
export YOLOV7_PORT=${YOLOV7_PORT:-12184}  # Changed from 12186 to match code expectations
# export SIGLIP2_PORT=${SIGLIP2_PORT:-12185}  # Changed from 12184 to avoid conflict
export CLIP_PORT=${CLIP_PORT:-12186}

session_name=vlm_servers_${RANDOM}

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window to create 4 panes (Grounding DINO, SAM, YOLOv7, SigLIP2)
tmux split-window -v -t ${session_name}:0
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

# Run commands in each pane with explicit conda activation and cd
# Using vlfm_v2 environment for all models (upgraded PyTorch + transformers for SigLIP2)
tmux send-keys -t ${session_name}:0.0 "conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
# tmux send-keys -t ${session_name}:0.1 "conda activate vlfm_v2 && cd /workspace/vlfm && ${VLFM_PYTHON} -m vlfm.vlm.blip2itm --port ${BLIP2ITM_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.clip_encoder --port ${CLIP_PORT}" C-m
tmux send-keys -t ${session_name}:0.2 "conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.sam_segmenter --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'. You must wait up to 90 seconds for the model weights to finish being loaded."
echo ""
echo "Model servers:"
echo "  - Grounding DINO: localhost:${GROUNDING_DINO_PORT}"
echo "  - CLIP:           localhost:${CLIP_PORT}    "
echo "  - SAM:            localhost:${SAM_PORT}"
echo "  - YOLOv7:         localhost:${YOLOV7_PORT}"
echo ""
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"