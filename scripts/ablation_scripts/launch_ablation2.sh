#!/usr/bin/env bash

# Ablation 2: OpenCLIP + Overlap (GPU 1)
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLFM_ROOT="$(dirname "$SCRIPT_DIR")"

export VLFM_PYTHON=${VLFM_PYTHON:-/data/nshreen1/anaconda3/envs/vlfm_v2/bin/python}
export GROUNDING_DINO_PORT=13181
export SAM_PORT=13183
export YOLOV7_PORT=13184
export CLIP_PORT=13186

session_name=vlm_servers_ablation2

tmux new-session -d -s ${session_name}
tmux split-window -v -t ${session_name}:0
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

tmux send-keys -t ${session_name}:0.0 "export CUDA_VISIBLE_DEVICES=1 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "export CUDA_VISIBLE_DEVICES=1 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.clip_encoder --port ${CLIP_PORT} --pretrained laion2b_s32b_b79k" C-m
tmux send-keys -t ${session_name}:0.2 "export CUDA_VISIBLE_DEVICES=1 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.sam_segmenter --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "export CUDA_VISIBLE_DEVICES=1 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo "Created tmux session '${session_name}' for Ablation 2 on GPU 1 (Ports 1318X)."
