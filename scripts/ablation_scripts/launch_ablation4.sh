#!/usr/bin/env bash

# Ablation 4: MetaCLIP + Overlap (GPU 3)
export CUDA_VISIBLE_DEVICES=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLFM_ROOT="$(dirname "$SCRIPT_DIR")"

export VLFM_PYTHON=${VLFM_PYTHON:-/data/nshreen1/anaconda3/envs/vlfm_v2/bin/python}
export GROUNDING_DINO_PORT=15181
export SAM_PORT=15183
export YOLOV7_PORT=15184
export CLIP_PORT=15186

session_name=vlm_servers_ablation4

tmux new-session -d -s ${session_name}
tmux split-window -v -t ${session_name}:0
tmux split-window -h -t ${session_name}:0.0
tmux split-window -h -t ${session_name}:0.2

tmux send-keys -t ${session_name}:0.0 "export CUDA_VISIBLE_DEVICES=3 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.grounding_dino --port ${GROUNDING_DINO_PORT}" C-m
tmux send-keys -t ${session_name}:0.1 "export CUDA_VISIBLE_DEVICES=3 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.clip_encoder --port ${CLIP_PORT} --pretrained metaclip_fullcc" C-m
tmux send-keys -t ${session_name}:0.2 "export CUDA_VISIBLE_DEVICES=3 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.object_centric.sam_segmenter --port ${SAM_PORT}" C-m
tmux send-keys -t ${session_name}:0.3 "export CUDA_VISIBLE_DEVICES=3 && conda activate vlfm_v2 && cd ${VLFM_ROOT} && ${VLFM_PYTHON} -m vlfm.vlm.yolov7 --port ${YOLOV7_PORT}" C-m

echo "Created tmux session '${session_name}' for Ablation 4 on GPU 3 (Ports 1518X)."
