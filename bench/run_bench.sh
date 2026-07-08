#!/usr/bin/env bash
# 1-minute-clip pipeline benchmark + quality gate.
#
# Usage:
#   bench/run_bench.sh <video.mp4> <workdir> <reference_fit3d.npz> [gpu]
#
# Extracts the first 60 s of <video.mp4> into <workdir>/frames, expects
# <workdir>/boxes.npz to exist (per-frame subject bbox - see README), then times
# every stage on one GPU and gates the resulting fit3d against the reference.
# Wall-clock target: <=8 min per video-minute on an RTX 3090 (measured 6:17 with
# defaults: SAM_INFER=body SAM_BATCH=16 VGGT_STRIDE=150 VGGT_FSTRIDE=2).
set -euo pipefail

VIDEO=$1; WORK=$2; REF=$3; GPU=${4:-0}
SAM_CKPT=${SAM_CKPT:-$HOME/.cache/modelscope/facebook/sam-3d-body-dinov3}
VGGT_CKPT=${VGGT_CKPT:-$HOME/.cache/modelscope/facebook/VGGT-Omega/vggt_omega_1b_512.pt}
export CUDA_VISIBLE_DEVICES=$GPU

if [ ! -d "$WORK/frames" ]; then
    mkdir -p "$WORK/frames"
    ffmpeg -nostdin -loglevel error -i "$VIDEO" -t 60 -q:v 2 -start_number 0 "$WORK/frames/f%05d.jpg"
fi
[ -f "$WORK/boxes.npz" ] || { echo "missing $WORK/boxes.npz"; exit 1; }

T0=$(date +%s)
echo "=== SAM-3D-Body ==="
POSE_WORK=$WORK python -m humanoid_motion_recon.pose_video "$SAM_CKPT" | tail -1
T1=$(date +%s); echo "sam: $((T1-T0)) s"
echo "=== VGGT-Omega ==="
DEPTH_WORK=$WORK VGGT_FSTRIDE=${VGGT_FSTRIDE:-2} python -m humanoid_motion_recon.depth_video "$VGGT_CKPT" | tail -1
T2=$(date +%s); echo "vggt: $((T2-T1)) s"
echo "=== lift + fit ==="
DEPTH_WORK=$WORK MR_OUT=$WORK python -m humanoid_motion_recon.lift_skeleton | tail -1
DEPTH_WORK=$WORK MR_OUT=$WORK python -m humanoid_motion_recon.fit_pose | tail -1
T3=$(date +%s); echo "lift+fit: $((T3-T2)) s"
echo "=== TOTAL: $((T3-T0)) s (target <=480 s per video-minute) ==="
echo "=== quality gate ==="
python "$(dirname "$0")/quality_gate.py" "$WORK/fit3d.npz" "$REF"
