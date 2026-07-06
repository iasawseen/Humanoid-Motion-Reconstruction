#!/usr/bin/env python
"""SAM-3D-Body kp2d skeleton (body + hands) over the raw video frames.

This is the ground-truth-scale overlay: kp2d is drawn directly in the processed-frame pixel
space it was predicted in - no reprojection, no focal/kappa considerations (see README; that
applies only to fit3d reprojection). Replaces the ephemeral one-off that built
figure_sam3d.mp4 body-only.

Perf port: there is no resize/compositing here to move to the GPU - the frame IS the native
decoded JPEG and the skeleton stays on PIL/skel_draw by design. The heavy per-frame work
(JPEG decode + supersampled skeleton draw, both GIL-releasing PIL C ops) runs in a bounded
thread pipeline so it overlaps the NVENC write; frames are still written strictly in order.

Usage: POSE_WORK=<workdir with frames/ + mhr/> [MR_FPS=..] [OUT=path.mp4] \
       python -m humanoid_motion_recon.sam2d_video
"""
import glob
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .fastvid import VideoWriter
from .skel_draw import draw_skeleton

WORK = os.environ["POSE_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.environ.get("OUT", os.path.join(OUTD, "figure_sam3d.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))

# MHR-70 -> skel_draw KPN order (body 18 + hands 40); keep in sync with fit_pose.KP
KP = {"nose": 0, "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lhip": 9, "rhip": 10,
      "lkne": 11, "rkne": 12, "lank": 13, "rank": 14, "lbtoe": 15, "lheel": 17,
      "rbtoe": 18, "rheel": 20, "rwri": 41, "lwri": 62, "neck": 69}
for _side, _base in (("r", 21), ("l", 42)):
    for _f, _fn in enumerate(("th", "ix", "md", "rg", "pk")):
        for _j, _jn in enumerate(("tip", "dst", "mid", "prx")):
            KP[f"{_side}_{_fn}_{_jn}"] = _base + 4 * _f + _j
KIDX = list(KP.values())

frames = sorted(glob.glob(os.path.join(WORK, "frames", "f*.jpg"))) or \
    sorted(glob.glob(os.path.join(WORK, "frames", "f*.png")))
W, H = Image.open(frames[0]).size


def render(fp):
    """Decode + (maybe) skeleton-draw one frame; independent per frame -> thread-safe."""
    img = np.asarray(Image.open(fp).convert("RGB"))
    key = os.path.basename(fp).rsplit(".", 1)[0]
    mf = os.path.join(WORK, "mhr", key + ".npz")
    if os.path.exists(mf):
        kp2d = np.load(mf)["kp2d"]
        if np.ptp(kp2d[:, 1]) <= 1.5 * H and np.ptp(kp2d[:, 0]) <= 1.5 * W:   # skip degenerate
            return draw_skeleton(img, kp2d[KIDX], scale=H / 1080 + 0.3), 1
    return img, 0


os.makedirs(os.path.dirname(OUTP) or ".", exist_ok=True)
wr = VideoWriter(OUTP, W, H, FPS)
drawn = 0
NW = min(8, os.cpu_count() or 4)
Q = 2 * NW                                               # bounded in-flight window
with ThreadPoolExecutor(max_workers=NW) as ex:
    pend = deque(ex.submit(render, fp) for fp in frames[:Q])
    for i in range(len(frames)):
        img, d = pend.popleft().result()
        if i + Q < len(frames):
            pend.append(ex.submit(render, frames[i + Q]))
        drawn += d
        wr.write(img)
        if i % 1200 == 0:
            print(f"[sam2d] {i}/{len(frames)}", flush=True)
wr.close()
print(f"[sam2d] WROTE {OUTP} ({len(frames)} frames, skeleton on {drawn})")
