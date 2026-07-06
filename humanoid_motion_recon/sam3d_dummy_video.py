#!/usr/bin/env python
"""Raw SAM-3D-Body 3D dummy for chirality debugging: [video | RAW kp3d dummy | flip-corrected].

The dummy is the per-frame kp3d skeleton rendered in 3D (up-calibrated camera frame, fixed 3/4
orthographic view, pelvis-centred) - NO fit, NO smoothing. Chirality is invisible in 2D overlays
(a mirrored pose projects identically) but obvious here: left limbs cyan / right orange, and a
yellow FORWARD arrow (hip-line x up) on the floor. SAM's front/back mirror shows as the colors
swapping + the arrow reversing while the video robot stands still. The third pane applies the
fitter's Viterbi flip decisions (<MR_OUT>/flips.npy) - it should hold steady.

Perf port (torch CUDA, humanoid_motion_recon.gpurender): the top-pane frame resize and the 3-pane
compositing run on the GPU; the two dummy panes stay on PIL/skel_draw (small draws). Per-frame
work (decode + panes + GPU compose) runs in a bounded thread pipeline overlapping the NVENC
write; frames are still written strictly in order.

Usage: POSE_WORK=<frames+mhr dir> [MR_OUT=mr_out] [OUT=...] \
       python -m humanoid_motion_recon.sam3d_dummy_video
"""
import glob
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageDraw

from .fastvid import VideoWriter
from .gpurender import from_np, resize, to_np
from .skel_draw import KPN, draw_skeleton

WORK = os.environ["POSE_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.environ.get("OUT", os.path.join(OUTD, "sam3d_dummy.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
VW, VH = 960, 540                                        # video pane
DW, DH = 480, 540                                        # dummy panes

KP = {"nose": 0, "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lhip": 9, "rhip": 10,
      "lkne": 11, "rkne": 12, "lank": 13, "rank": 14, "lbtoe": 15, "lheel": 17,
      "rbtoe": 18, "rheel": 20, "rwri": 41, "lwri": 62, "neck": 69}
for _side, _base in (("r", 21), ("l", 42)):
    for _f, _fn in enumerate(("th", "ix", "md", "rg", "pk")):
        for _j, _jn in enumerate(("tip", "dst", "mid", "prx")):
            KP[f"{_side}_{_fn}_{_jn}"] = _base + 4 * _f + _j
KIDX = list(KP.values())
LH, RH, NK = 9, 10, 69
SWAP70 = list(range(70))
for a, b in ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 18), (16, 19), (17, 20), (63, 64), (65, 66), (67, 68)]
             + [(21 + i, 42 + i) for i in range(21)]):
    SWAP70[a], SWAP70[b] = b, a

mfiles = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
frames_dir = os.path.join(WORK, "frames")
flips = np.load(os.path.join(OUTD, "flips.npy")) if os.path.exists(os.path.join(OUTD, "flips.npy")) \
    else None

# up calibration from a stand window (same as the fitter)
st0 = float(os.environ.get("MR_STAND0", "7.5")); st1 = float(os.environ.get("MR_STAND1", "9.5"))
ups = []
for f in mfiles:
    n = int(os.path.basename(f)[1:6])
    if st0 <= n / FPS <= st1:
        k3 = np.load(f)["kp3d"]
        ups.append(k3[NK] - 0.5 * (k3[LH] + k3[RH]))
up = np.mean(ups, 0); up /= np.linalg.norm(up)
# display basis: y' = up; view the scene from a fixed 3/4 yaw so forward changes are visible
zc = np.array([0.0, 0.0, 1.0]); zc = zc - (zc @ up) * up; zc /= np.linalg.norm(zc)
xc = np.cross(up, zc)
YAW = np.radians(35.0)
vx = np.cos(YAW) * xc + np.sin(YAW) * zc                 # screen-right axis
vz = -np.sin(YAW) * xc + np.cos(YAW) * zc                # into-screen axis (unused, ortho)
SC = 220.0                                               # px per meter


def dummy_pane(k3, label, col_bg=(18, 20, 26)):
    img = np.zeros((DH, DW, 3), np.uint8); img[:] = col_bg
    midhip = 0.5 * (k3[LH] + k3[RH])
    P = k3 - midhip
    u = DW / 2 + (P @ vx) * SC
    v = DH * 0.52 - (P @ up) * SC
    img = draw_skeleton(img, np.stack([u, v], 1)[[KIDX.index(j) for j in KIDX]], scale=0.55)
    # forward arrow on the "floor"
    l = k3[LH] - k3[RH]
    fwd = np.cross(l, up); fwd /= np.linalg.norm(fwd) + 1e-9
    base = np.array([DW / 2, DH * 0.52 + 0.95 * SC * 0.55 + 20])
    tip = base + np.array([(fwd @ vx) * 60, -(fwd @ up) * 60 * 0 - (fwd @ vz) * 30])
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    d.line([tuple(base), tuple(tip)], fill=(255, 220, 40), width=4)
    d.ellipse([tip[0] - 5, tip[1] - 5, tip[0] + 5, tip[1] + 5], fill=(255, 220, 40))
    d.rectangle([0, 0, 8 + 9 * len(label), 24], fill=(0, 0, 0))
    d.text((6, 4), label, fill=(240, 240, 240), font_size=15)
    return np.asarray(pil)


def render(f):
    """One output frame (or None if the source jpg is missing); independent -> thread-safe."""
    n = int(os.path.basename(f)[1:6])
    fp = os.path.join(frames_dir, f"f{n:05d}.jpg")
    if not os.path.exists(fp):
        return None
    top = resize(np.asarray(Image.open(fp)), VH, VW)     # GPU resize
    k3 = np.load(f)["kp3d"]
    raw = dummy_pane(k3, "RAW sam-3d-body")
    if flips is not None and n < len(flips) and flips[n]:
        kc = k3[SWAP70].copy()
        zp = 0.5 * (kc[LH, 2] + kc[RH, 2])
        kc[:, 2] = 2 * zp - kc[:, 2]
    else:
        kc = k3
    cor = dummy_pane(kc, "flip-corrected", col_bg=(16, 26, 18))
    # GPU pane compositing: video / [raw | corrected]
    return to_np(torch.cat([top, torch.cat([from_np(raw), from_np(cor)], 1)], 0))


wr = VideoWriter(OUTP, VW, VH * 2, FPS)                  # vertical: video / [raw | corrected]
n_total = 0
NW = min(8, os.cpu_count() or 4)
Q = 2 * NW                                               # bounded in-flight window
with ThreadPoolExecutor(max_workers=NW) as ex:
    pend = deque(ex.submit(render, f) for f in mfiles[:Q])
    for i in range(len(mfiles)):
        frame = pend.popleft().result()
        if i + Q < len(mfiles):
            pend.append(ex.submit(render, mfiles[i + Q]))
        if frame is None:
            continue
        wr.write(frame)
        n_total += 1
        if n_total % 1200 == 0:
            print(f"[dummy] {n_total}/{len(mfiles)}", flush=True)
wr.close()
print(f"[dummy] WROTE {OUTP} ({n_total} frames @ {FPS:.3f})")
