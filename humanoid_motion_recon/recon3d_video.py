#!/usr/bin/env python
"""Tilted-perspective video of the VGGT-Omega reconstruction + lifted Figure skeleton.

LEFT: source figure.mp4 frame. RIGHT: 3D perspective view of the reconstruction point cloud
(height-colored), the current skeleton (green bones / yellow joints), the pelvis trail
(orange), and the video-camera marker. The view orbits slowly (one revolution over the whole
video) at a fixed 35-degree tilt, so structure reads through motion parallax.

Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.recon3d_video
Output: $MR_OUT/recon3d.mp4 (default mr_out/recon3d.mp4)
"""
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.path.join(OUTD, "recon3d.mp4")
FPS = 24000 / 1001
H = W = 720
EL = np.radians(35.0)

z = np.load(os.path.join(WORK, "lift3d.npz"))
J, t, ok = z["joints_w"].astype(np.float64), z["t"], z["ok"]
P = z["scene"].astype(np.float64)
vcam = z["cam_pos"]

# trim outliers so the framing isn't blown up by stray points
lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
P = P[((P >= lo - 0.5) & (P <= hi + 0.5)).all(1)]

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]

center = np.array([*np.median(P[:, :2], 0), 0.6])
span = float(np.percentile(np.linalg.norm(P[:, :2] - center[:2], axis=1), 97))
RADIUS = 1.8 * span
FOC = 0.72 * H

# height coloring for the cloud (dark floor -> light tops)
zn = np.clip(P[:, 2] / 1.8, 0.0, 1.0)
COLS = np.stack([60 + 120 * zn, 70 + 110 * zn, 90 + 130 * zn], 1).astype(np.uint8)

mid = 0.5 * (J[:, I["lhip"]] + J[:, I["rhip"]])
N = len(t)


def view(az):
    fwd = -np.array([np.cos(EL) * np.cos(az), np.cos(EL) * np.sin(az), np.sin(EL)])
    eye = center - fwd * RADIUS
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, np.stack([right, up, fwd])          # rows: cam basis in world


def project(R, eye, pts):
    v = (pts - eye) @ R.T
    zc = np.maximum(v[:, 2], 1e-3)
    return np.stack([W / 2 + v[:, 0] / zc * FOC, H / 2 - v[:, 1] / zc * FOC], 1), v[:, 2]


os.makedirs(OUTD, exist_ok=True)
wr = imageio.get_writer(OUTP, fps=FPS, quality=7, macro_block_size=1)
for n in range(N):
    az = np.radians(-100.0) + 2 * np.pi * (n / N)   # one slow revolution over the video
    eye, R = view(az)
    img = np.zeros((H, W, 3), np.uint8); img[:] = (14, 14, 18)
    uv, zc = project(R, eye, P)
    order = np.argsort(-zc)                          # painter's: far first
    ui, vi = uv[order, 0].astype(int), uv[order, 1].astype(int)
    m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    img[vi[m], ui[m]] = COLS[order][m]
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    # video-camera marker
    (cu, cv), _ = project(R, eye, vcam[None, :])[0][0], None
    d.rectangle([cu - 4, cv - 4, cu + 4, cv + 4], outline=(240, 240, 240), width=2)
    # pelvis trail
    tr_idx = [k for k in range(0, n + 1, 4) if ok[k]]
    if len(tr_idx) > 1:
        tuv, _ = project(R, eye, mid[tr_idx])
        d.line([tuple(p) for p in tuv], fill=(230, 140, 30), width=2)
    if ok[n]:
        juv, _ = project(R, eye, J[n])
        for a, b in BONES:
            d.line([tuple(juv[I[a]]), tuple(juv[I[b]])], fill=(40, 220, 70), width=3)
        for p in juv:
            d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(250, 220, 40))
    d.text((12, 10), f"t = {t[n]:6.2f} s", fill=(250, 250, 250))
    right_pane = np.asarray(pil)
    src = Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg")).resize((640, 360))
    left = np.zeros((H, 640, 3), np.uint8)
    left[(H - 360) // 2:(H - 360) // 2 + 360] = np.asarray(src)
    wr.append_data(np.concatenate([left, right_pane], 1))
    if n % 1200 == 0:
        print(f"[recon3d] {n}/{N}", flush=True)
wr.close()
print(f"[recon3d] WROTE {OUTP}")
