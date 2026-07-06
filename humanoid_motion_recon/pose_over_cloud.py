#!/usr/bin/env python
"""Fitted 3D robot skeleton over the per-frame VGGT-Omega point cloud, +33 deg inclination.

TOP: source figure.mp4 frame. BOTTOM: that frame's cloud (RGB-colored, confidence-gated)
re-rendered from the original camera orbited +33 deg vertically about a mid-scene pivot, with
the rigid-fitter skeleton (fit3d.npz, green bones / yellow joints) drawn in the same view.

GPU pipeline (torch CUDA): depth backprojection, the tilt reprojection and the painter's-order
RGB splat (2x2 footprint) run on-device via gpurender; each source frame is decoded ONCE and
resized on GPU for both panes; the skel_draw/PIL skeleton renders into its bounding-box crop
only (identical pixels, ~5x less supersampled area than the full pane); NVENC encode.

Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.pose_over_cloud
Output: $MR_OUT/pose_over_cloud.mp4 (default mr_out/pose_over_cloud.mp4)
"""
import glob
import os

import numpy as np
import torch
from PIL import Image

from .skel_draw import draw_skeleton
from .fastvid import VideoWriter
from . import gpurender as gr
from .gpurender import DEV

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.environ.get("POC_OUT", os.path.join(OUTD, "pose_over_cloud.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
CONF_MIN = 1.2
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))

fz = np.load(os.environ.get("POSE_NPZ", os.path.join(OUTD, "fit3d.npz")))
J_w, t_arr, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"]
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
lz = np.load(os.environ.get("LIFT_NPZ", os.path.join(OUTD, "lift3d.npz")))
Rcw, scale, cam_pos = lz["R_cam2world"], float(lz["scale"]), lz["cam_pos"]
# world -> VGGT camera frame (inverse of the lift alignment; per-frame if the camera moves)
if "M_c2w" in lz:
    Minv = np.linalg.inv(lz["M_c2w"])
    J_cam = np.einsum("nij,nkj->nki", Minv, J_w - lz["c_c2w"][:, None])
else:
    J_cam = np.einsum("ji,nkj->nki", Rcw, (J_w - cam_pos[None, None]) / scale)

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]

d0 = np.load(os.path.join(WORK, "depth", "f00000.npz"))
h, w = d0["depth"].shape
SC = 960 / w
PW, PH = int(w * SC) // 2 * 2, int(h * SC) // 2 * 2
FW, FH = Image.open(os.path.join(WORK, "frames", "f00000.jpg")).size

# SAM's focal (processed-frame px): the fitted joints subtend SAM's ANGULAR sizes - projecting
# with VGGT's focal shrinks the drawn skeleton by f_vggt/f_sam. Same fix as collate_video.
_mhr = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
_f = [float(z["focal"]) for f in _mhr[:: max(1, len(_mhr) // 25)]
      for z in [np.load(f)] if "focal" in z.files]
F_SAM = float(np.median(_f)) if _f else 1468.6           # older caches: fixed SAM heuristic


def body_fix(Jc, fx, fy):
    """Rescale about mid-hip: undo kappa (world-only calibration) and the VGGT/SAM focal ratio;
    the pelvis stays on its kp2d ray so its pixel is untouched."""
    pelv = 0.5 * (Jc[I["lhip"]] + Jc[I["rhip"]])
    s = F_SAM * (w / FW) / np.sqrt(fx * fy) / KAPPA
    return pelv + (Jc - pelv) * s


def tilt_project(X, Y, Z, zmid, fx, fy):
    ca, sa = np.cos(TILT), np.sin(TILT)
    Yt = ca * Y - sa * (Z - zmid)
    Zt = np.maximum(sa * Y + ca * (Z - zmid) + zmid, 1e-4)
    return PW / 2 + X / Zt * fx * SC, PH / 2 + Yt / Zt * fy * SC, Zt


# GPU pixel grid + 2x2 splat footprint (same footprint/clamp semantics as fastvid.gpu_splat)
_vs, _us = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=DEV),
                          torch.arange(w, dtype=torch.float32, device=DEV), indexing="ij")
SQ2 = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=DEV)  # (dy,dx) offsets
_CA, _SA = float(np.cos(TILT)), float(np.sin(TILT))


def cloud_pane(D, C, src, fx, fy):
    """Backproject conf-gated depth, orbit +TILT about the median depth, splat far-to-near.
    D, C [h,w] float32 CUDA; src [h,w,3] uint8 CUDA -> (pane tensor, zmid)."""
    good = C > CONF_MIN
    Dg = D[good]
    zmid = float(torch.quantile(Dg, 0.5)) if Dg.numel() else 1.0
    X = (_us[good] - w / 2) / fx * Dg
    Y = (_vs[good] - h / 2) / fy * Dg
    Yt = _CA * Y - _SA * (Dg - zmid)
    Zt = (_SA * Y + _CA * (Dg - zmid) + zmid).clamp(min=1e-4)
    u = (PW / 2 + X / Zt * fx * SC).long().clamp(0, PW - 2)
    v = (PH / 2 + Yt / Zt * fy * SC).long().clamp(0, PH - 2)
    pane = gr.canvas(PH, PW)
    gr.splat_cloud(pane, torch.stack([u, v], 1).float(), Zt, src[good], SQ2)
    return pane, zmid


def skel_patch(pc, pts):
    """draw_skeleton confined to the joints' bounding box (+margin): identical pixels, far less
    PIL supersampling area. All ink (bones, rings, head circle) stays within the joint hull
    plus a few px of stroke radius, so a 64 px margin is conservative."""
    fin = pts[np.isfinite(pts).all(1)]
    if not len(fin):
        return pc
    x0 = max(int(fin[:, 0].min()) - 64, 0); x1 = min(int(fin[:, 0].max()) + 64, PW)
    y0 = max(int(fin[:, 1].min()) - 64, 0); y1 = min(int(fin[:, 1].max()) + 64, PH)
    if x0 >= x1 or y0 >= y1:
        return pc
    pc[y0:y1, x0:x1] = draw_skeleton(np.ascontiguousarray(pc[y0:y1, x0:x1]), pts - [x0, y0])
    return pc


wr = VideoWriter(OUTP, PW, PH * 2, FPS)
N = len(glob.glob(os.path.join(WORK, 'depth', 'f*.npz')))
for n in range(N):
    dz = np.load(os.path.join(WORK, "depth", f"f{n:05d}.npz"))
    D = gr.from_np(dz["depth"]).float()
    C = gr.from_np(dz["conf"]).float()
    pe = dz["pose_enc"]
    fy = (h / 2.0) / np.tan(pe[7] / 2.0)
    fx = (w / 2.0) / np.tan(pe[8] / 2.0)
    jt = gr.from_np(np.array(Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg")),
                             dtype=np.uint8))                # decode once, resize on GPU
    src = gr.resize(jt, h, w)                                # cloud colors
    left = gr.resize(jt, PH, PW)                             # top pane
    pane, zmid = cloud_pane(D, C, src, float(fx), float(fy))
    pc = gr.to_np(pane)
    if ok[n]:
        Jc = body_fix(J_cam[n], fx, fy)
        ju, jv, _ = tilt_project(Jc[:, 0], Jc[:, 1], Jc[:, 2], zmid, fx, fy)
        pc = skel_patch(pc, np.stack([ju, jv], 1))
    wr.write(np.concatenate([gr.to_np(left), pc], 0))        # vertical stack: source / cloud
    if n % 1200 == 0:
        print(f"[poc] {n}/{N}", flush=True)
wr.close()
print(f"[poc] WROTE {OUTP}")
