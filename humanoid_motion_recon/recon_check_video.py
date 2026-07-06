#!/usr/bin/env python
"""Raw VGGT-Omega reconstruction verification videos (no skeleton, no robot pose).

1) recon_points.mp4 - LEFT: source figure.mp4 frame; RIGHT: that frame's depth map unprojected
   to a 3D point cloud and re-rendered from the ORIGINAL camera angle (identity view, the
   pose_enc focal). A legit reconstruction reproduces the source; depth errors show as holes,
   smears, and flicker. Low-confidence points are dropped (black holes = model unsure).
2) depth_maps.mp4 - LEFT: source frame; RIGHT: the raw depth map colorized (turbo, fixed
   global range across the whole video, so brightness is comparable frame to frame).

GPU pipeline (torch CUDA, humanoid_motion_recon/gpurender.py): unprojection, tilt reprojection +
painter's 2x2 splats, the turbo LUT lookup, per-frame resizes, and pane compositing all run on
the GPU; one uint8 download per frame per writer feeds NVENC (fastvid).

Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.recon_check_video
"""
import glob as _g
import os

import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from . import gpurender as gp  # noqa: E402
from .fastvid import VideoWriter  # noqa: E402
from .gpurender import DEV  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
FPS = 24000 / 1001
CONF_MIN = 1.2
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))  # negative pitches the view downward
SKIP_DM = os.environ.get("SKIP_DM") == "1"

# depth-map resolution (VGGT) and output pane
d0 = np.load(os.path.join(WORK, "depth", "f00000.npz"))
h, w = d0["depth"].shape
SC = min(960 / w, 1080 / h)                              # cap both dims (NVENC max 4096)
PW, PH = int(w * SC) // 2 * 2, int(h * SC) // 2 * 2
PORTRAIT = h > w                                         # portrait: stack panes horizontally

# global depth range for stable colorization
samp = []
NALL = len(_g.glob(os.path.join(WORK, 'depth', 'f*.npz')))
for k in range(0, NALL, max(1, NALL // 53)):
    f = os.path.join(WORK, "depth", f"f{k:05d}.npz")
    if os.path.exists(f):
        samp.append(np.load(f)["depth"].astype(np.float32).ravel()[::7])
samp = np.concatenate(samp)
D0, D1 = np.percentile(samp, [2, 98])
print(f"[check] depth range [{D0:.3f}, {D1:.3f}] (raw units)", flush=True)
TURBO = torch.from_numpy(
    (np.asarray(cm.turbo(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)).to(DEV)

vs, us = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=DEV),
                        torch.arange(w, dtype=torch.float32, device=DEV), indexing="ij")


def splat22(u, v, zc, col):
    """Painter's-ordered 2x2-pixel splat matching the CPU renderer exactly: coords are
    truncated and CLIPPED to the pane (border points smear onto the edge), far points first."""
    order = torch.argsort(zc, descending=True)
    ui = u[order].long().clamp_(0, PW - 2)
    vi = v[order].long().clamp_(0, PH - 2)
    col = col[order]
    pc = torch.zeros((PH, PW, 3), dtype=torch.uint8, device=DEV)
    flat = pc.view(-1, 3)
    base = vi * PW + ui
    for off in (0, 1, PW, PW + 1):                       # 2x2 splats close most gaps
        flat[base + off] = col
    return pc


def resize_nearest(img, hh, ww):
    """uint8 HxWx3 CUDA tensor -> hh x ww x 3, nearest (matches PIL Image.NEAREST)."""
    t = img.permute(2, 0, 1)[None].float()
    t = torch.nn.functional.interpolate(t, size=(hh, ww), mode="nearest-exact")
    return t[0].permute(1, 2, 0).to(torch.uint8)


AX = 1 if PORTRAIT else 0
wr_pc = (VideoWriter(os.path.join(OUTD, "recon_points.mp4"), PW * 3, PH, FPS) if PORTRAIT
         else VideoWriter(os.path.join(OUTD, "recon_points.mp4"), PW, PH * 3, FPS))
wr_dm = VideoWriter(os.path.join(OUTD, "depth_maps.mp4"), PW * 2, PH, FPS)
N = NALL
for n in range(N):
    dz = np.load(os.path.join(WORK, "depth", f"f{n:05d}.npz"))
    D = torch.from_numpy(dz["depth"].astype(np.float32)).to(DEV)
    C = torch.from_numpy(dz["conf"].astype(np.float32)).to(DEV)
    pe = dz["pose_enc"]
    fy = float((h / 2.0) / np.tan(pe[7] / 2.0))
    fx = float((w / 2.0) / np.tan(pe[8] / 2.0))
    f0 = gp.from_np(np.array(Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))))
    src = gp.resize(f0, h, w)                            # point-cloud colors
    left = gp.resize(f0, PH, PW)                         # source pane

    # ---- point-cloud re-renders: the camera orbits a mid-scene pivot a little ABOVE and a
    # little BELOW the original viewpoint, so depth errors show as parallax distortion
    good = C > CONF_MIN
    Xs = ((us - w / 2) / fx * D)[good]
    Ys = ((vs - h / 2) / fy * D)[good]
    ds = D[good]
    cc = src[good]
    if ds.numel():
        dso, _ = torch.sort(ds)                          # np.median: mean of middle two
        zmid = float(0.5 * (dso[(dso.numel() - 1) // 2] + dso[dso.numel() // 2]))
    else:
        zmid = 1.0
    panes = [left]
    for tilt in (TILT, -TILT):                           # above, then below
        ca, sa = np.cos(tilt), np.sin(tilt)
        Yt = ca * Ys - sa * (ds - zmid)
        Zt = (sa * Ys + ca * (ds - zmid) + zmid).clamp(min=1e-4)
        u2 = PW / 2 + Xs / Zt * fx * SC
        v2 = PH / 2 + Yt / Zt * fy * SC
        panes.append(splat22(u2, v2, Zt, cc))
    wr_pc.write(gp.to_np(torch.cat(panes, AX)))          # source / above / below

    # ---- colorized depth map
    if SKIP_DM:
        if n % 1200 == 0:
            print(f"[check] {n}/{N}", flush=True)
        continue
    idx = ((D - float(D0)) / float(D1 - D0) * 255).clamp(0, 255).long()
    dm = TURBO[idx]
    dm = torch.where(good[:, :, None], dm, dm // 4)      # dim low-confidence regions
    dm = resize_nearest(dm, PH, PW)
    wr_dm.write(gp.to_np(torch.cat([left, dm], 1)))
    if n % 1200 == 0:
        print(f"[check] {n}/{N}", flush=True)
wr_pc.close(); wr_dm.close()
print(f"[check] WROTE {OUTD}/recon_points.mp4 and depth_maps.mp4", flush=True)
