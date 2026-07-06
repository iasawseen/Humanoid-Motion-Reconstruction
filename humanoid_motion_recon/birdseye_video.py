#!/usr/bin/env python
"""Bird's-eye video of the lifted Figure skeleton over the VGGT-Omega reconstruction.

TOP: source figure.mp4 frame. BOTTOM: top-down view - reconstruction cloud (gray, static),
pelvis trail (orange, grows over time), current skeleton (green bones / yellow joints),
camera marker. Same timeline as the source (23.976 fps).

GPU pipeline (torch CUDA): static cloud splatted once; the trail canvas is INCREMENTAL
(new segment per frame, not a full redraw - the old CPU version was O(N^2) in trail points);
bones/joints rasterized as CUDA scatter ops; source frames resized on GPU; NVENC encode.

Usage: DEPTH_WORK=<pose_full dir> [POSE_NPZ=...fit3d.npz] [BEV_OUT=...] python -m humanoid_motion_recon.birdseye_video
Output: <MR_OUT>/birdseye.mp4
"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from .fastvid import VideoWriter

WORK = os.environ["DEPTH_WORK"]
OUTP = os.environ.get("BEV_OUT", os.path.join(
    os.environ.get("MR_OUT", "mr_out"), "birdseye.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z = np.load(os.environ.get("POSE_NPZ", os.path.join(WORK, "lift3d.npz")))
J, t, ok, P = z["joints_w"], z["t"], z["ok"], z["scene"]
cam = z["cam_pos"]

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]
BIDX = torch.tensor([[I[a], I[b]] for a, b in BONES], device=DEV)
C_BONE = torch.tensor([100, 220, 120], dtype=torch.uint8, device=DEV)
C_JOINT = torch.tensor([250, 220, 80], dtype=torch.uint8, device=DEV)
C_TRAIL = torch.tensor([240, 140, 60], dtype=torch.uint8, device=DEV)

# ---- BEV mapping (equal scale, whole scene + camera in view)
H = W = 720
allx = np.concatenate([P[:, 0], [cam[0]]]); ally = np.concatenate([P[:, 1], [cam[1]]])
x0, x1 = np.percentile(allx, [0.5, 99.5]); y0, y1 = np.percentile(ally, [0.5, 99.5])
pad = 0.4
scale = min((W - 40) / (x1 - x0 + 2 * pad), (H - 40) / (y1 - y0 + 2 * pad))
cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def px_np(p):
    return (W / 2 + (p[..., 0] - cx) * scale, H / 2 - (p[..., 1] - cy) * scale)


def _disk(r):
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=DEV),
                            torch.arange(-r, r + 1, device=DEV), indexing="ij")
    m = (dy * dy + dx * dx) <= r * r
    return torch.stack([dy[m], dx[m]], 1)                # [K,2]


DISK1, DISK3 = _disk(1), _disk(3)


def splat(canvas, uv, color, disk):
    """Stamp disk-shaped points: uv [M,2] float pixels, color uint8[3]."""
    if uv.numel() == 0:
        return
    q = uv.round().long()                                # [M,2] (u,v)
    pts = q[:, None, :] + disk[None, :, [1, 0]]          # [M,K,2] (u,v)
    pts = pts.reshape(-1, 2)
    m = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    pts = pts[m]
    canvas[pts[:, 1], pts[:, 0]] = color


def lines(canvas, p0, p1, color, disk):
    """Rasterize segments p0->p1 ([B,2] pixel float) with disk-thick strokes."""
    L = (p1 - p0).norm(dim=1).clamp(min=1.0)
    npt = int(L.max().item()) + 2
    s = torch.linspace(0, 1, npt, device=DEV)
    pts = p0[:, None, :] + (p1 - p0)[:, None, :] * s[None, :, None]  # [B,npt,2]
    splat(canvas, pts.reshape(-1, 2), color, disk)


# static background: scene cloud (one-time GPU splat) + camera marker (PIL, one-time)
bg = torch.full((H, W, 3), 16, dtype=torch.uint8, device=DEV); bg[:, :, 2] = 20
Pg = torch.from_numpy(P[::2].astype(np.float32)).to(DEV)
u = W / 2 + (Pg[:, 0] - cx) * scale
v = H / 2 - (Pg[:, 1] - cy) * scale
m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
g = (70 + Pg[m, 2].clamp(min=0) * 40).clamp(max=130).to(torch.uint8)
bg[v[m].long(), u[m].long()] = g[:, None].expand(-1, 3)
_bgp = Image.fromarray(bg.cpu().numpy())
_d = ImageDraw.Draw(_bgp)
_cu, _cv = px_np(cam)
_d.rectangle([_cu - 5, _cv - 5, _cu + 5, _cv + 5], fill=(230, 230, 230))
_d.text((_cu + 8, _cv - 6), "camera", fill=(230, 230, 230))
BG = torch.from_numpy(np.asarray(_bgp)).to(DEV)

# time-label patches (small PIL, pre-rendered once per unique 0.01s label is overkill; per frame is tiny)
def t_patch(tv):
    p = Image.new("RGB", (150, 22), (16, 16, 20))
    ImageDraw.Draw(p).text((4, 4), f"t = {tv:6.2f} s", fill=(250, 250, 250))
    return torch.from_numpy(np.asarray(p)).to(DEV)


mid = 0.5 * (J[:, I["lhip"], :2] + J[:, I["rhip"], :2])
mu, mv = px_np(mid)
mid_px = torch.from_numpy(np.stack([mu, mv], 1).astype(np.float32)).to(DEV)
Jg = torch.from_numpy(J[:, :, :2].astype(np.float32)).to(DEV)
Ju = W / 2 + (Jg[:, :, 0] - cx) * scale
Jv = H / 2 - (Jg[:, :, 1] - cy) * scale
Jpx = torch.stack([Ju, Jv], 2)                           # [N,18,2]
okg = torch.from_numpy(ok.astype(bool)).to(DEV)

N = len(t)
_s0 = Image.open(os.path.join(WORK, "frames", "f00000.jpg")).size
_srch = 2 * ((W * _s0[1] // _s0[0]) // 2)
wr = VideoWriter(OUTP, W, H + _srch, FPS)
trail = torch.zeros((H, W), dtype=torch.bool, device=DEV)
last_tr = None
for n in range(N):
    if ok[n] and n % 3 == 0:                             # incremental trail update
        tc = torch.zeros((H, W, 3), dtype=torch.uint8, device=DEV)
        if last_tr is not None:
            lines(tc, mid_px[last_tr][None], mid_px[n][None], C_TRAIL, DISK1)
        else:
            splat(tc, mid_px[n][None], C_TRAIL, DISK1)
        trail |= tc[:, :, 0] > 0
        last_tr = n
    bev = BG.clone()
    bev[trail] = C_TRAIL
    if ok[n]:
        pts = Jpx[n]
        lines(bev, pts[BIDX[:, 0]], pts[BIDX[:, 1]], C_BONE, DISK1)
        splat(bev, pts, C_JOINT, DISK3)
    bev[6:28, 8:158] = t_patch(t[n])
    src = Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))
    st = torch.from_numpy(np.asarray(src)).to(DEV).permute(2, 0, 1)[None].float()
    st = torch.nn.functional.interpolate(st, size=(_srch, W), mode="bilinear", antialias=True)
    frame = torch.cat([st[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8), bev], 0)
    wr.write(frame.cpu().numpy())
    if n % 1200 == 0:
        print(f"[bev] {n}/{N}", flush=True)
wr.close()
print(f"[bev] WROTE {OUTP}")
