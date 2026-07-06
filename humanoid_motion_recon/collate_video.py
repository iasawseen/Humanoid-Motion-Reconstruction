#!/usr/bin/env python
"""One-pass MotionRecon collate: horizontal stack [raw + pose overlay | depth | points +33 | BEV].

Composes every pane per frame directly from the cached assets (frames/, depth/, fit3d/lift3d)
- no intermediate video decoding, no ffmpeg filter graph - and writes via NVENC (fastvid).
The raw pane carries the fitted humanoid skeleton reprojected onto the video.

GPU pipeline (torch CUDA, gpurender): source-frame resizes, the depth colormap (exact
matplotlib-turbo LUT + PIL-probed NEAREST index maps), the tilted point-cloud projection +
painter's splat, and the pane concatenation all run on-device. Skeletons/trail stay on
PIL/skel_draw but are drawn on bbox crops (pixel-identical, O(bbox) not O(pane)); the BEV
trail points are precomputed once (the fading alpha ramp is a function of trail length, so
the trail redraw itself stays per-frame by design).

Worker-parallel: the parent splits the frame range across WORKERS subprocesses (default 8 -
the consumer NVENC concurrent-session cap on current drivers), each renders its chunk to a
part file, and the parts are losslessly concatenated (ffmpeg -c copy). Frames are independent
(the BEV trail is a deterministic function of the frame index), so chunking is exact.

Layout: LAYOUT=row -> 1x4 horizontal strip [raw | depth | points | BEV] (default for portrait
sources); LAYOUT=grid -> 2x2 [raw, points / depth, BEV] in equal padded tiles (default for
landscape - e.g. figure gives 1920x1080).

Usage: DEPTH_WORK=<workdir> MR_OUT=<outdir> [MR_FPS=..] [TILT_DEG=33] [WORKERS=8] \\
       [LAYOUT=row|grid] python -m humanoid_motion_recon.collate_video
Output: <MR_OUT>/collate.mp4
"""
import glob
import os
import sys

import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from .fastvid import VideoWriter
from .gpurender import DEV, from_np, resize, to_np
from .skel_draw import KPN, I, draw_skeleton, draw_trail

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
TILT = np.radians(float(os.environ.get("TILT_DEG", "33")))
CONF_MIN = 1.2

fz = np.load(os.path.join(OUTD, "fit3d.npz"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
J_w, t_arr, ok = fz["joints_w"].astype(np.float64), fz["t"], fz["ok"]
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
Rcw, scale, cam_pos = lz["R_cam2world"], float(lz["scale"]), lz["cam_pos"]
P_scene = lz["scene"].astype(np.float64)
if "M_c2w" in lz:                                        # moving camera: per-frame inverse map
    Minv = np.linalg.inv(lz["M_c2w"])
    J_cam = np.einsum("nij,nkj->nki", Minv, J_w - lz["c_c2w"][:, None])
else:
    J_cam = np.einsum("ji,nkj->nki", Rcw, (J_w - cam_pos[None, None]) / scale)

# SAM's focal (processed-frame px): the fitted joints subtend SAM's ANGULAR sizes, which map to
# correct pixels only under this focal - projecting them with VGGT's pose_enc focal shrinks the
# drawn skeleton by f_vggt/f_sam (0.91 figure, 0.62 vera). Median over the mhr cache; older
# caches predate the focal key and used SAM's fixed heuristic 1468.6.
_mhr = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
_f = [float(z["focal"]) for f in _mhr[:: max(1, len(_mhr) // 25)]
      for z in [np.load(f)] if "focal" in z.files]
F_SAM = float(np.median(_f)) if _f else 1468.6


def body_fix(Jc, fx, fy):
    """Undo the two un-inverted scale factors on the reprojected skeleton: the kappa heel
    calibration (world-space, legitimate there) and the VGGT-vs-SAM focal ratio. Pure rescale
    about the mid-hip: the pelvis stays on its kp2d ray, so its pixel is untouched.
    (w/FW converts F_SAM from processed-frame px to depth-resolution px.)"""
    pelv = 0.5 * (Jc[I["lhip"]] + Jc[I["rhip"]])
    s = F_SAM * (w / FW) / np.sqrt(fx * fy) / KAPPA
    return pelv + (Jc - pelv) * s

dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
N = len(dfiles)
d0 = np.load(dfiles[0])["depth"]
h, w = d0.shape
FW, FH = Image.open(os.path.join(WORK, "frames", "f00000.jpg")).size
H = 1080 if h > w else 540                               # pane height by aspect
SCp = H / h                                              # cloud/depth pane scale
PW = int(w * SCp) // 2 * 2
RW = int(FW * H / FH) // 2 * 2                           # raw pane width
BW = H                                                   # BEV square

# depth colorization range (global)
samp = np.concatenate([np.load(dfiles[k])["depth"].astype(np.float32).ravel()[::9]
                       for k in range(0, N, max(1, N // 40))])
D0, D1 = (float(q) for q in np.percentile(samp, [2, 98]))
TURBO = (np.asarray(cm.turbo(np.linspace(0, 1, 256)))[:, :3] * 255).astype(np.uint8)
TURBO_T = from_np(TURBO)                                 # [256,3] uint8 LUT on GPU


def _pil_nn_map(nin, nout):
    """Exact PIL-NEAREST source-index map for one axis, probed through PIL itself so the
    fp tie-breaking at cell boundaries matches Image.resize bit-for-bit."""
    enc = np.zeros((1, nin, 3), np.uint8)
    enc[0, :, 0] = np.arange(nin) >> 8
    enc[0, :, 1] = np.arange(nin) & 255
    r = np.asarray(Image.fromarray(enc).resize((nout, 1), Image.NEAREST)).astype(np.int64)
    return torch.as_tensor((r[0, :, 0] << 8) | r[0, :, 1], device=DEV)


NN_V, NN_U = _pil_nn_map(h, H), _pil_nn_map(w, PW)       # depth-pane NEAREST resize maps


def skel_on(img, pts, sc):
    """draw_skeleton on a joint-bbox crop: PIL's 2x-supersampled overlay is O(pane area)
    even for a small skeleton (~69 ms on a 1080-high pane), while every drawn pixel lives
    within the joints' bbox + stroke/head/LANCZOS margin. Integer-translated crop => the
    rasterization grid is unchanged, so pixels match the full-pane call."""
    p = np.asarray(pts, np.float64)
    fin = np.isfinite(p).all(1)
    if not fin.any():
        return img
    Hh, Ww = img.shape[:2]
    mg = int(40 * sc) + 10                               # > joint ring + head circle + filter taps
    x0 = max(int(p[fin, 0].min()) - mg, 0); x1 = min(int(p[fin, 0].max()) + mg + 1, Ww)
    y0 = max(int(p[fin, 1].min()) - mg, 0); y1 = min(int(p[fin, 1].max()) + mg + 1, Hh)
    if x1 <= x0 or y1 <= y0:                             # bbox misses the pane: nothing visible
        return img
    img[y0:y1, x0:x1] = draw_skeleton(np.ascontiguousarray(img[y0:y1, x0:x1]),
                                      p - (x0, y0), scale=sc)
    return img


def trail_on(img, tr):
    """draw_trail on the trail-bbox crop (same argument as skel_on; width-3 strokes)."""
    if len(tr) < 2:
        return img
    p = np.asarray(tr, np.float64)
    Hh, Ww = img.shape[:2]
    x0 = max(int(p[:, 0].min()) - 8, 0); x1 = min(int(p[:, 0].max()) + 9, Ww)
    y0 = max(int(p[:, 1].min()) - 8, 0); y1 = min(int(p[:, 1].max()) + 9, Hh)
    if x1 <= x0 or y1 <= y0:
        return img
    img[y0:y1, x0:x1] = draw_trail(np.ascontiguousarray(img[y0:y1, x0:x1]),
                                   [(u - x0, v - y0) for u, v in tr])
    return img


def gpu_splat_t(uv, zv, cols, Hp, Wp, splat=2):
    """fastvid.gpu_splat semantics, but GPU-resident in and out (no numpy round trip):
    painter's-ordered 2x2 splat - far points first, later writes win."""
    order = torch.argsort(zv, descending=True)
    ui = uv[order, 0].long().clamp(0, Wp - splat)
    vi = uv[order, 1].long().clamp(0, Hp - splat)
    cc = cols[order]
    img = torch.zeros((Hp, Wp, 3), dtype=torch.uint8, device=DEV)
    flat = img.view(-1, 3)
    for dy in range(splat):
        for dx in range(splat):
            flat[(vi + dy) * Wp + (ui + dx)] = cc
    return img


# BEV mapping (from lift3d scene, like birdseye_video)
allx = np.concatenate([P_scene[:, 0], [cam_pos[0]]])
ally = np.concatenate([P_scene[:, 1], [cam_pos[1]]])
x0, x1 = np.percentile(allx, [0.5, 99.5]); y0, y1 = np.percentile(ally, [0.5, 99.5])
bscale = min((BW - 40) / (x1 - x0 + 0.8), (H - 40) / (y1 - y0 + 0.8))
bcx, bcy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def bpx(p):
    return (BW / 2 + (p[0] - bcx) * bscale, H / 2 - (p[1] - bcy) * bscale)


bev_bg = np.zeros((H, BW, 3), np.uint8); bev_bg[:] = (16, 16, 20)
uu = (BW / 2 + (P_scene[:, 0] - bcx) * bscale).astype(int)
vv = (H / 2 - (P_scene[:, 1] - bcy) * bscale).astype(int)
m = (uu >= 0) & (uu < BW) & (vv >= 0) & (vv < H)
g = (70 + np.clip(P_scene[:, 2], 0, 1.5) * 40).astype(np.uint8)
bev_bg[vv[m], uu[m]] = np.stack([g[m]] * 3, 1)
mid_w = 0.5 * (J_w[:, I["lhip"], :2] + J_w[:, I["rhip"], :2])
# trail points precomputed once; frame n uses the first TRAIL_CNT[n] of them (== the original
# [bpx(mid_w[k]) for k in range(0, n+1, 3) if ok[k]] rebuilt per frame)
TRAIL_PTS = [bpx(mid_w[k]) for k in range(0, N, 3) if ok[k]]
TRAIL_CNT = np.cumsum([1 if (k % 3 == 0 and ok[k]) else 0 for k in range(N)])

vs, us = np.mgrid[0:h, 0:w].astype(np.float32)
US, VS = from_np(us), from_np(vs)
LAYOUT = os.environ.get("LAYOUT", "row" if FH > FW else "grid")
TW = max(RW, PW, BW)                                     # grid tile width
OW, OH = (RW + PW * 2 + BW, H) if LAYOUT == "row" else (TW * 2, H * 2)


def tile(pane):
    out = torch.zeros((H, TW, 3), dtype=torch.uint8, device=DEV)
    x0 = (TW - pane.shape[1]) // 2
    out[:, x0:x0 + pane.shape[1]] = pane
    return out


def render_frame(n):
    dz = np.load(dfiles[n])
    D = from_np(dz["depth"].astype(np.float32))
    C = from_np(dz["conf"].astype(np.float32))
    pe = dz["pose_enc"]
    fy = float((h / 2.0) / np.tan(pe[7] / 2.0))
    fx = float((w / 2.0) / np.tan(pe[8] / 2.0))
    src = from_np(np.asarray(Image.open(os.path.join(WORK, "frames", f"f{n:05d}.jpg"))))
    Jc = body_fix(J_cam[n], fx, fy) if ok[n] else None
    # ---- pane 1: raw + humanoid pose overlay (GPU resize, PIL skeleton)
    raw = resize(src, H, RW)
    if ok[n]:
        ju = (Jc[:, 0] / Jc[:, 2] * fx + w / 2) * (RW / w)
        jv = (Jc[:, 1] / Jc[:, 2] * fy + h / 2) * (H / h)
        raw = from_np(skel_on(to_np(raw), np.stack([ju, jv], 1), H / 1080 + 0.3))
    # ---- pane 2: depth (GPU turbo LUT + confidence dim + exact NEAREST upscale)
    idx = ((D - D0) / (D1 - D0) * 255).clamp(0, 255).to(torch.uint8)
    dm = TURBO_T[idx.long()]
    dm = torch.where((C <= CONF_MIN)[:, :, None], dm // 4, dm)
    dm = dm[NN_V][:, NN_U]
    # ---- pane 3: points at TILT + skeleton (GPU projection + painter's splat)
    good = C > CONF_MIN
    X = (US - w / 2) / fx * D
    Y = (VS - h / 2) / fy * D
    Dg = D[good]
    if Dg.numel():                                       # numpy-median semantics (mid-pair mean)
        s, k = Dg.sort().values, Dg.numel()
        zmid = float(0.5 * (s[(k - 1) // 2] + s[k // 2]))
    else:
        zmid = 1.0
    ca, sa = float(np.cos(TILT)), float(np.sin(TILT))
    Yt = ca * Y - sa * (D - zmid)
    Zt = (sa * Y + ca * (D - zmid) + zmid).clamp(min=1e-4)
    u2 = (PW / 2 + X / Zt * fx * SCp)[good]
    v2 = (H / 2 + Yt / Zt * fy * SCp - (h * SCp - H) / 2)[good]
    pts = gpu_splat_t(torch.stack([u2, v2], 1), Zt[good], resize(src, h, w)[good], H, PW)
    if ok[n]:
        Yj = ca * Jc[:, 1] - sa * (Jc[:, 2] - zmid)
        Zj = np.maximum(sa * Jc[:, 1] + ca * (Jc[:, 2] - zmid) + zmid, 1e-4)
        ju = PW / 2 + Jc[:, 0] / Zj * fx * SCp
        jv = H / 2 + Yj / Zj * fy * SCp - (h * SCp - H) / 2
        pts = from_np(skel_on(to_np(pts), np.stack([ju, jv], 1), H / 1080 + 0.3))
    # ---- pane 4: BEV (PIL trail/skeleton; trail points precomputed)
    bev = bev_bg.copy()
    bev = trail_on(bev, TRAIL_PTS[:TRAIL_CNT[n]])
    if ok[n]:
        bev = skel_on(bev, np.array([bpx(J_w[n, i, :2])                      # body only: hands
                                     for i in range(min(18, J_w.shape[1]))]),  # are sub-px in BEV
                      H / 1080 + 0.3)
    bev = from_np(bev)
    if LAYOUT == "row":
        return to_np(torch.cat([raw, dm, pts, bev], 1))
    return to_np(torch.cat([torch.cat([tile(raw), tile(pts)], 1),
                            torch.cat([tile(dm), tile(bev)], 1)], 0))


def render_range(n0, n1, out_path):
    wr = VideoWriter(out_path, OW, OH, FPS)
    for n in range(n0, n1):
        wr.write(render_frame(n))
        if (n - n0) % 600 == 0:
            print(f"[collate] {n0}-{n1}: {n - n0}/{n1 - n0}", flush=True)
    wr.close()


if __name__ == "__main__":
    part = os.environ.get("COLLATE_PART")
    final = os.path.join(OUTD, "collate.mp4")
    if part:                                             # worker mode
        n0, n1 = map(int, part.split(":"))
        render_range(n0, n1, os.environ["PART_OUT"])
        sys.exit(0)
    K = int(os.environ.get("WORKERS", "8"))
    if K <= 1:
        render_range(0, N, final)
    else:
        import subprocess
        import tempfile
        tmpd = tempfile.mkdtemp(prefix="collate_")
        bounds = np.linspace(0, N, K + 1).astype(int)
        procs, parts = [], []
        for k in range(K):
            part_out = os.path.join(tmpd, f"part{k:02d}.mp4")
            parts.append(part_out)
            env = dict(os.environ, COLLATE_PART=f"{bounds[k]}:{bounds[k + 1]}",
                       PART_OUT=part_out)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "humanoid_motion_recon.collate_video"], env=env))
        for pr in procs:
            if pr.wait() != 0:
                sys.exit("[collate] worker failed")
        lst = os.path.join(tmpd, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{q}'\n" for q in parts)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
        for q in parts + [lst]:
            os.remove(q)
        os.rmdir(tmpd)
    print(f"[collate] WROTE {final}", flush=True)
