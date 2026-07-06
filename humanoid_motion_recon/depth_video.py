#!/usr/bin/env python
"""Per-frame depth reconstruction of figure.mp4 with VGGT-Omega (facebookresearch/vggt-omega).

Frames are processed in LARGE OVERLAPPING windows (one multi-view forward per window). Each
window is internally consistent; across windows VGGT renormalizes the scene (scale/shift), so
consecutive windows are stitched by a robust affine fitted on the SHARED frames and blended
linearly over the overlap. The result is one consistent depth normalization for the whole video
- no window seams - and the stitching uses only shared-frame correspondence, so it works for
moving cameras too (no static-scene assumption).

Camera extrinsics are chained the same way: each window's per-frame poses live in that window's
own coordinate frame (first frame ~ identity), so consecutive windows are aligned by a sim(3)
estimated on the shared frames (rotation: SVD-averaged relative orientation; scale: the depth
affine 'a'; translation: mean camera-center residual). Every frame gets cam_R/cam_T =
camera-from-CHAIN extrinsics whose units match the stitched depth - this is what makes lifted
skeletons world-consistent under a MOVING camera.

Usage (env kimodo or sam_3d_body, GPU 0):
    CUDA_VISIBLE_DEVICES=0 DEPTH_WORK=<workdir> python -m humanoid_motion_recon.depth_video <ckpt.pt>
Env: VGGT_WINDOW (default 160; 192 fits in 24 GiB), VGGT_STRIDE (default 120).
Outputs: <workdir>/depth/f%05d.npz  {depth[h,w] f16, conf[h,w] f16, pose_enc[9] f32,
                                     cam_R[3,3] f32, cam_T[3] f32}

Image preprocessing is a local PIL+torch replica of vggt_omega.utils.load_fn
.load_and_preprocess_images (bit-identical for its defaults: mode="balanced",
image_resolution=512, patch_size=16) - the upstream helper pulls in torchvision, which the
kimodo env does not have; the VGGTOmega model itself is torchvision-free.
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.environ.get(
    "VGGT_OMEGA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vggt-omega")))
from vggt_omega.models.vggt_omega import VGGTOmega           # noqa: E402


def load_and_preprocess_images(paths, image_resolution=512, patch_size=16):
    """Torchvision-free replica of vggt_omega.utils.load_fn.load_and_preprocess_images
    (mode="balanced"): center-crop extreme aspect ratios into [0.5, 2.0], resize so the
    token count stays near (image_resolution/patch_size)^2, ToTensor (float32/255)."""
    token_number = (image_resolution // patch_size) ** 2
    images, shapes = [], set()
    for p in paths:
        with Image.open(p) as im:
            if im.mode == "RGBA":
                im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
            im = im.convert("RGB")
        w, h = im.size
        ar = h / max(w, 1)
        if ar < 0.5:                                         # too wide -> crop width
            cw = min(w, max(1, int(round(h / 0.5))))
            left = max((w - cw) // 2, 0)
            im = im.crop((left, 0, left + cw, h))
        elif ar > 2.0:                                       # too tall -> crop height
            ch = min(h, max(1, int(round(w * 2.0))))
            top = max((h - ch) // 2, 0)
            im = im.crop((0, top, w, top + ch))
        w, h = im.size
        ar = h / max(w, 1)
        wp = max(1, int(np.round(np.sqrt(token_number / ar))))
        hp = max(1, int(np.round(token_number / np.sqrt(token_number / ar))))
        im = im.resize((wp * patch_size, hp * patch_size), Image.Resampling.BICUBIC)
        t = torch.from_numpy(np.array(im, np.uint8)).permute(2, 0, 1).float().div_(255.0)
        shapes.add((t.shape[1], t.shape[2]))
        images.append(t)
    if len(shapes) > 1:                                      # pad to common size (value 1.0)
        mh = max(s[0] for s in shapes); mw = max(s[1] for s in shapes)
        for k, t in enumerate(images):
            ph, pw = mh - t.shape[1], mw - t.shape[2]
            if ph > 0 or pw > 0:
                images[k] = torch.nn.functional.pad(
                    t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=1.0)
    return torch.stack(images)


WORK = os.environ["DEPTH_WORK"]
CKPT = sys.argv[1]
S = int(os.environ.get("VGGT_WINDOW", "160"))
T = int(os.environ.get("VGGT_STRIDE", "150"))                # 10 shared frames: the robust
# affine + sim(3) sample 4000 px per shared frame - plenty; the old default 120 spent 25%
# of compute on overlap with no measurable stitch gain
OV = S - T                                                   # overlap frames
FSTRIDE = int(os.environ.get("VGGT_FSTRIDE", "1"))           # frame subsampling (FAST: 2).
# Downstream reads ONE body-level depth per frame, median-filtered over k=19 (fit_pose), so
# half-rate depth is near-lossless; skipped frames get nearest-neighbor fill-in npz so every
# consumer sees a dense directory.
RES = int(os.environ.get("VGGT_RES", "512"))

frames_all = sorted(glob.glob(os.path.join(WORK, "frames", "f*.jpg")))
frames = frames_all[::FSTRIDE]
N = len(frames)
outd = os.path.join(WORK, "depth")
os.makedirs(outd, exist_ok=True)

model = VGGTOmega().eval()
state = torch.load(CKPT, map_location="cpu", weights_only=False)
state = state.get("model", state) if isinstance(state, dict) else state
model.load_state_dict(state)
model = model.cuda()
print(f"[depth] {N} frames, window={S} stride={T} overlap={OV}", flush=True)


def robust_affine(x, y):
    keep = np.ones(len(x), bool)
    a, b = 1.0, 0.0
    for _ in range(3):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = y - (a * x + b)
        s_ = np.std(r[keep])
        if s_ < 1e-6:
            break
        keep = np.abs(r) < 2.0 * s_
    return float(a), float(b)


def quat_to_mat(q):                                          # xyzw scalar-last (VGGT convention)
    x, y, z, r = q
    n = x * x + y * y + z * z + r * r
    s = 0.0 if n < 1e-12 else 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * r), s * (x * z + y * r)],
        [s * (x * y + z * r), 1 - s * (x * x + z * z), s * (y * z - x * r)],
        [s * (x * z - y * r), s * (y * z + x * r), 1 - s * (x * x + y * y)]])


prev = {}                                                    # frame idx -> stitched depth (f32)
prev_pose = {}                                               # frame idx -> (cam_R, cam_T) chained
rng = np.random.default_rng(0)
with torch.no_grad():
    for w0 in range(0, N, T):
        batch = frames[w0:w0 + S]
        images = load_and_preprocess_images(batch, image_resolution=RES).cuda()
        pred = model(images)
        depth = pred["depth"][0].float().cpu().numpy()[..., 0]   # [B,h,w]
        conf = pred["depth_conf"][0].float().cpu().numpy()
        pose = pred["pose_enc"][0].float().cpu().numpy()
        del pred, images
        R_new = [quat_to_mat(pose[k][3:7]) for k in range(len(batch))]  # cam-from-window-world
        T_new = [pose[k][:3].astype(np.float64) for k in range(len(batch))]
        # stitch to the running chain via the shared frames
        shared = [i for i in range(w0, w0 + len(batch)) if i in prev]
        if shared:
            xs, ys = [], []
            for i in shared:
                d_new = depth[i - w0].ravel()
                d_prev = prev[i].ravel()
                sel = rng.choice(len(d_new), 4000, replace=False)
                xs.append(d_new[sel]); ys.append(d_prev[sel])
            a, b = robust_affine(np.concatenate(xs).astype(np.float64),
                                 np.concatenate(ys).astype(np.float64))
            # sim(3) window -> chain: p_chain = a * R_rel @ p_win + t_rel
            # rotation: per shared frame R_rel(i) = R_chain_i^T @ R_new_i; SVD-average
            M = np.zeros((3, 3))
            for i in shared:
                M += prev_pose[i][0].T @ R_new[i - w0]
            U, _, Vt = np.linalg.svd(M)
            R_rel = U @ Vt
            if np.linalg.det(R_rel) < 0:
                R_rel = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
            cen_chain = [-prev_pose[i][0].T @ prev_pose[i][1] for i in shared]
            cen_new = [-R_new[i - w0].T @ T_new[i - w0] for i in shared]
            t_rel = np.mean([cc - a * R_rel @ cn for cc, cn in zip(cen_chain, cen_new)], 0)
        else:
            a, b = 1.0, 0.0
            R_rel, t_rel = np.eye(3), np.zeros(3)
        depth = a * depth + b
        # per-frame camera-from-chain: units of the corrected depth (scale folded into T)
        R_ch = [R_new[k] @ R_rel.T for k in range(len(batch))]
        T_ch = [a * T_new[k] - R_ch[k] @ t_rel for k in range(len(batch))]
        # blend the overlap, write everything this window covers
        prev_next = {}
        prev_pose_next = {}
        for k, i in enumerate(range(w0, w0 + len(batch))):
            d = depth[k]
            if i in prev:
                wgt = (k + 1) / (OV + 1)                     # ramps toward the new window
                d = (1.0 - wgt) * prev[i] + wgt * d
            np.savez_compressed(os.path.join(outd, os.path.basename(frames[i])[:-4] + ".npz"),
                                depth=d.astype(np.float16), conf=conf[k].astype(np.float16),
                                pose_enc=pose[k].astype(np.float32),
                                cam_R=R_ch[k].astype(np.float32), cam_T=T_ch[k].astype(np.float32))
            if i >= w0 + T:                                  # will be shared with the next window
                prev_next[i] = d.astype(np.float32)
                prev_pose_next[i] = (R_ch[k], T_ch[k])
        prev = prev_next
        prev_pose = prev_pose_next
        print(f"[depth] {min(w0 + S, N)}/{N}  a={a:.4f} b={b:+.4f}", flush=True)
        if w0 + S >= N:
            break

if FSTRIDE > 1:                                              # nearest-neighbor fill-in
    import shutil
    done = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(outd, "f*.npz"))}
    for i, f in enumerate(frames_all):
        key = os.path.basename(f)[:-4]
        if key in done:
            continue
        src_i = min(range(0, len(frames_all), FSTRIDE), key=lambda j: abs(j - i))
        src = os.path.join(outd, os.path.basename(frames_all[src_i])[:-4] + ".npz")
        shutil.copyfile(src, os.path.join(outd, key + ".npz"))
    print(f"[depth] fill-in complete ({len(frames_all) - N} copied)", flush=True)
print("DONE_DEPTH", flush=True)
