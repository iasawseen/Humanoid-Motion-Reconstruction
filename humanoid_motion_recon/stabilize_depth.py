#!/usr/bin/env python
"""Temporal stabilization of the VGGT-Omega depth stream (static camera).

Jitter sources: per-16-frame-window scale/shift renormalization + per-frame depth noise.
Fix, per frame:
  1. robust affine (a, b): D_n ~ a * D_ref + b over background inliers, where D_ref is the
     per-pixel temporal MEDIAN depth over the video (the robot moves around, so the median at
     each pixel is background); (a, b) median-filtered over time (k=19 straddles window seams);
  2. normalize D'_n = (D_n - b) / a  -> every frame in the canonical normalization;
  3. composite: pixels close to D_ref (static scene) take D_ref exactly (zero jitter);
     dynamic pixels (robot, racks, door) keep their aligned per-frame depth.

Writes <workdir>/depth_stab/f%05d.npz with the same keys (depth, conf, pose_enc), so every
renderer works unchanged by pointing at the stabilized directory.
Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.stabilize_depth
"""
import glob
import os

import numpy as np

WORK = os.environ["DEPTH_WORK"]
SRC = os.path.join(WORK, "depth")
DST = os.path.join(WORK, "depth_stab")
os.makedirs(DST, exist_ok=True)
TAU = 0.022                                              # static/dynamic threshold (raw units)

dfiles = sorted(glob.glob(os.path.join(SRC, "f*.npz")))
N = len(dfiles)

# ---- canonical background: per-pixel temporal median over ~250 sampled frames
samp_idx = np.linspace(0, N - 1, 250).astype(int)
stack = []
for i in samp_idx:
    stack.append(np.load(dfiles[i])["depth"].astype(np.float32))
stack = np.stack(stack)
D_ref = np.median(stack, 0)
del stack
h, w = D_ref.shape
print(f"[stab] reference median depth built ({h}x{w})", flush=True)

# ---- pass 1: per-frame robust affine to the reference
rng = np.random.default_rng(0)
sel = rng.choice(h * w, 30000, replace=False)
ref_flat = D_ref.ravel()[sel].astype(np.float64)
a_arr = np.ones(N); b_arr = np.zeros(N)
for n, f in enumerate(dfiles):
    d = np.load(f)["depth"].astype(np.float64).ravel()[sel]
    keep = np.abs(d - ref_flat) < 0.08                   # generous: excludes the robot region
    for _ in range(2):
        A = np.stack([ref_flat[keep], np.ones(keep.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, d[keep], rcond=None)
        r = d - (a * ref_flat + b)
        keep = np.abs(r) < 2.0 * max(np.std(r[keep]), 1e-4)
    a_arr[n], b_arr[n] = a, b
    if n % 1200 == 0:
        print(f"[stab] affine {n}/{N}", flush=True)
# temporal median of (a, b): straddle the 16-frame windows
k = 19
pad = k // 2
idx = np.arange(N)
a_s = np.array([np.median(a_arr[max(0, i - pad):i + pad + 1]) for i in idx])
b_s = np.array([np.median(b_arr[max(0, i - pad):i + pad + 1]) for i in idx])
print(f"[stab] affine spread: a {100 * a_arr.std():.2f}% -> smoothed {100 * a_s.std():.2f}%",
      flush=True)

# ---- pass 2: normalize + composite + write
for n, f in enumerate(dfiles):
    z = np.load(f)
    D = z["depth"].astype(np.float32)
    Dn = (D - b_s[n]) / a_s[n]
    static = np.abs(Dn - D_ref) < TAU
    Dn[static] = D_ref[static]
    np.savez_compressed(os.path.join(DST, os.path.basename(f)),
                        depth=Dn.astype(np.float16), conf=z["conf"], pose_enc=z["pose_enc"])
    if n % 1200 == 0:
        print(f"[stab] write {n}/{N} (static {static.mean()*100:.0f}%)", flush=True)
print("[stab] DONE", flush=True)
