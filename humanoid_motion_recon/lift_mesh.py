#!/usr/bin/env python
"""Lift SAM-3D-Body MESHES into the world frame and smooth them over time, exactly like the
skeleton fit (fit_pose.py) - same per-frame transform chain, same temporal filters.

Per frame (mirrors fit_pose's rigid re-lift; no recomputation - the fit already persisted its
scale chain): pelvis on the kp2d mid-hip ray at body depth b[n]; vertex offsets = SAM's
camera-frame verts minus kp3d mid-hip, scaled by a[n]; camera->world via per-frame M_c2w (or
the static legacy map); kappa rescale about mid-hip; then per-vertex temporal smoothing
(7-median + 11-box, the fit_pose filters) on GPU. Topology is constant (mhr/faces.npy), so
vertices are in correspondence across frames and smooth like joints.

Figure caveat (see README): SAM's front/back mirror flicker means flicker runs
average mirrored+true vertices - the smoothed mesh smears there until chirality is resolved
at the consumer layer. vera_short (human) is clean.

Usage (sam_3d_body or kimodo env, needs torch):
  DEPTH_WORK=<cache with depth/+mhr/> MR_OUT=<fit3d dir> python -m humanoid_motion_recon.lift_mesh
Output: <DEPTH_WORK>/mesh_w.npz {verts_w [N,V,3] f16, t, ok} (cache dir - too big for MR_OUT)
"""
import glob
import os

import numpy as np
import torch

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
T_MIN = float(os.environ.get("MR_TMIN", "4.0"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fz = np.load(os.path.join(OUTD, "fit3d.npz"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
a_s, b_s, kappa = fz["a"], fz["b"], float(fz["kappa"])

dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
N = len(dfiles)
tsec = np.array([int(os.path.basename(f)[1:6]) for f in dfiles]) / FPS

# frame size of the SAM processed space (kp2d pixels)
from PIL import Image
_f0 = glob.glob(os.path.join(WORK, "frames", "f*.jpg")) + \
      glob.glob(os.path.join(WORK, "frames", "f*.png"))
W_IMG, H_IMG = Image.open(sorted(_f0)[0]).size

V = None
verts_w = None
ok = np.zeros(N, bool)
for n, df in enumerate(dfiles):
    key = os.path.basename(df)[:-4]
    mf = os.path.join(WORK, "mhr", key + ".npz")
    if not os.path.exists(mf) or not np.isfinite(b_s[n]):
        continue
    mz = np.load(mf)
    if "verts" not in mz.files:
        continue
    dz = np.load(df)
    pe = dz["pose_enc"]
    # depth-map resolution + intrinsics (fit_pose convention)
    h, w = dz["depth"].shape
    fy = (h / 2.0) / np.tan(pe[7] / 2.0)
    fx = (w / 2.0) / np.tan(pe[8] / 2.0)
    kp2d, kp3d = mz["kp2d"], mz["kp3d"]
    if np.ptp(kp2d[:, 1]) > 1.5 * H_IMG or np.ptp(kp2d[:, 0]) > 1.5 * W_IMG:
        continue                                         # degenerate SAM frame
    su, sv = w / W_IMG, h / H_IMG
    up = 0.5 * (kp2d[9] + kp2d[10]) * np.array([su, sv])
    pelv = np.array([(up[0] - w / 2) / fx * b_s[n], (up[1] - h / 2) / fy * b_s[n], b_s[n]])
    midc = 0.5 * (kp3d[9] + kp3d[10])
    vc = pelv[None] + a_s[n] * (mz["verts"].astype(np.float32) - midc[None])
    if V is None:
        V = vc.shape[0]
        verts_w = np.full((N, V, 3), np.nan, np.float32)
    if "M_c2w" in lz:
        vw = vc @ lz["M_c2w"][n].T + lz["c_c2w"][n]
        pw = lz["M_c2w"][n] @ pelv + lz["c_c2w"][n]
    else:
        vw = float(lz["scale"]) * (vc @ lz["R_cam2world"].T) + lz["cam_pos"]
        pw = float(lz["scale"]) * (lz["R_cam2world"] @ pelv) + lz["cam_pos"]
    # kappa rescale about the lifted pelvis (world-space body-size calib, fit_pose convention;
    # the pelvis is the mapped mid-hip ray point by construction, so it is the fixed point)
    verts_w[n] = pw[None] + kappa * (vw - pw[None])
    ok[n] = True
    if n % 1000 == 0:
        print(f"[mesh-lift] {n}/{N}", flush=True)

# temporal smoothing on GPU: per-vertex-coord, 7-median then 11-box (fit_pose filters),
# gaps interpolated for filtering, kept invalid in the output
print("[mesh-lift] smoothing on", DEV)
oki = np.flatnonzero(ok)
t_idx = torch.arange(N, device=DEV, dtype=torch.float32)
X = torch.from_numpy(verts_w).to(DEV)                    # [N,V,3]
Xi = X.clone()
# linear gap interpolation along time (vectorized: searchsorted on valid indices)
vi = torch.from_numpy(oki).to(DEV)
pos = torch.searchsorted(vi, torch.arange(N, device=DEV).contiguous()).clamp(1, len(vi) - 1)
lo, hi = vi[pos - 1], vi[pos]
wgt = ((t_idx - lo.float()) / (hi - lo).float().clamp(min=1)).clamp(0, 1)[:, None, None]
Xi = X[lo] * (1 - wgt) + X[hi] * wgt
CH = 2048
for v0 in range(0, Xi.shape[1], CH):
    blk = Xi[:, v0:v0 + CH].reshape(N, -1).T             # [C,N]
    p = torch.nn.functional.pad(blk[None], (3, 3), mode="replicate")[0]
    blk = p.unfold(1, 7, 1).median(dim=2).values
    p = torch.nn.functional.pad(blk[None], (5, 5), mode="replicate")[0]
    blk = p.unfold(1, 11, 1).mean(dim=2)
    Xi[:, v0:v0 + CH] = blk.T.reshape(N, -1, 3)
sm = Xi.cpu().numpy()
sm[~ok] = np.nan
ok &= tsec >= T_MIN
outp = os.path.join(WORK, "mesh_w.npz")
np.savez_compressed(outp, verts_w=sm.astype(np.float16), t=tsec, ok=ok)
print(f"[mesh-lift] WROTE {outp} ({int(ok.sum())}/{N} frames, V={V})")
