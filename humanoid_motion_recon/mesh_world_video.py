#!/usr/bin/env python
"""Render the world-lifted, time-smoothed SAM mesh (lift_mesh.py output).

Per frame: [ source frame with the SMOOTHED world mesh projected back through the per-frame
camera | fixed 3/4 world view: smoothed mesh over the scene cloud ]. The left pane is the
apples-to-apples check against the raw per-frame mesh overlay (sam3d_mesh.mp4): if lifting +
smoothing is faithful, the mesh still hugs the subject, now without per-frame jitter.

GPU pipeline (torch CUDA, gpurender): the mesh surface is sampled ONCE (constant topology),
then per frame shaded + z-order-splatted (mesh_splat) directly in the OpenCV camera frame
(x right, y down, z forward - no pyrender axis flips); NVENC encode via fastvid.

Usage (torch CUDA):
  DEPTH_WORK=<cache> MR_OUT=<fit3d dir> [PLAY_FPS=..] python -m humanoid_motion_recon.mesh_world_video
Output: <MR_OUT>/mesh_world.mp4
"""
import glob
import os

import numpy as np
import torch
from PIL import Image

from .fastvid import VideoWriter  # noqa: E402
from . import gpurender as gr  # noqa: E402

WORK = os.environ["DEPTH_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.environ.get("OUT", os.path.join(OUTD, "mesh_world.mp4"))
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
PLAY_FPS = float(os.environ.get("PLAY_FPS", str(FPS)))

mz = np.load(os.path.join(WORK, "mesh_w.npz"))
verts_w, tsec, ok = mz["verts_w"], mz["t"], mz["ok"]
N = len(tsec)
faces = np.load(os.path.join(WORK, "mhr", "faces.npy"))
lz = np.load(os.path.join(OUTD, "lift3d.npz"))
scene_pts = lz["scene"]
fz = np.load(os.path.join(OUTD, "fit3d.npz"))
KAPPA = float(fz["kappa"]) if "kappa" in fz.files else 1.0
b_s = fz["b"]

dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
frames_dir = os.path.join(WORK, "frames")
mdir = os.path.join(WORK, "mhr")
_m0 = np.load(os.path.join(mdir, os.path.basename(dfiles[0])[:-4] + ".npz"))
F_SAM = float(_m0["focal"]) if "focal" in _m0.files else 1468.6
_ff = sorted(glob.glob(os.path.join(frames_dir, "f*.jpg")) +
             glob.glob(os.path.join(frames_dir, "f*.png")))
W_IMG = Image.open(_ff[0]).size[0]

# fixed 3/4 world camera: look at the median subject position from above-side
c0 = np.nanmedian(verts_w[ok].reshape(-1, 3), 0)
diag = np.nanpercentile(scene_pts, 97, axis=0) - np.nanpercentile(scene_pts, 3, axis=0)
R_VIEW = float(np.linalg.norm(diag[:2])) * 0.75 + 1.0
AZ, EL = np.radians(float(os.environ.get("MW_AZ", "135"))), np.radians(35.0)
eye = c0 + R_VIEW * np.array([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)])
fwd = (c0 - eye); fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
POSE_W = np.eye(4)
POSE_W[:3, 0], POSE_W[:3, 1], POSE_W[:3, 2], POSE_W[:3, 3] = right, up, -fwd, eye

# mesh material: matches the old pyrender baseColorFactor (0.75, 0.85, 0.95)
MESH_COLOR = (190, 217, 242)
# one-time surface sampling from the first valid frame (topology is constant); density
# sized for the native-res left pane (subject can fill most of the 720x1280 frame)
F_T, FIDX, BARY = gr.sample_mesh(faces, verts_w[int(np.argmax(ok))], density=24.0)


# static world-view background: scene cloud splatted through the fixed camera
def world_bg(w, h, K):
    Rt = np.linalg.inv(POSE_W)
    pc = scene_pts @ Rt[:3, :3].T + Rt[:3, 3]
    pc = pc[pc[:, 2] < -0.05]                            # in front (pyrender looks down -z)
    z = -pc[:, 2]
    u = (K[0] * pc[:, 0] / z + K[2]).astype(int)
    v = (-K[1] * pc[:, 1] / z + K[3]).astype(int)
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    bg = np.zeros((h, w, 3), np.uint8)
    g = np.clip(80 + 20 / np.sqrt(z[m]), 60, 140).astype(np.uint8)
    bg[v[m], u[m]] = g[:, None]
    return bg


PW, PH = 960, 540
K_W = (700.0, 700.0, PW / 2, PH / 2)
BG_W = gr.from_np(world_bg(PW, PH, K_W))

# fixed world camera, world -> OpenCV camera frame: inverse of POSE_W (a pyrender/GL camera:
# looks down -z, y up), then flip y,z to mesh_splat's convention (y down, z forward)
R_W = POSE_W[:3, :3].astype(np.float32)
EYE_W = POSE_W[:3, 3].astype(np.float32)
FLIP_YZ = np.array([1.0, -1.0, -1.0], np.float32)

writer = None
done = 0
for n in range(N):
    if not ok[n]:
        continue
    fp = os.path.join(frames_dir, f"f{n:05d}.jpg")
    if not os.path.exists(fp):
        fp = os.path.join(frames_dir, f"f{n:05d}.png")
    if not os.path.exists(fp):
        continue
    img = np.asarray(Image.open(fp).convert("RGB"))
    h, w = img.shape[:2]
    dz = np.load(dfiles[n])
    pe = dz["pose_enc"]
    dh, dw = dz["depth"].shape
    fy = (dh / 2.0) / np.tan(pe[7] / 2.0) * (h / dh)
    fx = (dw / 2.0) / np.tan(pe[8] / 2.0) * (w / dw)
    vw = verts_w[n].astype(np.float32)
    # per-frame camera pose in world (from the lift): camera->world
    if "M_c2w" in lz.files:
        Rc, tc = lz["M_c2w"][n], lz["c_c2w"][n]
        sc = np.cbrt(max(np.linalg.det(Rc), 1e-9))
        Rc = Rc / sc                                     # strip the affine scale for the pose
    else:
        sc = float(lz["scale"]); Rc = lz["R_cam2world"]; tc = lz["cam_pos"]
    # reprojection scale invariant (see README): world offsets carry kappa and the
    # VGGT/SAM focal ratio; against the VIDEO they must be body_fix'd about the pelvis.
    # Work in camera frame: Vc = R^-1 (Vw - t)/s, rescale offsets, render with VGGT intrinsics.
    vc = ((vw - tc) @ Rc) / sc
    # exact fixed point: the fitter's ray-point pelvis (kp2d mid-hip ray at body depth b[n])
    mm = np.load(os.path.join(mdir, os.path.basename(dfiles[n])[:-4] + ".npz"))
    up_px = 0.5 * (mm["kp2d"][9] + mm["kp2d"][10]) * (w / W_IMG)
    pelv_c = np.array([(up_px[0] - w / 2) / fx * b_s[n],
                       (up_px[1] - h * 0.5) / fy * b_s[n], b_s[n]], np.float32)
    s_fix = F_SAM * (w / W_IMG) / np.sqrt(fx * fy) / KAPPA
    vc = pelv_c + (vc - pelv_c) * s_fix
    # vc is already in the OpenCV camera frame (x right, y down, z forward) = mesh_splat's
    left = gr.from_np(img)
    gr.mesh_splat(left, torch.as_tensor(vc, dtype=torch.float32, device=gr.DEV),
                  F_T, FIDX, BARY, (fx, fy, w / 2, h / 2), base_color=MESH_COLOR)
    vcw = ((vw - EYE_W) @ R_W) * FLIP_YZ
    right_p = BG_W.clone()
    gr.mesh_splat(right_p, torch.as_tensor(vcw, dtype=torch.float32, device=gr.DEV),
                  F_T, FIDX, BARY, K_W, base_color=MESH_COLOR)
    left = gr.resize(left, int(960 * h / w / 2) * 2, 960)
    pane = torch.cat([left, right_p], 1) if left.shape[0] == PH else \
        torch.cat([gr.resize(left, PH, 960), right_p], 1)
    if writer is None:
        writer = VideoWriter(OUTP, pane.shape[1], pane.shape[0], PLAY_FPS)
    writer.write(gr.to_np(pane))
    done += 1
    if done % 500 == 0:
        print(f"[mesh-world] {done}", flush=True)
writer.close()
print(f"[mesh-world] WROTE {OUTP} ({done} frames)")
