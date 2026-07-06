#!/usr/bin/env python
"""SAM-3D-Body 3D humanoid MESH video: [ frame with mesh overlay | 90-deg side view ].

RAW estimates - no fit, no flips, no smoothing. On the shaped body front vs back is unambiguous
(chest/face vs back geometry), so SAM's front/back mirror errors are directly visible: the
side-view body snaps between facing left and right while the video subject stands still.
If <MR_OUT>/flips.npy exists (fit_pose fitter, MR_FLIPFIX), the frame border shows the fitter's
decision: RED = "this frame is mirrored", GREEN = clean.

CACHE-FIRST: renders from `verts` saved in mhr/f*.npz (pose_video.py with mesh keys) + mhr/faces.npy
- no model load. Frames missing `verts` are skipped (run PV_FORCE=1
python -m humanoid_motion_recon.pose_video to upgrade).

Usage (torch CUDA mesh splatter - no pyrender/cv2): POSE_WORK=<cache> \
    [MR_OUT=...] [PLAY_FPS=10] [FPS_SAMPLE=2.0] [MR_FPS=23.976] python -m humanoid_motion_recon.sam3d_mesh_video
Output: <MR_OUT>/sam3d_mesh.mp4
"""
import glob
import os

import numpy as np
import torch
from PIL import Image

from .fastvid import VideoWriter                                         # noqa: E402
from .gpurender import DEV, canvas, from_np, mesh_splat, sample_mesh, to_np  # noqa: E402

WORK = os.environ["POSE_WORK"]
OUTD = os.environ.get("MR_OUT", "mr_out")
OUTP = os.environ.get("OUT", os.path.join(OUTD, "sam3d_mesh.mp4"))
FPS_SAMPLE = float(os.environ.get("FPS_SAMPLE", "2.0"))   # cache sampling rate
FULL_FPS = float(os.environ.get("MR_FPS", 24000 / 1001))  # flips.npy index rate
PLAY_FPS = float(os.environ.get("PLAY_FPS", "10"))

# pyrender mesh_base_color equivalents, as visual RGB 0-255
C_FRONT = (217, 217, 242)                                # was (0.85, 0.85, 0.95)
C_SIDE = (242, 217, 191)                                 # was (0.95, 0.85, 0.75)
C_FLIP = torch.tensor((255, 60, 60), dtype=torch.uint8, device=DEV)   # red border
C_CLEAN = torch.tensor((60, 200, 90), dtype=torch.uint8, device=DEV)  # green border
# +90 deg about the vertical (y) axis of the camera frame: (x,y,z) -> (z,y,-x),
# same apparent turn as the pyrender side view (subject facing camera -> faces image-left).
RY90 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], device=DEV)

faces = np.load(os.path.join(WORK, "mhr", "faces.npy"))
flips_path = os.path.join(OUTD, "flips.npy")
flips = np.load(flips_path) if os.path.exists(flips_path) else None

mfiles = sorted(glob.glob(os.path.join(WORK, "mhr", "f*.npz")))
writer = None
samp = None                                              # (F, fidx, bary): topology is constant
done = skipped = 0
for mf in mfiles:
    key = os.path.basename(mf)[:-4]
    z = np.load(mf)
    if "verts" not in z.files:
        skipped += 1
        continue
    img_path = os.path.join(WORK, "frames", key + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(WORK, "frames", key + ".jpg")
    if not os.path.exists(img_path):
        continue
    img = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    verts = torch.as_tensor(z["verts"].astype(np.float32), device=DEV)
    cam_t = torch.as_tensor(z["cam_t"].astype(np.float32), device=DEV)
    if samp is None:
        samp = sample_mesh(faces, z["verts"].astype(np.float32))
    F, fidx, bary = samp
    K = (float(z["focal"]), float(z["focal"]), W / 2.0, H / 2.0)
    front = from_np(img)
    mesh_splat(front, verts + cam_t, F, fidx, bary, K, base_color=C_FRONT)
    side = canvas(H, W)
    c = verts.mean(0)                                    # vertical axis through the centroid
    mesh_splat(side, (verts - c) @ RY90.T + c + cam_t, F, fidx, bary, K, base_color=C_SIDE)
    pane = torch.cat([front, side], 1)
    if flips is not None:
        nfull = min(int(round(int(key[1:]) / FPS_SAMPLE * FULL_FPS)), len(flips) - 1)
        col = C_FLIP if flips[nfull] else C_CLEAN                    # RGB: red / green
        pane[:14, :] = col; pane[-14:, :] = col; pane[:, :14] = col; pane[:, -14:] = col
    if writer is None:
        writer = VideoWriter(OUTP, pane.shape[1], pane.shape[0], PLAY_FPS)
    writer.write(to_np(pane))                            # RGB throughout
    done += 1
    if done % 60 == 0:
        print(f"[mesh] {done}/{len(mfiles)}", flush=True)
if writer is not None:
    writer.close()
print(f"[mesh] WROTE {OUTP} ({done} frames, {skipped} lacked verts"
      + (", border = fitter flip decision)" if flips is not None else ")"))
