#!/usr/bin/env python
"""Convert NVIDIA soma-retargeter G1 CSV output to a 36-col MuJoCo qpos CSV
(root xyz + wxyz quat + 29 hinges, Unitree G1 joint order).

Companion to export_soma_bvh: fit3d -> BVH -> (soma-retargeter) -> CSV -> this -> qpos.

Their CSV: Frame, root_translateXYZ (cm, Z-up), root_rotateXYZ (deg extrinsic-xyz euler,
soma_retargeter/assets/csv.py), 29 G1 DOF (deg) in exactly our g1.xml hinge order. Ours:
root xyz (m) + wxyz quat + 29 hinge rad, in the MotionRecon fit world (+X forward).
Their "Mujoco" facing conversion is Rx(+90): BVH (x,y,z) -> robot (x,-z,y), which puts
BVH-forward (+Z, the SOMA sample convention our exporter matches) on -Y; we yaw the whole
world by +90 deg to put it back on +X.

Optional SRC_FPS/DST_FPS env vars resample the output timeline (the BVH is upsampled to
120 fps for the retargeter's per-frame smoothing; resample back to the scenario fps here).

Usage: python -m humanoid_motion_recon.import_soma_csv in.csv out.csv [euler_seq (default xyz)]
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

inp, outp = sys.argv[1], sys.argv[2]
seq = sys.argv[3] if len(sys.argv) > 3 else "xyz"

raw = np.genfromtxt(inp, delimiter=",", skip_header=1)
SRC, DST = float(os.environ.get("SRC_FPS", "0")), float(os.environ.get("DST_FPS", "0"))
if SRC and DST and abs(SRC - DST) > 1e-3:
    t_src = np.arange(len(raw)) / SRC
    t_dst = np.arange(int(np.ceil(t_src[-1] * DST - 1e-9)) + 1) / DST  # keep the last frame
    lin = np.stack([np.interp(t_dst, t_src, raw[:, c]) for c in range(raw.shape[1])], 1)
    sl = Slerp(t_src, Rotation.from_euler(seq, raw[:, 4:7], degrees=True))
    lin[:, 4:7] = sl(np.clip(t_dst, t_src[0], t_src[-1])).as_euler(seq, degrees=True)
    raw = lin
Rz90 = Rotation.from_euler("z", 90, degrees=True)
pos = Rz90.apply(raw[:, 1:4] / 100.0)
quat = (Rz90 * Rotation.from_euler(seq, raw[:, 4:7], degrees=True)).as_quat()  # xyzw
quat = quat[:, [3, 0, 1, 2]]                                          # -> wxyz
dofs = np.radians(raw[:, 7:36])
qpos = np.concatenate([pos, quat, dofs], 1)
np.savetxt(outp, qpos, delimiter=",", fmt="%.6f")
print(f"[csv2qpos] {qpos.shape} {inp} -> {outp} (euler {seq})")
