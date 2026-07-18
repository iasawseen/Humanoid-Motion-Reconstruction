#!/usr/bin/env python
"""Ground a robot qpos trajectory: put the planted foot soles on z=0.

The NVIDIA soma-retargeter scales SOMA Hips targets by a calibration measured on the
retargeter's own zero-pose template; a source BVH whose standing pelvis sits higher than
that template makes the whole solved trajectory ride a few cm above the retargeter's
floor — feet rigid (the stabilizer holds them) but hovering. Since the subject world's
floor is exactly z=0 (heels-=-floor calibration), the hover shows up in replays and
ghost overlays as walking on air.

Fix: FK every frame, take the lowest world-AABB corner of the foot collision geoms
(bodies matching ankle|foot, geoms with contype!=0 or 'collision' in the name), and
subtract a constant = the PCTL percentile (default median) of that per-frame minimum.
Walking always has a planted foot, so per-frame min-z IS the sole height; the median is
robust to swing-phase and penetration outliers. A constant shift preserves the motion.

Usage: ROBOT_XML=<robot mjcf> [PCTL=50] \
           python -m humanoid_motion_recon.ground_qpos in_qpos.csv out_qpos.csv
qpos CSV layout: root xyz + wxyz quat + hinges (as written by import_soma_csv).
"""
import os
import sys

import numpy as np
import mujoco

inp, outp = sys.argv[1], sys.argv[2]
PCTL = float(os.environ.get("PCTL", "50"))

m = mujoco.MjModel.from_xml_path(os.environ["ROBOT_XML"])
d = mujoco.MjData(m)
qpos = np.loadtxt(inp, delimiter=",")

foot_bodies = [b for b in range(m.nbody)
               if any(k in (m.body(b).name or "") for k in ("ankle", "foot"))]
geoms = [g for g in range(m.ngeom)
         if m.geom_bodyid[g] in foot_bodies
         and (m.geom_contype[g] != 0 or "collision" in (m.geom(g).name or ""))]
assert geoms, "no foot collision geoms found (bodies matching ankle|foot)"
print(f"[ground] foot geoms: {[m.geom(g).name or f'#{g}' for g in geoms]}")

corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                   dtype=np.float64)
minz = np.empty(len(qpos))
nq = min(m.nq, qpos.shape[1])
for n in range(len(qpos)):
    d.qpos[:] = 0.0
    d.qpos[:nq] = qpos[n, :nq]
    mujoco.mj_kinematics(m, d)
    lo = np.inf
    for g in geoms:
        c, s = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        w = (c + corners * s) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lo = min(lo, w[:, 2].min())
    minz[n] = lo

off = float(np.percentile(minz, PCTL))
qpos = qpos.copy()
qpos[:, 2] -= off
np.savetxt(outp, qpos, delimiter=",", fmt="%.6f")
print(f"[ground] sole min-z: mean {minz.mean():.4f} std {minz.std():.4f} "
      f"p5/p50/p95 {np.percentile(minz, 5):.4f}/{np.percentile(minz, 50):.4f}/"
      f"{np.percentile(minz, 95):.4f}")
print(f"[ground] offset {off:.4f} m (PCTL {PCTL:.0f}); {inp} -> {outp} ({len(qpos)} frames)")
