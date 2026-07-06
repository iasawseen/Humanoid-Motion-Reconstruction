#!/usr/bin/env python
"""One-pass MotionRecon MESH collate: [raw + SAM mesh | depth | points + world mesh | BEV mesh].

Identical pane layout, sizes, colors, depth pane, BEV background/trail, timeline and env
semantics as collate_video.py - imported as a module, so its module-level state (pane
geometry, depth colormap, BEV mapping, trail precompute, world->camera maps, F_SAM/kappa)
IS this renderer's state - but every skeleton overlay is replaced by the body MESH:

- RAW pane: the raw per-frame SAM mesh (mhr `verts` + `cam_t` in the SAM camera frame,
  npz `focal`, cx/cy = frame centre - the sam3d_mesh_video construction) splatted over the
  source frame. No body_fix: raw SAM in its own frame with its own focal projects correctly.
- POINTS pane: the SMOOTHED world mesh (<DEPTH_WORK>/mesh_w.npz) through the SAME
  world->camera map + body_fix rescale (about the fit3d camera-frame mid-hip, same fixed
  point and scale as collate_video.body_fix; see README) the skeleton used. The
  pane's tilt projection is a rigid rotation about the x-axis through (y=0, z=zmid), so the
  vertices are rotated into the tilted frame and gpurender.mesh_splat's pinhole projection
  reproduces the pane's exact formulas - shaded surface samples, z-ordered against
  themselves, drawn over the cloud.
- BEV pane: cloud background + fading pelvis trail unchanged; the world mesh surface samples
  go through the pane's existing BEV pixel mapping, shaded top-down (headlight |n_z|),
  z-sorted so HIGHER points win (splat_cloud with sort key -z).

Mesh surface sampling (gpurender.sample_mesh) runs ONCE - MHR topology is constant; the
area-weighted sample counts are scale-invariant, so one sampling serves raw + world meshes.
Worker-parallel exactly like collate_video (WORKERS chunks, lossless -c copy concat).

Usage: DEPTH_WORK=<workdir> MR_OUT=<outdir> [MR_FPS=..] [TILT_DEG=33] [WORKERS=8] \\
       [LAYOUT=row|grid] [MESH_DENSITY=14] python -m humanoid_motion_recon.collate_mesh_video
Output: <MR_OUT>/collate_mesh.mp4
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

from . import collate_video as cw                        # noqa: E402  shared pane state
from . import gpurender as gr                            # noqa: E402
from .fastvid import VideoWriter                         # noqa: E402
from .skel_draw import I                                 # noqa: E402

DENS = float(os.environ.get("MESH_DENSITY", "14"))
AMB = 0.45                                               # mesh_splat's ambient term
C_RAW = (217, 217, 242)                                  # sam3d_mesh_video front color
C_WORLD = (190, 217, 242)                                # mesh_world_video smoothed color

mz = np.load(os.path.join(cw.WORK, "mesh_w.npz"))
VERTS_W, MOK = mz["verts_w"], mz["ok"]
faces = np.load(os.path.join(cw.WORK, "mhr", "faces.npy"))
# one-time surface sampling (constant topology); seed verts = first valid world frame
F_T, FIDX, BARY = gr.sample_mesh(faces, VERTS_W[int(np.argmax(MOK))].astype(np.float32),
                                 density=DENS)


def world2cam(V, n):
    """The same world->camera map collate_video applies to joints_w (row-vector form)."""
    if hasattr(cw, "Minv"):                              # moving camera: per-frame inverse
        return (V - cw.lz["c_c2w"][n]) @ cw.Minv[n].T
    return ((V - cw.cam_pos) / cw.scale) @ cw.Rcw


def mesh_surf(verts_t):
    """Surface sample points + unit face normals (mesh_splat's own construction)."""
    tri = verts_t[F_T[FIDX]]                             # [M,3,3]
    pts = (tri * BARY[:, :, None]).sum(1)
    fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return pts, fn / (fn.norm(dim=1, keepdim=True) + 1e-9)


def render_frame(n):
    dz = np.load(cw.dfiles[n])
    D = gr.from_np(dz["depth"].astype(np.float32))
    C = gr.from_np(dz["conf"].astype(np.float32))
    pe = dz["pose_enc"]
    fy = float((cw.h / 2.0) / np.tan(pe[7] / 2.0))
    fx = float((cw.w / 2.0) / np.tan(pe[8] / 2.0))
    src = gr.from_np(np.asarray(Image.open(os.path.join(cw.WORK, "frames", f"f{n:05d}.jpg"))))
    world_ok = bool(cw.ok[n]) and bool(MOK[n])
    # ---- pane 1: raw + RAW SAM mesh (own camera frame, own focal - no body_fix)
    raw = gr.resize(src, cw.H, cw.RW)
    mf = os.path.join(cw.WORK, "mhr", os.path.basename(cw.dfiles[n]))
    mm = np.load(mf) if os.path.exists(mf) else None
    if mm is not None and "verts" in mm.files:
        v = torch.as_tensor(mm["verts"].astype(np.float32) + mm["cam_t"].astype(np.float32),
                            device=gr.DEV)
        f_s = float(mm["focal"])                         # processed-frame px; scale per axis
        gr.mesh_splat(raw, v, F_T, FIDX, BARY,
                      (f_s * cw.RW / cw.FW, f_s * cw.H / cw.FH, cw.RW / 2.0, cw.H / 2.0),
                      base_color=C_RAW)
    # ---- pane 2: depth (identical to collate_video)
    idx = ((D - cw.D0) / (cw.D1 - cw.D0) * 255).clamp(0, 255).to(torch.uint8)
    dm = cw.TURBO_T[idx.long()]
    dm = torch.where((C <= cw.CONF_MIN)[:, :, None], dm // 4, dm)
    dm = dm[cw.NN_V][:, cw.NN_U]
    # ---- pane 3: points at TILT + SMOOTHED WORLD mesh
    good = C > cw.CONF_MIN
    X = (cw.US - cw.w / 2) / fx * D
    Y = (cw.VS - cw.h / 2) / fy * D
    Dg = D[good]
    if Dg.numel():                                       # numpy-median semantics
        s, k = Dg.sort().values, Dg.numel()
        zmid = float(0.5 * (s[(k - 1) // 2] + s[k // 2]))
    else:
        zmid = 1.0
    ca, sa = float(np.cos(cw.TILT)), float(np.sin(cw.TILT))
    Yt = ca * Y - sa * (D - zmid)
    Zt = (sa * Y + ca * (D - zmid) + zmid).clamp(min=1e-4)
    u2 = (cw.PW / 2 + X / Zt * fx * cw.SCp)[good]
    v2 = (cw.H / 2 + Yt / Zt * fy * cw.SCp - (cw.h * cw.SCp - cw.H) / 2)[good]
    pts = cw.gpu_splat_t(torch.stack([u2, v2], 1), Zt[good],
                         gr.resize(src, cw.h, cw.w)[good], cw.H, cw.PW)
    if world_ok:
        Vc = world2cam(VERTS_W[n].astype(np.float64), n)
        # body_fix, same fixed point + scale as collate_video.body_fix on the joints: the
        # mesh rescales about the JOINTS' camera-frame mid-hip so it lands where the
        # skeleton's (kp2d-exact) pelvis pixel was.
        pelv = 0.5 * (cw.J_cam[n][I["lhip"]] + cw.J_cam[n][I["rhip"]])
        Vc = pelv + (Vc - pelv) * (cw.F_SAM * (cw.w / cw.FW) / np.sqrt(fx * fy) / cw.KAPPA)
        # rigid tilt about the x-axis through (y=0, z=zmid): after it the pane's tilt
        # projection IS a pinhole, so mesh_splat reproduces u2/v2 exactly (and z-orders
        # the mesh against itself, over the already-drawn cloud)
        Vt3 = np.stack([Vc[:, 0],
                        ca * Vc[:, 1] - sa * (Vc[:, 2] - zmid),
                        sa * Vc[:, 1] + ca * (Vc[:, 2] - zmid) + zmid], 1)
        gr.mesh_splat(pts, torch.as_tensor(Vt3, dtype=torch.float32, device=gr.DEV),
                      F_T, FIDX, BARY,
                      (fx * cw.SCp, fy * cw.SCp, cw.PW / 2.0,
                       cw.H / 2.0 - (cw.h * cw.SCp - cw.H) / 2.0),
                      base_color=C_WORLD)
    # ---- pane 4: BEV - background + trail unchanged, mesh top-down (higher points win)
    bev = cw.bev_bg.copy()
    bev = gr.from_np(cw.trail_on(bev, cw.TRAIL_PTS[:cw.TRAIL_CNT[n]]))
    if world_ok:
        p3, fn = mesh_surf(torch.as_tensor(VERTS_W[n].astype(np.float32), device=gr.DEV))
        shade = AMB + (1 - AMB) * fn[:, 2].abs().clamp(0, 1)
        cols = (torch.tensor(C_WORLD, dtype=torch.float32, device=gr.DEV)[None]
                * shade[:, None]).clamp(0, 255).to(torch.uint8)
        uv = torch.stack([cw.BW / 2 + (p3[:, 0] - cw.bcx) * cw.bscale,
                          cw.H / 2 - (p3[:, 1] - cw.bcy) * cw.bscale], 1)  # == cw.bpx
        gr.splat_cloud(bev, uv, -p3[:, 2], cols, gr.DISK0)  # -z: highest drawn last -> wins
    if cw.LAYOUT == "row":
        return gr.to_np(torch.cat([raw, dm, pts, bev], 1))
    return gr.to_np(torch.cat([torch.cat([cw.tile(raw), cw.tile(pts)], 1),
                               torch.cat([cw.tile(dm), cw.tile(bev)], 1)], 0))


def render_range(n0, n1, out_path):
    wr = VideoWriter(out_path, cw.OW, cw.OH, cw.FPS)
    for n in range(n0, n1):
        wr.write(render_frame(n))
        if (n - n0) % 600 == 0:
            print(f"[collate-mesh] {n0}-{n1}: {n - n0}/{n1 - n0}", flush=True)
    wr.close()


if __name__ == "__main__":
    part = os.environ.get("COLLATE_PART")
    final = os.path.join(cw.OUTD, "collate_mesh.mp4")
    if part:                                             # worker mode
        n0, n1 = map(int, part.split(":"))
        render_range(n0, n1, os.environ["PART_OUT"])
        sys.exit(0)
    K = int(os.environ.get("WORKERS", "8"))
    if K <= 1:
        render_range(0, cw.N, final)
    else:
        import subprocess
        import tempfile
        tmpd = tempfile.mkdtemp(prefix="collate_mesh_")
        bounds = np.linspace(0, cw.N, K + 1).astype(int)
        procs, parts = [], []
        for k in range(K):
            part_out = os.path.join(tmpd, f"part{k:02d}.mp4")
            parts.append(part_out)
            env = dict(os.environ, COLLATE_PART=f"{bounds[k]}:{bounds[k + 1]}",
                       PART_OUT=part_out)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "humanoid_motion_recon.collate_mesh_video"], env=env))
        for pr in procs:
            if pr.wait() != 0:
                sys.exit("[collate-mesh] worker failed")
        lst = os.path.join(tmpd, "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{q}'\n" for q in parts)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
        for q in parts + [lst]:
            os.remove(q)
        os.rmdir(tmpd)
    print(f"[collate-mesh] WROTE {final}", flush=True)
