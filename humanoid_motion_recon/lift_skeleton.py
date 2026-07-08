#!/usr/bin/env python
"""Lift the SAM-3D-Body 2D skeletons of figure.mp4 into 3D using VGGT-Omega depth, and render a
bird's-eye view of the skeletons through the video.

Per frame: joint_3d = depth(u,v) * K^-1 [u,v,1] in the (static) camera frame; depth is sampled
as a confidence-gated 5x5 median around each keypoint, with the pelvis depth as fallback for
occluded joints. The world frame is gravity-aligned by fitting the floor plane on the robot-free
title frames; the floor normal becomes +Z(up), and the counter-run direction (dominant horizontal
wall orientation) becomes +X.

Outputs (in <workdir>):
    lift3d.npz        joints_w [N,21,3] world (gravity-aligned, floor z=0), t [N] seconds, ok [N]
    birdseye.png      top-down poses + pelvis trajectory, time-colored
Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.lift_skeleton
"""
import glob
import os
import sys

import numpy as np

WORK = os.environ["DEPTH_WORK"]
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
T_MIN = float(os.environ.get("MR_TMIN", "4.0"))          # discard unreliable head of the video
# Calibration windows. MR_AUTO=1 auto-detects them from the subject (leg-extension ratio
# = upright, + pose stillness = standing; dominant upright facing = yaw anchor; heel-plane
# up leveling). Default (MR_AUTO unset) keeps the env-var windows below - auto mode is
# validated for cross-resolution stability (~3 cm) but not yet at parity with hand-tuned
# windows (see bench/): the lift-space vs fit-space heel discrepancy is unresolved.
AUTO = os.environ.get("MR_AUTO", "0") == "1"
STAND_SET = not AUTO or "MR_STAND0" in os.environ or "MR_STAND1" in os.environ
YAW_SET = not AUTO or "MR_YAW0" in os.environ or "MR_YAW1" in os.environ
UP_SET = not AUTO or "MR_UP0" in os.environ or "MR_UP1" in os.environ
STAND0 = float(os.environ.get("MR_STAND0", "7.5"))       # upright-stance calibration window
STAND1 = float(os.environ.get("MR_STAND1", "9.5"))
YAW0 = float(os.environ.get("MR_YAW0", "8.0"))           # facing-anchor window (+X)
YAW1 = float(os.environ.get("MR_YAW1", "13.5"))
UP0 = float(os.environ.get("MR_UP0", "2.6"))             # torso-up estimation window
UP1 = float(os.environ.get("MR_UP1", "9.5"))
PELVIS_H = float(os.environ.get("MR_PELVIS_H", "0.72"))  # standing pelvis height (world units)
from PIL import Image as _Image
import glob as _glob
W_IMG, H_IMG = _Image.open(sorted(_glob.glob(os.path.join(WORK, "frames", "f*.jpg")))[0]).size
KP = {"nose": 0, "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lhip": 9, "rhip": 10,
      "lkne": 11, "rkne": 12, "lank": 13, "rank": 14, "lbtoe": 15, "lheel": 17,
      "rbtoe": 18, "rheel": 20, "rwri": 41, "lwri": 62, "neck": 69}
KIDX = list(KP.values())
BONES = [("neck", "nose"), ("lsho", "rsho"), ("neck", "lsho"), ("neck", "rsho"),
         ("lsho", "lelb"), ("lelb", "lwri"), ("rsho", "relb"), ("relb", "rwri"),
         ("lhip", "rhip"), ("neck", "lhip"), ("neck", "rhip"),
         ("lhip", "lkne"), ("lkne", "lank"), ("rhip", "rkne"), ("rkne", "rank")]


def quat_to_mat(q):
    x, y, z, w = q  # vggt convention: check; assume xyzw
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n < 1e-12 else 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def cam_of(pose_enc, h, w):
    fov_h, fov_w = pose_enc[7], pose_enc[8]
    fy = (h / 2.0) / np.tan(fov_h / 2.0)
    fx = (w / 2.0) / np.tan(fov_w / 2.0)
    return fx, fy, w / 2.0, h / 2.0


def sample_depth(D, C, u, v, conf_min=1.5):
    h, w = D.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return np.nan
    u0, u1 = max(0, ui - 2), min(w, ui + 3)
    v0, v1 = max(0, vi - 2), min(h, vi + 3)
    d = D[v0:v1, u0:u1].astype(np.float64).ravel()
    c = C[v0:v1, u0:u1].astype(np.float64).ravel()
    good = c > conf_min
    if good.sum() < 3:
        return np.nan
    return float(np.percentile(d[good], 25))             # near side: the robot is always the
                                                         # closest surface on its own silhouette


def main():
    dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
    N = len(dfiles)
    joints_cam = np.full((N, len(KIDX), 3), np.nan)
    kp3d_all = np.full((N, len(KIDX), 3), np.nan)        # SAM camera-frame kps (calibration)
    kp2d_all = np.full((N, len(KIDX), 2), np.nan)
    cams = np.full((N, 6), np.nan)
    d_body_raw = np.full(N, np.nan)
    djoint_raw = np.full((N, len(KIDX)), np.nan)
    tsec = np.zeros(N)
    # per-frame camera-from-chain extrinsics (moving camera); identity when depth_video
    # predates the pose chaining (static-camera runs like figure)
    camR = np.tile(np.eye(3), (N, 1, 1))
    camT = np.zeros((N, 3))
    for n, df in enumerate(dfiles):
        key = os.path.basename(df)[:-4]
        k = int(key[1:])
        tsec[n] = k / FPS
        with np.load(df) as _dz:
            if "cam_R" in _dz:
                camR[n] = _dz["cam_R"].astype(np.float64)
                camT[n] = _dz["cam_T"].astype(np.float64)
        mf = os.path.join(WORK, "mhr", key + ".npz")
        if not os.path.exists(mf):
            continue
        dz = np.load(df)
        D, C, pe = dz["depth"].astype(np.float32), dz["conf"].astype(np.float32), dz["pose_enc"]
        h, w = D.shape
        fx, fy, cx, cy = cam_of(pe, h, w)
        su, sv = w / W_IMG, h / H_IMG
        mz = np.load(mf)
        kp2d = mz["kp2d"]
        kp2d_all[n] = kp2d[KIDX]
        kp3d_all[n] = mz["kp3d"][KIDX]
        cams[n] = [fx, fy, cx, cy, su, sv]
        # body depth: hips only (widest part of the silhouette; neck/shoulders bleed at edges)
        core = [KP["lhip"], KP["rhip"]]
        dcore = [sample_depth(D, C, kp2d[i][0] * su, kp2d[i][1] * sv) for i in core]
        dcore = [d for d in dcore if np.isfinite(d)]
        if dcore:
            d_body_raw[n] = float(np.median(dcore))
        djoint = np.full(len(KIDX), np.nan)
        for j, ki in enumerate(KIDX):
            d = sample_depth(D, C, kp2d[ki][0] * su, kp2d[ki][1] * sv)
            if np.isfinite(d):
                djoint[j] = d
        djoint_raw[n] = djoint

    # temporal median of the body depth (static camera, slow motion), then per-joint band
    d_body = np.array([np.nanmedian(d_body_raw[max(0, n - 5):n + 6]) for n in range(N)])
    for n in range(N):
        if not np.isfinite(d_body[n]) or not np.isfinite(cams[n, 0]):
            continue
        fx, fy, cx, cy, su, sv = cams[n]
        band = 0.08 * d_body[n]
        for j in range(len(KIDX)):
            d = djoint_raw[n, j]
            if not np.isfinite(d) or abs(d - d_body[n]) > band:
                d = d_body[n]
            u, v = kp2d_all[n, j, 0] * su, kp2d_all[n, j, 1] * sv
            joints_cam[n, j] = [(u - cx) / fx * d, (v - cy) / fy * d, d]

    # camera frame -> chain frame (one consistent frame across camera motion):
    # p_chain = cam_R^T @ (p_cam - cam_T); identity for static-camera runs
    joints_cam = np.einsum("nji,nkj->nki", camR, joints_cam - camT[:, None])

    # ---- auto-calibration (when MR_STAND*/MR_UP*/MR_YAW* are unset): detect upright and
    # standing frames from the subject itself. Leg-extension ratio (hip->heel distance over
    # summed leg segment lengths, from SAM kp3d) is gravity-free, so there is no
    # chicken-and-egg with the unknown up vector; stillness = pelvis-centred pose velocity.
    ki = {name: KIDX.index(KP[name]) for name in KP}
    midhip3 = 0.5 * (kp3d_all[:, ki["lhip"]] + kp3d_all[:, ki["rhip"]])
    rel3 = kp3d_all - midhip3[:, None]
    def _ext(side):
        L = (np.linalg.norm(rel3[:, ki[side + "hip"]] - rel3[:, ki[side + "kne"]], axis=1)
             + np.linalg.norm(rel3[:, ki[side + "kne"]] - rel3[:, ki[side + "ank"]], axis=1)
             + np.linalg.norm(rel3[:, ki[side + "ank"]] - rel3[:, ki[side + "heel"]], axis=1))
        return np.linalg.norm(rel3[:, ki[side + "hip"]] - rel3[:, ki[side + "heel"]], axis=1) /             np.maximum(L, 1e-6)
    # STANCE-side geometry: during walking only the stance leg is straight and planted, so
    # per-frame we score the more-extended side (mean-of-sides has no upright frames at all
    # on walking-only clips; mid-heel alignment likewise breaks during stride).
    ext_l, ext_r = _ext("l"), _ext("r")
    ext = np.maximum(ext_l, ext_r)
    stance_side = np.where(ext_l >= ext_r, ki["lheel"], ki["rheel"])
    # straight legs are NOT enough: the subject bends at the hips with straight legs (rack
    # work), which tilts the torso while ext stays high. Standing straight means torso
    # anti-parallel to the (stance) leg - frame-internal, no gravity needed.
    torso3 = rel3[:, ki["neck"]]
    legdir3 = rel3[np.arange(N), stance_side]
    align = -np.einsum("ni,ni->n", torso3, legdir3) / np.maximum(
        np.linalg.norm(torso3, axis=1) * np.linalg.norm(legdir3, axis=1), 1e-9)
    vel = np.full(N, np.nan)
    dv = np.linalg.norm(np.diff(rel3, axis=0), axis=2)   # [N-1, K]
    vel[1:] = np.nanmedian(dv, axis=1) * FPS
    e95 = np.nanpercentile(ext, 95)
    upright = (ext > 0.92 * e95) & (align > 0.90)        # straight stance leg, unbent torso
    still = vel < np.nanpercentile(vel[np.isfinite(vel)], 35)
    # heels must be depth-VISIBLE, not inpainted: their sampled depth has to sit inside the
    # body band (occluded heels - e.g. behind the open dishwasher door - read scene depth and
    # poison the floor estimate; an auto window at t~30 s cost 7 cm of floor height)
    heel_j = [KIDX.index(KP["lheel"]), KIDX.index(KP["rheel"])]
    heels_seen = np.isfinite(djoint_raw[:, heel_j]).all(1)
    heels_seen &= np.abs(djoint_raw[:, heel_j] - d_body[:, None]).max(1) < 0.06 * np.where(
        np.isfinite(d_body), d_body, np.inf)
    # feet together matters: leg splay lowers the pelvis even with straight legs and an
    # unbent torso (work stance), which dilutes the pelvis-height median -> wrong kappa
    leglen = 0.5 * (np.linalg.norm(rel3[:, ki["lhip"]] - rel3[:, ki["lheel"]], axis=1)
                    + np.linalg.norm(rel3[:, ki["rhip"]] - rel3[:, ki["rheel"]], axis=1))
    width = np.linalg.norm(rel3[:, ki["lheel"]] - rel3[:, ki["rheel"]], axis=1)
    narrow = width < 0.45 * np.maximum(leglen, 1e-6)
    standable = upright & (ext > 0.96 * e95) & still & (tsec >= T_MIN) & heels_seen & narrow
    if standable.sum() < max(5, int(0.5 * FPS)):
        # heel depth-band visibility is an occlusion guard, not a hard requirement - on
        # low-confidence floors (dark studio) VGGT never confirms the heels and the guard
        # would starve the stance set (per-joint depth falls back to the body depth anyway)
        relaxed = upright & (ext > 0.96 * e95) & still & (tsec >= T_MIN) & narrow
        if relaxed.sum() > standable.sum():
            print(f"[lift] auto stance: heel-visibility guard relaxed "
                  f"({int(standable.sum())} -> {int(relaxed.sum())} frames)")
            standable = relaxed
    # window choice is deferred until world joints exist: candidate windows are ranked by
    # how well their implied FLOOR generalizes to all other standing frames (an occluder in
    # front of the feet - e.g. the pulled-out dishwasher rack - passes the depth-band check
    # but yields a floor that strands every other standing frame's heels ~15 cm in the air)
    stand_auto = None                                    # resolved below, needs Jw
    up_auto = upright.copy()
    if up_auto.sum() < 10:
        print("[lift] WARN: too few upright frames - falling back to MR_UP defaults")
        up_auto = (tsec >= UP0) & (tsec <= UP1)

    # ---- gravity alignment: floor plane from robot-free title frames (dense unprojection)
    pts = []
    for k in range(0, 4):
        df = os.path.join(WORK, "depth", f"f{k:05d}.npz")
        if not os.path.exists(df):
            continue
        dz = np.load(df)
        D, C, pe = dz["depth"].astype(np.float32), dz["conf"].astype(np.float32), dz["pose_enc"]
        h, w = D.shape
        fx, fy, cx, cy = cam_of(pe, h, w)
        vs, us = np.mgrid[0:h:4, 0:w:4]
        d = D[::4, ::4]
        c = C[::4, ::4]
        m = c > 2.0
        pc = np.stack([(us[m] - cx) / fx * d[m], (vs[m] - cy) / fy * d[m], d[m]], 1)
        pts.append((pc - camT[k]) @ camR[k])             # into the chain frame
    P = np.concatenate(pts, 0)
    # UP from the robot itself: the walking/standing torso is the most reliable plumb line here
    # (VGGT depth on the textureless floor is bowed - a floor-plane normal came out ~30 deg off,
    # measured by the standing robot's torso). Floor points only set the height offset below.
    upr = (tsec >= UP0) & (tsec <= UP1) if UP_SET else up_auto
    tor = joints_cam[upr, KIDX.index(KP["neck"])] - 0.5 * (
        joints_cam[upr, KIDX.index(KP["lhip"])] + joints_cam[upr, KIDX.index(KP["rhip"])])
    tor = tor[np.isfinite(tor[:, 0])]
    tor /= np.linalg.norm(tor, axis=1, keepdims=True)
    up = np.median(tor, 0)
    up /= np.linalg.norm(up)
    z = up
    tmp = np.array([0.0, 0.0, 1.0])
    x = tmp - (tmp @ z) * z; x /= np.linalg.norm(x)      # provisional (refined below)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 0)                           # cam -> world rows
    Jw = np.einsum("ij,nkj->nki", R, joints_cam)
    # floor + scale from the ROBOT itself (scene-histogram peaks proved unreliable):
    # feet touch the floor while standing, and the standing pelvis height is known
    if not STAND_SET:
        # ALL qualifying frames vote on floor/scale: medians over a few hundred spread
        # frames resist a locally-corrupted span (rack in front of the feet for ~2 s),
        # where any single elected window can silently inherit it. Fit-side calibration
        # (pelvis-plateau kappa + floor re-anchor) covers the residuals.
        if standable.sum() >= max(5, int(0.5 * FPS)):
            stand_auto = standable
            tt = tsec[standable]
            print(f"[lift] auto stance: {int(standable.sum())} frames spread over "
                  f"t={tt.min():.2f}..{tt.max():.2f} s")
        else:
            stand_auto = (tsec >= STAND0) & (tsec <= STAND1)
            if not stand_auto.any():                     # default window beyond a short clip
                stand_auto = upright & still if (upright & still).any() else \
                    np.ones(N, bool)
            print(f"[lift] WARN: no standing frames detected - fallback stance mask "
                  f"({int(stand_auto.sum())} frames)")
    stand = (tsec >= STAND0) & (tsec <= STAND1) if STAND_SET else stand_auto

    if not STAND_SET:
        # UP refinement: the torso is only a proxy (the subject leans toward its work on
        # average - the residual showed up as an 8-12 deg standing-heel floor tilt across
        # the visited area). The standing heels themselves ARE the level reference: fit a
        # plane to the lower-heel world points and rotate its normal to +Z. Needs xy
        # spread; skipped when the subject never moves.
        for _ in range(1):                               # single pass: reselecting the
                                                         # lower heel after rotation can oscillate
            lh, rh = KIDX.index(KP["lheel"]), KIDX.index(KP["rheel"])
            lo = np.where(Jw[:, lh, 2] < Jw[:, rh, 2], lh, rh)
            H = Jw[np.arange(len(Jw)), lo][stand]
            H = H[np.isfinite(H[:, 2])]
            if len(H) < 30:
                print("[lift] up refine: skipped (insufficient standing heels)")
                break
            span = np.ptp(H[:, :2], axis=0)
            if max(span) < 1e-3 or min(span) < 0.03:
                print("[lift] up refine: skipped (insufficient heel spread)")
                break
            A = np.hstack([H[:, :2], np.ones((len(H), 1))])
            coef, *_ = np.linalg.lstsq(A, H[:, 2], rcond=None)
            nrm = np.array([-coef[0], -coef[1], 1.0]); nrm /= np.linalg.norm(nrm)
            tilt = np.degrees(np.arccos(np.clip(nrm[2], -1, 1)))
            ax = np.cross(nrm, [0.0, 0.0, 1.0])
            sa = np.linalg.norm(ax)
            if sa < 1e-8:
                break
            ax /= sa; ca = nrm[2]
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R2 = np.eye(3) + sa * Kx + (1 - ca) * (Kx @ Kx)
            Jw = np.einsum("ij,nkj->nki", R2, Jw)
            R = R2 @ R                                   # keep the camera chain consistent
            print(f"[lift] up refine: heel-plane tilt {tilt:.2f} deg leveled "
                  f"({len(H)} standing heels)")
            if tilt < 0.3:
                break

    heels = Jw[stand][:, [KIDX.index(KP["lheel"]), KIDX.index(KP["rheel"])], 2]
    zfloor = float(np.nanmedian(heels))
    pelv_z = float(np.nanmedian(0.5 * (Jw[stand, KIDX.index(KP["lhip"]), 2]
                                       + Jw[stand, KIDX.index(KP["rhip"]), 2])))
    best_in = 0
    Jw[..., 2] -= zfloor

    # metric scale: standing pelvis height = 0.72 G1-world units (G1-scaled Figure stance)
    Pw = P @ R.T
    Pw[:, 2] -= zfloor
    cam_h = -zfloor
    scale = PELVIS_H / (pelv_z - zfloor)
    Jw *= scale
    Pw *= scale
    print(f"[lift] stand pelvis raw {pelv_z - zfloor:.3f} -> scale {scale:.2f}, "
          f"cam height {cam_h * scale:.2f}u")

    # kitchen frame: +X = the robot's facing during dishwasher work (median hip-line x up)
    names = list(KP)
    hipv = Jw[:, names.index("lhip")] - Jw[:, names.index("rhip")]
    if YAW_SET:
        work = (tsec >= YAW0) & (tsec <= YAW1)
        hv = hipv[work]
        hv = hv[np.isfinite(hv[:, 0])]
        hmed = np.median(hv, 0); hmed[2] = 0.0
        fwd = np.cross(hmed, [0, 0, 1.0])                # facing = hipline x up (z-up frame)
        fwd = -fwd / np.linalg.norm(fwd)
    else:
        # auto yaw = DOMINANT upright facing, chirality-proof. Raw hip lines carry SAM's
        # front/back mirror (~18% of frames flip them 180 deg), so a plain median lands
        # mid-mixture: use the AXIAL mean (angle doubling - sign-invariant), then resolve
        # the 180 deg sign by the walking direction (pelvis velocity while moving fast).
        m = up_auto & (tsec >= T_MIN) & np.isfinite(hipv[:, 0])
        f_n = np.cross(hipv[m], [0, 0, 1.0]); f_n = -f_n
        ang = np.arctan2(f_n[:, 1], f_n[:, 0])
        a2 = np.angle(np.mean(np.exp(2j * ang))) / 2.0   # axial mean, modulo pi
        axial = np.array([np.cos(a2), np.sin(a2), 0.0])
        pelv_xy = 0.5 * (Jw[:, names.index("lhip"), :2] + Jw[:, names.index("rhip"), :2])
        v_xy = np.full((N, 2), np.nan)
        v_xy[1:] = np.diff(pelv_xy, axis=0) * FPS
        spd = np.linalg.norm(v_xy, axis=1)
        # walking = sustained displacement, not sway: absolute speed floor (units: scaled
        # world, standing pelvis = PELVIS_H) plus direction consistency, else the sign is
        # depth-noise (it flipped between VGGT resolutions on sway-only frames)
        fastm = np.isfinite(spd) & (spd > 0.3 * PELVIS_H) & (tsec >= T_MIN)
        # longest contiguous moving run; its NET displacement direction is arc-robust
        # (mean per-frame direction fails on curved walk-ins and sign-flips across runs)
        r0 = r1 = b0 = b1 = -1
        for n in range(N + 1):
            if n < N and fastm[n]:
                r1 = n; r0 = n if r0 < 0 or not fastm[n - 1] else r0
            elif r0 >= 0:
                if r1 - r0 > b1 - b0:
                    b0, b1 = r0, r1
                r0 = -1
        if b1 - b0 + 1 >= int(0.5 * FPS):
            net = pelv_xy[min(b1 + 1, N - 1)] - pelv_xy[b0]
            sgn = np.sign(net @ axial[:2])
            src = f"walk net displacement (t={tsec[b0]:.1f}..{tsec[b1]:.1f} s)"
        else:                                            # planted subject: majority raw sign
            sgn = np.sign(np.median(f_n[:, :2] @ axial[:2]))
            src = "majority hip-line sign"
        fwd = (sgn if sgn != 0 else 1.0) * axial
        print(f"[lift] auto yaw: facing ({fwd[0]:+.2f},{fwd[1]:+.2f}), sign from {src}, "
              f"{int(m.sum())} frames")
    # rotate so fwd -> +X
    c, s2 = fwd[0], fwd[1]
    Rz = np.array([[c, s2, 0], [-s2, c, 0], [0, 0, 1.0]])
    Jw = np.einsum("ij,nkj->nki", Rz, Jw)
    Pw = Pw @ Rz.T
    ok = np.isfinite(Jw[:, :, 0]).all(1)
    ok &= tsec >= T_MIN                                  # unreliable head discarded
    # camera in the same frame (needed by the kitchen fit / camera solve). With per-frame
    # chained extrinsics: world(p_cam, n) = M_c2w[n] @ p_cam + c_c2w[n]
    # where the static part is world(p_chain) = Rz @ (scale * (R @ p_chain - [0,0,zfloor])).
    Rcw = Rz @ R                                         # chain-basis rows in world coords
    M_c2w = scale * np.einsum("ij,njk->nik", Rcw, np.transpose(camR, (0, 2, 1)))
    c_c2w = -np.einsum("nij,nj->ni", M_c2w, camT) - np.array([0.0, 0.0, zfloor * scale])
    cam_pos_w = np.median(c_c2w, 0)                      # legacy static key (exact when static)
    fovs = []
    for k in range(0, 40):
        df = os.path.join(WORK, "depth", f"f{k:05d}.npz")
        if os.path.exists(df):
            fovs.append(np.load(df)["pose_enc"][7])
    np.savez_compressed(os.path.join(WORK, "lift3d.npz"),
                        joints_w=Jw.astype(np.float32), t=tsec, ok=ok,
                        scene=Pw[::3].astype(np.float32),
                        cam_pos=cam_pos_w.astype(np.float64), R_cam2world=Rcw.astype(np.float64),
                        scale=np.float64(scale), fov_h=np.float64(np.median(fovs)),
                        stand_mask=stand, upright_mask=(upright & (tsec >= T_MIN)),
                        M_c2w=M_c2w.astype(np.float64), c_c2w=c_c2w.astype(np.float64))
    print(f"[lift] {ok.sum()}/{N} frames lifted; cam at {cam_pos_w.round(2)}, "
          f"cam travel {np.linalg.norm(c_c2w - c_c2w[0], axis=1).max():.2f}u")

    # ---- bird's-eye view
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(Pw[::4, 0], Pw[::4, 1], s=0.4, c="0.8", zorder=0)   # scene (walls/counters)
    mid = 0.5 * (Jw[:, list(KP).index("lhip")] + Jw[:, list(KP).index("rhip")])
    good = ok & (tsec >= 2.0) & (tsec <= 210.0)
    ax.plot(mid[good, 0], mid[good, 1], "-", lw=1, color="0.6", zorder=1)
    names = list(KP)
    for tv in np.arange(3.0, 210.0, 3.0):
        n = int(np.argmin(np.abs(tsec - tv)))
        if not ok[n]:
            continue
        col = cm.viridis((tv - 3.0) / 207.0)
        for a, b in BONES:
            pa, pb = Jw[n, names.index(a)], Jw[n, names.index(b)]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], "-", color=col, lw=1.2, alpha=0.8, zorder=2)
    sc = ax.scatter([], [], c=[])
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title("figure.mp4 lifted skeletons, bird's-eye (VGGT-Omega depth), color = time")
    fig.colorbar(cm.ScalarMappable(cmap="viridis"), ax=ax, label="0 .. 210 s", shrink=0.6)
    fig.savefig(os.path.join(WORK, "birdseye.png"), dpi=110, bbox_inches="tight")
    print("[lift] birdseye.png saved")


if __name__ == "__main__":
    main()
