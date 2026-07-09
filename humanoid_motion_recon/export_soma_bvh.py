#!/usr/bin/env python
"""Export a MotionRecon fit (fit3d.npz) as a SOMA-skeleton BVH for NVIDIA's soma-retargeter
(https://github.com/NVIDIA/soma-retargeter: SOMA BVH -> Unitree G1 CSV via Newton/Warp IK).

The template hierarchy (77-joint SOMA rig, cm units, Y-up, +Z facing = the retargeter's
"Mujoco" facing convention, bone-local rest offsets) is copied verbatim from a sample BVH in
the soma-retargeter checkout; only the MOTION section is generated.

Anatomy is CALIBRATED from the template's own reference frame (frame 0 of a standing-start
sample): how the spine line and hip line sit in the Hips/Chest local frames (the raw
Spine1 bone offset is ~7 deg off the true pelvis-to-neck line - anchoring to it pitches the
whole robot), the mid-hip <-> Hips-joint offset (the SOMA Hips joint sits ~8.5 cm above the
hip line; conflating them inflates the skeleton ~9% and makes the retargeter crouch), and
the local rotations of all untracked joints (neck chain, clavicles, fingers) which keep the
reference frame's natural posture instead of zeros. Direction transfer: two-vector Kabsch
frames for Hips/Chest, slerp for Spine1/2, and rig-exact two-vector frames for every limb
segment (measured bone direction + measured bend axis; the SOMA rig's convention is
segment-local +Z = bend axis, bone = +/-X - NVIDIA's own clips carry ForeArm/Shin locals
as exact pure positive-Z hinges). Twist anchored to the template's world heading (minrot
transport / world-anchored mesh deltas) is poison: it rails the retargeter's hip/shoulder
yaws for subjects facing away from the template. When MR_MOCAP_NPZ carries `rots_w`
(+ `rot_names`/`ref_n`), wrist pronation is taken from the mesh.

Scale + floor are self-measured from the fit (pelvis-plateau -> template reference mid-hip
height; lower-heel p5 -> y=0), so any scenario calibration convention works.

Usage:
    MR_OUT=<dir with fit3d.npz> [MR_FPS=30] [SOMA_RETARGETER_DIR=../soma-retargeter] \
        python -m humanoid_motion_recon.export_soma_bvh <out.bvh>
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
I = {n: i for i, n in enumerate(KPN)}
FPS = float(os.environ.get("MR_FPS", "30"))
# the retargeter's smoothing/stabilization objectives are per-frame and tuned on 120 fps
# SEED data - low-fps input gets over-smoothed in wall time (turn lag). Default: upsample.
FPS_OUT = float(os.environ.get("BVH_FPS", "120"))

W2B = 100.0 * np.array([[0.0, 1.0, 0.0],                 # bvh x        = world y
                        [0.0, 0.0, 1.0],                 # bvh y (up)   = world z (up)
                        [1.0, 0.0, 0.0]])                # bvh z (fwd)  = world x (fwd), m->cm


def unit(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def minrot(a, b):
    """[T,3,3] minimal rotations taking unit vectors a -> b (batched Rodrigues)."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = np.einsum("ni,ni->n", a, b)
    s2 = np.einsum("ni,ni->n", v, v)
    R = np.tile(np.eye(3), (len(a), 1, 1))
    m = s2 > 1e-12
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    R[m] = (np.eye(3) + K[m] + K[m] @ K[m] * ((1 - c[m]) / s2[m])[:, None, None])
    return R


def frame_align(a1, a2, b1, b2):
    """[T,3,3] rotations R with R@a1 ~ b1(n), R@a2 ~ b2(n); a* are fixed rest vectors.

    Orthonormal-triad Kabsch: primary direction is matched exactly, the secondary fixes
    the twist. a1/a2 are 3-vectors; b1/b2 are [T,3].
    """
    def triad(p, s):
        x = unit(p)
        z = unit(np.cross(p, s))
        y = np.cross(z, x)
        return np.stack([x, y, z], -1)                   # columns
    A = triad(np.tile(a1, (len(b1), 1)), np.tile(a2, (len(b1), 1)))
    B = triad(b1, b2)
    return np.einsum("nij,nkj->nik", B, A)               # B @ A^T


def slerp_batch(Ra, Rb, t):
    out = np.empty_like(Ra)
    for n in range(len(Ra)):
        s = Slerp([0, 1], Rotation.from_matrix([Ra[n], Rb[n]]))
        out[n] = s([t]).as_matrix()[0]
    return out


def parse_template(path):
    """Hierarchy text (verbatim), joint list [(name, parent, offset, nch)], ref frame 0."""
    text = open(path).read()
    hier, mot = text.split("MOTION")
    joints, stack = [], []
    for line in hier.splitlines():
        t = line.strip()
        if t.startswith(("ROOT", "JOINT")):
            joints.append([t.split()[1], stack[-1] if stack else None, None, 0])
        elif t.startswith("OFFSET"):
            v = np.array(list(map(float, t.split()[1:])))
            if joints and joints[-1][2] is None:
                joints[-1][2] = v
        elif t.startswith("CHANNELS"):
            joints[-1][3] = int(t.split()[1])
        elif t == "{":
            stack.append(joints[-1][0] if joints else None)
        elif t == "}":
            stack.pop()
    ref_row = np.array(list(map(float, mot.strip().splitlines()[3].split())))
    return hier, [(n, p, o, c) for n, p, o, c in joints], ref_row


def fk_reference(joints, row):
    """FK one BVH motion row -> per-joint global rots, positions, local rots."""
    G, P, L, i = {}, {}, {}, 0
    for n, p, o, nch in joints:
        if nch == 6:
            pos, eul = row[i:i + 3], row[i + 3:i + 6]
            i += 6
        else:
            pos, eul = None, row[i:i + 3]
            i += 3
        Rl = Rotation.from_euler("ZYX", eul, degrees=True).as_matrix()
        L[n] = Rl
        if p is None:
            G[n], P[n] = Rl, (pos if pos is not None else o)
        else:
            G[n] = G[p] @ Rl
            P[n] = P[p] + (G[p] @ o if pos is None else G[p] @ pos)
    return G, P, L


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "motion.bvh"
    srd = os.environ.get("SOMA_RETARGETER_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "soma-retargeter"))
    cands = sorted(glob.glob(os.path.join(srd, "assets/motions/bvh/*.bvh")))
    walk = [c for c in cands if "walk_forward" in c]     # clean standing-start reference
    tmpl = os.environ.get("SOMA_TEMPLATE_BVH", (walk or cands)[0])
    hier, joints, ref_row = parse_template(tmpl)
    names = [j[0] for j in joints]
    parent = {j[0]: j[1] for j in joints}
    off = {j[0]: j[2] for j in joints}
    Gr, Pr, Lr = fk_reference(joints, ref_row)           # anatomical reference (standing)
    u = lambda v: v / np.linalg.norm(v)
    midhip_r = 0.5 * (Pr["LeftLeg"] + Pr["RightLeg"])
    # anatomy in local frames at the reference: spine/hip-line in Hips, spine/shoulder-line
    # in Chest, and the constant mid-hip -> Hips-joint offset (in the Hips local frame)
    a_sp_hips = Gr["Hips"].T @ u(Pr["Neck1"] - midhip_r)
    a_hl_hips = Gr["Hips"].T @ u(Pr["LeftLeg"] - Pr["RightLeg"])
    a_sp_chest = Gr["Chest"].T @ u(Pr["Neck1"] - midhip_r)
    a_sl_chest = Gr["Chest"].T @ u(Pr["LeftArm"] - Pr["RightArm"])
    v_mh_local = Gr["Hips"].T @ (Pr["Hips"] - midhip_r)

    outd = os.environ.get("MR_OUT", "mr_out")
    # fit3d articulation carries SAM's front/back mirror flickers smoothed in - fine for
    # chirality-stable subjects, but flicker-prone clips need chirality-CORRECTED joints
    # (MR_MOCAP_NPZ: npz with mocap [N,58,3] world joints + ok, e.g. a hypothesis-selection
    # + Viterbi pass output). Hip-line direction flips ~180 deg on mirrored frames and the
    # retargeted robot pivots with it.
    mocap = os.environ.get("MR_MOCAP_NPZ", "")
    if mocap:
        fz = np.load(mocap)
        Jw, ok = fz["mocap"][:, :18].astype(np.float64), fz["ok"].astype(bool)
    else:
        fz = np.load(os.path.join(outd, "fit3d.npz"))
        Jw, ok = fz["joints_w"][:, :18].astype(np.float64), fz["ok"].astype(bool)
    # MHR segment rotations (world, chirality-corrected; saved by the retarget pass): the
    # TWIST source. Bone directions alone cannot see twist about the bone - without these,
    # limb twist, palm and head orientation stay at the template's reference values.
    Rw, RN, ref_n = None, {}, 0
    if "rots_w" in fz.files:
        Rw = fz["rots_w"].astype(np.float64)
        RN = {str(n): i for i, n in enumerate(fz["rot_names"])}
        ref_n = int(fz["ref_n"])
        print(f"[bvh] MHR rotation twist source: {len(RN)} joints, anchor frame {ref_n}")
    first, last = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
    Jw = Jw[first:last + 1]
    T = len(Jw)
    if Rw is not None:
        Rw = Rw[first:last + 1]
    for j in range(18):                                  # interpolate interior gaps
        for c in range(3):
            v = Jw[:, j, c]
            m = np.isfinite(v)
            if not m.all():
                Jw[:, j, c] = np.interp(np.arange(T), np.flatnonzero(m), v[m])

    # temporal smoothing (the corrected mocap is raw per-frame articulation; our own
    # retarget smooths AFTER IK in qpos space - this path must smooth the input instead)
    if os.environ.get("MR_SMOOTH", "1") != "0":
        k = max(3, 2 * int(round(0.10 * FPS / 2)) + 1)   # ~0.2 s box, odd
        pad = np.pad(Jw, ((k // 2, k // 2), (0, 0), (0, 0)), mode="edge")
        Jw = np.stack([pad[i:i + len(Jw)] for i in range(k)], 0).mean(0)

    # planted-foot clamp: the fit's heels drift up several cm during bends (kappa artifact);
    # the retargeter's feet stabilizer pins contacts and fights the floating targets, which
    # shows up as sitting-back crouches. Pin each foot's height to its planted level over
    # stationary runs (our qpos pipeline does the same via ground_clamp before rendering).
    for ank, toe, heel in (("lank", "lbtoe", "lheel"), ("rank", "rbtoe", "rheel")):
        a = Jw[:, I[ank]]
        v = np.zeros(len(a))
        v[1:] = np.linalg.norm(np.diff(a[:, :2], axis=0), axis=1) * FPS
        vk = max(3, 2 * int(round(0.1 * FPS / 2)) + 1)
        v = np.convolve(np.pad(v, (vk // 2, vk // 2), mode="edge"), np.ones(vk) / vk, "valid")
        z_lo = np.nanpercentile(a[:, 2], 20)
        planted = (v < 0.25) & (a[:, 2] < z_lo + 0.10)
        idx = np.flatnonzero(planted)
        if len(idx):
            runs = [r for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                    if len(r) >= int(0.2 * FPS)]
            # one baseline per foot joint: all planted runs sit at the foot's ground level
            # (a run's own median would pin bend-phase heel FLOAT in place)
            base = {jn: float(np.percentile(Jw[np.concatenate(runs), I[jn], 2], 20))
                    for jn in (ank, toe, heel)} if runs else {}
            for r in runs:
                for jn in (ank, toe, heel):
                    zmed = base[jn]
                    Jw[r, I[jn], 2] = zmed
                    for e, drn in ((r[0], -1), (r[-1], +1)):   # short blend at run edges
                        for b in range(1, int(0.1 * FPS) + 1):
                            t_ = e + drn * b
                            if 0 <= t_ < len(Jw) and not planted[min(max(t_, 0), len(Jw) - 1)]:
                                w = 1.0 - b / (int(0.1 * FPS) + 1)
                                Jw[t_, I[jn], 2] = w * zmed + (1 - w) * Jw[t_, I[jn], 2]

    if FPS_OUT and abs(FPS_OUT - FPS) > 1e-3:            # resample to the retargeter fps
        t_src = np.arange(T) / FPS
        t_dst = np.arange(int(t_src[-1] * FPS_OUT) + 1) / FPS_OUT
        Jw = np.stack([[np.interp(t_dst, t_src, Jw[:, j, c]) for c in range(3)]
                       for j in range(18)], 0).transpose(2, 0, 1)
        if Rw is not None:
            Rw = np.stack([Slerp(t_src, Rotation.from_matrix(Rw[:, j]))(t_dst).as_matrix()
                           for j in range(Rw.shape[1])], 1)
        T = len(Jw)
    # rotation-anchor frame index in the (trimmed, resampled) timebase
    ref_i = int(np.clip(round((ref_n - first) / FPS * FPS_OUT), 0, T - 1)) \
        if FPS_OUT and abs(FPS_OUT - FPS) > 1e-3 else int(np.clip(ref_n - first, 0, T - 1))

    # self-measured scale (pelvis plateau -> template reference MID-HIP height, not the
    # Hips joint - it sits ~8.5 cm higher) + floor re-anchor
    pel_w = 0.5 * (Jw[:, I["lhip"], 2] + Jw[:, I["rhip"], 2])
    stand_h = float(np.median(pel_w[pel_w > np.percentile(pel_w, 85) * 0.97]))
    ref_midhip_y = float(midhip_r[1])
    s = (ref_midhip_y / 100.0) / stand_h
    J = np.einsum("ij,ntj->nti", W2B, Jw) * s            # -> bvh frame (cm)
    heel_lo = np.minimum(J[:, I["lheel"], 1], J[:, I["rheel"], 1])
    J[:, :, 1] -= float(np.percentile(heel_lo[np.isfinite(heel_lo)], 5))

    def d(a, b):
        return J[:, I[b]] - J[:, I[a]]

    midhip = 0.5 * (J[:, I["lhip"]] + J[:, I["rhip"]])
    hipline = J[:, I["lhip"]] - J[:, I["rhip"]]
    sholine = J[:, I["lsho"]] - J[:, I["rsho"]]
    spine = J[:, I["neck"]] - midhip

    G = {}
    G["Hips"] = frame_align(a_sp_hips, a_hl_hips, unit(spine), unit(hipline))
    G["Chest"] = frame_align(a_sp_chest, a_sl_chest, unit(spine), unit(sholine))
    G["Spine1"] = slerp_batch(G["Hips"], G["Chest"], 1 / 3)
    G["Spine2"] = slerp_batch(G["Hips"], G["Chest"], 2 / 3)
    # ---- limb segment frames: rig-exact two-vector construction.
    # HARD RIG CONVENTION (verified on the template reference and all 10 NVIDIA sample
    # clips: measured bend axis . segment-local +Z = +1.000; their ForeArm/Shin locals
    # are EXACT pure positive-Z hinges, off-axis 0.00): every limb segment has local
    # +Z = bend axis (positive flexion) and local +/-X = bone. Each segment frame is
    # therefore fully determined by the MEASURED bone direction + MEASURED bend axis -
    # heading-correct by construction. (The previous minrot transport pinned segment
    # twist to the template's +Z facing: a subject standing 180 deg from it carried
    # ~180 deg of twist on every limb link, which the retargeter's orientation
    # objectives - hand r=1.2, foot r=2.0 + FeetStabilizer at 1.0 - turned into railed
    # hip/shoulder yaws, frozen ankles and permanently bent knees, all pinned at exact
    # g1 limits by their joint_limit clamper from frame 0.)
    # Straight limbs have no measurable bend axis: blend toward the torso-riding
    # template axis (heading-aligned via the measured Hips/Chest frames) by bend angle.
    EZ = np.array([0.0, 0.0, 1.0])
    D_hips = np.einsum("nij,kj->nik", G["Hips"], Gr["Hips"])
    D_chest = np.einsum("nij,kj->nik", G["Chest"], Gr["Chest"])

    def bend_axis(d_par, d_chi, seg, D_T):
        ax = np.cross(d_par, d_chi)
        s = np.linalg.norm(ax, axis=1)
        bend = np.degrees(np.arcsin(np.clip(s, 0.0, 1.0)))
        ax_fb = np.einsum("nij,j->ni", D_T, Gr[seg] @ EZ)
        w = np.clip(bend / 15.0, 0.0, 1.0)[:, None]
        ax_m = np.where(s[:, None] > 1e-8, ax / np.maximum(s, 1e-9)[:, None], ax_fb)
        return unit(w * ax_m + (1.0 - w) * ax_fb)

    # mesh wrist pronation (the one twist DOF bend-axis frames cannot see): twist of the
    # mesh wrist-vs-forearm delta about the forearm bone axis, relative to REF
    def pronation(p_):
        if Rw is None or (p_ + "wri") not in RN or (p_ + "_forearm") not in RN:
            return None
        jf, jw = RN[p_ + "_forearm"], RN[p_ + "wri"]
        R_rel = np.einsum("nji,njk->nik", Rw[:, jf], Rw[:, jw])   # forearm-local wrist
        dR = np.einsum("nij,kj->nik", R_rel, R_rel[ref_i])
        el, wr = ("lelb", "lwri") if p_ == "l" else ("relb", "rwri")
        a = Jw[ref_i, I[wr]] - Jw[ref_i, I[el]]
        a = Rw[ref_i, jf].T @ (a / np.linalg.norm(a))             # bone axis, forearm-local
        qq = Rotation.from_matrix(dR).as_quat()                   # [T,4] xyzw
        tw = 2.0 * np.arctan2(qq[:, :3] @ a, qq[:, 3])
        return (tw + np.pi) % (2 * np.pi) - np.pi

    for S_, sh, el, wr in (("Left", "lsho", "lelb", "lwri"), ("Right", "rsho", "relb", "rwri")):
        p_ = S_[0].lower()
        d_ua, d_fa = unit(d(sh, el)), unit(d(el, wr))
        ax = bend_axis(d_ua, d_fa, S_ + "Arm", D_chest)
        # elbow-bend floor: a fully straight arm makes the IK's elbow branch degenerate
        # (the retargeter hops elbow-up/elbow-down frame to frame). Keep >= ~10 deg of
        # bend by rotating the forearm direction about the bend axis.
        bend = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", d_ua, d_fa), -1, 1)))
        need = np.clip(10.0 - bend, 0.0, None)
        m_st = need > 0
        if m_st.any():
            rv = ax[m_st] * np.radians(need[m_st])[:, None]
            d_fa[m_st] = np.einsum("nij,nj->ni",
                                   Rotation.from_rotvec(rv).as_matrix(), d_fa[m_st])
        G[S_ + "Arm"] = frame_align(u(off[S_ + "ForeArm"]), EZ, d_ua, ax)
        G[S_ + "ForeArm"] = frame_align(u(off[S_ + "Hand"]), EZ, d_fa, ax)
        G[S_ + "Hand"] = np.einsum("nij,jk->nik", G[S_ + "ForeArm"],
                                   Gr[S_ + "ForeArm"].T @ Gr[S_ + "Hand"])
        tw = pronation(p_)
        if tw is not None:
            Rtw = Rotation.from_rotvec(d_fa * tw[:, None]).as_matrix()
            G[S_ + "Hand"] = np.einsum("nij,njk->nik", Rtw, G[S_ + "Hand"])
    for S_, hp, kn, an, to in (("Left", "lhip", "lkne", "lank", "lbtoe"),
                               ("Right", "rhip", "rkne", "rank", "rbtoe")):
        d_th, d_sh, d_ft = unit(d(hp, kn)), unit(d(kn, an)), unit(d(an, to))
        axk = bend_axis(d_th, d_sh, S_ + "Leg", D_hips)
        G[S_ + "Leg"] = frame_align(u(off[S_ + "Shin"]), EZ, d_th, axk)
        G[S_ + "Shin"] = frame_align(u(off[S_ + "Foot"]), EZ, d_sh, axk)
        axf = unit(axk - np.einsum("ni,ni->n", axk, d_ft)[:, None] * d_ft)
        G[S_ + "Foot"] = frame_align(u(off[S_ + "ToeBase"]), EZ, d_ft, axf)
        G[S_ + "ToeBase"] = np.einsum("nij,jk->nik", G[S_ + "Foot"],
                                      Gr[S_ + "Foot"].T @ Gr[S_ + "ToeBase"])

    # untracked joints (Root, neck chain, clavicles, fingers, toes-ends, eyes/jaw) keep the
    # reference frame's LOCAL rotations - the natural posture the retargeter was tuned on
    Gall = {n: G.get(n) for n in names}
    for n in names:
        if Gall[n] is None:
            Rl = np.tile(Lr[n], (T, 1, 1))
            Gall[n] = Rl if parent[n] is None else np.einsum("nij,njk->nik", Gall[parent[n]], Rl)

    # locals + euler (channel order Zrotation Yrotation Xrotation = intrinsic ZYX)
    eul = {}
    for n in names:
        p = parent[n]
        Rl = Gall[n] if p is None else np.einsum("nji,njk->nik", Gall[p], Gall[n])
        eul[n] = Rotation.from_matrix(Rl).as_euler("ZYX", degrees=True)

    # Hips position channel: the Hips JOINT, i.e. measured mid-hip + the (constant, local)
    # mid-hip -> Hips offset carried through the animated Hips frame
    hips_pos = midhip + np.einsum("nij,j->ni", Gall["Hips"], v_mh_local)

    lines = []
    for f in range(T):
        row = []
        for n, p, o, nch in joints:
            if nch == 6:
                pos = hips_pos[f] if n == "Hips" else np.zeros(3)
                row += [f"{v:.6f}" for v in pos]
            row += [f"{v:.6f}" for v in eul[n][f]]
        lines.append(" ".join(row))
    with open(out_path, "w") as fo:
        fo.write(hier)
        fo.write(f"MOTION\nFrames: {T}\nFrame Time: {1.0 / (FPS_OUT or FPS):.6f}\n")
        fo.write("\n".join(lines) + "\n")

    # ---- validation: template-consistent FK, compare bone dirs vs measured
    P = {names[0]: np.zeros((T, 3))}
    Gf = {names[0]: Gall[names[0]]}
    for n, p, o, nch in joints[1:]:
        P[n] = hips_pos if n == "Hips" else P[p] + np.einsum("nij,j->ni", Gf[p], o)
        Gf[n] = Gall[n]
    errs = []
    for a, b, jparent, jchild in [("lsho", "lelb", "LeftArm", "LeftForeArm"),
                                  ("lelb", "lwri", "LeftForeArm", "LeftHand"),
                                  ("rsho", "relb", "RightArm", "RightForeArm"),
                                  ("relb", "rwri", "RightForeArm", "RightHand"),
                                  ("lhip", "lkne", "LeftLeg", "LeftShin"),
                                  ("lkne", "lank", "LeftShin", "LeftFoot"),
                                  ("rhip", "rkne", "RightLeg", "RightShin"),
                                  ("rkne", "rank", "RightShin", "RightFoot")]:
        fkd = unit(np.einsum("nij,j->ni", Gf[jparent], off[jchild]))
        ang = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", fkd, unit(d(a, b))), -1, 1)))
        errs.append((float(np.median(ang)), jparent))
    heels = np.minimum(P["LeftFoot"][:, 1], P["RightFoot"][:, 1])
    fk_mh = 0.5 * (P["LeftLeg"] + P["RightLeg"])
    print(f"[bvh] {T} frames @ {FPS_OUT or FPS:g} fps (src {FPS:g}) -> {out_path} "
          f"(scale x{s:.3f}, stand_h {stand_h:.3f} m)")
    print("[bvh] FK-vs-measured bone dirs (median deg): "
          + "  ".join(f"{nm}:{a:.1f}" for a, nm in errs))
    print(f"[bvh] FK midhip y median {np.median(fk_mh[:,1]):.1f} cm (ref {midhip_r[1]:.1f}); "
          f"midhip consistency {np.median(np.linalg.norm(fk_mh - midhip, axis=1)):.2f} cm; "
          f"ankle y p5 {np.percentile(heels, 5):.1f} cm (ref "
          f"{min(Pr['LeftFoot'][1], Pr['RightFoot'][1]):.1f})")


if __name__ == "__main__":
    main()
