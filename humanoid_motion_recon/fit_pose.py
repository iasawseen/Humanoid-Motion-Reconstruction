#!/usr/bin/env python
"""Fit the rigid SAM-3D-Body skeleton to the VGGT-Omega point cloud, per frame.

Instead of lifting each keypoint from a single (silhouette-adjacent) depth pixel, this uses ALL
robot pixels: SAM's kinematically-consistent skeleton predicts a surface-depth map over a tube
around its 2D bones (bone-axis depth minus part radius); the VGGT depth over the same pixels is
then explained by a per-frame affine depth map  d_vggt = a * d_sam + b  (VGGT windows are
normalized similarity transforms of the metric scene, so 2 parameters absorb exactly the
per-window scale/shift drift). Joints are re-lifted through the smoothed (a, b), preserving
SAM's bone lengths by construction.

Inputs:  <workdir>/mhr/f*.npz (kp2d, kp3d, cam_t), <workdir>/depth/f*.npz (depth, conf,
         pose_enc), $MR_OUT/lift3d.npz (world alignment: R_cam2world, scale,
         cam_pos - so fit3d lands in the same frame as lift3d).
Outputs: $MR_OUT/fit3d.npz  {joints_w [N,18,3], t, ok, a, b, npix}
         + prints bone-length / seam-jump stats vs the naive lift.
Usage: DEPTH_WORK=<pose_full dir> python -m humanoid_motion_recon.fit_pose
"""
import glob
import os

import numpy as np

WORK = os.environ["DEPTH_WORK"]
OUTD = "mr_out"
FPS = float(os.environ.get("MR_FPS", 24000 / 1001))
T_MIN = float(os.environ.get("MR_TMIN", "4.0"))
from PIL import Image as _Image
import glob as _glob
W_IMG, H_IMG = _Image.open(sorted(_glob.glob(os.path.join(WORK, "frames", "f*.jpg")))[0]).size
CONF_MIN = 1.1

KP = {"nose": 0, "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lhip": 9, "rhip": 10,
      "lkne": 11, "rkne": 12, "lank": 13, "rank": 14, "lbtoe": 15, "lheel": 17,
      "rbtoe": 18, "rheel": 20, "rwri": 41, "lwri": 62, "neck": 69}
# MHR-70 hands: 5 fingers x 4 joints per hand, each finger chain ordered TIP->proximal
# (verified empirically: distances from the wrist decrease within each 4-block).
# Right hand block 21-40 (wrist 41), left hand block 42-61 (wrist 62).
for _side, _base in (("r", 21), ("l", 42)):
    for _f, _fn in enumerate(("th", "ix", "md", "rg", "pk")):        # thumb..pinky
        for _j, _jn in enumerate(("tip", "dst", "mid", "prx")):      # tip -> proximal
            KP[f"{_side}_{_fn}_{_jn}"] = _base + 4 * _f + _j
KPN = list(KP)
KIDX = list(KP.values())
# finger bone chains: wrist -> prx -> mid -> dst -> tip
HAND_BONES = []
for _side, _w in (("r", "rwri"), ("l", "lwri")):
    for _fn in ("th", "ix", "md", "rg", "pk"):
        HAND_BONES += [(_w, f"{_side}_{_fn}_prx"), (f"{_side}_{_fn}_prx", f"{_side}_{_fn}_mid"),
                       (f"{_side}_{_fn}_mid", f"{_side}_{_fn}_dst"), (f"{_side}_{_fn}_dst", f"{_side}_{_fn}_tip")]
# tube bones with part radii (meters, SAM metric): surface sits ~radius in front of the axis
TUBE = [("lhip", "rhip", 0.10), ("neck", "lhip", 0.11), ("neck", "rhip", 0.11),
        ("neck", "nose", 0.09), ("lsho", "rsho", 0.09),
        ("lsho", "lelb", 0.055), ("lelb", "lwri", 0.05),
        ("rsho", "relb", 0.055), ("relb", "rwri", 0.05),
        ("lhip", "lkne", 0.075), ("lkne", "lank", 0.055),
        ("rhip", "rkne", 0.075), ("rkne", "rank", 0.055),
        ("lank", "lheel", 0.045), ("lheel", "lbtoe", 0.045),
        ("rank", "rheel", 0.045), ("rheel", "rbtoe", 0.045)]

_DISC = {}


def disc(r):
    r = max(1, int(round(r)))
    if r not in _DISC:
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        m = dy * dy + dx * dx <= r * r
        _DISC[r] = (dy[m], dx[m])
    return _DISC[r]


def robust_affine(x, y):
    """y ~ a*x + b with 2 rounds of sigma clipping."""
    keep = np.ones(len(x), bool)
    a = b = None
    for _ in range(3):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (a, b), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        r = y - (a * x + b)
        s = np.std(r[keep])
        if s < 1e-6:
            break
        keep = np.abs(r) < 2.0 * s
        if keep.sum() < 100:
            break
    return float(a), float(b), int(keep.sum())


def main():
    dfiles = sorted(glob.glob(os.path.join(WORK, "depth", "f*.npz")))
    N = len(dfiles)
    tsec = np.array([int(os.path.basename(f)[1:6]) for f in dfiles]) / FPS
    npix = np.zeros(N, int)
    wx, wy = {}, {}
    uv_all = np.full((N, len(KIDX), 2), np.nan)
    zs_all = np.full((N, len(KIDX)), np.nan)
    rel_all = np.full((N, len(KIDX), 3), np.nan)
    cams = np.full((N, 2), np.nan)                       # fx, fy at depth res

    for n, df in enumerate(dfiles):
        key = os.path.basename(df)[:-4]
        mf = os.path.join(WORK, "mhr", key + ".npz")
        if not os.path.exists(mf):
            continue
        dz = np.load(df)
        D = dz["depth"].astype(np.float32)
        C = dz["conf"].astype(np.float32)
        pe = dz["pose_enc"]
        h, w = D.shape
        fy = (h / 2.0) / np.tan(pe[7] / 2.0)
        fx = (w / 2.0) / np.tan(pe[8] / 2.0)
        cams[n] = (fx, fy)
        mz = np.load(mf)
        kp2d = mz["kp2d"]
        if np.ptp(kp2d[:, 1]) > 1.5 * H_IMG or np.ptp(kp2d[:, 0]) > 1.5 * W_IMG:
            continue                                     # degenerate SAM frame (e.g. figure f05188)
        Zs = (mz["kp3d"] + mz["cam_t"][None])[:, 2]      # SAM camera-frame joint depths (m)
        su, sv = w / W_IMG, h / H_IMG
        uv = np.stack([kp2d[KIDX, 0] * su, kp2d[KIDX, 1] * sv], 1)
        uv_all[n] = uv
        zs_all[n] = Zs[KIDX]
        k3 = mz["kp3d"][KIDX]
        rel_all[n] = k3 - 0.5 * (k3[KPN.index("lhip")] + k3[KPN.index("rhip")])

        # SAM surface-depth canvas over the bone tube
        dsam = np.full((h, w), np.inf, np.float32)
        for a_n, b_n, rad in TUBE:
            pa, pb = uv[KPN.index(a_n)], uv[KPN.index(b_n)]
            za, zb = Zs[KP[a_n]], Zs[KP[b_n]]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            steps = max(2, int(np.hypot(*(pb - pa)) / 2))
            for s in np.linspace(0.0, 1.0, steps):
                u0, v0 = pa + s * (pb - pa)
                zz = za + s * (zb - za)
                if zz <= 0.1:
                    continue
                r_px = rad / zz * fx
                dy, dx = disc(r_px)
                vv = (int(round(v0)) + dy).clip(0, h - 1)
                uu = (int(round(u0)) + dx).clip(0, w - 1)
                zsurf = zz - rad
                m = dsam[vv, uu] > zsurf
                dsam[vv[m], uu[m]] = zsurf
        valid = np.isfinite(dsam) & (C > CONF_MIN)
        if valid.sum() < 100:
            continue
        x, y = dsam[valid].astype(np.float64), D[valid].astype(np.float64)
        # drop gross background grabs
        med = np.median(y)
        keep = np.abs(y - med) < 0.35 * med
        if keep.sum() < 60:
            continue
        x, y = x[keep], y[keep]
        if len(x) > 2000:                                # cap per-frame contribution
            sel = np.random.default_rng(n).choice(len(x), 2000, replace=False)
            x, y = x[sel], y[sel]
        wx.setdefault(n, []).append(x)
        wy.setdefault(n, []).append(y)
        npix[n] = len(x)

    # ---- per-frame BODY-LEVEL depth: the diagnostics were decisive - VGGT depth over the
    # thin robot carries no usable internal-shape signal (context inpainting; depth-coef flips
    # sign frame to frame), so the only thing read from the cloud is ONE robust body depth per
    # frame (median over the interior tube), median-filtered for the window seams. Articulation
    # comes entirely from SAM's rigid kp3d.
    d_body = np.full(N, np.nan)
    for n in sorted(wx):
        d_body[n] = np.median(np.concatenate(wy[n]))
    okd = np.isfinite(d_body)
    idx = np.arange(N)
    d_i = np.interp(idx, idx[okd], d_body[okd])
    k = 19                                               # > window length 16: straddles seams
    pad = k // 2
    d_s = np.array([np.median(d_i[max(0, i - pad):i + pad + 1]) for i in idx])
    print(f"[fit] body depth on {okd.sum()}/{N} frames, median {np.nanmedian(d_body):.3f} raw")

    # meters -> raw scale ratio, smoothed hard (it varies only with the window normalization)
    LH, RH = KPN.index("lhip"), KPN.index("rhip")
    zp_sam = 0.5 * (zs_all[:, LH] + zs_all[:, RH])
    ratio = d_s / zp_sam
    okr = np.isfinite(ratio)
    r_i = np.interp(idx, idx[okr], ratio[okr])
    kr = 49
    pr = kr // 2
    r_s = np.array([np.median(r_i[max(0, i - pr):i + pr + 1]) for i in idx])

    # rigid re-lift: pelvis on its ray at the body depth; joints = ratio-scaled SAM offsets
    z = np.load(os.path.join(os.environ.get("MR_OUT", OUTD), "lift3d.npz"))
    Rcw, scale, cam_pos = z["R_cam2world"], float(z["scale"]), z["cam_pos"]
    h, w = np.load(dfiles[0])["depth"].shape
    Jc = np.full((N, len(KIDX), 3), np.nan)
    for n in range(N):
        if not np.isfinite(uv_all[n, 0, 0]) or not np.isfinite(cams[n, 0]):
            continue
        fx, fy = cams[n]
        up = 0.5 * (uv_all[n, LH] + uv_all[n, RH])
        pelv = np.array([(up[0] - w / 2) / fx * d_s[n], (up[1] - h / 2) / fy * d_s[n], d_s[n]])
        Jc[n] = pelv[None] + r_s[n] * rel_all[n]
    a_s, b_s = r_s, d_s                                  # kept in the npz for inspection
    # camera -> world: per-frame affine when the camera moves (M_c2w from the chained VGGT
    # extrinsics), the static legacy map otherwise
    if "M_c2w" in z:
        Jw = np.einsum("nij,nkj->nki", z["M_c2w"], Jc) + z["c_c2w"][:, None]
    else:
        Jw = scale * np.einsum("ij,nkj->nki", Rcw, Jc) + cam_pos[None, None]
    # body-size calibration from the subject: the pelvis rides the depth ray (height is right
    # by construction), but the OFFSETS are sized by SAM's absolute depth, which wobbles.
    # During the stand window the heels must touch the floor -> uniform rescale about mid-hip.
    midw = 0.5 * (Jw[:, LH] + Jw[:, RH])
    fin = np.isfinite(Jw[:, :, 2]).all(1)
    env_stand = os.environ.get("MR_AUTO", "0") != "1" or \
        "MR_STAND0" in os.environ or "MR_STAND1" in os.environ
    m_up = None
    if not env_stand and "upright_mask" in z and len(z["upright_mask"]) == N:
        m_up = z["upright_mask"] & fin                   # robust kappa over ALL upright frames
        m_st = z["stand_mask"] & fin if "stand_mask" in z and len(z["stand_mask"]) == N else m_up
    elif not env_stand and "stand_mask" in z and len(z["stand_mask"]) == N:
        m_st = z["stand_mask"] & fin                     # same frames lift_skeleton calibrated on
    else:
        st0 = float(os.environ.get("MR_STAND0", "7.5"))
        st1 = float(os.environ.get("MR_STAND1", "9.5"))
        m_st = (tsec >= st0) & (tsec <= st1) & fin
    kappa = 1.0
    if m_up is not None:
        # SAM's offset-scale error varies across the video, so a single stance window can be
        # a local outlier (a rack in front of the feet reads as heels near the floor), and
        # SAM-geometry "upright" still includes near-stands with heels physically raised.
        # Robust estimator: implied per-frame kappa over the PELVIS-PLATEAU frames (the
        # subject at full standing height => heels on the floor), measured in fit space -
        # the median zeroes the standing heel float by construction.
        pel_z = midw[:, 2]
        m_tall = fin & (pel_z > np.nanpercentile(pel_z[fin], 85) * 0.97)
        # exclude tiptoe frames (reach-ups raise the pelvis ABOVE the stand plateau on
        # legitimately lifted heels -> implied kappa explodes); flat feet have heel ~ toe z,
        # and the gap is only mildly kappa-sensitive so it works pre-correction
        gap = np.maximum(Jw[:, KPN.index("lheel"), 2] - Jw[:, KPN.index("lbtoe"), 2],
                         Jw[:, KPN.index("rheel"), 2] - Jw[:, KPN.index("rbtoe"), 2])
        m_tall &= gap < 0.05
        pz = pel_z[m_tall]
        hz = np.minimum(Jw[m_tall, KPN.index("lheel"), 2], Jw[m_tall, KPN.index("rheel"), 2])
        # re-anchor the floor here rather than trusting lift's stance window (a corrupted
        # window shifts zfloor, and kappa cannot repair a wrong floor): standing pelvis
        # sits at PELVIS_H by convention, so the plateau median fixes the offset; kappa
        # then zeroes the standing heels against THAT floor.
        pelvis_h = float(os.environ.get("MR_PELVIS_H", "0.72"))
        dz = float(np.median(pz)) - pelvis_h
        k_n = (pz - dz) / np.maximum(pz - hz, 1e-6)
        k_n = k_n[np.isfinite(k_n) & (k_n > 0.5) & (k_n < 2.0)]
        if len(k_n) >= 30:
            kappa = float(np.median(k_n))
            Jw[:, :, 2] -= dz
            midw = 0.5 * (Jw[:, LH] + Jw[:, RH])
            Jw = midw[:, None] + kappa * (Jw - midw[:, None])
            print(f"[fit] body-size calib: floor re-anchor {dz:+.3f}u, robust kappa "
                  f"{kappa:.3f} (median over {len(k_n)} pelvis-plateau frames)")
    if kappa == 1.0 and m_st.sum() >= 3:
        pelv_z = np.median(midw[m_st, 2])
        heel_z = np.median(0.5 * (Jw[m_st, KPN.index("lheel"), 2]
                                  + Jw[m_st, KPN.index("rheel"), 2]))
        kappa = pelv_z / max(pelv_z - heel_z, 1e-6)
        if 0.5 < kappa < 2.0:
            Jw = midw[:, None] + kappa * (Jw - midw[:, None])
            print(f"[fit] body-size calib: heel z {heel_z:+.3f} -> kappa {kappa:.3f}")
        else:
            kappa = 1.0
    # temporal smoothing of the assembled motion, in WORLD space (camera motion must not leak
    # into the joints): per joint/coord, median (kills single-frame SAM spikes/flips) then a
    # short moving average (~0.4 s) - keeps walking/reaching dynamics, removes jitter
    from numpy.lib.stride_tricks import sliding_window_view
    idx = np.arange(N)
    flat = Jw.reshape(N, -1)
    for c in range(flat.shape[1]):
        v = flat[:, c]
        m = np.isfinite(v)
        if m.sum() < 10:
            continue
        vi = np.interp(idx, idx[m], v[m])
        for kk, op in ((7, "med"), (11, "box")):
            pad = np.pad(vi, kk // 2, mode="edge")
            wins = sliding_window_view(pad, kk)
            vi = np.median(wins, 1) if op == "med" else wins.mean(1)
        v[m] = vi[m]                                     # keep gaps invalid
    ok = np.isfinite(Jw[:, :, 0]).all(1)
    ok &= tsec >= T_MIN                                  # unreliable head discarded
    # kappa is a WORLD-space calibration; overlay renderers that reproject joints_w against the
    # video must divide the mid-hip-relative offsets by it (see collate_video / pose_over_cloud)
    np.savez_compressed(os.path.join(os.environ.get("MR_OUT", OUTD), "fit3d.npz"),
                        joints_w=Jw.astype(np.float32), t=tsec, ok=ok, kappa=np.float32(kappa),
                        a=a_s, b=b_s, npix=npix, scene=z["scene"], cam_pos=cam_pos)
    print(f"[fit] {ok.sum()}/{N} frames -> fit3d.npz")

    # ---- verification vs the naive lift
    Jl = z["joints_w"].astype(np.float64)
    okl = z["ok"]
    m = ok & okl & (tsec > 2.5) & (tsec < 200)
    I = {n_: i for i, n_ in enumerate(KPN)}
    print(f"{'bone':12s} {'lift std%':>10s} {'fit std%':>10s}")
    for a_n, b_n in [("lhip", "lkne"), ("lkne", "lank"), ("lsho", "lelb"), ("lhip", "rhip")]:
        for J_, tag in ((Jl, "lift"), (Jw, "fit")):
            L = np.linalg.norm(J_[m, I[a_n]] - J_[m, I[b_n]], axis=1)
            if tag == "lift":
                s1 = 100 * L.std() / L.mean()
            else:
                s2 = 100 * L.std() / L.mean()
        print(f"{a_n}-{b_n:8s} {s1:9.0f}% {s2:9.0f}%")
    for J_, ok_, tag in ((Jl, okl, "lift"), (Jw, ok, "fit")):
        mid = 0.5 * (J_[:, I["lhip"]] + J_[:, I["rhip"]])
        d = np.linalg.norm(np.diff(mid, axis=0), axis=1)
        okd = ok_[1:] & ok_[:-1] & (tsec[1:] > 8) & (tsec[1:] < 200)
        fi = np.arange(1, N)
        seam = (fi % 16 == 0)[:len(okd)]
        print(f"[{tag}] pelvis step: within-window {np.median(d[okd & ~seam])*100:.1f} cm, "
              f"window seam {np.median(d[okd & seam])*100:.1f} cm")


if __name__ == "__main__":
    main()
