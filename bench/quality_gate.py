#!/usr/bin/env python
"""Regression gate: compare a candidate fit3d.npz against a reference fit3d.npz.

Every pipeline optimization so far was accepted/rejected by exactly this comparison
(e.g. SAM body-mode + batching + VGGT stride/fstride landed at 1.02 cm local MPJPE vs
the full-quality reference — an order of magnitude under a typical retarget error floor).

Metrics:
  - LOCAL inter-pipeline MPJPE (pelvis-centred pose; both worlds share the subject-
    calibrated convention, so no rotation alignment is applied) + per-joint breakdown
  - trajectory shape (median-centred pelvis xy path)
  - hip-line yaw agreement (chirality/heading consistency)
  - kappa (scale calibration stability)

Usage:
    python bench/quality_gate.py <candidate_fit3d.npz> <reference_fit3d.npz> [--max-frames N]

Exit code 1 if the default thresholds fail (local MPJPE > 3 cm, yaw p95 > 5 deg,
kappa relative diff > 2%) - override with --loose to report without gating.
"""
import argparse
import sys

import numpy as np

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--max-frames", type=int, default=0, help="compare first N frames only")
    ap.add_argument("--loose", action="store_true", help="report only, never fail")
    ap.add_argument("--align-yaw", action="store_true",
                    help="remove the median hip-line yaw offset before scoring (use when the "
                         "two runs use different +X conventions, e.g. auto vs pinned MR_YAW; "
                         "the offset itself is reported)")
    args = ap.parse_args()

    fb, fr = np.load(args.candidate), np.load(args.reference)
    N = min(len(fb["t"]), len(fr["t"]))
    if args.max_frames:
        N = min(N, args.max_frames)
    Jb = fb["joints_w"][:N, :18].astype(np.float64)
    Jr = fr["joints_w"][:N, :18].astype(np.float64)
    ok = fb["ok"][:N] & fr["ok"][:N]
    if ok.sum() < 10:
        print(f"[gate] FAIL: only {int(ok.sum())} comparable frames")
        sys.exit(1)
    print(f"[gate] frames compared: {int(ok.sum())}/{N}")

    kb, kr = float(fb["kappa"]), float(fr["kappa"])
    print(f"[gate] kappa candidate {kb:.3f} vs reference {kr:.3f}")

    if args.align_yaw:
        hb0, hr0 = Jb[:, 5] - Jb[:, 6], Jr[:, 5] - Jr[:, 6]
        off = np.median(np.angle(np.exp(1j * (np.arctan2(hr0[:, 1], hr0[:, 0])
                                              - np.arctan2(hb0[:, 1], hb0[:, 0])))[ok]))
        c, s_ = np.cos(off), np.sin(off)
        Jb = Jb @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]).T
        print(f"[gate] yaw convention offset removed: {np.degrees(off):+.1f} deg")

    pb, pr = 0.5 * (Jb[:, 5] + Jb[:, 6]), 0.5 * (Jr[:, 5] + Jr[:, 6])
    Lb, Lr = Jb - pb[:, None], Jr - pr[:, None]
    E = np.linalg.norm(Lb - Lr, axis=2)[ok]
    mpjpe, p95 = E.mean(), np.percentile(E, 95)
    print(f"[gate] LOCAL inter-pipeline MPJPE {mpjpe*100:.2f} cm  p95 {p95*100:.2f} cm")
    per = sorted(((E[:, j].mean(), KPN[j]) for j in range(18)), reverse=True)
    print("[gate] per-joint cm:", "  ".join(f"{n}:{v*100:.2f}" for v, n in per))

    tb = (pb - np.median(pb[ok], 0))[ok][:, :2]
    tr = (pr - np.median(pr[ok], 0))[ok][:, :2]
    td = np.linalg.norm(tb - tr, axis=1)
    print(f"[gate] trajectory dev mean {td.mean()*100:.2f} cm  p95 {np.percentile(td,95)*100:.2f} cm")
    print("[gate] note: trajectory reflects independent VGGT normalization chains when the "
          "runs saw different video spans - judge pose via LOCAL MPJPE first")

    hb, hr = Jb[:, 5] - Jb[:, 6], Jr[:, 5] - Jr[:, 6]
    dy = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(hb[:, 1], hb[:, 0])
                                                 - np.arctan2(hr[:, 1], hr[:, 0]))))))[ok]
    yaw95 = np.percentile(dy, 95)
    print(f"[gate] hip-line yaw diff mean {dy.mean():.2f} deg  p95 {yaw95:.2f} deg")

    fails = []
    if mpjpe > 0.03:
        fails.append(f"local MPJPE {mpjpe*100:.2f} cm > 3 cm")
    if yaw95 > 5.0:
        fails.append(f"yaw p95 {yaw95:.2f} deg > 5 deg")
    if abs(kb - kr) / max(kr, 1e-6) > 0.02:
        fails.append(f"kappa diff {abs(kb-kr)/kr*100:.1f}% > 2%")
    if fails and not args.loose:
        print("[gate] FAIL:", "; ".join(fails))
        sys.exit(1)
    print("[gate] PASS" if not fails else "[gate] (loose) issues: " + "; ".join(fails))


if __name__ == "__main__":
    main()
