# humanoid_motion_recon

Monocular RGB video → **world-frame humanoid skeletons + meshes**. Combines
[SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body) (per-frame MHR keypoints,
mesh, per-joint rotations) with [VGGT-Omega](https://github.com/facebookresearch/vggt)
(per-frame metric-ish depth from large stitched windows), then rigidly fits the per-frame
person reconstruction into a single world frame and smooths over time. Works with moving
cameras and non-human subjects (tested extensively on humanoid robots, which defeat
off-the-shelf human detectors — bounding boxes are an explicit input).

## Pipeline

```
frames/f%05d.jpg + boxes.npz              # you provide: extracted frames + per-frame person bbox
  └─ pose_video      SAM-3D-Body per frame → mhr/f%05d.npz (kp2d/kp3d/mesh/rotations)
  └─ depth_video     VGGT-Omega windows, robust-affine stitched → depth/f%05d.npz
      └─ lift_skeleton   world alignment (subject-calibrated) + naive lift → lift3d.npz
          └─ fit_pose    rigid fitter + temporal smoothing → fit3d.npz
              └─ lift_mesh  per-vertex world lift + smoothing → mesh_w.npz
```

World alignment is calibrated **from the subject itself** (torso-up, heels = floor,
standing pelvis height = scale) — scene-plane fits measurably fail. Depth over the subject
is context-inpainted by VGGT; the fitter therefore uses ONE robust body-level depth per
frame plus SAM's rigid articulation, never per-joint depth.

## Install & run

Two environments are needed in practice: one with `mujoco`-free torch + PIL for rendering
(`kimodo`-style), one with the SAM-3D-Body / VGGT-Omega deps (`cv2`, torch+cu126). Model
checkouts are discovered at `../sam-3d-body` and `../vggt-omega` relative to this repo, or
via `SAM3D_BODY_DIR` / `VGGT_OMEGA_DIR`.

```bash
pip install -e .

export DEPTH_WORK=/path/to/workdir      # holds frames/, boxes.npz; outputs land here + MR_OUT
export MR_OUT=/path/to/outputs          # default: ./mr_out

python -m humanoid_motion_recon.pose_video  <sam3d_ckpt_dir>   # POSE_WORK=$DEPTH_WORK
python -m humanoid_motion_recon.depth_video <vggt_ckpt.pt>
python -m humanoid_motion_recon.lift_skeleton
python -m humanoid_motion_recon.fit_pose
python -m humanoid_motion_recon.lift_mesh
```

### Speed modes (single RTX 3090: ≤6.3× real time)

| Knob | Default | Meaning |
|---|---|---|
| `SAM_INFER` | `body` | `body` = body decoder only (fast); `full` = + per-hand refinement (~3×) |
| `SAM_BATCH` | `16` | cross-frame batching through the model's person dimension |
| `SAM_WRIST_THRESH` | `1.4` | refined-hand acceptance gate (bistable on robot grippers — raise to pin open) |
| `VGGT_WINDOW` / `VGGT_STRIDE` | `160` / `150` | window size / stride (10 shared frames stitch fine) |
| `VGGT_FSTRIDE` | `1` | frame subsampling for depth (2 is near-lossless; fill-in npz written) |
| `VGGT_RES` | `512` | VGGT input resolution |

1 minute of 720p/24 video ≈ 6:17 wall on one 3090 (`body`), 9:18 with hands (`full`).

### Calibration env vars

`MR_FPS` (default 23.976), `MR_TMIN`, `MR_STAND0/1` (standing-window seconds for scale),
`MR_YAW0/1`, `MR_UP0/1` (yaw/up calibration windows), `MR_PELVIS_H` (standing pelvis height,
meters). Set per scenario; defaults suit a ~1.7 m subject.

**Experimental `MR_AUTO=1`**: auto-detects the stance/up/yaw windows from the subject
(leg-extension + torso-alignment uprightness, pose-velocity stillness, heel depth-band
visibility; yaw = dominant upright facing with walk-displacement sign; heel-plane up
leveling; fit-side floor re-anchor + pelvis-plateau robust kappa). Validated
cross-resolution stable (~3 cm inter-config MPJPE, yaw ~1.3 deg) but NOT yet at parity
with hand-tuned windows (~10 cm local MPJPE vs a tuned reference; root cause: lift-space
depth heels vs fit-space SAM-offset heels disagree systematically, so lift-side leveling
cannot zero fit-side tilt). Individual env vars still override single windows. Gate any
change with `bench/quality_gate.py`.

## Visualization / QA renderers

All torch-CUDA (no OpenGL): `collate_video` (4-pane: overlay | depth | skeleton-over-cloud |
BEV), `collate_mesh_video`, `pose_over_cloud`, `birdseye_video`, `recon_check_video`,
`recon3d_video`, `mesh_world_video`, `sam2d_video`, `sam3d_mesh_video`, `sam3d_dummy_video`
(raw SAM outputs — chirality evidence), `stabilize_depth` (static-camera depth de-jitter).
Shared infra: `gpurender` (splat/lines/cloud/mesh rasterizer), `fastvid` (NVENC writer),
`skel_draw`.

```bash
python -m humanoid_motion_recon.collate_video        # DEPTH_WORK + MR_OUT set as above
```

## Known limitations

- Front/back chirality of faceless subjects is ambiguous to SAM on ~half of frames; the
  fit stays faithful to SAM (2D-consistent). Correct chirality downstream on raw per-frame
  estimates before temporal smoothing (hypothesis selection + Viterbi works well).
- The refined-hand gate (`full` mode) flips refined↔body-decoded hands frame-to-frame on
  subjects with non-human wrists; temporal smoothing absorbs it, or pin the gate.
- Depth for thin/moving structures is inpainted — never read per-joint depth.
