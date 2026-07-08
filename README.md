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

**`MR_AUTO=1`** auto-detects all calibration from the subject: stance/uprightness =
leg-extension + torso-leg alignment + stillness + heel depth-band visibility (all
qualifying frames vote on lift floor/scale); yaw = chirality-proof axial dominant facing,
sign from walk net displacement; kappa = per-frame implied kappa over pelvis-plateau
frames (tiptoes excluded); **up = standing-heel-plane leveling measured in fit space**,
with the rigid correction exported back into `lift3d.npz` (scene, camera chain, lift
joints) so all consumers share one world. Division of ownership: floor + metric scale are
lift-owned (scene must meet the feet), kappa is fit-owned (offsets vs depth ray), up is
fit-measured and propagated to both. Validated: cross-resolution (512/448) stable to
2 cm / 1.3 deg / 0.3% kappa; floor level to ~1.6 deg with +0.6 cm standing heel float
(the hand-tuned legacy windows leave an 8.5 deg floor tilt on the same data); re-runs of
fit converge (each pass re-levels the residual). Auto's +X convention is the subject's
true dominant facing, which can differ from hand-pinned `MR_YAW` conventions - compare
across conventions with `bench/quality_gate.py --align-yaw`. Individual env vars still
override single windows. Validated on a second scenario (5 s walking-only clip, dark
studio, tracking camera, different robot) with zero manual windows: straight-line pelvis
path (straightness 1.00), height flat to 0.7 cm, facing = walk direction within 4 deg,
heels on floor to +-1 cm. Walking clips exercise the stance-side gates: uprightness is
scored on the more-extended (stance) leg, and the heel depth-band visibility guard is
advisory (dark floors never confirm heels).

## Videos

`videos/` (gitignored, local-only) is the conventional home for source clips and rendered
QA videos: `videos/input/<name>.mp4` and `videos/output/<scenario>/*.mp4`. Renderers write
wherever `MR_OUT` points; move keepers here.

## NVIDIA soma-retargeter bridge (fit -> SOMA BVH -> Unitree G1 CSV)

Two modules bridge to [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter)
(SOMA BVH -> G1 29-DOF CSV via Newton/Warp IK; needs its own py3.12 env and a checkout at
`../soma-retargeter` or `SOMA_RETARGETER_DIR`):

```bash
MR_OUT=<fitdir> python -m humanoid_motion_recon.export_soma_bvh out.bvh   # fit -> SOMA BVH
# run their converter (their env): python app/bvh_to_csv_converter.py --config ... --viewer null
python -m humanoid_motion_recon.import_soma_csv out.csv qpos.csv          # CSV -> 36-col qpos
```

`export_soma_bvh` copies the template hierarchy verbatim from a sample BVH in their checkout
and calibrates all anatomy conventions from that sample's standing reference frame (spine /
hip-line directions in the Hips/Chest local frames, the mid-hip to Hips-joint offset — the
SOMA Hips joint sits ~8.5 cm above the hip line, and conflating them inflates the skeleton
~9% and makes the retargeter crouch — and natural local rotations for untracked joints).
Scale + floor are self-measured from the fit. **Chirality**: fit3d articulation carries
SAM's smoothed-in mirror flickers; for flicker-prone clips pass `MR_MOCAP_NPZ` (npz with
chirality-corrected `mocap` [N,58,3] + `ok`, e.g. from a hypothesis-selection + Viterbi
pass) or the retargeted robot pivots ~180 deg on mirrored stretches. `import_soma_csv`
undoes their "Mujoco"-facing world convention (BVH +Z forward lands on -Y; we yaw it back
to +X) and converts cm/degrees to m/radians (their extrinsic-xyz root euler, G1 DOF order
identical to the g1.xml hinge order).

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
