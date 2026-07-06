# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`humanoid_motion_recon` — standalone package: monocular RGB video → world-frame humanoid
skeletons + meshes. SAM-3D-Body (per-frame MHR) + VGGT-Omega (stitched window depth) +
subject-calibrated world lift + rigid fit + temporal smoothing, plus torch-CUDA
visualization renderers. README.md is the user-facing doc; keep it current.

## Hard-won invariants (do not rediscover)

- **Two conda envs**: `sam_3d_body` (torch+cu126, cv2, pyrender) runs `pose_video` and
  `depth_video`; `kimodo` (torch CUDA, PIL, imageio, **no cv2**) runs everything else.
  Model checkouts: `../sam-3d-body`, `../vggt-omega` (or `SAM3D_BODY_DIR`/`VGGT_OMEGA_DIR`).
  Gated checkpoints come from ModelScope mirrors, not HF.
- **Invocation**: `python -m humanoid_motion_recon.<tool>` (relative imports; plain
  `python file.py` does not work). `ffmpeg` in loops needs `-nostdin`. NVENC max frame
  dimension 4096 px (`fastvid`).
- **Depth is body-level only.** VGGT inpaints depth over thin/moving subjects — one robust
  body depth per frame + SAM's rigid articulation; never per-joint depth.
- **World alignment is subject-calibrated** (torso-up, heels=floor, standing pelvis height
  = scale). Scene-plane fits fail; don't reintroduce them.
- **kp2d is in processed-frame pixel space** (auto-detected from `frames/f00000.jpg`).
- **Chirality**: SAM mirrors front/back on faceless subjects ~half the time; the fit stays
  2D-faithful by design. Correction belongs downstream (raw estimates → hypothesis
  selection + Viterbi → then smooth). Do not "fix" chirality inside fit_pose — corrected 3D
  that disagrees with kp2d breaks every image-space consumer.
- **SAM batching**: frames ride the model's person dimension; `full` mode additionally
  needs the frame-aware `prepare_batch` patch in `pose_video` (upstream crops all persons'
  hands from ONE image). A `[pose] WARN` prints if the patch fails to engage — treat as a bug.
- **Refined-hand gate is bistable** on non-human wrists (`SAM_WRIST_THRESH`).

## Etiquette

- This repo stands alone. Do not reference private notes, other repos' internals, or
  anything outside this tree in committed files.
- Commit only when explicitly asked.
- Work dirs (`DEPTH_WORK`, `MR_OUT`, frame caches) are large and live outside the repo;
  never commit them.
- When you learn something that contradicts or extends README or this file, update it in
  the same change.
