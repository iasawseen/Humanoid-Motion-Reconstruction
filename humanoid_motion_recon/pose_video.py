#!/usr/bin/env python
"""Estimate 3D robot poses from figure.mp4 frames with SAM 3D Body (facebookresearch/sam-3d-body).

The Figure robot defeats the human detector (white shell / black joints on a B&W kitchen), so the
detector is bypassed: bboxes come from median-background subtraction (see README) stored
in <workdir>/boxes.npz. Frames were extracted at 2 fps into <workdir>/frames.

Outputs per frame into <workdir>/mhr/f%05d.npz:
    kp3d [70,3]  MHR-70 keypoints, camera frame     kp2d [70,2]  image px
    cam_t [3]    person->camera translation          focal        px
plus an overlay jpg every SPOT frames into <workdir>/overlay for visual QA.

Usage (env sam_3d_body, GPU 1):
    CUDA_VISIBLE_DEVICES=1 python -m humanoid_motion_recon.pose_video <checkpoint_dir> [workdir]
"""
import os
import sys

import cv2
import numpy as np
import torch

WORK = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("POSE_WORK", "")
CKPT_DIR = sys.argv[1]
SPOT = 20

sys.path.insert(0, os.environ.get(
    "SAM3D_BODY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sam-3d-body")))
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator  # noqa: E402

device = torch.device("cuda")
model, model_cfg = load_sam_3d_body(
    os.path.join(CKPT_DIR, "model.ckpt"), device=device,
    mhr_path=os.path.join(CKPT_DIR, "assets", "mhr_model.pt"),
)
estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Two modes, both cross-frame BATCHED (SAM_BATCH, default 16):
#   SAM_INFER=body (default, FAST)  - body decoder only; hand kps still produced by the body
#       decode but coarser (~5.6 cm; wrists up to ~13 cm on occasional frames)
#   SAM_INFER=full (body+hands)     - adds the two per-hand refinement passes (~3x body cost)
# Batching rides the model's person dimension; for "full", the hand crops are made
# frame-aware by patching prepare_batch inside the meta_arch (upstream crops all persons'
# hands from ONE image; we substitute each person's own frame, flip detected via stride).
INFER = os.environ.get("SAM_INFER", "body")
BATCH = int(os.environ.get("SAM_BATCH", "16"))
# Wrist-angle gate for accepting refined hands (upstream default 1.4 rad). On the figure
# robot the gate is bistable (~2/3 of consecutive frames flip refined<->body in upstream
# sequential too) - raise to always trust refinement, lower to always use body decode.
WRIST_THRESH = float(os.environ.get("SAM_WRIST_THRESH", "1.4"))

boxes = np.load(os.path.join(WORK, "boxes.npz"))
os.makedirs(os.path.join(WORK, "mhr"), exist_ok=True)
os.makedirs(os.path.join(WORK, "overlay"), exist_ok=True)
np.save(os.path.join(WORK, "mhr", "faces.npy"), estimator.faces)   # constant mesh topology
FORCE = os.environ.get("PV_FORCE", "0") == "1"          # re-run frames that lack the new keys

def img_path_of(key):
    p = os.path.join(WORK, "frames", key + ".png")
    return p if os.path.exists(p) else os.path.join(WORK, "frames", key + ".jpg")


def save_one(key, o, kp2d_field="pred_keypoints_2d"):
    np.savez_compressed(
        os.path.join(WORK, "mhr", key + ".npz"),
        kp3d=o["pred_keypoints_3d"].astype(np.float32),
        kp2d=o["pred_keypoints_2d"].astype(np.float32),
        cam_t=o["pred_cam_t"].astype(np.float32),
        focal=np.float32(o["focal_length"]),
        bbox=boxes[key].astype(np.float32),
        # full MHR body: mesh + explicit DIRECTIONS (orientation / per-joint global rotations)
        verts=o["pred_vertices"].astype(np.float16),
        global_rot=o["global_rot"].astype(np.float32),
        joint_rots=o["pred_global_rots"].astype(np.float16),
        joint_coords=o["pred_joint_coords"].astype(np.float32),
        body_pose=o["body_pose_params"].astype(np.float32),
        shape=o["shape_params"].astype(np.float32),
        scale=o["scale_params"].astype(np.float32),
    )


def spot_overlay(key, kp2d):
    img = cv2.imread(img_path_of(key))
    x0, y0, x1, y1 = boxes[key].astype(int)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
    for u, v in kp2d[:21]:                              # body kps only
        cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(WORK, "overlay", key + ".jpg"), img)


def needs(key):
    p = os.path.join(WORK, "mhr", key + ".npz")
    if not os.path.exists(p):
        return True
    if not FORCE:
        return False
    try:
        return "verts" not in np.load(p).files          # idempotent force
    except Exception:
        return True


keys = [k for k in sorted(boxes.files) if needs(k)]
print(f"[pose] {len(keys)} frames to process (infer={INFER}, batch={BATCH})", flush=True)

if BATCH <= 1:
    for n, key in enumerate(keys):
        outputs = estimator.process_one_image(img_path_of(key),
                                              bboxes=boxes[key][None].astype(np.float32),
                                              use_mask=False, inference_type=INFER)
        if not outputs:
            print(f"[pose] {key}: NO OUTPUT")
            continue
        save_one(key, outputs[0])
        if n % SPOT == 0:
            spot_overlay(key, outputs[0]["pred_keypoints_2d"])
            print(f"[pose] {n+1}/{len(keys)}", flush=True)
else:
    # cross-frame batching through the model's person dimension (frames are independent
    # crops; batch invariance vs the single path verified at 1 px / 6.7 mm)
    from torch.utils.data._utils.collate import default_collate
    from sam_3d_body.utils.dist import recursive_to
    import sam_3d_body.models.meta_arch.sam3d_body as _meta

    _FRAME_IMGS = []                                    # per-person source frames (batched full)
    _ORIG_PREPARE = _meta.prepare_batch

    def _framewise_prepare(img, transform, boxes_arr, masks=None, masks_score=None,
                           cam_int=None):
        """prepare_batch that crops person idx's hand from ITS OWN frame. Upstream passes one
        shared image; flipped calls arrive as negative-stride views."""
        if not _FRAME_IMGS or len(_FRAME_IMGS) != boxes_arr.shape[0]:
            print(f"[pose] WARN: frame-wise hand crop NOT engaged "
                  f"({len(_FRAME_IMGS)} imgs vs {boxes_arr.shape[0]} boxes)", flush=True)
            return _ORIG_PREPARE(img, transform, boxes_arr, masks, masks_score, cam_int)
        flipped = img.strides[1] < 0
        parts = []
        for i in range(boxes_arr.shape[0]):
            fi = _FRAME_IMGS[i][:, ::-1] if flipped else _FRAME_IMGS[i]
            parts.append(_ORIG_PREPARE(fi, transform, boxes_arr[i:i + 1],
                                       None, None, cam_int))
        out = parts[0]
        for key in ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale", "bbox",
                    "affine_trans", "mask", "mask_score", "person_valid"]:
            if key in out:
                out[key] = torch.cat([p[key] for p in parts], dim=1)
        return out

    _meta.prepare_batch = _framewise_prepare

    per_key = ["focal_length", "pred_keypoints_3d", "pred_keypoints_2d", "pred_vertices",
               "pred_cam_t", "global_rot", "pred_joint_coords"]
    # raw out["mhr"] key -> the estimator's all_out name that save_one expects
    alias = {"pred_global_rots": "joint_global_rots", "body_pose_params": "body_pose",
             "shape_params": "shape", "scale_params": "scale"}
    for c0 in range(0, len(keys), BATCH):
        chunk = keys[c0:c0 + BATCH]
        data_list, imgs = [], []
        for key in chunk:
            img = cv2.cvtColor(cv2.imread(img_path_of(key)), cv2.COLOR_BGR2RGB)
            imgs.append(img)
            h, w = img.shape[:2]
            data_list.append(estimator.transform(dict(
                img=img, bbox=boxes[key].astype(np.float32), bbox_format="xyxy",
                mask=np.zeros((h, w, 1), np.uint8), mask_score=np.array(0.0, np.float32))))
        batch = default_collate(data_list)
        for bk in ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale", "bbox",
                   "affine_trans", "mask", "mask_score"]:
            if bk in batch:
                batch[bk] = batch[bk].unsqueeze(0).float()
        if "mask" in batch:
            batch["mask"] = batch["mask"].unsqueeze(2)
        batch["person_valid"] = torch.ones((1, len(data_list)))
        h, w = imgs[0].shape[:2]
        f = (h * h + w * w) ** 0.5
        batch["cam_int"] = torch.tensor([[[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]]]).float()
        batch = recursive_to(batch, "cuda")
        model._initialize_batch(batch)
        _FRAME_IMGS[:] = imgs
        with torch.no_grad():
            out = model.run_inference(imgs[0], batch, inference_type=INFER,
                                      transform_hand=estimator.transform_hand,
                                      thresh_wrist_angle=WRIST_THRESH)
        if INFER == "full":
            out = out[0]                                # (pose_output, batch_lhand, batch_rhand, _, _)
        o = recursive_to(recursive_to(out["mhr"], "cpu"), "numpy")
        for i, key in enumerate(chunk):
            oi = {k: o[k][i] for k in per_key}
            oi.update({dst: o[src][i] for dst, src in alias.items()})
            save_one(key, oi)
        if (c0 // BATCH) % 4 == 0:
            spot_overlay(chunk[0], o["pred_keypoints_2d"][0])
            print(f"[pose] {min(c0 + BATCH, len(keys))}/{len(keys)}", flush=True)
print("DONE_POSE")
