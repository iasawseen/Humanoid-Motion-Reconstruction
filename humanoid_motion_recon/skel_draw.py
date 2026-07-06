"""Pretty skeleton drawing shared by the pose visualization videos.

Mocap-style color coding: left limbs cyan, right limbs orange, spine/center white-green.
Bones get a dark underlay (contrast on busy backgrounds) and everything is rendered at 2x
supersampling onto an RGBA overlay, then LANCZOS-downsampled and alpha-composited.
"""
import numpy as np
from PIL import Image, ImageDraw

KPN = ["nose", "lsho", "rsho", "lelb", "relb", "lhip", "rhip", "lkne", "rkne",
       "lank", "rank", "lbtoe", "lheel", "rbtoe", "rheel", "rwri", "lwri", "neck"]
# hands appended AFTER the body 18 (same order as fit_pose.KP): 5 fingers x 4 joints per hand,
# finger chains tip->proximal. Older 18-joint fit3d files simply have no hand rows to draw.
for _side in ("r", "l"):
    for _fn in ("th", "ix", "md", "rg", "pk"):
        for _jn in ("tip", "dst", "mid", "prx"):
            KPN.append(f"{_side}_{_fn}_{_jn}")
I = {n: i for i, n in enumerate(KPN)}

C_L = (60, 190, 255)          # left limbs
C_R = (255, 150, 40)          # right limbs
C_C = (170, 255, 170)         # spine / center
C_JOINT = (255, 245, 200)

# (a, b, color, width)
BONES = [("neck", "nose", C_C, 4), ("lsho", "rsho", C_C, 5),
         ("neck", "lhip", C_C, 5), ("neck", "rhip", C_C, 5), ("lhip", "rhip", C_C, 5),
         ("lsho", "lelb", C_L, 4), ("lelb", "lwri", C_L, 4),
         ("lhip", "lkne", C_L, 5), ("lkne", "lank", C_L, 4),
         ("lank", "lheel", C_L, 3), ("lheel", "lbtoe", C_L, 3),
         ("rsho", "relb", C_R, 4), ("relb", "rwri", C_R, 4),
         ("rhip", "rkne", C_R, 5), ("rkne", "rank", C_R, 4),
         ("rank", "rheel", C_R, 3), ("rheel", "rbtoe", C_R, 3)]
for _side, _w, _c in (("r", "rwri", C_R), ("l", "lwri", C_L)):
    for _fn in ("th", "ix", "md", "rg", "pk"):
        BONES += [(_w, f"{_side}_{_fn}_prx", _c, 2),
                  (f"{_side}_{_fn}_prx", f"{_side}_{_fn}_mid", _c, 2),
                  (f"{_side}_{_fn}_mid", f"{_side}_{_fn}_dst", _c, 2),
                  (f"{_side}_{_fn}_dst", f"{_side}_{_fn}_tip", _c, 2)]


def draw_skeleton(img_arr, pts, scale=1.0, alpha=235):
    """Composite a pretty skeleton onto img_arr (H,W,3 uint8). pts: [K,2] pixel coords in KPN
    order; K may be 18 (body-only, older fit3d) or 58 (body + hands) - missing rows are skipped."""
    H, W = img_arr.shape[:2]
    S = 2
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    p = np.asarray(pts, dtype=np.float64) * S
    K = len(p)
    ws = scale * S
    # dark underlay first (all bones), then color pass
    for a, b, col, w in BONES:
        if I[a] >= K or I[b] >= K:
            continue
        pa, pb = p[I[a]], p[I[b]]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        d.line([tuple(pa), tuple(pb)], fill=(10, 12, 16, 220), width=int((w + 3) * ws))
    for a, b, col, w in BONES:
        if I[a] >= K or I[b] >= K:
            continue
        pa, pb = p[I[a]], p[I[b]]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        d.line([tuple(pa), tuple(pb)], fill=(*col, alpha), width=int(w * ws))
        for q in (pa, pb):
            r = 0.5 * w * ws
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=(*col, alpha))
    # joints: dark ring + light core (fingers: bone strokes only, no dots - too cluttered)
    for k in range(min(K, len(KPN))):
        if "_" in KPN[k]:                    # finger joints (body names have no underscore)
            continue
        q = p[k]
        if not np.isfinite(q).all():
            continue
        r = (5 if KPN[k] in ("lhip", "rhip", "neck") else 3.6) * ws
        d.ellipse([q[0] - r - 1.6 * ws, q[1] - r - 1.6 * ws,
                   q[0] + r + 1.6 * ws, q[1] + r + 1.6 * ws], fill=(10, 12, 16, 200))
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=(*C_JOINT, 255))
    # head
    nose = p[I["nose"]]
    if np.isfinite(nose).all():
        r = 9 * ws
        d.ellipse([nose[0] - r, nose[1] - r, nose[0] + r, nose[1] + r],
                  outline=(*C_C, alpha), width=int(2.5 * ws))
    ov = ov.resize((W, H), Image.LANCZOS)
    base = Image.fromarray(img_arr).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))


def draw_trail(img_arr, pts_seq, color=(255, 170, 60), width=3, fade=0.85):
    """Fading polyline trail: pts_seq [(x, y)] oldest->newest; alpha ramps to full at the head."""
    H, W = img_arr.shape[:2]
    if len(pts_seq) < 2:
        return img_arr
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    n = len(pts_seq)
    for i in range(1, n):
        a = int(30 + (255 - 30) * ((i / (n - 1)) ** 2) * fade)
        d.line([tuple(pts_seq[i - 1]), tuple(pts_seq[i])], fill=(*color, a), width=width)
    base = Image.fromarray(img_arr).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))
