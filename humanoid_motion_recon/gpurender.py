#!/usr/bin/env python
"""Shared torch-CUDA rasterization utilities for the humanoid_motion_recon video renderers.

Design (see README): GPU handles the HEAVY per-frame ops - point-cloud splats,
image resize, pane compositing; small overlays (58-joint skeletons, text) stay on
PIL/skel_draw where they are already correct and cost ~1 ms. Canvases are HxWx3 uint8
CUDA tensors; transfer to numpy only once per frame for the NVENC writer (fastvid).

All functions are frame-rate-agnostic and hold no global state.
"""
import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canvas(h, w, color=(0, 0, 0)):
    c = torch.empty((h, w, 3), dtype=torch.uint8, device=DEV)
    c[:] = torch.tensor(color, dtype=torch.uint8, device=DEV)
    return c


def disk(r):
    """Pixel-offset stencil [(dy,dx)...] of a radius-r disk."""
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1, device=DEV),
                            torch.arange(-r, r + 1, device=DEV), indexing="ij")
    m = (dy * dy + dx * dx) <= r * r
    return torch.stack([dy[m], dx[m]], 1)


DISK0, DISK1, DISK2, DISK3 = disk(0), disk(1), disk(2), disk(3)


def splat(cv, uv, color, stencil=DISK0):
    """Stamp stencil-shaped points. uv [M,2] float (u,v) pixels; color uint8[3] or [M,3]."""
    if uv.numel() == 0:
        return
    H, W = cv.shape[:2]
    q = uv.round().long()
    pts = q[:, None, :] + stencil[None, :, [1, 0]]       # [M,K,2] (u,v)
    K = stencil.shape[0]
    if torch.is_tensor(color) and color.dim() == 2:
        col = color[:, None, :].expand(-1, K, -1).reshape(-1, 3)
    else:
        col = torch.as_tensor(color, dtype=torch.uint8, device=DEV)
    pts = pts.reshape(-1, 2)
    m = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    if torch.is_tensor(col) and col.dim() == 2:
        cv[pts[m][:, 1], pts[m][:, 0]] = col[m]
    else:
        cv[pts[m][:, 1], pts[m][:, 0]] = col


def lines(cv, p0, p1, color, stencil=DISK1):
    """Rasterize thick segments p0->p1 ([B,2] float pixels)."""
    if p0.numel() == 0:
        return
    L = (p1 - p0).norm(dim=1).clamp(min=1.0)
    npt = int(L.max().item()) + 2
    s = torch.linspace(0, 1, npt, device=DEV)
    pts = p0[:, None, :] + (p1 - p0)[:, None, :] * s[None, :, None]
    splat(cv, pts.reshape(-1, 2), color, stencil)


def splat_cloud(cv, uv, z, rgb, stencil=DISK0):
    """Painter's-order point cloud: farthest first so near points overwrite.
    uv [M,2] float pixels, z [M] depth (bigger = farther), rgb [M,3] uint8."""
    order = torch.argsort(z, descending=True)
    splat(cv, uv[order], rgb[order], stencil)


def resize(img, h, w):
    """uint8 HxWx3 (numpy or tensor) -> h x w x 3 uint8 CUDA tensor, antialiased."""
    t = torch.as_tensor(np.asarray(img)).to(DEV) if not torch.is_tensor(img) else img.to(DEV)
    t = t.permute(2, 0, 1)[None].float()
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", antialias=True)
    return t[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8)


def to_np(cv):
    return cv.cpu().numpy()


def from_np(a):
    return torch.as_tensor(np.ascontiguousarray(a)).to(DEV)


def colormap_turbo(x):
    """x [H,W] float 0..1 -> HxWx3 uint8 CUDA (compact turbo-like polynomial ramp)."""
    x = x.clamp(0, 1)
    r = (0.14 + 4.5 * x - 4.0 * x * x).clamp(0, 1)
    g = (torch.sin(x * 3.1416) ** 1.5).clamp(0, 1)
    b = (1.0 - 2.4 * (x - 0.25).abs()).clamp(0, 1) * (x < 0.65) + (x >= 0.65) * (3.0 * (x - 0.65)).clamp(0, 0.35)
    return (torch.stack([r, g, b], -1) * 255).to(torch.uint8)


# ---- torch-native mesh rendering (surface splatting; replaces pyrender for matte meshes)

def sample_mesh(faces_np, verts0, density=8.0):
    """One-time surface sampling: area-weighted barycentric points on each face.
    Returns (face_idx [M], bary [M,3]) - fixed across frames (constant topology)."""
    F = torch.as_tensor(faces_np, dtype=torch.long, device=DEV)
    v = torch.as_tensor(verts0, dtype=torch.float32, device=DEV)
    a = v[F[:, 1]] - v[F[:, 0]]
    b = v[F[:, 2]] - v[F[:, 0]]
    area = 0.5 * torch.linalg.norm(torch.linalg.cross(a, b), dim=1)
    n = (area / area.mean() * density).clamp(1, 400).long()
    fidx = torch.repeat_interleave(torch.arange(len(F), device=DEV), n)
    g = torch.Generator(device="cpu").manual_seed(0)     # deterministic sampling
    r1 = torch.rand(len(fidx), generator=g).to(DEV)
    r2 = torch.rand(len(fidx), generator=g).to(DEV)
    s = r1.sqrt()
    bary = torch.stack([1 - s, s * (1 - r2), s * r2], 1)
    return F, fidx, bary


def mesh_splat(cv, verts, F, fidx, bary, K, base_color=(190, 217, 242),
               ambient=0.45, stencil=None):
    """Shade + z-ordered splat of a camera-frame mesh (x right, y down, z forward).
    verts [V,3] float tensor; K = (fx, fy, cx, cy); headlight Lambertian, two-sided."""
    tri = verts[F[fidx]]                                 # [M,3,3]
    pts = (tri * bary[:, :, None]).sum(1)
    fn = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-9)
    z = pts[:, 2]
    m = z > 0.05
    pts, fn, z = pts[m], fn[m], z[m]
    fx, fy, cx, cy = K
    uv = torch.stack([fx * pts[:, 0] / z + cx, fy * pts[:, 1] / z + cy], 1)
    view = pts / (pts.norm(dim=1, keepdim=True) + 1e-9)
    shade = ambient + (1 - ambient) * (fn * view).sum(1).abs().clamp(0, 1)
    col = (torch.as_tensor(base_color, dtype=torch.float32, device=DEV)[None]
           * shade[:, None]).clamp(0, 255).to(torch.uint8)
    splat_cloud(cv, uv, z, col, stencil if stencil is not None else DISK1)
