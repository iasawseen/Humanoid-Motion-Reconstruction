"""Fast video rendering utilities: NVENC hardware encoding + torch-GPU point splatting.

The frame-loop renderers were CPU-bound in two places: x264 encoding of noise-like point panes
(2-3 fps even on 11 cores) and numpy per-pixel splatting. NVENC encodes at hundreds of fps
regardless of content entropy; splatting on the GPU turns the projection + z-order + scatter
into a few tensor ops.
"""
import subprocess

import numpy as np

try:
    import torch
    _DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:                                        # torch-less env: splat falls back
    torch = None
    _DEV = "cpu"


class VideoWriter:
    """ffmpeg subprocess writer: rawvideo in -> h264_nvenc (fallback libx264 veryfast)."""

    def __init__(self, path, w, h, fps, cq=27, nvenc=True):
        self.path, self.w, self.h = path, w, h
        codec = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
                 if nvenc else
                 ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(cq - 4)])
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}",
               "-i", "-", "-pix_fmt", "yuv420p"] + codec + [path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.nvenc = nvenc

    def write(self, frame):
        assert frame.shape == (self.h, self.w, 3), frame.shape
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc} for {self.path}")


def gpu_splat(uv, z, colors, H, W, splat=2, bg=(0, 0, 0)):
    """Painter's-ordered point splat on the GPU. uv [N,2] float, z [N] (far = drawn first),
    colors [N,3] uint8 -> (H,W,3) uint8 numpy."""
    if torch is None:
        raise RuntimeError("torch unavailable")
    uvt = torch.as_tensor(uv, dtype=torch.float32, device=_DEV)
    zt = torch.as_tensor(z, dtype=torch.float32, device=_DEV)
    ct = torch.as_tensor(colors, dtype=torch.uint8, device=_DEV)
    order = torch.argsort(zt, descending=True)           # far first; later writes win
    ui = uvt[order, 0].long().clamp(0, W - splat)
    vi = uvt[order, 1].long().clamp(0, H - splat)
    cc = ct[order]
    img = torch.empty((H, W, 3), dtype=torch.uint8, device=_DEV)
    img[:, :, 0] = bg[0]; img[:, :, 1] = bg[1]; img[:, :, 2] = bg[2]
    flat = img.view(-1, 3)
    for dy in range(splat):
        for dx in range(splat):
            flat[(vi + dy) * W + (ui + dx)] = cc
    return img.cpu().numpy()
