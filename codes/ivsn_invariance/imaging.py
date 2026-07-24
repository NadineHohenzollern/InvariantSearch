"""Imaging for IVSN invariance experiments."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from . import runtime
from .data import load_rgba
from .domain import TransformSpec
from .runtime import IMAGE_SIZE


def affine_from_spec(spec: TransformSpec) -> Tuple[float, float, float, float, float, float]:
    cx = (runtime.OBJ_SIZE - 1) / 2.0
    cy = (runtime.OBJ_SIZE - 1) / 2.0
    sx = max(1e-06, spec.scale)
    sy = max(1e-06, spec.scale)
    kx = math.tan(math.radians(spec.skew_x_deg))
    ky = math.tan(math.radians(spec.skew_y_deg))
    theta = math.radians(spec.rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    K = np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]], dtype=np.float64)
    R = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    T_center = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    T_uncenter = np.array([[1, 0, cx + spec.shift_x], [0, 1, cy + spec.shift_y], [0, 0, 1]], dtype=np.float64)
    M_forward = T_uncenter @ R @ K @ S @ T_center
    M_inv = np.linalg.inv(M_forward)
    a, b, c = (M_inv[0, 0], M_inv[0, 1], M_inv[0, 2])
    d, e, f = (M_inv[1, 0], M_inv[1, 1], M_inv[1, 2])
    return (float(a), float(b), float(c), float(d), float(e), float(f))


def make_boundary_cosine_taper_mask(alpha: np.ndarray, width_px: float) -> np.ndarray:
    silhouette = alpha > 0
    if width_px <= 0 or not silhouette.any():
        return silhouette.astype(np.float32)
    dist = distance_transform_edt(silhouette)
    mask = np.ones_like(dist, dtype=np.float32)
    taper = (dist > 0) & (dist < width_px)
    t = dist[taper] / width_px
    mask[taper] = 0.5 * (1.0 - np.cos(np.pi * t))
    mask[~silhouette] = 0.0
    return mask


def apply_edge_taper_rgba(img: Image.Image, width_px: float) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    taper = make_boundary_cosine_taper_mask(alpha, width_px)
    alpha_new = np.clip(alpha * taper, 0.0, 1.0)
    arr[:, :, 3] = (255.0 * alpha_new).astype(np.uint8)
    return Image.fromarray(arr, mode='RGBA')


def apply_alpha_boundary_softening_rgba(img: Image.Image, blur_radius: float) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)
    alpha = Image.fromarray(arr[:, :, 3], mode='L')
    alpha_blur = alpha.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    arr[:, :, 3] = np.array(alpha_blur, dtype=np.uint8)
    return Image.fromarray(arr, mode='RGBA')


def apply_boundary_mask_rgba(img: Image.Image, args) -> Image.Image:
    if args.smoothing_mode == 'none':
        return img
    if args.smoothing_mode == 'alpha':
        return apply_alpha_boundary_softening_rgba(img, blur_radius=args.alpha_soften_blur_radius)
    if args.smoothing_mode == 'cosine':
        return apply_edge_taper_rgba(img, width_px=args.edge_taper_width_px)
    raise ValueError(f'Unsupported smoothing mode: {args.smoothing_mode}')


def apply_degradations_rgba(img: Image.Image, spec: TransformSpec) -> Image.Image:
    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3].copy()
    if spec.blur_radius > 0:
        rgb_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode='RGB')
        rgb_img = rgb_img.filter(ImageFilter.GaussianBlur(radius=float(spec.blur_radius)))
        rgb = np.array(rgb_img, dtype=np.float32)
    if spec.noise_std > 0:
        noise = np.random.normal(0.0, 255.0 * float(spec.noise_std), size=(rgb.shape[0], rgb.shape[1], 1))
        rgb = rgb + noise
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    out = np.zeros_like(arr)
    out[:, :, :3] = rgb
    out[:, :, 3] = alpha
    return Image.fromarray(out, mode='RGBA')


def apply_transform_rgba(img: Image.Image, spec: TransformSpec, args, apply_smoothing: bool) -> Image.Image:
    coeffs = affine_from_spec(spec)
    out = img.transform((runtime.OBJ_SIZE, runtime.OBJ_SIZE), Image.AFFINE, coeffs, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    out = apply_degradations_rgba(out, spec)
    if apply_smoothing:
        out = apply_boundary_mask_rgba(out, args)
    return out


def alpha_paste_rgb(canvas_rgb: Image.Image, obj_rgba: Image.Image, center_xy: Tuple[int, int]) -> Image.Image:
    x = int(center_xy[0] - obj_rgba.width // 2)
    y = int(center_xy[1] - obj_rgba.height // 2)
    canvas_rgba = canvas_rgb.convert('RGBA')
    canvas_rgba.alpha_composite(obj_rgba, (x, y))
    return canvas_rgba.convert('RGB')


def render_cue(cue_path: Path, args, transform_spec: Optional[TransformSpec]=None) -> Tuple[Image.Image, Image.Image]:
    cue_rgba_orig = load_rgba(cue_path).resize((runtime.OBJ_SIZE, runtime.OBJ_SIZE), Image.BICUBIC)
    if transform_spec is None:
        transform_spec = TransformSpec()
    cue_rgba_t = apply_transform_rgba(cue_rgba_orig, transform_spec, args, apply_smoothing=args.smooth_cue)
    cue_orig = Image.new('RGB', (runtime.OBJ_SIZE, runtime.OBJ_SIZE), (128, 128, 128))
    cue_orig = alpha_paste_rgb(cue_orig, cue_rgba_orig, (runtime.OBJ_SIZE // 2, runtime.OBJ_SIZE // 2))
    cue_t = Image.new('RGB', (runtime.OBJ_SIZE, runtime.OBJ_SIZE), (128, 128, 128))
    cue_t = alpha_paste_rgb(cue_t, cue_rgba_t, (runtime.OBJ_SIZE // 2, runtime.OBJ_SIZE // 2))
    return (cue_orig, cue_t)


def render_search_display(target_path: Path, distractor_paths: List[Path], target_position: int, target_transform: TransformSpec, distractor_transforms: List[TransformSpec], args) -> Image.Image:
    canvas = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    target_rgba = apply_transform_rgba(load_rgba(target_path).resize((runtime.OBJ_SIZE, runtime.OBJ_SIZE), Image.BICUBIC), target_transform, args, apply_smoothing=args.smooth_target)
    canvas = alpha_paste_rgb(canvas, target_rgba, runtime.POSITIONS[target_position])
    remaining_positions = [i for i in range(runtime.N_POSITIONS) if i != target_position]
    for dpath, pidx, dspec in zip(distractor_paths, remaining_positions, distractor_transforms):
        dist_rgba = apply_transform_rgba(load_rgba(dpath).resize((runtime.OBJ_SIZE, runtime.OBJ_SIZE), Image.BICUBIC), dspec, args, apply_smoothing=args.smooth_distractors)
        canvas = alpha_paste_rgb(canvas, dist_rgba, runtime.POSITIONS[pidx])
    return canvas
