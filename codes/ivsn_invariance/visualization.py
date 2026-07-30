"""Visualization for IVSN invariance experiments."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import runtime
from .domain import Trial
from .models import BaseAttentionModel
from .runtime import ensure_dir, load_font
from .search import ivsn_fixation_search, render_trial_from_struct
from .imaging import render_target
from .domain import TransformSpec


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = (arr.min(), arr.max())
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def make_attention_heatmap(attn_np: np.ndarray) -> Image.Image:
    attn_norm = normalize_map(attn_np)
    cmap = plt.get_cmap('jet')
    rgba = cmap(attn_norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def make_attention_overlay(search_img: Image.Image, attn_np: np.ndarray, alpha: float=0.45) -> Image.Image:
    heatmap = make_attention_heatmap(attn_np).resize(search_img.size)
    return Image.blend(search_img.convert('RGB'), heatmap, alpha=alpha)


def draw_fixation_path(search_img: Image.Image, fixations: List[Tuple[int, int]], target_position: int) -> Image.Image:
    img = search_img.copy().convert('RGB')
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    for i, (x, y) in enumerate(runtime.POSITIONS):
        color = (0, 200, 0) if i == target_position else (255, 255, 255)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
        draw.text((x + 8, y - 10), f'P{i}', fill=color, font=font)
    for i in range(len(fixations) - 1):
        x1, y1 = fixations[i]
        x2, y2 = fixations[i + 1]
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=3)
    for step, (x, y) in enumerate(fixations, start=1):
        r = 9
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=3)
        draw.text((x + 10, y - 10), str(step), fill=(255, 0, 0), font=font)
    return img


def save_combined_figure(cue_img: Image.Image, search_img: Image.Image, overlay_img: Image.Image, fix_img: Image.Image, out_path: Path, title: str):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=11)
    axs = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    for ax, img, ttl in zip(axs, [cue_img, search_img, overlay_img, fix_img], ['Cue', 'Search image', 'Attention overlay', 'IVSN path']):
        ax.imshow(np.array(img))
        ax.set_title(ttl)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_cue_compare(cue_orig: Image.Image, cue_t: Image.Image, out_path: Path, title: str):
    fig = plt.figure(figsize=(6, 3))
    fig.suptitle(title, fontsize=11)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(np.array(cue_orig))
    ax1.set_title('Cue')
    ax1.axis('off')
    ax2.imshow(np.array(cue_t))
    ax2.set_title('Target')
    ax2.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_examples(model: BaseAttentionModel, trials: List[Trial], out_dir: Path, save_examples: int, args):

    ex_root = out_dir / 'examples'
    ensure_dir(ex_root / 'combined')
    ensure_dir(ex_root / 'cue_compare')
    n = min(save_examples, len(trials))

    for idx in range(n):
        trial = trials[idx]

        cue_orig, cue_t, search_img = render_trial_from_struct(trial, args)
        target_orig, target_t = render_target(Path(trial.target_path), args, TransformSpec(**trial.target_transform))

        attn_np, scores = model.position_scores(cue_t, search_img, runtime.POSITIONS)

        _, fix_centers, found, _ = ivsn_fixation_search(scores, trial.target_position, runtime.MAX_FIXATIONS)

        overlay = make_attention_overlay(search_img, attn_np, alpha=0.45)
        fix_img = draw_fixation_path(search_img, fix_centers, trial.target_position)

        stem = f'trial_{idx:03d}'
        title = f'{stem} | model={args.model_kind} | gist_size={args.gist_image_size} | type={trial.trial_type} | cat={trial.target_category} | target=P{trial.target_position}'

        save_combined_figure(cue_t, search_img, overlay, fix_img, ex_root / 'combined' / f'{stem}_combined.png', title)
        save_cue_compare(cue_t, target_t, ex_root / 'cue_compare' / f'{stem}_cue_compare.png', title)
