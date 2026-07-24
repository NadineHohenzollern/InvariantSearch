"""Runtime for IVSN invariance experiments."""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import math
import numpy as np
import random


ALT_CATEGORY_NAMES = {'teddybears': ['teddybears', 'teddy_bears']}


ALL_CATEGORIES = ['sheep', 'cattle', 'cats', 'horses', 'teddybears', 'kites', 'dogs', 'elephants', 'birds', 'bears', 'zebras', 'giraffes', 'umbrellas', 'backpacks', 'frisbees', 'suitcases']


IMAGE_SIZE = 720


OBJ_SIZE = 156 # default for circle arrangement


MIN_MARGIN = 10


RADIUS_6 = 220


RADIUS_8 = 240


CATEGORIES = None


N_POSITIONS = None


RADIUS = None


POSITIONS = None


MAX_FIXATIONS = None


# MAX_SIZE_CUTOUT = None


JITTER = 0.1  # percentage of container size -> determines how much center of cutoff can drift from center of container


DEFAULT_N_IDENTICAL = 150


DEFAULT_N_DIFFERENT = 100


EARLY_SUCCESS_FIXATIONS = 3


ORACLE_WINDOW = 45


SEED = 0


DEFAULT_GIST_CONFIG = dict(in_channels=1, mode='dynamic', fmax=0.35, fratio=1.7, k=0.52, n_scales=4, n_orientations=6, n_phases=2, scale=25, gaussian=True, gaussian_inverse=False, n_stds=3, dc_compensate=True, stride=4, energy=True, energy_mode='substitute', divisive_norm=False, pooling=None, pool_size=16, pool_stride=16, flatten=False)


CODES_DIR = Path(__file__).resolve().parents[1]
MODEL_WEIGHTS_DIR = CODES_DIR / 'model_weights'

DEFAULT_GIST_CHECKPOINTS = {
    'vgg_gist_pretrained': MODEL_WEIGHTS_DIR / 'vgg_gist_model_epoch_25.pth',
    'conv_gist': MODEL_WEIGHTS_DIR / 'conv_gist_model_epoch_15.pth',
    'conv_gist_mlp': MODEL_WEIGHTS_DIR / 'conv_gist_mlp_model_epoch_10.pth',
    'vgg_gist_imagenet64': MODEL_WEIGHTS_DIR / 'vgg_gist_imagenet64_epoch25.pth',
}


def get_categories(n_objects: int):
    if n_objects <= 16:
        return random.sample(ALL_CATEGORIES, n_objects)
    raise ValueError(f'Unsupported n_objects: {n_objects}. Maximal 16 allowed.')


def circle_positions(image_size: int, radius: int, n: int):
    center = image_size // 2
    pts = []
    start_angle = -math.pi / 2.0
    for i in range(n):
        angle = start_angle + 2 * math.pi * i / n
        x = center + int(round(math.cos(angle) * radius))
        y = center + int(round(math.sin(angle) * radius))
        pts.append((x, y))
    return pts

def _add_jitter(x: int, y: int, container_size: int):
    epsilon = round(JITTER * container_size)
    x += random.randint(-epsilon, epsilon)
    y += random.randint(-epsilon, epsilon)
    return (int(round(x)), int(round(y)))

def grid_positions(image_size: int, n_matrix: int, container_size: int = None):
    """Determines the 2D center coordinates of the object to be pasted on the 
    search image in a nxn grid map arrangement. 

    Parameters: 
        image_size     (int): width and length of the search image 
        n_matrix       (int): dimension of the nxn matrix 
        container_size (int): width and length of each object container 

    Returns: 
        []: list of object cutoff center coordinates in grid map fashion 
    """

    # determine container and margin size 
    if container_size is None:
        # pick values such that container_size = 2 * margin 
        margin = image_size / (3 * n_matrix + 1)
        container_size = 2 * margin
    else:
        margin = (image_size - n_matrix * container_size) / (n_matrix + 1)
        if margin < MIN_MARGIN:
            container_size = (image_size - (n_matrix + 1) * MIN_MARGIN) / n_matrix
            margin = MIN_MARGIN

    # determine max size of object cutout
    global OBJ_SIZE
    OBJ_SIZE = int(round(container_size))

    # determine object positions (cutout center) in grip map arrangement
    pts = []
    for row in range(n_matrix):
        y = margin + row * (container_size + margin) + container_size / 2
        for col in range(n_matrix):
            x = margin + col * (container_size + margin) + container_size / 2
            pts.append(_add_jitter(int(round(x)), int(round(y)), container_size))
    return pts

def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_font(size=18):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except Exception:
        return ImageFont.load_default()


def set_runtime_geometry_circle(n_objects: int):
    global POSITIONS
    _set_runtime_geometry(n_objects)
    radius = RADIUS_6 if n_objects == 6 else RADIUS_8
    POSITIONS = circle_positions(IMAGE_SIZE, radius, N_POSITIONS)

def set_runtime_geometry_grid(n_matrix: int, container_size: int):
    global POSITIONS
    _set_runtime_geometry(n_matrix**2)
    POSITIONS = grid_positions(IMAGE_SIZE, n_matrix, container_size)

def _set_runtime_geometry(n_objects: int):
    global CATEGORIES, N_POSITIONS, POSITIONS, MAX_FIXATIONS
    CATEGORIES = get_categories(n_objects)
    N_POSITIONS = n_objects
    MAX_FIXATIONS = n_objects

