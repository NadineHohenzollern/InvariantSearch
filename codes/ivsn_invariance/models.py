"""Models for IVSN invariance experiments."""

import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from gist import Gist
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from . import runtime
from .runtime import DEFAULT_GIST_CHECKPOINTS, DEFAULT_GIST_CONFIG, IMAGE_SIZE, ORACLE_WINDOW


class PlainVGGFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights)
        self.features = nn.Sequential(*list(vgg.features.children())[:30])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class VGGGistPretrainedFeatureExtractor(nn.Module):
    """
    Older VGG+Gist model:
    gist -> fusion(25->128) -> VGG tail from layer 10 onward
    """

    def __init__(self, gist_config: dict):
        super().__init__()
        self.gist = Gist(**gist_config)
        self.fusion = nn.Sequential(nn.Conv2d(self.gist.out_channels, 128, kernel_size=1, stride=1), nn.ReLU())
        vgg = models.vgg16(weights='DEFAULT')
        self.features = nn.Sequential(*list(vgg.features.children())[10:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gist(x)
        x = self.fusion(x)
        x = self.features(x)
        return x


class VGGNetGistFeatureExtractor(nn.Module):
    """
    Older VGG+Gist model:
    gist -> fusion(25->128) -> VGG tail from layer 10 onward
    """

    def __init__(self, gist_config: dict):
        super().__init__()
        self.gist = Gist(**gist_config)
        self.fusion = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(self.gist.out_channels, 128, kernel_size=1, stride=1, bias=False)),
            ("relu", nn.ReLU())
        ]))
        vgg = models.vgg16(weights='DEFAULT')
        self.features = nn.Sequential(*list(vgg.features.children())[10:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gist(x)
        x = self.fusion(x)
        x = self.features(x)
        return x


class VGGGistImageNet64FeatureExtractor(nn.Module):
    """
    New 64-trained VGG+Gist model from colleague example:
    gist -> fusion(25->64) -> VGG layers [5:9] + [10:]
    """

    def __init__(self, gist_config: dict):
        super().__init__()
        self.gist = Gist(**gist_config)
        vgg = models.vgg16(weights='DEFAULT')
        self.fusion = nn.Sequential(nn.Conv2d(self.gist.out_channels, 64, kernel_size=1, stride=1), nn.ReLU())
        self.features = nn.Sequential(*list(vgg.features.children())[5:9], *list(vgg.features.children())[10:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gist(x)
        x = self.fusion(x)
        x = self.features(x)
        return x


class ConvGistFeatureExtractor(nn.Module):

    def __init__(self, gist_config: dict, conv_bias: bool=True):
        super().__init__()
        out_channels_base = 128
        self.gist = Gist(**gist_config)
        self.features = nn.Sequential(OrderedDict([
            ('block1_conv', nn.Conv2d(self.gist.out_channels, out_channels_base, kernel_size=5, stride=2, padding=2, bias=conv_bias)),
            ('block1_relu', nn.ReLU()),
            ('block1_norm', nn.BatchNorm2d(out_channels_base)),
            ('block2_conv', nn.Conv2d(out_channels_base, out_channels_base * 2, kernel_size=3, stride=2, padding=1, bias=conv_bias)),
            ('block2_relu', nn.ReLU()),
            ('block2_norm', nn.BatchNorm2d(out_channels_base * 2)),
            ('block3_conv', nn.Conv2d(out_channels_base * 2, out_channels_base * 4, kernel_size=3, stride=2, padding=1, bias=conv_bias)),
            ('block3_relu', nn.ReLU()),
            ('block3_norm', nn.BatchNorm2d(out_channels_base * 4)),
            ('block4_conv', nn.Conv2d(out_channels_base * 4, out_channels_base * 4, kernel_size=3, stride=1, padding=1, bias=conv_bias)),
            ('block4_relu', nn.ReLU()),
            ('block4_norm', nn.BatchNorm2d(out_channels_base * 4))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gist(x)
        x = self.features(x)
        return x


class ConvGistMLPFeatureExtractor(nn.Module):

    def __init__(self, gist_config: dict, conv_bias: bool=True):
        super().__init__()
        out_channels_base = 128
        self.gist = Gist(**gist_config)
        self.features = nn.Sequential(OrderedDict([
            ('block1_conv', nn.Conv2d(self.gist.out_channels, out_channels_base, kernel_size=5, stride=2, padding=2, bias=conv_bias)),
            ('block1_relu', nn.ReLU()),
            ('block1_norm', nn.BatchNorm2d(out_channels_base)),
            ('block2_conv', nn.Conv2d(out_channels_base, out_channels_base * 2, kernel_size=3, stride=2, padding=1, bias=conv_bias)),
            ('block2_relu', nn.ReLU()),
            ('block2_norm', nn.BatchNorm2d(out_channels_base * 2)),
            ('block3_conv', nn.Conv2d(out_channels_base * 2, out_channels_base * 4, kernel_size=3, stride=2, padding=1, bias=conv_bias)),
            ('block3_relu', nn.ReLU()),
            ('block3_norm', nn.BatchNorm2d(out_channels_base * 4)),
            ('block4_conv', nn.Conv2d(out_channels_base * 4, out_channels_base * 4, kernel_size=3, stride=1, padding=1, bias=conv_bias)),
            ('block4_relu', nn.ReLU()),
            ('block4_norm', nn.BatchNorm2d(out_channels_base * 4))
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gist(x)
        x = self.features(x)
        return x


def load_feature_extractor_weights(feature_extractor: nn.Module, checkpoint_path: Path, allowed_prefixes: Tuple[str, ...], device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model_state = feature_extractor.state_dict()
    cleaned = {}
    skipped = []
    for k, v in state_dict.items():
        nk = k.replace('module.', '')
        if not nk.startswith(allowed_prefixes):
            continue
        if nk not in model_state:
            skipped.append((nk, 'missing_in_model'))
            continue
        if model_state[nk].shape != v.shape:
            skipped.append((nk, f'shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[nk].shape)}'))
            continue
        cleaned[nk] = v
    missing, unexpected = feature_extractor.load_state_dict(cleaned, strict=False)
    print(f'Loaded checkpoint from: {checkpoint_path}')
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)
    if skipped:
        print('Skipped keys:')
        for name, reason in skipped:
            print(f'  {name}: {reason}')


class BaseAttentionModel:

    def __init__(self, device: str = 'cpu', attention_padding: int = 0):

        self.device = torch.device(device)
        self.attention_padding = attention_padding

        self.backbone: nn.Module
        self.maxpool: nn.Module
        self.cue_transform: nn.Module
        self.search_transform: nn.Module

    def preprocess(self, img: Image.Image, cue: bool = False) -> torch.Tensor:
        if cue:
            img = self.cue_transform(img)
        else:
            img = self.search_transform(img)

        return img.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def feature_map(self, img: Image.Image, cue: bool = False) -> torch.Tensor:
        x = self.preprocess(img, cue=cue)
        x = self.backbone(x)
        if cue:
            x = self.maxpool(x)
        return x
    
    @torch.no_grad()
    def position_scores(
        self,
        cue_img: Image.Image,
        search_img: Image.Image,
        positions: List[Tuple[int, int]]
    ):
        cue_feat = self.feature_map(cue_img, cue=True)
        search_feat = self.feature_map(search_img)
    
        attn = (search_feat * cue_feat).sum(dim=1, keepdim=True)
        attn = F.relu(attn)

        if self.attention_padding > 0:
            p = self.attention_padding
            attn = F.pad(attn, (p, p, p, p), mode='constant', value=0)

        attn = attn / (attn.max() + 1e-8)
        attn_up = F.interpolate(attn, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bicubic', align_corners=False)
        attn_np = attn_up.squeeze().detach().cpu().numpy()

        scores = []
        half = ORACLE_WINDOW // 2
        for x, y in positions:
            x1 = max(0, x - half)
            x2 = min(IMAGE_SIZE, x + half)
            y1 = max(0, y - half)
            y2 = min(IMAGE_SIZE, y + half)
            scores.append(float(attn_np[y1:y2, x1:x2].mean()))

        return (attn_np, np.asarray(scores, dtype=np.float32))


class VGGAttentionModel(BaseAttentionModel):

    def __init__(
        self,
        device: str = 'cpu',
        attention_padding: int = 0,
    ):
        super().__init__(device=device, attention_padding=attention_padding)
        
        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features.eval().to(self.device)
        self.backbone = vgg[:30].eval().to(self.device)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        for p in self.backbone.parameters():
            p.requires_grad = False

        t = weights.transforms()
        self.search_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=t.mean, std=t.std)
        ])
        self.cue_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((32, 32)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=t.mean, std=t.std)
        ])


class VOneNetAttentionModel(BaseAttentionModel):

    def __init__(
        self,
        backbone: str = "alexnet",
        device: str = 'cpu',
        attention_padding: int = 0,
    ):
        super().__init__(device=device, attention_padding=attention_padding)

        from vonenet import get_model

        vonenet_model = get_model(
            model_arch = backbone,
            pretrained = True,
            map_location = device,
        ).module

        if backbone == "alexnet":

            self.backbone = nn.Sequential(
                vonenet_model.vone_block,
                vonenet_model.bottleneck,
                vonenet_model.model.features[:-1]  # Exclude the last maxpool layer,
            )
            self.maxpool = vonenet_model.model.features[-1]  # The last maxpool layer

        elif backbone == "resnet50":
            self.backbone = nn.Sequential(
                vonenet_model.vone_block,
                vonenet_model.bottleneck,
                vonenet_model.model.layer1,
                vonenet_model.model.layer2,
                vonenet_model.model.layer3,
            )
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) 

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = self.backbone.eval().to(self.device)
        self.maxpool = self.maxpool.eval().to(self.device)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.search_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.cue_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((32, 32)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


class GistAttentionModel(BaseAttentionModel):

    def __init__(
        self,
        feature_extractor: nn.Module,
        device: str = 'cpu',
        grayscale_input: bool = True,
        image_size: int = 224,
        attention_padding: int = 0
    ):
        super().__init__(device=device, attention_padding=attention_padding)

        self.backbone = feature_extractor.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        if isinstance(self.backbone.features[-1], nn.MaxPool2d):
            # If the last layer is a maxpool (VGG Model), we separate it from the backbone to allow for
            # different processing of cue and search images.
            self.maxpool = self.backbone.features[-1]
            self.backbone.features[-1] = nn.Identity() 
        else:
            # Otherwise the model is ConvGist, which does not have a maxpool at the end,
            # so we dont use a maxpool. Now search image output will be 7 x 7 instead.
            self.maxpool = nn.Identity()

        search_transform = [transforms.ToImage(), transforms.Resize((image_size, image_size))]
        if grayscale_input:
            search_transform.append(transforms.Grayscale(num_output_channels=1))
        search_transform.append(transforms.ToDtype(torch.float32, scale=True))
        self.search_transform = transforms.Compose(search_transform)

        cue_transform = [transforms.ToImage(), transforms.Resize((image_size // 7, image_size // 7))]
        if grayscale_input:
            cue_transform.append(transforms.Grayscale(num_output_channels=1))
        cue_transform.append(transforms.ToDtype(torch.float32, scale=True))
        self.cue_transform = transforms.Compose(cue_transform)


def get_gist_config(args) -> dict:
    
    gist_config = deepcopy(DEFAULT_GIST_CONFIG)
    if args.model_kind == 'vgg_gist_imagenet64' and args.gist_image_size == 64:
        gist_config['stride'] = 1

    return gist_config


def build_attention_model(args) -> BaseAttentionModel:

    if args.model_kind == 'vgg':
        return VGGAttentionModel(
            device=args.device,
            attention_padding=args.attention_padding
        )

    if args.model_kind == 'vonenet':
        return VOneNetAttentionModel(
            backbone=args.vonenet_backbone,
            device=args.device,
            attention_padding=args.attention_padding
        )
    
    gist_config = get_gist_config(args)

    if "gist" in args.model_kind:

        if args.model_kind == 'vgg_gist_pretrained':
            feature_extractor = VGGGistPretrainedFeatureExtractor(gist_config)
            checkpoint = Path(args.vgg_gist_checkpoint or DEFAULT_GIST_CHECKPOINTS['vgg_gist_pretrained'])
            allowed_prefixes = ('gist.', 'fusion.', 'features.')


        elif args.model_kind == 'vgg_gist_new':
            feature_extractor = VGGNetGistFeatureExtractor(gist_config)
            checkpoint = Path(args.vgg_gist_checkpoint or DEFAULT_GIST_CHECKPOINTS['vgg_gist_new'])
            allowed_prefixes = ('gist.', 'fusion.', 'features.')

        elif args.model_kind == 'vgg_gist_imagenet64':
            feature_extractor = VGGGistImageNet64FeatureExtractor(gist_config)
            checkpoint = Path(args.vgg_gist_imagenet64_checkpoint or DEFAULT_GIST_CHECKPOINTS['vgg_gist_imagenet64'])
            allowed_prefixes = ('gist.', 'fusion.', 'features.')

        elif args.model_kind == 'conv_gist':
            feature_extractor = ConvGistFeatureExtractor(gist_config)
            checkpoint = Path(args.conv_gist_checkpoint or DEFAULT_GIST_CHECKPOINTS['conv_gist'])
            allowed_prefixes = ('gist.', 'features.')

        elif args.model_kind == 'conv_gist_mlp':
            feature_extractor = ConvGistMLPFeatureExtractor(gist_config)
            checkpoint = Path(args.conv_gist_mlp_checkpoint or DEFAULT_GIST_CHECKPOINTS['conv_gist_mlp'])
            allowed_prefixes = ('gist.', 'features.')

        else:
            raise ValueError(f'Unsupported model_kind: {args.model_kind}')

        load_feature_extractor_weights(
            feature_extractor,
            checkpoint,
            allowed_prefixes=allowed_prefixes,
            device=torch.device(args.device)
        )

        return GistAttentionModel(
            feature_extractor,
            device=args.device,
            grayscale_input=True,
            image_size=args.gist_image_size,
            attention_padding=args.attention_padding
        )

    raise ValueError(f'Unsupported model_kind: {args.model_kind}')
