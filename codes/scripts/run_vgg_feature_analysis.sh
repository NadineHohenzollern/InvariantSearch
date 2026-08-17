#!/bin/bash

#SBATCH --job-name=vgg_features
#SBATCH --output=logs/vgg_features_%j.out
#SBATCH --error=logs/vgg_features_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_DIR}"

source venv/bin/activate
mkdir -p logs

python analyze_vgg_features.py \
  --data-root data/coco_crops_transparent_8cat \
  --out-dir results/vgg_feature_rotation \
  --transform-mode rotation \
  --rotation-values 0 30 60 90 120 150 180 \
  --n-objects 8 \
  --device cuda \
  --smoothing-mode alpha \
  --smooth-target \
  --smooth-distractors \
  --layer-windows 16:5 23:3 30:1 \
  --erf-examples-per-group 1
