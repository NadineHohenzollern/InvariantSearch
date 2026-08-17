# IVSN transformation-invariance experiments

This repository contains the Invariant Visual Search Network (IVSN) code and an
experiment for measuring robustness to rotations, scaling, shifts, skew, noise,
and blur. The experiment supports the original VGG backbone as well as several
Gist-based feature extractors.

## Installation

Python 3.8 or newer is recommended. Create and activate a virtual environment,
then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiment imports `Gist` with:

```python
from gist import Gist
```

`gist` is a project-specific model implementation, not part of Python's standard
library. Install the repository/package that provides this class, or add its
directory to `PYTHONPATH`, before running one of the Gist models. The plain VGG
model does not use a Gist checkpoint, although `models.py` currently imports the
class when the model module is loaded.

Main third-party libraries:

- PyTorch and torchvision for neural-network backbones and inference
- NumPy for numerical operations
- Pillow for stimulus transformations and rendering
- Matplotlib and pandas for plots and tabular summaries

Choose PyTorch and torchvision builds that match your CUDA installation. For a
CPU-only run, pass `--device cpu`.

## Project structure

```text
IVSN/
├── README.md
├── requirements.txt
├── images/
├── GIF/
└── codes/
    ├── ivsn_invariant_search.py   # backward-compatible CLI entry point
    ├── ivsn_invariance/
    │   ├── cli.py                 # arguments and experiment orchestration
    │   ├── runtime.py             # constants, seeds, and runtime geometry
    │   ├── domain.py              # trial and transformation data classes
    │   ├── data.py                # dataset and manifest I/O
    │   ├── imaging.py             # transformations and stimulus rendering
    │   ├── trials.py              # conditions and trial generation
    │   ├── models.py              # feature extractors and attention models
    │   ├── search.py              # IVSN fixation search
    │   ├── visualization.py       # attention and example figures
    │   └── reporting.py           # summaries, CSV files, and plots
    ├── model_weights/             # add local checkpoints here (not in Git)
    └── scripts/                   # SLURM run and test scripts
```

## Model weights (not included)

Model checkpoints are intentionally excluded from Git because they are too large
for a normal GitHub repository. After cloning the repository, copy the available
weights into:

```text
codes/model_weights/
```

Use these exact filenames:

```text
codes/model_weights/vgg_gist_model_epoch_25.pth
codes/model_weights/conv_gist_model_epoch_15.pth
codes/model_weights/conv_gist_mlp_model_epoch_10.pth
codes/model_weights/vgg_gist_imagenet64_epoch25.pth
```

For example, on Linux or macOS:

```bash
cp /path/to/weights/*.pth codes/model_weights/
```

On Windows PowerShell:

```powershell
Copy-Item C:\path\to\weights\*.pth codes\model_weights\
```

The default paths are resolved relative to the source files, not the current
working directory. The program therefore finds the weights whether it is started
from the repository root or from `codes/`. A custom location can still be passed
through the corresponding `--*-checkpoint` option.

## Dataset

The invariance experiment expects one folder per category below `--data-root`.
Each category folder must contain PNG images. Supported configurations contain
either six or eight categories:

```text
sheep, cattle, cats, horses, teddybears, kites[, dogs, elephants]
```

The alias `teddy_bears` is also accepted for `teddybears`.

The dataset used by the original IVSN experiments is available
[here](https://drive.google.com/file/d/1ti0MT860zGEUu18BCCe9QEBHa46yBnC_/view?usp=drive_link).

## Running the invariance experiment

From the repository root:

```bash
python codes/ivsn_invariant_search.py \
  --data-root /path/to/dataset \
  --out-dir codes/outputs/rotation_vgg \
  --transform-mode rotation \
  --model-kind vgg \
  --device cuda
```

The package entry point is equivalent when run from `codes/`:

```bash
cd codes
python -m ivsn_invariance --help
```

Available model kinds are `vgg`, `vgg_gist_pretrained`, `conv_gist`,
`conv_gist_mlp`, and `vgg_gist_imagenet64`. The scripts in `codes/scripts/`
provide complete SLURM examples for these variants. They determine the `codes/`
directory automatically and can therefore be submitted from any directory, for
example with `sbatch codes/scripts/run_vgg.sh`.

## Outputs

Each experiment writes trial manifests, per-trial JSON/CSV results, grouped
summaries, plots, and optional example visualizations below `--out-dir`.

## VGG feature-representation analysis

The feature-analysis entry point compares two search displays that are
pixel-identical except for the target transformation. It measures corresponding
target-region feature vectors at VGG feature-module cut points 16, 23, and 30,
using 5 x 5, 3 x 3, and 1 x 1 regions respectively. It also computes effective
receptive fields (ERFs) by taking the absolute input gradient for each selected
spatial cell and superimposing the resulting saliency maps.

```bash
python codes/analyze_vgg_features.py \
  --data-root /path/to/dataset \
  --out-dir codes/outputs/feature_rotation \
  --transform-mode rotation \
  --rotation-values 0 30 60 90 120 150 180 \
  --n-objects 8 \
  --device cuda
```

The output contains:

- `feature_cell_distances.csv`: Euclidean, channel-normalized RMS Euclidean,
  and cosine distances for every corresponding spatial cell.
- `feature_trial_distances.csv`: target-region means for every trial and layer.
- `grouped_feature_distances.csv`: means and 95% confidence intervals for all,
  target-identical, and target-different trials.
- `plots/`: grouped bar charts for strict cellwise, pooled, and spatially
  tolerant feature distances.
- `cue_target_cell_metrics.csv`, `cue_target_trial_metrics.csv`, and
  `grouped_cue_target_metrics.csv`: comparisons between the pooled cue and the
  original/transformed search-target representation. These make the
  target-identical versus target-different split directly meaningful.
- `cue_target_plots/`: cue-to-target similarity, similarity-loss, and distance
  plots for each layer.
- `search_performance.csv` and `grouped_search_performance.csv`: layer-30
  IVSN-compatible target attention scores, margins, ranks, probabilities, and
  fixation counts before and after the target transformation.
- `feature_performance_correlations.csv` and `correlation_plots/`: Pearson and
  Spearman associations between representation changes and search performance.
- `distance_matrix_examples/`: explicit 1 x 1, 3 x 3, and 5 x 5 cell-distance
  heatmaps for the selected examples.
- `erf_examples/`: paired ERF figures and compressed raw saliency arrays.
- `erf_example_metrics.csv`: scale-independent ERF distances plus mass overlap,
  centroid displacement, spread, and 90%-mass area for visualized examples.

ERF computation is substantially more expensive than feature extraction. Use
`--erf-examples-per-group 0` for quantitative-only runs, or choose another
positive count for more qualitative examples. Layer regions can be overridden,
for example with `--layer-windows 16:5 23:3 30:1`.

The strict corresponding-cell metric measures both feature change and spatial
rearrangement. Mean/max-pooled metrics and symmetric nearest-cell distances are
reported alongside it to distinguish invariance from local equivariance. Cue
features use a 32 x 32 input and adaptive max pooling by default, matching the
original VGG IVSN cue path at layer 30; use `--cue-size` only when intentionally
testing a different cue resolution.

## Original IVSN publication

The original IVSN model was published in *Nature Communications*:
[Finding any Waldo with zero-shot invariant and efficient visual search](https://www.nature.com/articles/s41467-018-06217-x).

```bibtex
@article{zhang2018finding,
  title={Finding any Waldo with zero-shot invariant and efficient visual search},
  author={Zhang, Mengmi and Feng, Jiashi and Ma, Keng Teck and Lim, Joo Hwee and Zhao, Qi and Kreiman, Gabriel},
  journal={Nature Communications},
  volume={9},
  number={1},
  pages={3730},
  year={2018},
  publisher={Nature Publishing Group UK London}
}
```

## License

Licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
Commercial use requires formal permission.
