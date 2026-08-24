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

## Isolated-target activation and ERF analysis

`analyze_target_features.py` implements a complementary target-only analysis.
Unlike `analyze_vgg_features.py`, it does not render a search display. Each
original and transformed target is rendered on the same gray target canvas,
resized to 32 x 32, and passed through ImageNet VGG16.

To reuse exactly the 300 targets from an existing feature-analysis run, provide
its per-trial CSV. Repeated condition/layer rows are deduplicated by `unique_id`:

```bash
python codes/analyze_target_features.py \
  --data-root /path/to/dataset \
  --targets-csv codes/outputs/vgg_feature_rotation_extended/feature_trial_distances.csv \
  --out-dir codes/outputs/target_activation_rotation \
  --transform-mode rotation \
  --rotation-values 0 30 60 90 120 150 180 \
  --layers 16 23 30 \
  --input-size 32 \
  --erf-images-per-class 3 \
  --device cuda
```

The same entry point supports `scale`, `shift_x`, `shift_y`, `skew_x`,
`skew_y`, `noise`, `blur`, and `mixed` transformation modes. If neither
`--targets-csv` nor `--load-base-manifest` is supplied, a new set of 120
target-identical and 180 target-different base trials is sampled and saved.

The output contains:

- `target_trials.csv`: the deduplicated target trials used by the run.
- `target_activation_trial_metrics.csv`: per-target, per-condition and
  per-layer elementwise activation differences. Mean absolute difference is
  the primary measure; RMS, relative mean absolute difference and cosine
  distance are included as complementary measures.
- `grouped_target_activation_metrics.csv`: all/target-identical/
  target-different means, across-trial standard deviations and 95% confidence
  intervals.
- `activation_plots/`: per-layer transformation plots with standard-deviation
  error bars.
- `selected_erf_targets.csv`: deterministic unique targets selected per class.
- `erf_examples/`: original-image ERF overlay, transformed-image ERF overlay,
  absolute normalized difference, and compressed raw arrays for every selected
  layer and transformation.
- `target_erf_metrics.csv`: quantitative ERF overlap, distance, centroid,
  spread, and area summaries for the qualitative examples.
- `erf_aligned_examples/`: presentation-oriented 3 x 3 figures (one row per
  layer). The first two columns show the original and transformed ERFs. In the
  third column, the transformed ERF is inverse-warped into the original object
  coordinates: green is shared sensitivity, blue is decreased/lost
  sensitivity, and orange is increased/new sensitivity.
- `grouped_target_erf_metrics.csv` and `erf_alignment_plots/`: aligned versus
  unaligned ERF overlap/similarity and the gain produced by geometric
  alignment. These help separate a spatially moved response from a genuine
  change in what drives the layer.

For the default 32 x 32 input, the selected activations have shapes 256 x 8 x 8
at layer 16, 512 x 4 x 4 at layer 23, and 512 x 2 x 2 at layer 30. An ERF is
computed for every spatial cell in a layer by summing that cell over channels,
taking the absolute input gradient, and then summing the resulting cell maps.
Thus the three default layers superimpose 64, 16, and 4 cell maps respectively.

The aligned figures can also be added to a completed target-feature run without
running VGG again, because they reuse the saved ERF arrays:

```bash
python codes/visualize_aligned_target_erfs.py \
  --result-dir /path/to/existing/target_feature_results
```

This post-processing command writes `aligned_target_erf_metrics.csv`,
`grouped_aligned_target_erf_metrics.csv`, `erf_aligned_examples/`, and
`erf_alignment_plots/` inside the existing result directory.

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
