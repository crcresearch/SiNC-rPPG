# SiNC-rPPG

# Non-Contrastive Unsupervised Learning of Physiological Signals from Video

## Highlight paper in Conference on Computer Vision and Pattern Recognition (CVPR) 2023

### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Speth_Non-Contrastive_Unsupervised_Learning_of_Physiological_Signals_From_Video_CVPR_2023_paper.pdf) | [Video](https://www.youtube.com/watch?v=Bg7VkxWcOhQ)

<figure>
  <img src="./teaser.png" style="width:100%">
  <figcaption>Figure 1: Overview of the SiNC framework for rPPG compared with traditional supervised and unsupervised learning. Supervised and contrastive losses use distance metrics to the ground truth or other samples. Our framework applies the loss directly to the prediction by shaping the frequency spectrum, and encouraging variance over a batch of inputs. Power outside of the bandlimits is penalized to learn invariances to irrelevant frequencies. Power within the bandlimits is encouraged to be sparsely distributed near the peak frequency.</figcaption>                                                                                                          </figure>

## Contents

* **New to the codebase?** See the step-by-step **[INSTRUCTIONS.md](INSTRUCTIONS.md)** (data layout, PURE preprocessing, train/test commands, **`experiment_root` and output layout**, [registries](INSTRUCTIONS.md#2-registries-names-and-python-classes), Hydra knobs, troubleshooting).
* Preprocessing code for the PURE dataset is in src/preprocessing/PURE
* Training code is in src/train.py
* Testing code is in src/test.py
* Hydra configuration is under [conf/](conf/) (model, dataset, training, paths); run logging helpers are in src/args.py
* Training and evaluation orchestration live in [src/engine/](src/engine/) (`Trainer`, `run_evaluation`). **Registries** map short string names (for example `pure_unsupervised`, `physnet`) to dataset/model classes; see [INSTRUCTIONS.md §2](INSTRUCTIONS.md#2-registries-names-and-python-classes). Implementation: [src/utils/registry.py](src/utils/registry.py), [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py), [src/utils/model_registry.py](src/utils/model_registry.py)
* Default data layout is documented in [data/README.md](data/README.md)
* Loss functions are in src/utils/losses.py
* Model architectures are in src/models/
* Dataloaders are in src/datasets/
* Cross-platform experiment runner: [scripts/run_experiments.py](scripts/run_experiments.py)
* TODO: preprocessing code for [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg), [DDPM](https://cvrl.nd.edu/projects/data/#remote-pulse-detection-21), and [HKBU-MARs](https://rds.comp.hkbu.edu.hk/mars/).

## Requirements

* **Python 3.11+** (tested with 3.11.x)
* This repository is managed with **[uv](https://docs.astral.sh/uv/)**; Conda is not required.

## Installation (recommended: uv)

From the repository root, create or reuse a virtual environment and install the project plus dev tools:

```bash
uv sync --group dev
```

This installs runtime dependencies from [pyproject.toml](pyproject.toml), installs the package in editable mode, and adds `pytest` and `ruff` for development.

## Installation (pip and venv, without uv)

If you do not use uv, create a virtual environment with the standard library so packages are isolated from your system Python:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows cmd
pip install -U pip
pip install -e ".[dev]"
```

Runtime-only install (no dev tools):

```bash
pip install -e .
```

You can also install pinned-style runtime libraries from [requirements.txt](requirements.txt) (`pip install -r requirements.txt`), then install dev tools manually (`pip install pytest ruff`) if you prefer not to use the `[dev]` extra.

## Configuration (Hydra)

Training and testing use [Hydra](https://hydra.cc/). Defaults live under [conf/config.yaml](conf/config.yaml) with groups `model`, `dataset`, `training`, and `paths`. Override from the CLI, for example:

```bash
cd src
uv run python train.py experiment_root=/path/to/experiments K=0 training.epochs=50
uv run python test.py experiment_root=/path/to/experiments
```

`hydra.job.chdir` is set to `false` so relative paths stay anchored to the process working directory when needed; metadata and preprocessed roots are resolved to **absolute paths under the repository root** via [src/repo_paths.py](src/repo_paths.py).

### Hydra cheat sheet

| Goal | Example |
|------|---------|
| Show composed config and overrides | `cd src && uv run python train.py --help` |
| Print full resolved config and exit | `uv run python train.py --cfg job` |
| Top-level run keys (flattened into `arg_obj`) | `experiment_root=...`, `K=3`, `debug=1`, `continue_training=1` |
| Swap a **config group** (file under `conf/<group>/`) | `model=rpnet`, `dataset=pure_testing`, `training=default` |
| Nested keys (YAML hierarchy) | `training.lr=3e-4`, `training.batch_size=16`, `dataset.fps=90`, `training.use_lightning=true`, `training.lightning_gradient_clip=1.0` |
| Cross-domain validation (top-level) | `validation_dataset=ubfc_unsupervised`, `validation_fps=30` |
| Paths relative to **repo root** (see `conf/paths/default.yaml`) | `paths.metadata_dir=data/metadata`, `paths.preprocessed_dir=/scratch/rppg/preprocessed` |
| Add a key not in YAML (use sparingly) | `+some_new_flag=1` (see [Hydra docs](https://hydra.cc/docs/advanced/override_grammar/basic/) for `+` vs `++`) |

**Notes:** Run `train.py` / `test.py` from `src/` (as in the examples above) so `config_path="../conf"` resolves correctly. `scripts/run_experiments.py` already sets `cwd=src` and passes overrides like `experiment_root=...` and `dataset=...`. Override values with spaces may need quoting in your shell.

### Where training writes files (`experiment_root`)

In this repo, an **experiment** is a **named directory** that groups related training outputs—typically **one training job per cross-validation fold** `K`, all evaluated together. **`experiment_root`** is the **path to that parent directory** (for example `experiments/PURE_exper` under your clone). The trainer creates subfolders inside it, such as `fold0_seed0`, containing checkpoints, `arg_obj.txt`, TensorBoard logs, and manifests.

[`scripts/run_experiments.py`](scripts/run_experiments.py) always passes `experiment_root` into Hydra. If you omit `--experiment-root`, the default is **`experiments/PURE_exper`** relative to the **repository root** (not the filesystem root `/`).

If you call `train.py` with Hydra leaving `experiment_root` unset (`null` in [conf/config.yaml](conf/config.yaml)), the code falls back to auto-numbered folders under `paths.experiments_dir`; that path differs from the K-fold layout and affects how evaluation discovers runs. See **[Experiment root and outputs](INSTRUCTIONS.md#5-experiment-root-and-outputs)** in [INSTRUCTIONS.md](INSTRUCTIONS.md) for details.

## Experiment tracking (TensorBoard and manifests)

Training logs **per-loss scalars** under `train/loss/...` and validation under `val/loss/...`, plus **learning rate** as `train/lr`, in TensorBoard (`runs/` under the experiment directory). A one-time **flat config** snapshot is written as `config/flat_args` text in the same run.

**Weights & Biases** ([wandb](https://wandb.ai)) is not integrated in this repository; both the default training loop and the optional Lightning path use TensorBoard and the manifest files below for run history.

Each epoch appends a line to **`checkpoints_manifest.jsonl`** (epoch, checkpoint path, `val_loss` breakdown, `is_best`, trainer backend). When training finishes, **`training_summary.json`** records the best epoch, best checkpoint path, and best validation total.

View logs:

```bash
tensorboard --logdir /path/to/experiment_dir/runs
```

**PyTorch Lightning (optional):** set `training.use_lightning=true` to run the same loss logic and **identical `torch.save` checkpoints** (compatible with `test.py`) via Lightning, with logs under `runs/lightning/`. Optional `training.lightning_gradient_clip` sets global-norm clipping when greater than zero.

## Research extensions

* **Face detectors (preprocessing):** [src/preprocessing/face_detector.py](src/preprocessing/face_detector.py) defines a `FaceDetector` ABC plus `get_face_detector(name)`. `mediapipe` is implemented; `retinaface` and `mtcnn` are stubs raising `NotImplementedError`. PURE and UBFC `make_dataset.py` accept `--detector` (default `mediapipe`). Shared landmark indexing lives in [src/preprocessing/mesh_common.py](src/preprocessing/mesh_common.py).
* **Signal metrics:** [src/utils/metrics.py](src/utils/metrics.py) provides `mean_absolute_error` and `snr_db` for NumPy or Torch tensors.
* **Cross-domain validation:** Top-level Hydra keys `validation_dataset` and optional `validation_fps` (see [conf/config.yaml](conf/config.yaml)) select a **different** registered dataset for the validation split only (e.g. train on PURE, validate on UBFC). Example: `validation_dataset=ubfc_unsupervised` with `dataset=pure_unsupervised`.

## Data layout

See [data/README.md](data/README.md). By default, CSV metadata is read from `data/metadata/` and `.npz` clips from `data/preprocessed/<PURE|UBFC|...>/`. Existing CSVs that store absolute `path` values continue to work; otherwise filenames are resolved under `data/preprocessed/<Dataset>/`.

## To run

1. To prepare the data for training, [download PURE](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure) and follow the steps in `src/preprocessing/PURE`.

2. Train K folds (default K = 0..14, same as the original shell script) from the repo root:

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper
```

Optional flags: `--dataset pure_unsupervised`, `--k-min`, `--k-max`.

3. Evaluate saved experiments:

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper
```

## Development

* **Tests:** `uv run pytest`
* **Lint / format:** `uv run ruff check src tests scripts` and `uv run ruff format src tests scripts`

## Notes

When new dataloaders are added, register them in [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py). For **evaluation** across held-out test corpora, you can still extend `src/test.py` (e.g. its testing-dataset list). For **training-time** validation on another corpus, use `validation_dataset` / `validation_fps` as above.

### Citation

If you use any part of our code or data, please cite our paper.

```
@inproceedings{speth2023sinc,
  title={Non-Contrastive Unsupervised Learning of Physiological Signals from Video},
  author={Speth, Jeremy and Vance, Nathan and Flynn, Patrick and Czajka, Adam},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```
