# SiNC-rPPG

# Non-Contrastive Unsupervised Learning of Physiological Signals from Video

## Highlight paper in Conference on Computer Vision and Pattern Recognition (CVPR) 2023

### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Speth_Non-Contrastive_Unsupervised_Learning_of_Physiological_Signals_From_Video_CVPR_2023_paper.pdf) | [Video](https://www.youtube.com/watch?v=Bg7VkxWcOhQ)

<figure>
  <img src="./teaser.png" style="width:100%">
  <figcaption>
    Figure 1: Overview of the SiNC framework for rPPG compared with traditional supervised and unsupervised learning. Supervised and contrastive losses use distance metrics to the ground truth or other samples. Our framework applies the loss directly to the prediction by shaping the frequency spectrum, and encouraging variance over a batch of inputs. Power outside of the bandlimits is penalized to learn invariances to irrelevant frequencies. Power within the bandlimits is encouraged to be sparsely distributed near the peak frequency.
  </figcaption>
</figure>

## Contents

* **[INSTRUCTIONS.md](INSTRUCTIONS.md)** — Main guide: preprocessing, training and testing, **`experiment_root`**, [Hydra overrides](INSTRUCTIONS.md#hydra-override-cheat-sheet), [multi-dataset training](INSTRUCTIONS.md#multi-dataset-training-for-mixed-pure-and-ubfc), registries, troubleshooting.
* **[docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md)** — Checklist for adding a new corpus (data paths, loader, registry, Hydra, when to touch core code).
* **Repo layout:** [conf/](conf/) (Hydra), [src/train.py](src/train.py), [src/test.py](src/test.py), [src/engine/](src/engine/), [src/datasets/](src/datasets/); default data layout: [data/README.md](data/README.md).
* **K-fold wrapper:** [scripts/run_experiments.py](scripts/run_experiments.py).
* **TODO:** preprocessing for [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg), [DDPM](https://cvrl.nd.edu/projects/data/#remote-pulse-detection-21), [HKBU-MARs](https://rds.comp.hkbu.edu.hk/mars/).

## Requirements

* **Python 3.11+** (tested with 3.11.x)
* This repository is managed with **[uv](https://docs.astral.sh/uv/)**.

## Installation (recommended: uv)

From the repository root:

```bash
uv sync --group dev
```

This installs runtime dependencies from [pyproject.toml](pyproject.toml), editable package mode, and dev tools (`pytest`, `ruff`).

## Installation (pip and venv, without uv)

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux / macOS
pip install -U pip
pip install -e ".[dev]"
```

Runtime-only: `pip install -e .` or [requirements.txt](requirements.txt) plus `pytest` / `ruff` manually.

## Configuration (Hydra)

Training and testing use [Hydra](https://hydra.cc/) with defaults in [conf/config.yaml](conf/config.yaml) (`model`, `dataset`, `training`, `paths`). Run `train.py` / `test.py` from **`src/`** (or use `scripts/run_experiments.py`, which sets `cwd=src`).

```bash
cd src
uv run python train.py experiment_root=/path/to/experiments K=0 training.epochs=50
uv run python test.py experiment_root=/path/to/experiments
```

**Details:** [Hydra override cheat sheet](INSTRUCTIONS.md#hydra-override-cheat-sheet), [`experiment_root` and outputs](INSTRUCTIONS.md#5-experiment-root-and-outputs), path resolution in [src/config_merge.py](src/config_merge.py) / [src/repo_paths.py](src/repo_paths.py).

## Running training and evaluation

1. **Data:** [Download PURE](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure) and preprocess (see [INSTRUCTIONS.md](INSTRUCTIONS.md) and `src/preprocessing/PURE/`).

2. **Train** (repo root; default K = 0..14):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper
```

3. **Evaluate:**

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper
```

`--dataset`, `--k-min` / `--k-max`, and Hydra overrides after `--` are documented in [INSTRUCTIONS.md](INSTRUCTIONS.md) (e.g. [§7](INSTRUCTIONS.md#7-configuration-and-hydra-overrides), [quick commands](INSTRUCTIONS.md#12-quick-command-cheat-sheet-pure)).

## Experiment tracking

TensorBoard, checkpoint manifest, and summaries live under each fold directory — see [INSTRUCTIONS.md §9](INSTRUCTIONS.md#9-monitoring-and-artifacts).

## Development

* **Tests:** `uv run pytest`
* **Lint / format:** `uv run ruff check src tests scripts` and `uv run ruff format src tests scripts`

## Notes

Register new dataset modes in [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py). Step-by-step for a new corpus: [docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md). Training-time validation on another corpus: Hydra `validation_dataset` / `validation_fps` ([INSTRUCTIONS.md §7](INSTRUCTIONS.md#7-configuration-and-hydra-overrides)).

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
