# SiNC-rPPG instructions

This document describes the SiNC-rPPG framework from repository layout through training and evaluation, using **PURE** and **UBFC** as primary examples. It covers file locations, pipeline stages, and configuration options for new experiments.

For a short entry point, see [README.md](README.md).

---

## Contents

| Section | Topic |
|---------|--------|
| [§1](#1-pipeline-overview) | Pipeline overview |
| [§2](#2-registries-names-and-python-classes) | Registries (names and Python classes) |
| [§3](#3-prerequisites) | Prerequisites |
| [§4](#4-where-everything-lives-directory-map) | Directory map and `.npz` conventions |
| [§5](#5-experiment-root-and-outputs) | Experiment root and outputs |
| [§6](#6-end-to-end-workflow-pure) | Preprocessing, training, evaluation (PURE, UBFC, multi-dataset) |
| [§7](#7-configuration-and-hydra-overrides) | Configuration and Hydra overrides |
| [§8](#8-training-vs-validation-vs-testing-terminology) | Training / validation / testing terminology |
| [§9](#9-monitoring-and-artifacts) | Monitoring, artifacts, optional extensions |
| [§10](#10-troubleshooting) | Troubleshooting |
| [§11](#11-development-checks-no-dataset-required) | Development checks |
| [§12](#12-quick-command-cheat-sheet-pure) | Quick command cheat sheet (PURE) |

**Related:** [Adding a dataset](docs/ADDING_A_DATASET.md) — checklist for wiring a new corpus (data layout, loader, registry, Hydra, preprocessing).

---

## 1. Pipeline overview

- **Preprocessing** converts raw PURE video into **`.npz`** clips (cropped faces, fixed resolution) and writes a **CSV metadata** file listing each clip.
- **Training** reads the metadata CSV, loads clips for fold **`K`**, optimizes SiNC losses, writes **checkpoints** under an **experiment root** ([§5](#5-experiment-root-and-outputs)), and may log to **TensorBoard**.
- **Testing** (`test.py`) loads checkpoints from each completed fold under the same experiment root and runs a **supervised test** loader from the registry (default **`pure_testing`**; override with **`testing_dataset`** / **`testing_datasets`** in Hydra—see [Step F — Evaluate (test)](#step-f--evaluate-test)).

A full loss-by-loss understanding is not required to run the pipeline once end-to-end; [README.md](README.md) and `conf/` provide a compact overview and citation.

---

## 2. Registries (names and Python classes)

A **registry** maps a **string name** in configuration (typically Hydra) to a **Python class** constructed at runtime.

* **[`src/utils/registry.py`](src/utils/registry.py)** — shared `Registry` type (`register`, `get`, case-insensitive keys).
* **[`src/datasets/dataset_registry.py`](src/datasets/dataset_registry.py)** — dataset modes (e.g. `pure_unsupervised` → `PUREUnsupervised`, `pure_testing` → PURE supervised test loader, UBFC equivalents). The string `dataset=pure_unsupervised` resolves through this registry rather than a long conditional chain.
* **[`src/utils/model_registry.py`](src/utils/model_registry.py)** — networks (e.g. `physnet` → `PhysNet`, `rpnet` → `RPNet`) for `model=...` in Hydra.

Hydra presets under `conf/dataset/` and `conf/model/` align with these registry keys. **Multi-corpus training** uses `mixed_unsupervised` / `mixed_supervised` ([Multi-dataset training](#multi-dataset-training-for-mixed-pure-and-ubfc)).

**Extending the codebase:** implement a new class, add `.register("key", Class)` in the appropriate registry module, and add a matching YAML file under `conf/dataset/` or `conf/model/`. A concise step-by-step is in **[docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md)**; [README.md — Notes](README.md#notes) summarizes dataset registration.

---

## 3. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.11+** | As stated in [README.md](README.md). |
| **[uv](https://docs.astral.sh/uv/)** or **pip** | UV is recommended; installs from [pyproject.toml](pyproject.toml). |
| **Disk space** | Raw PURE plus preprocessed clips; plan tens of GB depending on retention. |
| **Compute device** | PyTorch selects backends in order: **CUDA** (NVIDIA / ROCm), **Apple MPS**, **Intel XPU** (when supported), then **CPU**—see [`src/utils/torch_device.py`](src/utils/torch_device.py). Training targets GPU-class hardware; **CPU** is supported but often impractical for full runs. |
| **PURE access** | Dataset request/download: [official PURE page](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure). |

Install dependencies from the **repository root**:

```bash
uv sync --group dev
```

---

## 4. Where everything lives (directory map)

| Location | Purpose |
|----------|---------|
| `data/raw/` | Optional archive of **unaltered** downloads (e.g. `data/raw/PURE/...`). |
| `data/preprocessed/` | **Model input**: `.npz` files (e.g. `data/preprocessed/PURE/01-01.npz`). |
| `data/metadata/` | **CSVs** listing subjects, sessions, and paths to `.npz` files (e.g. `PURE.csv` at 30 fps). |
| `experiments/` | **Training outputs**: per-fold folders, checkpoints, TensorBoard, manifests. Layout depends on `experiment_root` ([§5](#5-experiment-root-and-outputs)). |
| `predictions/` | Pickled outputs from `test.py` (default). |
| `results/` | Text logs from evaluation (default). |
| `conf/` | **Hydra** YAML: `model`, `dataset`, `training`, `paths`, plus top-level keys in [conf/config.yaml](conf/config.yaml). |
| `src/` | `train.py`, `test.py`, engine, datasets, losses. |
| `src/preprocessing/PURE/` | Scripts to build PURE `.npz` clips and metadata. |

To **inspect** a predictions pickle without ad-hoc code, run [`src/utils/read_predictions_pickle.py`](src/utils/read_predictions_pickle.py) from `src/` (same working directory as `train.py` / `test.py`): `uv run python utils/read_predictions_pickle.py ../predictions/<name>.pkl`, or `uv run python utils/read_predictions_pickle.py --path <file>` when the path starts with `-`.

Further detail: [data/README.md](data/README.md).

### Preprocessed `.npz` keys: `wave` vs supervised and unsupervised modes

Clips normally include **`video`** (face crop stack) and often **`wave`** (oximeter waveform). **Supervised** modes (`pure_supervised`, `ubfc_supervised`, **`pure_testing`**, and other loaders that read `wave` for labels or metrics) **require** `wave`; a missing key raises **`ValueError`** with the file path. **Unsupervised** modes (`pure_unsupervised`, `ubfc_unsupervised`) require **`video`**; **`wave` may be omitted**, with clip boundaries taken from **`video.shape[0]`** (a length-only internal placeholder). Omitting `wave` does not supply ground truth where supervised evaluation or `test.py` expects real waveforms—those paths still require authentic **`wave`** data.

---

## 5. Experiment root and outputs

### What “experiment” means here

An **experiment** is the **logical run kept together**: one model configuration and protocol (e.g. PURE unsupervised K-fold), with **one full training run per fold** `K` (many epochs each). All fold directories share one **parent path**, the **experiment root**.

### What `experiment_root` is and where it lives

**`experiment_root`** is a Hydra **top-level** key in [conf/config.yaml](conf/config.yaml) (default `null` if unset). It must be a **directory path**: absolute, or **relative to the repository root**, resolved by [`src/config_merge.py`](src/config_merge.py).

Training does not fix a single folder name. Example layout:

```text
<repository-root>/experiments/PURE_exper/     # experiment_root
├── fold0_seed0/
├── fold1_seed0/
└── ...
```

[`src/engine/trainer.py`](src/engine/trainer.py) creates `foldK_seedS`; `seed` is derived from `K` (`seed = K // 5`). Each fold folder contains `saved_models/`, `best_saved_models/`, `arg_obj.txt`, TensorBoard `runs/`, `checkpoints_manifest.jsonl`, etc.

**`test.py` / evaluation** expects `experiment_root` to point at this **same parent**: it lists subdirectories named like `fold0_seed0`, reads each fold’s `arg_obj.txt` and best checkpoint ([`src/engine/evaluation.py`](src/engine/evaluation.py)).

### Using `scripts/run_experiments.py`

[`scripts/run_experiments.py`](scripts/run_experiments.py) **always** passes `experiment_root=...` into `train.py` and `test.py` when that script is used.

* With `--experiment-root experiments/my_run`, outputs resolve under **`<repo>/experiments/my_run`**.
* If `--experiment-root` is omitted, the default **`experiments/PURE_exper`** (under the **repository root**, not filesystem `/`) applies.

The **same** `experiment_root` must be passed to `test` as to `train` so evaluation finds completed folds.

### Calling `train.py` directly: unset `experiment_root`

If `train.py` runs with Hydra and **`experiment_root` remains `null`**, [`src/engine/trainer.py`](src/engine/trainer.py) uses `paths.experiments_dir` (default `experiments/` from [conf/paths/default.yaml](conf/paths/default.yaml)) and creates a **sequentially numbered** subdirectory:

```text
experiments/exper_0000/
experiments/exper_0001/
```

via `_get_experiment_dir` (**`exper_XXXX`**, not `foldK_seedS`).

The stock **`test.py` K-fold loop** expects **`foldK_seedS`** under an explicit experiment root. Auto `exper_XXXX` layouts are **not** discovered by default evaluation unless evaluation is adapted or `experiment_root` is always set for K-fold training. For the standard PURE protocol, **`experiment_root` should be set explicitly** (as `run_experiments.py` does) for both training and testing.

---

## 6. End-to-end workflow (PURE)

### 6.1 PURE preprocessing and metadata

#### Step A — Obtain raw PURE

Download PURE per license/instructions. Example layout:

```text
data/raw/PURE/<official archive layout>
```

Preprocessing only requires a path to the **root of the downloaded PURE tree** passed to `make_dataset.py`.

#### Step B — Build preprocessed clips (faces, 64×64)

From the repository root:

```bash
cd src/preprocessing/PURE
uv run python make_dataset.py /path/to/downloaded/PURE /absolute/or/relative/path/to/data/preprocessed/PURE --detector mediapipe
```

- **First argument:** raw PURE root.
- **Second argument:** output folder for **`.npz`** files; `data/preprocessed/PURE` matches [data/README.md](data/README.md).
- **`--detector`:** default `mediapipe`; other backends may be stubs ([§9.1 — Research extensions](#91-research-extensions)).

This step is CPU-heavy and may run for a long time.

#### Step C — Generate metadata CSV

Still under `src/preprocessing/PURE`:

```bash
uv run python make_metadata.py /path/to/data/preprocessed/PURE ../../../data/metadata/PURE.csv
```

- **First argument:** folder containing `.npz` files (e.g. `../../../data/preprocessed/PURE` from this directory).
- **Second argument:** output CSV. For **30 fps** (default `dataset.fps: 30`), the PURE loader expects **`data/metadata/PURE.csv`**. For **90 fps**, a matching pipeline and **`PURE_90fps.csv`** are required ([`src/datasets/PURE.py`](src/datasets/PURE.py)).

CSV columns include `subj_id`, `sess_id`, and `path` (often absolute paths to each `.npz`).

### 6.2 Hydra sanity check (optional)

From the **repository root**:

```bash
uv run python src/train.py --cfg job
```

Merged YAML should list `dataset.fps`, `training.epochs`, `paths.metadata_dir`, etc. If this fails, verify the environment (`uv sync`) before training.

### 6.3 Training

**`scripts/run_experiments.py train`** runs one `train.py` subprocess per fold in a K range (`--k-min`–`--k-max`), supports a single fold via `--k-min=0 --k-max=0`, and forwards Hydra overrides after `--`. **`train.py` directly** invokes the same training logic once per call (typically with cwd `src/`).

**Recommended:** run [`scripts/run_experiments.py`](scripts/run_experiments.py) from the **repository root**. Each subprocess uses **working directory `src/`** (equivalent to `cd src` then `python train.py ...`). No manual `cd src` is required when using this runner.

Hydra overrides (e.g. `training.epochs`, `training.batch_size`, `training.lr`) go **after `--`** and are forwarded unchanged to each fold’s `train.py`.

Default **full K-fold** (`K=0`…`14`):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper
```

**Single fold** (smoke test):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0
```

Optional `--dataset` (must match a registered Hydra dataset / registry name):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper --dataset pure_unsupervised
uv run python scripts/run_experiments.py train --experiment-root experiments/UBFC_exper --dataset ubfc_unsupervised
```

Forwarded overrides:

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0 -- training.epochs=5 training.batch_size=8 training.lr=3e-4
```

**Outputs:** per-fold directories, checkpoints, TensorBoard under `runs/`, `checkpoints_manifest.jsonl`, `training_summary.json` ([README.md](README.md)).

**Manual Hydra** (from `src/`):

```bash
cd src
uv run python train.py experiment_root=/absolute/path/to/experiments/PURE_manual K=0 training.epochs=5
```

A valid **`experiment_root`** is required for the standard K-fold + evaluation flow ([§5](#5-experiment-root-and-outputs)).

### Step F — Evaluate (test)

**`scripts/run_experiments.py test`** runs `test.py` once for a completed `experiment_root`. **`test.py` directly** runs the same evaluation logic from `src/`.

After at least one fold has finished:

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper
```

Forwarded overrides to `test.py`:

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper -- paths.results_dir=results_alt paths.predictions_dir=predictions_alt window_size=12
```

Evaluation discovers fold subdirectories, reads each fold’s `arg_obj.txt`, loads checkpoints, and runs the test loop.

**Supervised test loader** selection comes from [conf/config.yaml](conf/config.yaml) and CLI overrides—it is **not** read from each fold’s `arg_obj.txt`.

| Hydra key | Purpose |
|-----------|--------|
| **`testing_dataset`** | Single registry name; default `pure_testing` (e.g. `ubfc_testing` for UBFC held-out metrics). |
| **`testing_datasets`** | Optional list; if non-empty, **replaces** the single-key case and runs each test set in one `test.py` invocation. |

**Examples (repository root):**

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_smoke

uv run python scripts/run_experiments.py test --experiment-root experiments/UBFC_smoke -- testing_dataset=ubfc_testing

cd src
uv run python test.py experiment_root=/absolute/path/to/experiments/UBFC_smoke testing_dataset=ubfc_testing
cd ..

uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper -- 'testing_datasets=[pure_testing,ubfc_testing]'
```

**PURE training, UBFC testing:** the test loader is controlled only by **`testing_dataset`**, not by the training `dataset=` stored in each fold. Example:

```bash
uv run python scripts/run_experiments.py test \
  --experiment-root experiments/PURE_to_ubfc_test \
  -- testing_dataset=ubfc_testing
```

UBFC must be preprocessed with **`data/metadata/UBFC.csv`** (or the 90 fps variant); `dataset.fps` for the test run should match UBFC metadata (often `30`).

**Split caveat ([`src/engine/evaluation.py`](src/engine/evaluation.py)):** `test_split` follows the **train** vs **test** registry name prefix (e.g. `pure` vs `pure` → fold **`test`** group). If prefixes **differ** (PURE train → **`ubfc_testing`**), evaluation uses the **`all`** split for the test dataset (all subjects in **`UBFC.csv`**), not UBFC’s single-fold **test** bucket. That answers “score on UBFC data” but differs from “UBFC test-fold-only for this `K`.” For the stricter protocol, train with a matching corpus key (e.g. **`ubfc_unsupervised`**) or adjust evaluation.

Non-default `data/preprocessed` / `data/metadata` locations: pass `paths.preprocessed_dir=...`, `paths.metadata_dir=...` after `--`. See [`src/engine/evaluation.py`](src/engine/evaluation.py) (`_resolve_testing_datasets`).

### UBFC — same machinery, different dataset preset

UBFC follows the same registry pattern as PURE (`ubfc_unsupervised`, `ubfc_supervised`, `ubfc_testing` in [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py)).

| Preset file | Typical use |
|-------------|-------------|
| [conf/dataset/ubfc_unsupervised.yaml](conf/dataset/ubfc_unsupervised.yaml) | Unsupervised training / validation |
| [conf/dataset/ubfc_supervised.yaml](conf/dataset/ubfc_supervised.yaml) | Supervised training |
| [conf/dataset/ubfc_testing.yaml](conf/dataset/ubfc_testing.yaml) | Supervised test-style loader; `testing_dataset=ubfc_testing` for `test.py`. |

**Paths:** `data/preprocessed/UBFC/`, metadata **`data/metadata/UBFC.csv`** at 30 fps (or **`UBFC_90fps.csv`** at `dataset.fps=90`). Layout: [data/README.md](data/README.md).

**Preprocessing** ([`src/preprocessing/UBFC-rPPG/README.md`](src/preprocessing/UBFC-rPPG/README.md)):

```bash
cd src/preprocessing/UBFC-rPPG
uv run python make_dataset.py /path/to/downloaded/UBFC /path/to/data/preprocessed/UBFC
uv run python make_metadata.py /path/to/data/preprocessed/UBFC ../../../data/metadata/UBFC.csv
```

**Training (repository root):**

```bash
uv run python scripts/run_experiments.py train \
  --experiment-root experiments/UBFC_smoke \
  --k-min 0 --k-max 0 \
  -- dataset=ubfc_unsupervised

uv run python scripts/run_experiments.py train \
  --experiment-root experiments/UBFC_smoke \
  --k-min 0 --k-max 0 \
  -- dataset=ubfc_supervised training.epochs=50
```

**Manual `train.py`** from `src/`:

```bash
cd src
uv run python train.py experiment_root=/absolute/path/to/experiments/UBFC_manual K=0 dataset=ubfc_unsupervised
```

After UBFC training, pass **`testing_dataset=ubfc_testing`** for evaluation ([Step F](#step-f--evaluate-test)). Cross-corpus **validation** during training is separate: `validation_dataset=ubfc_unsupervised` with PURE training is configured in [§7](#7-configuration-and-hydra-overrides).

### Multi-dataset training for mixed PURE and UBFC

**Multi-corpus training** selects a **mixed** Hydra preset. **`mixed_sub_datasets`** lists **two or more** children (each: registered **`dataset`** key plus optional **`weight`**). **All children inherit the same top-level Hydra fields** from that preset (`fps`, `fpc`, `step`, frame size, paths); the implementation does **not** assign per-child **`fps`**—a single frame rate and matching CSVs/preprocessed data apply to every corpus in the mix.

[`MixedTrainDataset`](src/datasets/mixed_train.py) builds children via the registry, concatenates clips, and—with a **`weight`** on every child—uses [`WeightedRandomSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) so long-run sampling mass matches **normalized** weights (bundled example: **PURE / UBFC** at **0.7 / 0.3**). Weights are **targets**, not a fixed per-batch ratio (replacement sampling; discrete clip counts).

| Preset | Role |
|--------|------|
| [conf/dataset/mixed_unsupervised.yaml](conf/dataset/mixed_unsupervised.yaml) | Unsupervised SiNC on PURE + UBFC train folds |
| [conf/dataset/mixed_supervised.yaml](conf/dataset/mixed_supervised.yaml) | Supervised training on both corpora |

**Requirements:** `paths.metadata_dir` and `paths.preprocessed_dir` must contain **`PURE.csv`**, **`UBFC.csv`**, and the usual **`PURE/`** / **`UBFC/`** trees ([UBFC — same machinery](#ubfc--same-machinery-different-dataset-preset)). If **any** child omits **`weight`**, training uses **uniform shuffle** over the concat (mixing proportional to **dataset sizes**, not YAML weights).

**Examples:**

```bash
cd src
uv run python train.py dataset=mixed_unsupervised experiment_root=/path/to/exper K=0
uv run python train.py dataset=mixed_supervised experiment_root=/path/to/exper K=0
```

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/mixed_smoke --k-min 0 --k-max 0 --dataset mixed_unsupervised
```

**Validation** follows **`validation_dataset`** when set, otherwise [`validation_arg_obj`](src/datasets/cross_domain.py) behavior; mixed presets affect the **training** loader only.

---

## 7. Configuration and Hydra overrides

**Scope:** This section covers **Hydra composition and CLI overrides** for training and evaluation when using **already registered** dataset and model keys (e.g. `training.epochs`, `paths.preprocessed_dir`, `dataset=mixed_unsupervised`, `testing_dataset=ubfc_testing`). **Integrating a new corpus** (new loader class, metadata layout, registry entry, preprocessing scripts) is documented in **[docs/ADDING_A_DATASET.md](docs/ADDING_A_DATASET.md)**.

Hydra composes [conf/config.yaml](conf/config.yaml) from groups under `conf/`. Overrides are CLI tokens `key=value` or `group.key=value` for nested YAML.

### Hydra override cheat sheet

| Goal | Example |
|------|---------|
| Show composed config and overrides | `cd src && uv run python train.py --help` |
| Print full resolved config and exit | `uv run python train.py --cfg job` |
| Top-level run keys (flattened into `arg_obj`) | `experiment_root=...`, `K=3`, `debug=1`, `continue_training=1` |
| Swap a **config group** (`conf/<group>/`) | `model=rpnet`, `dataset=pure_testing`, `training=default` |
| Nested keys | `training.lr=3e-4`, `training.batch_size=16`, `dataset.fps=90`, `training.use_lightning=true`, `training.lightning_gradient_clip=1.0` |
| Cross-domain validation (top-level) | `validation_dataset=ubfc_unsupervised`, `validation_fps=30` |
| Paths relative to **repo root** ([conf/paths/default.yaml](conf/paths/default.yaml)) | `paths.metadata_dir=data/metadata`, `paths.preprocessed_dir=/scratch/rppg/preprocessed` |
| Add a key not in YAML (sparingly) | `+some_new_flag=1` ([Hydra override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for `+` vs `++`) |

**Working directory:** `train.py` / `test.py` expect cwd **`src/`** so `config_path="../conf"` resolves. [`scripts/run_experiments.py`](scripts/run_experiments.py) sets `cwd=src` and passes `experiment_root=...` (and for `train`, `K=...`, `dataset=...`). Extra Hydra tokens after `--`:

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0 -- training.epochs=5 training.batch_size=8 training.lr=3e-4
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_smoke -- paths.results_dir=results_smoke paths.predictions_dir=predictions_smoke
```

Shell quoting may be required for values containing spaces. `hydra.job.chdir` is `false` in [conf/config.yaml](conf/config.yaml); metadata and preprocessed paths resolve to absolute paths under the repo via [`src/config_merge.py`](src/config_merge.py) and [`src/repo_paths.py`](src/repo_paths.py).

**Common overrides:**

| Goal | Example overrides |
|------|-------------------|
| Shorter dry run | `training.epochs=5` |
| Batch size / LR | `training.batch_size=16`, `training.lr=3e-4` |
| Frame rate (must match CSV + preprocessed data) | `dataset.fps=30` or `dataset.fps=90` |
| Different model | `model=physnet` or `model=rpnet` (see `conf/model/`) |
| Metadata or preprocessed roots on another disk | `paths.metadata_dir=/scratch/metadata`, `paths.preprocessed_dir=/scratch/preprocessed` |
| Optional PyTorch Lightning | `training.use_lightning=true` |
| Train on PURE, validate on another corpus | `validation_dataset=ubfc_unsupervised`, `validation_fps=30` |
| **`test.py`**: supervised evaluation corpus | `testing_dataset=pure_testing` (default), `testing_dataset=ubfc_testing`, or `testing_datasets=[pure_testing,ubfc_testing]` |
| Train on PURE, then **test** on UBFC in one `test.py` run | `testing_dataset=ubfc_testing` (see [Step F](#step-f--evaluate-test) **split caveat** when train/test corpus prefixes differ) |
| **Multi-corpus training** | `dataset=mixed_unsupervised` or `dataset=mixed_supervised` (edit `mixed_sub_datasets` in YAML; [multi-dataset](#multi-dataset-training-for-mixed-pure-and-ubfc)) |

Registered dataset **names** (`dataset=...`, `validation_dataset=...`, **`testing_dataset`**) are defined in [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py).

**Supervised presets** (`pure_supervised`, `ubfc_supervised`, `mixed_supervised`) set **`optimization_step`**, **`validation_step`**, **`losses`**, and **`validation_loss`** for supervised SiNC (not the default unsupervised `bsv` stack). Selecting `dataset=pure_supervised` (or the others) is sufficient unless those fields are intentionally overridden again.

### Quick override templates

```bash
cd src
uv run python train.py experiment_root=/absolute/path/to/experiments/PURE_manual K=0 training.epochs=5 training.batch_size=8 training.lr=3e-4

cd ..
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0 -- training.epochs=5 training.batch_size=8 training.lr=3e-4

uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_smoke -- paths.results_dir=results_smoke paths.predictions_dir=predictions_smoke window_size=12

uv run python scripts/run_experiments.py test --experiment-root experiments/UBFC_smoke -- testing_dataset=ubfc_testing

uv run python scripts/run_experiments.py train --experiment-root experiments/UBFC_smoke --k-min 0 --k-max 0 -- dataset=ubfc_unsupervised
```

**Notes:**

- For `run_experiments.py`, place Hydra overrides **after `--`** so argparse does not consume them.
- The same forwarded overrides apply to every fold in the K-loop.
- During evaluation, some `test.py` values are taken from each fold’s `arg_obj.txt` (e.g. `K`, `fps`, `fpc`, `step`, `model_type`). The **supervised test loader** comes from Hydra **`testing_dataset`** / **`testing_datasets`** ([conf/config.yaml](conf/config.yaml) defaults), not from the fold’s training `dataset=...` string.

---

## 8. Training vs validation vs testing (terminology)

| Stage | In this repo | Typical inputs |
|--------|----------------|----------------|
| **Training** | `train.py` / `Trainer` | Unsupervised (or supervised) **train** split for fold `K` |
| **Validation** | Same run, validation split and losses | Held-out subjects within the training corpus, or another corpus if `validation_dataset` is set |
| **Testing** | `test.py` / `run_evaluation` | Loads saved checkpoints; default `testing_dataset=pure_testing` (override e.g. `ubfc_testing` in Hydra) |

---

## 9. Monitoring and artifacts

| Artifact | Where | What it is |
|----------|--------|------------|
| TensorBoard | `<experiment_dir>/runs/` | Scalars for losses, LR, etc. |
| Flat config snapshot | TensorBoard text `config/flat_args` | One-time resolved training settings |
| Checkpoint manifest | `checkpoints_manifest.jsonl` | Per-epoch checkpoint paths and validation metrics |
| Best summary | `training_summary.json` | Best epoch and checkpoint pointer |

```bash
tensorboard --logdir /path/to/experiment_dir/runs
```

**Optional Lightning:** `training.use_lightning=true` preserves the same loss logic and **`torch.save`** checkpoints compatible with `test.py`; logs under `runs/lightning/`. `training.lightning_gradient_clip` enables global-norm clipping when greater than zero.

### 9.1 Research extensions

* **Face detectors (preprocessing):** [`src/preprocessing/face_detector.py`](src/preprocessing/face_detector.py) — `FaceDetector` ABC and `get_face_detector(name)`. `mediapipe` is implemented; `retinaface` and `mtcnn` raise `NotImplementedError`. PURE and UBFC `make_dataset.py` accept `--detector` (default `mediapipe`). Landmarks: [`src/preprocessing/mesh_common.py`](src/preprocessing/mesh_common.py).
* **Signal metrics:** [`src/utils/metrics.py`](src/utils/metrics.py) — `mean_absolute_error`, `snr_db` for NumPy or Torch tensors.
* **Cross-domain validation:** Hydra `validation_dataset` and optional `validation_fps` ([conf/config.yaml](conf/config.yaml)) select another registered dataset for the **validation** split only (e.g. `validation_dataset=ubfc_unsupervised` with `dataset=pure_unsupervised`).

---

## 10. Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `FileNotFoundError` for CSV or `.npz` | `paths.metadata_dir`, `paths.preprocessed_dir`, and that `PURE.csv` paths resolve to real files. |
| Wrong fps | `dataset.fps` must be `30` or `90` for PURE; CSV must be `PURE.csv` vs `PURE_90fps.csv`. |
| Hydra cannot find config | Run `train.py` with cwd `src/`, or use `scripts/run_experiments.py` (sets `cwd=src`). |
| CUDA OOM | Reduce `training.batch_size`, `training.fpc`, or `training.step` ([conf/training/default.yaml](conf/training/default.yaml)). |
| Preprocessing errors | Confirm raw PURE path; `--detector mediapipe`; write permissions on the output folder. |

---

## 11. Development checks (no dataset required)

From the repository root:

```bash
uv run pytest
uv run ruff check src tests scripts
```

---

## 12. Quick command cheat sheet (PURE)

```bash
# 0) Install
uv sync --group dev

# 1) Preprocess (adjust paths)
cd src/preprocessing/PURE
uv run python make_dataset.py <RAW_PURE_ROOT> ../../../data/preprocessed/PURE
uv run python make_metadata.py ../../../data/preprocessed/PURE ../../../data/metadata/PURE.csv
cd ../../..

# 2) Train all folds
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper

# 3) Evaluate
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper
```

The same layout applies to UBFC or new corpora after adding preprocessing, metadata CSVs, dataset classes, and registry entries ([§2](#2-registries-names-and-python-classes), [README.md — Notes](README.md#notes)).
