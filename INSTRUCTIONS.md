# SiNC-rPPG instructions

This document walks you from “I cloned the repo” to training and evaluation on the **PURE** dataset, and explains where files live, what each step does, and which settings you typically change for new experiments.

If you only need a short reference, see [README.md](README.md).

---

## 1. What you are running (mental model)

- **Preprocessing** turns raw PURE videos into small **`.npz`** clips (cropped faces, fixed resolution) and builds a **CSV metadata** file listing every clip.
- **Training** reads that CSV, loads clips for a given **fold** `K`, optimizes the SiNC losses, saves **checkpoints** under an **experiment root** (see [section 5](#5-experiment-root-and-outputs)), and optionally logs to **TensorBoard**.
- **Testing** (`test.py`) loads checkpoints from each completed fold under the same experiment root and runs evaluation on a **supervised test** dataset configuration (for PURE, the code uses `pure_testing` internally).

You do **not** need to understand every loss name on day one; follow the pipeline once end-to-end, then explore [README.md](README.md) and `conf/` for deeper changes.

---

## 2. Registries (names and Python classes)

A **registry** is a small lookup table: a **string name** you put in config (often Hydra) maps to a **Python class** the training code will actually construct.

* **[`src/utils/registry.py`](src/utils/registry.py)** defines the shared `Registry` type (essentially a dict with `register(name, cls)` and `get(name)`, case-insensitive keys).
* **[`src/datasets/dataset_registry.py`](src/datasets/dataset_registry.py)** registers dataset modes such as `pure_unsupervised` → `PUREUnsupervised`, `pure_testing` → the PURE supervised test loader, and UBFC equivalents. When you set `dataset=pure_unsupervised` on the command line or in YAML, the dataloader code resolves that string through this registry instead of using a long chain of `if` / `elif` statements.
* **[`src/utils/model_registry.py`](src/utils/model_registry.py)** does the same for networks, for example `physnet` → `PhysNet`, `rpnet` → `RPNet`, matching `model=...` in Hydra.

**As a user**, you usually only choose among the **already registered** names in `conf/dataset/` and `conf/model/`; those names line up with the keys in the registries.

**To add a new dataset or model**, you implement the class, then add one `.register("your_key", YourClass)` line in the appropriate registry module (and add a matching Hydra config file under `conf/dataset/` or `conf/model/`). See also [README.md](README.md) under **Notes** for registering new dataloaders.

---

## 3. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.11+** | As stated in [README.md](README.md). |
| **[uv](https://docs.astral.sh/uv/)** or **pip** | UV is recommended; installs the project from [pyproject.toml](pyproject.toml). |
| **Disk space** | Raw PURE plus preprocessed clips; plan tens of GB depending on retention. |
| **Compute device** | PyTorch picks the best available backend in order: **CUDA** (NVIDIA / ROCm), **Apple MPS**, **Intel XPU** (Intel GPU when supported), then **CPU**—see [`src/utils/torch_device.py`](src/utils/torch_device.py). Training is aimed at GPU-class hardware; **CPU** is supported but usually impractically slow for full runs. |
| **PURE access** | Request/download the dataset from the [official PURE page](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure). |

Install dependencies from the **repository root**:

```bash
uv sync --group dev
```

---

## 4. Where everything lives (directory map)

| Location | Purpose |
|----------|---------|
| `data/raw/` | Optional archive of **unaltered** downloads (e.g. `data/raw/PURE/...`). Keeps a clear separation from processed data. |
| `data/preprocessed/` | **Model input**: `.npz` files (e.g. `data/preprocessed/PURE/01-01.npz`). |
| `data/metadata/` | **CSVs** listing subjects, sessions, and paths to `.npz` files (e.g. `PURE.csv` for 30 fps). |
| `experiments/` | **Training outputs**: per-fold folders, checkpoints, TensorBoard runs, manifests. Naming depends on whether you set `experiment_root`; see [section 5](#5-experiment-root-and-outputs). |
| `predictions/` | Pickled prediction outputs from `test.py` (default). |
| `results/` | Text logs from evaluation (default). |
| `conf/` | **Hydra** YAML defaults: `model`, `dataset`, `training`, `paths`, plus top-level keys in `conf/config.yaml`. |
| `src/` | Training (`train.py`), testing (`test.py`), engine, datasets, losses. |
| `src/preprocessing/PURE/` | Scripts to build PURE `.npz` clips and metadata. |

More detail: [data/README.md](data/README.md).

---

## 5. Experiment root and outputs

### What “experiment” means here

An **experiment** is not a single training step or a single epoch. It is the **logical run you want to keep together**: the same model configuration and protocol (for example PURE unsupervised K-fold), with **one full training run per fold** `K` (many epochs each). All fold directories share one **parent path**, the **experiment root**.

### What `experiment_root` is and where it lives

**`experiment_root`** is a Hydra **top-level** key (see [conf/config.yaml](conf/config.yaml); default `null` if you do not override it). It must be a **directory path**: either absolute, or **relative to the repository root**, which [`src/config_merge.py`](src/config_merge.py) resolves to an absolute path.

Training does **not** use a single fixed folder name. You choose the root, for example:

```text
<your clone>/experiments/PURE_exper/          # experiment_root
├── fold0_seed0/                             # one fold’s artifacts
├── fold1_seed0/
└── ...
```

Each `foldK_seedS` folder is created by [`src/engine/trainer.py`](src/engine/trainer.py); `seed` is derived from `K` (`seed = K // 5`). Inside a fold folder you will find `saved_models/`, `best_saved_models/`, `arg_obj.txt`, TensorBoard `runs/`, `checkpoints_manifest.jsonl`, and so on.

**`test.py` / evaluation** expects `experiment_root` to point at this **same parent**: it lists immediate subdirectories and assumes names like `fold0_seed0` so it can read each fold’s `arg_obj.txt` and best checkpoint (see [`src/engine/evaluation.py`](src/engine/evaluation.py)).

### Using `scripts/run_experiments.py`

[`scripts/run_experiments.py`](scripts/run_experiments.py) **always** passes `experiment_root=...` into `train.py` and `test.py`; it is never omitted at the Hydra layer when you use this script.

* If you pass `--experiment-root experiments/my_run`, outputs go under **`<repo>/experiments/my_run`** (relative paths are resolved against the repository root).
* If you **omit** `--experiment-root`, argparse supplies the default **`experiments/PURE_exper`**, again under the **repository root**—not the machine’s filesystem root (`/`).

You must pass the **same** `experiment_root` to `test` as you used for `train` so evaluation finds the completed folds.

### Calling `train.py` directly: what if `experiment_root` is not set?

If you run `train.py` with Hydra and do **not** set `experiment_root` (so it stays `null`), [`src/engine/trainer.py`](src/engine/trainer.py) uses `paths.experiments_dir` (default `experiments/` from [conf/paths/default.yaml](conf/paths/default.yaml)) and creates a **new sequentially numbered** subdirectory:

```text
experiments/exper_0000/
experiments/exper_0001/
...
```

via `_get_experiment_dir`. That layout uses **names like `exper_0000`**, not `foldK_seedS`.

The **K-fold evaluation** path in `test.py` is written for directories named **`foldK_seedS`** under a single explicit experiment root. If you rely on the auto `exper_XXXX` layout, **do not expect the stock `test.py` loop to discover those runs** unless you adapt evaluation or always set `experiment_root` for K-fold training. For the standard PURE protocol, **set `experiment_root` explicitly** (as `run_experiments.py` does) for both training and testing.

---

## 6. End-to-end workflow (PURE)

### Step A — Obtain raw PURE

Download PURE per the dataset license/instructions. Place the extracted tree somewhere stable, for example:

```text
data/raw/PURE/<official PURE folder layout>
```

The exact subfolder names depend on the archive you receive; the preprocessing script only needs a path to the **root of the downloaded PURE tree** you will pass to `make_dataset.py`.

### Step B — Build preprocessed clips (faces, 64×64)

From the repository root:

```bash
cd src/preprocessing/PURE
uv run python make_dataset.py /path/to/your/downloaded/PURE /absolute/or/relative/path/to/data/preprocessed/PURE --detector mediapipe
```

- **First argument**: path to **raw** PURE (as provided by the dataset).
- **Second argument**: output folder for **`.npz`** files. Using `data/preprocessed/PURE` matches the default layout in [data/README.md](data/README.md).
- **`--detector`**: default is `mediapipe`. Other backends may be stubs; see [Research extensions](README.md#research-extensions) in the README.

This step is CPU-heavy and can take a long time.

### Step C — Generate metadata CSV

Still under `src/preprocessing/PURE`:

```bash
uv run python make_dataset.py ...   # (if not already done)

uv run python make_metadata.py /path/to/data/preprocessed/PURE ../../../data/metadata/PURE.csv
```

- **First argument**: the same folder that contains the `.npz` files (e.g. repo-relative `../../../data/preprocessed/PURE` from this directory).
- **Second argument**: output CSV path. For **30 fps** training (default `dataset.fps: 30` in Hydra), the PURE loader expects **`data/metadata/PURE.csv`**. For **90 fps**, you need a matching preprocessing pipeline and **`PURE_90fps.csv`** (see [src/datasets/PURE.py](src/datasets/PURE.py)).

The CSV columns include `subj_id`, `sess_id`, and `path` (absolute paths to each `.npz`).

### Step D — Sanity-check Hydra config (optional)

From the **repository root** (this repo’s `train.py` resolves `conf/` correctly):

```bash
uv run python src/train.py --cfg job
```

You should see merged YAML including `dataset.fps`, `training.epochs`, `paths.metadata_dir`, etc. If this fails, fix your environment (`uv sync`) before training.

### Step E — Train (single fold or full K-fold)

**Quick difference:**
- `scripts/run_experiments.py train`: wrapper that runs one `train.py` subprocess per fold in a K range (K-fold) (`--k-min` to `--k-max`). It also supports a single-fold run by setting `--k-min=0 --k-max=0`, and forwarded Hydra overrides via --.
- `train.py` directly: same training logic, one fold/run per invocation, called manually (typically from `src/`).

**Recommended:** Run [`scripts/run_experiments.py`](scripts/run_experiments.py) from the **repository root** (the folder that contains `scripts/` and `src/`), using the `train` subcommand in the examples below. For each fold index `K`, the script starts a separate training process that runs **`train.py`** with Hydra arguments such as `experiment_root=...`, `K=...`, and `dataset=...`. That process is launched with **working directory `src/`** (so the same layout as in the README: `cd src` then `python train.py ...`). You do not need to `cd src` yourself when using this runner.

If you want to pass extra Hydra overrides (for example `training.epochs`, `training.batch_size`, `training.lr`) through this runner, place them after `--`. They are forwarded unchanged to each `train.py` subprocess in the K-fold loop.

Train **all default folds** (`K=0` through `K=14`, matching the original paper-style protocol):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper
```

Train **one fold** (fast smoke test on your machine):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0
```

Optional dataset flag (must match a registered Hydra dataset group / registry name):

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_exper --dataset pure_unsupervised
```

Forward additional Hydra overrides to `train.py`:

```bash
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0 -- training.epochs=5 training.batch_size=8 training.lr=3e-4
```

**What gets created**: under `experiments/...` you will see per-fold directories (names depend on training seeds and fold index), checkpoints, TensorBoard logs under `runs/`, `checkpoints_manifest.jsonl`, and `training_summary.json` (see [README.md](README.md)).

**Alternative (manual Hydra)**: from `src/`:

```bash
cd src
uv run python train.py experiment_root=/absolute/path/to/experiments/PURE_manual K=0 training.epochs=5
```

You must pass a valid **`experiment_root`** for the standard K-fold + evaluation flow; training uses it to write outputs. See [section 5](#5-experiment-root-and-outputs) for defaults, directory layout, and what happens if Hydra leaves `experiment_root` unset.

### Step F — Evaluate (test)

**Quick difference:**
- `scripts/run_experiments.py test`: wrapper that runs one `test.py` process for a completed `experiment_root` from the repo root.
- `test.py` directly: same evaluation logic, but called manually from `src/`.

After at least one fold has finished and checkpoints exist:

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper
```

You can also forward Hydra overrides to `test.py` after `--`:

```bash
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_exper -- paths.results_dir=results_alt paths.predictions_dir=predictions_alt window_size=12
```

Evaluation discovers fold subdirectories under `experiment_root`, reads each fold’s saved `arg_obj.txt`, loads the corresponding checkpoint, and runs the test loop. Default internal testing dataset list includes **`pure_testing`** (see [src/engine/evaluation.py](src/engine/evaluation.py)).

---

## 7. Configuration: what to change for “new data” or new runs

Hydra composes [conf/config.yaml](conf/config.yaml) from groups under `conf/`. Common overrides (CLI `key=value`):

| Goal | Example overrides |
|------|-------------------|
| Shorter dry run | `training.epochs=5` |
| Batch size / LR | `training.batch_size=16`, `training.lr=3e-4` |
| Frame rate (must match CSV + preprocessed data) | `dataset.fps=30` or `dataset.fps=90` |
| Different model | `model=physnet` or `model=rpnet` (see `conf/model/`) |
| Metadata or preprocessed roots on another disk | `paths.metadata_dir=/scratch/metadata`, `paths.preprocessed_dir=/scratch/preprocessed` |
| Optional PyTorch Lightning backend | `training.use_lightning=true` |
| Train on PURE, validate on another corpus | `validation_dataset=ubfc_unsupervised`, `validation_fps=30` |

Registered dataset **names** (for `dataset=...` and `validation_dataset=...`) are defined in [src/datasets/dataset_registry.py](src/datasets/dataset_registry.py).

### Quick override templates (copy/paste)

Use these as starting points and replace values as needed.

```bash
# train.py directly (single fold, full Hydra flexibility)
cd src
uv run python train.py experiment_root=/absolute/path/to/experiments/PURE_manual K=0 training.epochs=5 training.batch_size=8 training.lr=3e-4

# run_experiments.py train (K-fold wrapper) + forwarded Hydra overrides after `--`
cd ..
uv run python scripts/run_experiments.py train --experiment-root experiments/PURE_smoke --k-min 0 --k-max 0 -- training.epochs=5 training.batch_size=8 training.lr=3e-4

# run_experiments.py test + forwarded Hydra overrides after `--`
uv run python scripts/run_experiments.py test --experiment-root experiments/PURE_smoke -- paths.results_dir=results_smoke paths.predictions_dir=predictions_smoke window_size=12
```

Notes:
- For `run_experiments.py`, put extra Hydra overrides **after `--`** so argparse does not try to parse them.
- The same forwarded overrides are applied to every fold in the K-loop.
- Some `test.py` keys are overwritten from each fold's saved `arg_obj.txt` during evaluation (for example `K`, `fps`, `fpc`, `step`, `model_type`, and internal test dataset selection), so path/output-related overrides are usually the most useful there.

---

## 8. Training vs validation vs testing (terminology)

| Stage | In this repo | Typical inputs |
|--------|----------------|----------------|
| **Training** | `train.py` / `Trainer` | Unsupervised (or supervised) **train** split for fold `K` |
| **Validation** | Same run, validation split and losses | Held-out subjects within the training corpus, or another corpus if `validation_dataset` is set |
| **Testing** | `test.py` / `run_evaluation` | Loads saved checkpoints; uses **`pure_testing`** (and similar patterns for other corpora) as coded in evaluation |

---

## 9. Monitoring and artifacts

| Artifact | Where | What it is |
|----------|--------|------------|
| TensorBoard | `<experiment_dir>/runs/` | Scalars for losses, LR, etc. |
| Flat config snapshot | TensorBoard text `config/flat_args` | One-time record of resolved training settings |
| Checkpoint manifest | `checkpoints_manifest.jsonl` | Per-epoch checkpoint paths and validation metrics |
| Best summary | `training_summary.json` | Best epoch and checkpoint pointer |

View TensorBoard:

```bash
tensorboard --logdir /path/to/your/experiment_dir/runs
```

---

## 10. Troubleshooting

| Symptom | Things to check |
|---------|------------------|
| `FileNotFoundError` for CSV or `.npz` | `paths.metadata_dir`, `paths.preprocessed_dir`, and that `PURE.csv` paths point to real files. |
| Wrong fps | `dataset.fps` must be `30` or `90` for PURE; CSV must be `PURE.csv` vs `PURE_90fps.csv`. |
| Hydra cannot find config | Run `train.py` with cwd `src/` **or** use `scripts/run_experiments.py`, which sets `cwd=src`. If invoking `src/train.py` from root works in your install, that is fine too (`--cfg job` smoke test). |
| CUDA OOM | Lower `training.batch_size`, `training.fpc`, or `training.step` after reading [conf/training/default.yaml](conf/training/default.yaml). |
| Preprocessing errors | Confirm raw PURE path; try `--detector mediapipe`; ensure write permissions on the output folder. |

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

Once this works once, reuse the same layout for UBFC or new corpora by adding preprocessing, metadata CSVs, dataset classes, and registry entries as described in [README.md](README.md).
