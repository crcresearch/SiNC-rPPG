# Adding a new dataset

This guide lists the **minimum steps** to plug a new rPPG-style corpus into the existing SiNC-rPPG layout. For end-to-end training, paths, and Hydra usage, see [INSTRUCTIONS.md](../INSTRUCTIONS.md) (especially [§2 Registries](../INSTRUCTIONS.md#2-registries-names-and-python-classes), [§4](../INSTRUCTIONS.md#4-where-everything-lives-directory-map), [§6](../INSTRUCTIONS.md#6-end-to-end-workflow-pure), [§7](../INSTRUCTIONS.md#7-configuration-and-hydra-overrides)).

---

## 1. Data layout

| Location | Role |
|----------|------|
| `data/preprocessed/<Corpus>/` | One **`.npz`** per clip (or per subject/session, as designed). Same layout as PURE/UBFC: see [data/README.md](../data/README.md). |
| `data/metadata/<Corpus>.csv` | Rows point to `.npz` files (paths relative to `preprocessed_dir/<Corpus>/` or absolute). |

**Loader convention:** registry keys like `foo_unsupervised` use the substring **before the first `_`** to resolve the preprocessed folder name via [`preprocessed_subdir_for_dataset`](../src/datasets/path_utils.py) (`foo` → `FOO` if not listed in the explicit `mapping`). Add a mapping entry there only if the folder name must differ from `prefix.upper()`.

---

## 2. Dataset Python class

1. Add a module under [`src/datasets/`](../src/datasets/) (e.g. mirror [`PURE.py`](../src/datasets/PURE.py) / [`UBFC.py`](../src/datasets/UBFC.py)): subclass [`BaseRPPGDataset`](../src/datasets/base.py) or an existing pattern, implement `load_data`, `set_augmentations`, `__getitem__`.
2. Match the **batch tuple** expected by the training step:
   - **Unsupervised:** same as `PUREUnsupervised` (clip, subject index, frame indices, speed).
   - **Supervised:** same as `PURESupervised` (clip, wave, HR, subject, speed).
3. **Supervised / test loaders** that read `wave` must enforce presence of `wave` in each `.npz` (see [INSTRUCTIONS §4 — Preprocessed `.npz` keys](../INSTRUCTIONS.md#4-where-everything-lives-directory-map)).

**Core training code:** [`Trainer`](../src/engine/trainer.py), [`optimization.py`](../src/utils/optimization.py), and Hydra merge typically **do not** need edits if the new loader follows the same tuple contract as PURE/UBFC.

---

## 3. Registry

Edit [`src/datasets/dataset_registry.py`](../src/datasets/dataset_registry.py):

```python
DATASET_REGISTRY.register("foo_unsupervised", FooUnsupervised)
```

Use a **lowercase** stable string; it becomes `arg_obj.dataset` and the Hydra `dataset=` override.

**Mixed training:** `mixed_sub_datasets` may reference this key once it is registered. No change to [`MixedTrainDataset`](../src/datasets/mixed_train.py) beyond registry membership.

---

## 4. Hydra preset

Add [`conf/dataset/foo_unsupervised.yaml`](../conf/dataset/) (copy from [`pure_unsupervised.yaml`](../conf/dataset/pure_unsupervised.yaml) or [`ubfc_unsupervised.yaml`](../conf/dataset/ubfc_unsupervised.yaml)):

- Set `dataset: foo_unsupervised` (must match the registry key).
- Set `fps`, `frame_width`, `frame_height` to match the CSV and clips.
- For **supervised** presets, include the same `optimization_step` / `losses` / … overrides as [`pure_supervised.yaml`](../conf/dataset/pure_supervised.yaml) (dataset group is merged **after** training in [`config_merge.py`](../src/config_merge.py)).

Select at runtime: `dataset=foo_unsupervised` or `--dataset foo_unsupervised` with [`scripts/run_experiments.py`](../scripts/run_experiments.py).

---

## 5. Preprocessing

- Prefer a dedicated folder under `src/preprocessing/<YourCorpus>/` (see [`src/preprocessing/PURE/`](../src/preprocessing/PURE/) and [`UBFC-rPPG/`](../src/preprocessing/UBFC-rPPG/)).
- Scripts should write `.npz` files and emit a CSV compatible with the new loader’s `load_data()` (column names and paths).

Preprocessing is **not** invoked automatically by training; it is a separate step documented per corpus.

---

## 6. Optional: evaluation / testing

- **Training-time validation on another corpus:** Hydra `validation_dataset` / `validation_fps` ([INSTRUCTIONS §7](../INSTRUCTIONS.md#7-configuration-and-hydra-overrides)).
- **`test.py`:** add a **supervised** test-style class if needed (see `pure_testing` / `ubfc_testing` pattern), register it, add `conf/dataset/foo_testing.yaml`, and set `testing_dataset=foo_testing` when running evaluation.

---

## 7. Checklist

| Step | Action |
|------|--------|
| 1 | `data/preprocessed/<Corpus>/` + `data/metadata/<Corpus>.csv` |
| 2 | `src/datasets/*.py` — loader class(es), tuple contract |
| 3 | `dataset_registry.py` — `.register(...)` |
| 4 | `conf/dataset/<name>.yaml` — `dataset:` matches registry |
| 5 | `src/preprocessing/<Corpus>/` — optional but recommended |
| 6 | `path_utils.preprocessed_subdir_for_dataset` — only if default `prefix.upper()` is wrong |
| 7 | `pytest` — smoke load for `train`/`val` split |

---

## When core files *do* need edits

| Situation | Likely touch |
|-----------|----------------|
| New preprocessed folder naming rule | [`path_utils.py`](../src/datasets/path_utils.py) `mapping` |
| New loss or batch structure | `optimization.py`, `losses.py`, loaders **together** |
| New global Hydra flag | [`conf/config.yaml`](../conf/config.yaml), [`config_merge.py`](../src/config_merge.py) if not already flattened |

For a **third corpus** with the same clip format and PURE-like CSV logic, steps **1–5** above are usually sufficient.
