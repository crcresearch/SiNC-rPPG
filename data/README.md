# Data layout

Place datasets here so paths are stable and independent of the process working directory.

```text
data/
├── raw/                      # Unmodified originals (e.g. PURE PNGs / UBFC videos)
│   ├── PURE/
│   └── UBFC/
├── preprocessed/             # Model-ready clips (e.g. combined .npz files)
│   ├── PURE/
│   │   ├── 01-01.npz
│   │   └── ...
│   └── UBFC/
└── metadata/                 # CSVs listing subjects and paths to preprocessed samples
    ├── PURE.csv
    ├── PURE_90fps.csv       # optional, when using fps=90
    ├── UBFC.csv
    └── UBFC_90fps.csv
```

## Migration from `src/datasets/metadata/`

If you previously kept CSVs under `src/datasets/metadata/`, copy or symlink them into `data/metadata/` and regenerate CSVs so the `path` column either points to existing files or to filenames under `data/preprocessed/<DATASET>/` (see `src/preprocessing/PURE/README.md`).

Hydra defaults (`conf/paths/default.yaml`) already reference `data/metadata` and `data/preprocessed`. Override at runtime if needed, for example:

`paths.metadata_dir=/other/metadata paths.preprocessed_dir=/other/preprocessed`
