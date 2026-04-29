import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.base import BaseRPPGDataset
from datasets.path_utils import preprocessed_subdir_for_dataset, resolve_npz_path


class PURE(BaseRPPGDataset, ABC):
    """PURE dataset: subject/session metadata and NPZ clips."""

    def _register_subject_wave(self, d: dict, npz_path: Path) -> None:
        """Append this subject's timeline to ``self.waves`` (length drives clip indexing)."""
        path = str(npz_path)
        if "wave" not in d:
            raise ValueError(
                f"NPZ missing required key 'wave' for PURE supervised/testing loaders: {path}"
            )
        self.waves.append(d["wave"])

    def load_data(self):
        meta_dir = Path(self.arg_obj.metadata_dir)
        if self.fps == 30:
            meta = pd.read_csv(meta_dir / "PURE.csv")
        elif self.fps == 90:
            meta = pd.read_csv(meta_dir / "PURE_90fps.csv")
        else:
            print("Invalid fps for PURE loader. Must be in [30,90]. Exiting.")
            sys.exit(-1)

        use_mods = set()
        if self.split == "train":
            use_mods.add(self.round_robin_index % 5)
            use_mods.add((self.round_robin_index + 1) % 5)
            use_mods.add((self.round_robin_index + 2) % 5)
        elif self.split == "val":
            use_mods.add((self.round_robin_index + 3) % 5)
        elif self.split == "test":
            use_mods.add((self.round_robin_index + 4) % 5)
        elif self.split == "all":
            use_mods = set([0, 1, 2, 3, 4])
        else:
            print("Invalid split specified to PURE dataloader:", self.split, "Exiting.")
            sys.exit(-1)

        data = []
        self.waves = []
        remove_samples = set(["7/7", "7/2"])  ## Same samples removed by Gideon et al. 2021
        subdir = preprocessed_subdir_for_dataset(self.arg_obj.dataset)
        pre_dir = Path(self.arg_obj.preprocessed_dir)
        for idx, row in meta.iterrows():
            subj_id = row["subj_id"]
            sess_id = row["sess_id"]
            if (subj_id % 5 in use_mods) and (f"{subj_id}/{sess_id}" not in remove_samples):
                npz_path = resolve_npz_path(str(row["path"]), pre_dir, subdir)
                npz = np.load(npz_path)
                d = {k: npz[k] for k in npz.files}
                d["subj_id"] = subj_id
                d["sess_id"] = sess_id
                d["path"] = str(npz_path)
                self._register_subject_wave(d, npz_path)
                data.append(d)
        self.data = data

    @abstractmethod
    def set_augmentations(self) -> None:
        """Configure augmentation flags for this split."""

    @abstractmethod
    def __getitem__(self, idx):
        """Return a batch element (see concrete PURE task classes)."""
