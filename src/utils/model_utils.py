import sys
from pathlib import Path

import torch
from natsort import natsorted


def get_last_checkpoint(save_root):
    save_root = Path(save_root)
    model_files = natsorted([p.name for p in save_root.iterdir() if p.is_file()])
    if not model_files:
        return None, -1
    last_epoch = len(model_files) - 1
    checkpoint = str(save_root / model_files[-1])
    return checkpoint, last_epoch


def get_best_loss(best_save_root):
    best_save_root = Path(best_save_root)
    model_files = [p.name for p in best_save_root.iterdir() if p.is_file()]
    if len(model_files) != 1:
        print("Zero or more than one best model when trying to load best model. Exiting.")
        sys.exit(-1)
    checkpoint_path = best_save_root / model_files[0]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    best_loss = checkpoint["loss"]["total"]
    return float(best_loss.item()) if hasattr(best_loss, "item") else float(best_loss)
