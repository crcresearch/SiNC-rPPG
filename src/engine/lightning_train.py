"""Optional PyTorch Lightning training path.

Checkpoints remain plain ``torch.save`` dicts compatible with ``test.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.optim as optim
from lightning.pytorch.loggers import TensorBoardLogger

import args
from utils import model_utils
from utils import validate as validate_utils
from utils.checkpoint_manifest import append_checkpoint_record, write_training_summary
from utils.losses import select_loss, select_validation_loss
from utils.model_selector import select_model
from utils.optimization import select_optimization_step, select_validation_step


def _use_lightning(arg_obj) -> bool:
    v = getattr(arg_obj, "use_lightning", False)
    if isinstance(v, (int, float)):
        return bool(int(v))
    return bool(v)


def _gradient_clip(arg_obj) -> float | None:
    v = float(getattr(arg_obj, "lightning_gradient_clip", 0.0) or 0.0)
    return v if v > 0 else None


class SiNCLightningModule(L.LightningModule):
    def __init__(
        self,
        arg_obj,
        val_set,
        experiment_dir: Path,
        save_root: Path,
        best_save_root: Path,
        manifest_path: Path,
    ):
        super().__init__()
        self.arg_obj = arg_obj
        self.val_set = val_set
        self.experiment_dir = Path(experiment_dir)
        self.save_root = Path(save_root)
        self.best_save_root = Path(best_save_root)
        self.manifest_path = Path(manifest_path)

        self.model = select_model(arg_obj)
        self.train_criterions = select_loss(arg_obj)
        self.validation_criterion = select_validation_loss(arg_obj)
        self.optimization_step_fn = select_optimization_step(arg_obj)
        self.validation_step_fn = select_validation_step(arg_obj)

        self._best_loss = float("inf")
        self._val_totals: list[float] = []
        self._model_paths: list[str] = []

        if bool(arg_obj.continue_training):
            ckpt, _ = model_utils.get_last_checkpoint(str(self.save_root))
            if ckpt is not None and Path(ckpt).is_file():
                d = torch.load(ckpt, map_location="cpu", weights_only=False)
                self.model.load_state_dict(d["model_state_dict"])
                print(f"Lightning: loaded model weights from {ckpt} (optimizer not restored).")

    def training_step(self, batch, batch_idx):
        losses_dict = self.optimization_step_fn(
            self.model, batch, self.train_criterions, self.device, self.arg_obj
        )
        bs = batch[0].shape[0] if hasattr(batch[0], "shape") else 1
        for k, v in losses_dict.items():
            self.log(
                f"train/loss/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=(k == "total"),
                batch_size=bs,
            )
        return losses_dict["total"]

    def configure_optimizers(self):
        o = self.arg_obj
        if bool(o.scheduler):
            opt = optim.SGD(self.model.parameters(), lr=o.lr, momentum=0.9)
            sch = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        if o.optimizer == "adam":
            return optim.Adam(self.model.parameters(), lr=o.lr)
        if o.optimizer == "sgd":
            return optim.SGD(self.model.parameters(), lr=o.lr, momentum=0.9)
        return optim.AdamW(self.model.parameters(), lr=o.lr)

    def val_dataloader(self):
        dummy = torch.zeros(1, 1)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy), batch_size=1)

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        val_loss = validate_utils.infer_over_dataset_training(
            self.model,
            self.val_set,
            self.validation_step_fn,
            self.validation_criterion,
            self.device,
            self.arg_obj,
            str(self.experiment_dir),
            self.current_epoch,
        )
        for k, v in val_loss.items():
            self.log(f"val/loss/{k}", float(_tensor_to_float(v)), prog_bar=(k == "total"))

        opt = self.optimizers()
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", float(lr), prog_bar=False)

        epoch = int(self.current_epoch)
        save_path = str(self.save_root / f"{self.arg_obj.model_type}_e{epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": val_loss,
            },
            save_path,
        )
        self._model_paths.append(save_path)
        total = float(_tensor_to_float(val_loss["total"]))
        self._val_totals.append(total)

        is_best = total < self._best_loss
        if is_best:
            self._best_loss = total
            best_path = str(self.best_save_root / self.arg_obj.model_type)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": val_loss,
                },
                best_path,
            )

        manifest_record = {
            "epoch": epoch,
            "checkpoint": save_path,
            "val_loss": {k: _tensor_to_float(v) for k, v in val_loss.items()},
            "is_best": is_best,
            "trainer": "lightning",
        }
        append_checkpoint_record(self.manifest_path, manifest_record)

        print("Validation Loss: %.6f" % (total))
        print("************************")
        print("")


def _tensor_to_float(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def run_lightning_training(
    arg_obj,
    device: torch.device,
    experiment_dir: Path,
    save_root: Path,
    best_save_root: Path,
    train_loader,
    val_set,
) -> None:
    manifest_path = experiment_dir / "checkpoints_manifest.jsonl"
    summary_path = experiment_dir / "training_summary.json"

    pl_module = SiNCLightningModule(
        arg_obj, val_set, experiment_dir, save_root, best_save_root, manifest_path
    )

    tb = TensorBoardLogger(
        save_dir=str(experiment_dir / "runs"),
        name="lightning",
        default_hp_metric=False,
    )
    clip = _gradient_clip(arg_obj)
    trainer = L.Trainer(
        max_epochs=int(arg_obj.epochs),
        accelerator="auto",
        devices=1,
        logger=tb,
        enable_checkpointing=False,
        gradient_clip_val=clip,
        num_sanity_val_steps=0,
    )

    args.print_args(arg_obj)
    print("Using PyTorch device:", device)
    print("Using PyTorch Lightning trainer.")

    start = time.time()
    trainer.fit(pl_module, train_loader)
    print("Finished Training.")

    val_losses = np.array(pl_module._val_totals, dtype=float)
    best_epoch = int(np.argmin(val_losses))
    best_path = pl_module._model_paths[best_epoch]
    best_loss = float(val_losses[best_epoch])
    print(f"The best model was epoch {best_epoch} saved to {best_path}.")
    print(f"validation loss for best model ({arg_obj.validation_loss}):", round(best_loss, 4))
    print("")
    print("Took %.3f seconds total." % (time.time() - start))

    write_training_summary(
        summary_path,
        best_epoch=best_epoch,
        best_checkpoint=best_path,
        best_val_total=best_loss,
        manifest_path=manifest_path,
        experiment_dir=str(experiment_dir),
    )


__all__ = ["SiNCLightningModule", "run_lightning_training", "_use_lightning"]
