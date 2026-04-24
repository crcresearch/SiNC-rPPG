"""Full SiNC training loop (epochs, checkpoints, validation)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import args
from datasets.cross_domain import is_cross_domain, validation_arg_obj
from datasets.utils import get_dataset
from engine.lightning_train import _use_lightning, run_lightning_training
from utils import model_utils
from utils import validate as validate_utils
from utils.checkpoint_manifest import append_checkpoint_record, write_training_summary
from utils.losses import select_loss, select_validation_loss
from utils.model_selector import select_model
from utils.optimization import optimization_loop, select_optimization_step, select_validation_step
from utils.train_logger import TrainLogger


class Trainer:
    def __init__(self, arg_obj, device: torch.device):
        self.arg_obj = arg_obj
        self.device = device

    def run(self) -> None:
        arg_obj = self.arg_obj
        device = self.device
        args.print_args(arg_obj)
        print("Using PyTorch device:", device)

        seed = int(arg_obj.K / 5)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if arg_obj.experiment_root is not None:
            experiment_root = Path(arg_obj.experiment_root)
            experiment_dir = experiment_root / f"fold{arg_obj.K}_seed{seed}"
            if not experiment_dir.is_dir():
                experiment_dir.mkdir(parents=True, exist_ok=True)
            else:
                if not bool(arg_obj.continue_training):
                    print("Directory already exists:", experiment_dir, "Exiting.")
                    sys.exit(-1)
        else:
            experiment_root = Path(arg_obj.experiments_dir)
            experiment_dir = _get_experiment_dir(experiment_root)

        print("Saving experiment to: ", experiment_dir)
        save_root = experiment_dir / "saved_models"
        best_save_root = experiment_dir / "best_saved_models"
        save_root.mkdir(parents=True, exist_ok=True)
        best_save_root.mkdir(parents=True, exist_ok=True)

        args.log_args(arg_obj, str(experiment_dir / "arg_obj.txt"))
        manifest_path = experiment_dir / "checkpoints_manifest.jsonl"
        summary_path = experiment_dir / "training_summary.json"

        train_set = get_dataset("train", arg_obj)
        val_arg = validation_arg_obj(arg_obj)
        val_set = get_dataset("val", val_arg)
        if is_cross_domain(arg_obj):
            print(
                "Cross-domain validation: train dataset",
                arg_obj.dataset,
                "| val dataset",
                val_arg.dataset,
            )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=arg_obj.batch_size, shuffle=True, num_workers=arg_obj.num_workers
        )

        if _use_lightning(arg_obj):
            run_lightning_training(
                arg_obj,
                device,
                experiment_dir,
                save_root,
                best_save_root,
                train_loader,
                val_set,
            )
            return

        logger = TrainLogger(str(experiment_dir), arg_obj, print_iter=1)
        logger.log_config_once()

        model = select_model(arg_obj)
        model = model.float().to(device)

        if bool(arg_obj.scheduler):
            optimizer = optim.SGD(model.parameters(), lr=arg_obj.lr, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        else:
            if arg_obj.optimizer == "adam":
                optimizer = optim.Adam(model.parameters(), lr=arg_obj.lr)
            elif arg_obj.optimizer == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=arg_obj.lr, momentum=0.9)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=arg_obj.lr)

        train_criterions = select_loss(arg_obj)
        optimization_step = select_optimization_step(arg_obj)
        validation_criterion = select_validation_loss(arg_obj)
        validation_step = select_validation_step(arg_obj)

        best_loss = np.inf
        if bool(arg_obj.continue_training):
            checkpoint_path, last_epoch = model_utils.get_last_checkpoint(str(save_root))
            if checkpoint_path is not None and Path(checkpoint_path).is_file():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                val_loss = checkpoint["loss"]
                best_files = [p for p in best_save_root.iterdir() if p.is_file()]
                if len(best_files) == 1:
                    best_loss = float(model_utils.get_best_loss(str(best_save_root)))
                else:
                    best_loss = np.inf
                print(
                    f"Continuing model training from {checkpoint_path} (epoch {last_epoch}) "
                    f"with best_loss of {best_loss}."
                )
                print("val_loss:", val_loss)
                print("best_loss:", best_loss)

        val_losses = []
        model_paths = []
        start = time.time()

        global_i = 0
        for epoch in range(arg_obj.epochs):
            model.train()
            model, optimizer, logger, global_i = optimization_loop(
                model,
                train_loader,
                optimizer,
                optimization_step,
                train_criterions,
                logger,
                global_i,
                epoch,
                device,
                arg_obj,
            )

            model.eval()
            val_loss = validate_utils.infer_over_dataset_training(
                model,
                val_set,
                validation_step,
                validation_criterion,
                device,
                arg_obj,
                str(experiment_dir),
                epoch,
            )
            val_losses.append(float(_tensor_to_float(val_loss["total"])))

            print("Validation Loss: %.6f" % (val_loss["total"]))
            print("Took %.3f seconds." % (time.time() - start))
            print("************************")
            print("")

            logger.log_validation(val_loss, epoch)
            lr = optimizer.param_groups[0]["lr"]
            logger.log_learning_rate(float(lr), epoch)

            save_path = _create_save_path(save_root, epoch, arg_obj)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                save_path,
            )
            model_paths.append(save_path)

            is_best = float(val_loss["total"]) < best_loss
            if is_best:
                best_loss = float(val_loss["total"])
                save_path_best = _create_best_save_path(best_save_root, arg_obj)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    save_path_best,
                )

            append_checkpoint_record(
                manifest_path,
                {
                    "epoch": int(epoch),
                    "checkpoint": save_path,
                    "val_loss": {k: _tensor_to_float(v) for k, v in val_loss.items()},
                    "is_best": is_best,
                    "trainer": "classic",
                },
            )

            if bool(arg_obj.scheduler):
                scheduler.step()

        print("Finished Training.")
        best_epoch = int(np.argmin(val_losses))
        best_path = model_paths[best_epoch]
        best_loss_final = float(_tensor_to_float(val_losses[best_epoch]))
        print(f"The best model was epoch {best_epoch} saved to {best_path}.")
        print(
            f"validation loss for best model ({arg_obj.validation_loss}):",
            round(best_loss_final, 4),
        )
        print("")
        print("Took %.3f seconds total." % (time.time() - start))
        logger.close()

        write_training_summary(
            summary_path,
            best_epoch=best_epoch,
            best_checkpoint=best_path,
            best_val_total=best_loss_final,
            manifest_path=manifest_path,
            experiment_dir=str(experiment_dir),
        )


def _tensor_to_float(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def _create_save_path(root: Path, epoch, arg_obj) -> str:
    file_name = f"{arg_obj.model_type}_e{epoch}"
    return str(root / file_name)


def _create_best_save_path(root: Path, arg_obj) -> str:
    return str(root / arg_obj.model_type)


def _get_experiment_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(subdirs) > 0:
        last_number = int(subdirs[-1].name.split("_")[-1])
        last_number += 1
    else:
        last_number = 0
    experiment_dir = root / f"exper_{last_number:04d}"
    experiment_dir.mkdir(exist_ok=False)
    return experiment_dir
