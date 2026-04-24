from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TrainLogger:
    """TensorBoard logging for SiNC training (spectral / composite losses)."""

    def __init__(self, experiment_dir, arg_obj, print_iter=200):
        self.log_dir = str(Path(experiment_dir) / "runs")
        self.experiment_dir = Path(experiment_dir)
        self.arg_obj = arg_obj
        self.print_iter = print_iter
        self.running_losses = {}
        self.running_losses["total"] = 0.0
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=f"{arg_obj.model_type}")
        self._hparams_written = False

    def log_config_once(self) -> None:
        """Write a one-time text summary of flat run settings for TensorBoard."""
        if self._hparams_written:
            return
        self._hparams_written = True
        lines = []
        for key in sorted(vars(self.arg_obj)):
            val = getattr(self.arg_obj, key)
            if val is None:
                continue
            lines.append(f"{key}: {val}")
        text = "\n".join(lines)
        self.writer.add_text("config/flat_args", text, 0)

    def log(self, epoch, global_step, step, current_losses):
        for criterion in current_losses:
            if criterion not in self.running_losses:
                self.running_losses[criterion] = 0.0
            self.running_losses[criterion] += current_losses[criterion]
        if global_step % self.print_iter == (self.print_iter - 1):
            for criterion in current_losses:
                print(
                    f"[{epoch}, {global_step:5d}] Train loss ({criterion}): {current_losses[criterion]:.6f}"
                )
                tag = f"train/loss/{criterion}"
                self.writer.add_scalar(
                    tag,
                    self.running_losses[criterion] / self.print_iter,
                    global_step,
                )
                self.running_losses[criterion] = 0.0

    def log_validation(self, val_loss, epoch):
        for k in val_loss.keys():
            self.writer.add_scalar(f"val/loss/{k}", val_loss[k], epoch)

    def log_learning_rate(self, lr: float, epoch: int) -> None:
        self.writer.add_scalar("train/lr", lr, epoch)

    def close(self):
        self.writer.close()

    def symlink_logfile(self):
        log_path = self.arg_obj.log_path
        if log_path is not None:
            log_path = Path(log_path)
            if log_path.is_file():
                tail = log_path.name
                log_dir = self.experiment_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_symlink = log_dir / tail
                log_symlink.symlink_to(log_path)
            else:
                print("WARNING: Invalid log_path parameter, file does not exist.")
