import hydra
from omegaconf import DictConfig

from config_merge import hydra_cfg_to_arg_namespace
from engine.evaluation import run_evaluation
from utils.torch_device import select_torch_device

device = select_torch_device()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    arg_obj = hydra_cfg_to_arg_namespace(cfg)
    run_evaluation(arg_obj, device)


if __name__ == "__main__":
    main()
