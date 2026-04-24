"""Cross-fold evaluation on saved checkpoints."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from datasets.utils import get_dataset
from utils import validate as validate_utils
from utils.model_selector import select_model


def parse_log(experiment_dir: Path) -> dict:
    config_path = experiment_dir / "arg_obj.txt"
    d = {}
    with open(config_path, "r") as infile:
        for line in infile:
            clean_line = line.rstrip()
            splits = clean_line.split(" ")
            d[splits[0]] = splits[-1]
    return d


def run_evaluation(arg_obj, device: torch.device) -> None:
    print("Using PyTorch device:", device)
    load_pickle = False

    experiment_root = Path(arg_obj.experiment_root)
    print(experiment_root)

    prediction_dir = Path(arg_obj.predictions_dir)
    log_dir = Path(arg_obj.results_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = prediction_dir / (experiment_root.name + ".pkl")
    log_path = log_dir / (experiment_root.name + ".txt")

    testing_datasets = ["pure_testing"]
    fold_dirs = sorted([p.name for p in experiment_root.iterdir() if p.is_dir()])

    if load_pickle:
        with open(output_path, "rb") as infile:
            dataset_exper = pickle.load(infile)
    else:
        dataset_exper = {}
        for testing_dataset in testing_datasets:
            print("Using test set:", testing_dataset)
            print()
            exper = {}
            testing_dataset_name = testing_dataset.split("_")[0]
            for fold_dir in fold_dirs:
                print()
                str_splits = fold_dir.split("_")
                fold = str_splits[0][4:]
                seed = str_splits[1][4:]
                print(fold_dir, fold, seed)

                if seed not in exper:
                    exper[seed] = {}
                if fold not in exper[seed]:
                    exper[seed][fold] = {}

                experiment_dir = experiment_root / fold_dir
                config_dict = parse_log(experiment_dir)

                training_dataset_name = config_dict["dataset"].split("_")[0]
                print("training, testing datasets:", training_dataset_name, testing_dataset_name)
                testing_split_idx = 2 if training_dataset_name == testing_dataset_name else 3
                test_split = ["train", "val", "test", "all"][testing_split_idx]
                print("testing split:", test_split)
                arg_obj.dataset = testing_dataset
                arg_obj.K = int(config_dict["K"])
                arg_obj.fps = float(config_dict["fps"])
                arg_obj.fpc = int(config_dict["fpc"])
                arg_obj.step = int(config_dict["step"])
                test_set = get_dataset(test_split, arg_obj)

                arg_obj.model_type = config_dict["model_type"]
                model = select_model(arg_obj)

                dummy_criterion = nn.MSELoss()

                model_dir = experiment_dir / "best_saved_models"
                model_files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
                model_tag = model_files[0]
                model_path = model_dir / model_tag
                save_tag = model_tag + f"_{test_split}"
                print("best_model_path, save_tag:", model_path, save_tag)

                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.float().to(device)
                model.eval()

                _ave_loss, pred_waves, pred_HRs, gt_waves, gt_HRs = (
                    validate_utils.infer_over_dataset_testing(
                        model, test_set, dummy_criterion, device, arg_obj
                    )
                )
                exper[seed][fold]["pred_waves"] = pred_waves
                exper[seed][fold]["pred_HRs"] = pred_HRs
                exper[seed][fold]["gt_waves"] = gt_waves
                exper[seed][fold]["gt_HRs"] = gt_HRs

                ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave = validate_utils.evaluate_predictions(
                    pred_waves, pred_HRs, gt_waves, gt_HRs
                )
                print("ME, MAE, RMSE, r")
                print(f"{ME_HR:.3f} & {MAE_HR:.3f} & {RMSE_HR:.3f} & {r_HR:.3f}")
                print()

                exper[seed][fold]["ME"] = ME_HR
                exper[seed][fold]["MAE"] = MAE_HR
                exper[seed][fold]["RMSE"] = RMSE_HR
                exper[seed][fold]["r"] = r_HR

            dataset_exper[testing_dataset] = exper

        with open(output_path, "wb") as outfile:
            pickle.dump(dataset_exper, outfile)

    print()
    print("Whole Dataset Values:")
    for testing_dataset in dataset_exper.keys():
        print("Testing dataset:", testing_dataset)
        exper = dataset_exper[testing_dataset]
        exper_errors = {"ME": [], "MAE": [], "RMSE": [], "r": []}
        for seed in exper.keys():
            single_exper = exper[seed]
            pred_waves = []
            gt_waves = []
            pred_HRs = []
            gt_HRs = []
            for fold in single_exper.keys():
                pred_waves.append(single_exper[fold]["pred_waves"])
                gt_waves.append(single_exper[fold]["gt_waves"])
                pred_HRs.append(single_exper[fold]["pred_HRs"])
                gt_HRs.append(single_exper[fold]["gt_HRs"])
            pred_waves = np.hstack(pred_waves)
            gt_waves = np.hstack(gt_waves)
            pred_HRs = np.hstack(pred_HRs)
            gt_HRs = np.hstack(gt_HRs)
            ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave = validate_utils.evaluate_predictions(
                pred_waves, pred_HRs, gt_waves, gt_HRs
            )
            print("ME, MAE, RMSE, r")
            print(f"{ME_HR:.3f} & {MAE_HR:.3f} & {RMSE_HR:.3f} & {r_HR:.3f}")
            exper_errors["ME"].append(ME_HR)
            exper_errors["MAE"].append(MAE_HR)
            exper_errors["RMSE"].append(RMSE_HR)
            exper_errors["r"].append(r_HR)
        print()
        ME_mu = np.mean(exper_errors["ME"])
        MAE_mu = np.mean(exper_errors["MAE"])
        RMSE_mu = np.mean(exper_errors["RMSE"])
        r_mu = np.mean(exper_errors["r"])
        ME_std = np.std(exper_errors["ME"])
        MAE_std = np.std(exper_errors["MAE"])
        RMSE_std = np.std(exper_errors["RMSE"])
        r_std = np.std(exper_errors["r"])

        with open(log_path, "a+") as outfile:
            outfile.write(f"{testing_dataset}\n")
            outfile.write("ME, MAE, RMSE, r\n")
            outfile.write(
                f"{ME_mu:.2f} $\\pm$ {ME_std:.2f} & {MAE_mu:.2f} $\\pm$ {MAE_std:.2f} & {RMSE_mu:.2f} $\\pm$ {RMSE_std:.2f} & {r_mu:.2f} $\\pm$ {r_std:.2f}\n"
            )
            outfile.write("\n")

        print(testing_dataset)
        print("ME, MAE, RMSE, r")
        print(
            rf"{ME_mu:.2f} $\pm$ {ME_std:.2f} & {MAE_mu:.2f} $\pm$ {MAE_std:.2f} & {RMSE_mu:.2f} $\pm$ {RMSE_std:.2f} & {r_mu:.2f} $\pm$ {r_std:.2f}"
        )
        print()

    print("Done.")
