"""Ragged per-subject wave lists / object arrays must flatten without np.asarray rags."""

from __future__ import annotations

import numpy as np

from utils.validate import evaluate_predictions, flatten_ragged_object_1d


def test_flatten_list_of_variable_length_arrays() -> None:
    rag = [np.array([1.0, 2.0]), np.array([3.0])]
    flat = flatten_ragged_object_1d(rag)
    np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 3.0]))


def test_flatten_object_ndarray() -> None:
    rag = np.array(
        [np.array([1.0, 2.0]), np.array([3.0])],
        dtype=object,
    )
    flat = flatten_ragged_object_1d(rag)
    np.testing.assert_array_equal(flat, np.array([1.0, 2.0, 3.0]))


def test_evaluate_predictions_with_list_gt_waves() -> None:
    pred_w = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    gt_w = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    pred_h = [np.array([60.0]), np.array([61.0])]
    gt_h = [np.array([60.0]), np.array([61.0])]
    ME_HR, MAE_HR, RMSE_HR, r_HR, r_wave = evaluate_predictions(pred_w, pred_h, gt_w, gt_h)
    assert MAE_HR == 0.0
    assert abs(r_wave - 1.0) < 1e-9
