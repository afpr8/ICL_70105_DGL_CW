import numpy as np
import pytest
import torch

from src.utils import metrics


def _make_small_graph_pair():
    """
    Create a small pair of adjacency matrices for testing

    Returns:
        pred_mat, gt_mat: np.ndarray of shape (3,3)
    """
    pred_mat = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]], dtype=float)
    gt_mat = np.array([[0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0]], dtype=float)
    return pred_mat, gt_mat


def test_compute_centrality_errors_matches_shapes():
    pred_mat, gt_mat = _make_small_graph_pair()
    gt_cent = metrics.precompute_gt_centralities(np.array([gt_mat]))[0]
    mae_bc, mae_ec, mae_pc, ge, ac  = metrics._compute_centrality_errors(
        pred_mat,
        gt_mat,
        gt_cent
    )
    # Should return non-negative floats
    assert all(
        isinstance(x, float) and x >= 0
        for x in (mae_bc, mae_ec, mae_pc, ge, ac)
    )


def test_compute_adjacency_metrics_basic():
    pred = np.array([0, 1, 2], dtype=float)
    gt = np.array([0, 1, 2], dtype=float)
    result = metrics._compute_adjacency_metrics(pred, gt)
    assert np.isclose(result["mae"], 0.0)
    assert np.isclose(result["pcc"], 1.0)
    assert np.isclose(result["js_distance"], 0.0)


def test_get_metrics_without_centralities():
    pred_mat, gt_mat = _make_small_graph_pair()
    pred_tensor = torch.from_numpy(np.expand_dims(pred_mat, axis=0)).float()
    gt_tensor = torch.from_numpy(np.expand_dims(gt_mat, axis=0)).float()
    output_graphs_arr = [(pred_tensor, gt_tensor)]
    res = metrics.get_metrics(output_graphs_arr, final_metrics=False)
    assert all(k in res for k in ["mae", "pcc", "js_distance"])
    assert res["mae"] >= 0


def test_get_metrics_with_centralities():
    pred_mat, gt_mat = _make_small_graph_pair()
    pred_tensor = torch.from_numpy(np.expand_dims(pred_mat, axis=0)).float()
    gt_tensor = torch.from_numpy(np.expand_dims(gt_mat, axis=0)).float()
    output_graphs_arr = [(pred_tensor, gt_tensor)]
    gt_cent_cache = metrics.precompute_gt_centralities(np.array([gt_mat]))
    res = metrics.get_metrics(
        output_graphs_arr,
        gt_centralities_cache=gt_cent_cache,
        final_metrics=True
    )
    for k in ["mae", "pcc", "js_distance", "mae_bc", "mae_ec", "mae_pc"]:
        assert k in res
        assert isinstance(res[k], (float, np.floating))


def test_get_metrics_requires_cache_when_final_true():
    pred_mat, gt_mat = _make_small_graph_pair()
    pred_tensor = torch.from_numpy(np.expand_dims(pred_mat, axis=0)).float()
    gt_tensor = torch.from_numpy(np.expand_dims(gt_mat, axis=0)).float()
    output_graphs_arr = [(pred_tensor, gt_tensor)]
    with pytest.raises(ValueError):
        metrics.get_metrics(output_graphs_arr, final_metrics=True)


def test_get_metrics_with_larger_graph():
    """
    Test get_metrics with a slightly larger 5x5 graph to ensure centrality MAEs
        are non-negative
    """
    # Construct a simple 5-node undirected graph
    pred_mat = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ], dtype=float)

    gt_mat = np.array([
        [0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ], dtype=float)

    pred_tensor = torch.from_numpy(np.expand_dims(pred_mat, axis=0)).float()
    gt_tensor = torch.from_numpy(np.expand_dims(gt_mat, axis=0)).float()
    output_graphs_arr = [(pred_tensor, gt_tensor)]

    # Precompute centralities for GT
    hr_stack = np.stack([gt_mat], axis=0)
    gt_centralities_cache = metrics.precompute_gt_centralities(hr_stack)

    # Compute metrics
    res = metrics.get_metrics(
        output_graphs_arr,
        gt_centralities_cache=gt_centralities_cache,
        final_metrics=True
    )

    # Assert keys exist
    expected_keys = ["mae", "pcc", "js_distance", "mae_bc", "mae_ec", "mae_pc"]
    for k in expected_keys:
        assert k in res
        # Centrality MAEs and MAE should be >= 0
        assert res[k] >= 0
