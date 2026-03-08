from unittest.mock import patch

# Local imports
from src.training.logging import log_metrics_fold

def test_log_metrics_fold_basic():
    epoch = 2
    fold_id = 1
    train_metrics = {"loss": 0.5, "accuracy": 0.8}
    val_metrics = {"loss": 0.6, "accuracy": 0.75}

    with patch("mlflow.log_metric") as mock_log:
        log_metrics_fold(epoch, fold_id, train_metrics, val_metrics)

        # Expected calls count = number of metrics in train + val
        assert mock_log.call_count == len(train_metrics) + len(val_metrics)

        # Check one train metric call
        mock_log.assert_any_call(
            "fold_1_train_loss", 0.5, step=epoch+1
        )
        mock_log.assert_any_call(
            "fold_1_train_accuracy", 0.8, step=epoch+1
        )

        # Check one val metric call
        mock_log.assert_any_call(
            "fold_1_val_loss", 0.6, step=epoch+1
        )
        mock_log.assert_any_call(
            "fold_1_val_accuracy", 0.75, step=epoch+1
        )

def test_log_metrics_fold_empty_dicts():
    epoch = 0
    fold_id = 0
    train_metrics = {}
    val_metrics = {}

    with patch("mlflow.log_metric") as mock_log:
        # Should not raise, and no calls made
        log_metrics_fold(epoch, fold_id, train_metrics, val_metrics)
        mock_log.assert_not_called()

def test_log_metrics_fold_single_metric():
    epoch = 1
    fold_id = 3
    train_metrics = {"mse": 0.123}
    val_metrics = {"mse": 0.456}

    with patch("mlflow.log_metric") as mock_log:
        log_metrics_fold(epoch, fold_id, train_metrics, val_metrics)
        mock_log.assert_any_call("fold_3_train_mse", 0.123, step=epoch+1)
        mock_log.assert_any_call("fold_3_val_mse", 0.456, step=epoch+1)
