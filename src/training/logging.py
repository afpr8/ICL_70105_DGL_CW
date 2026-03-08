import mlflow

def log_metrics_fold(
        epoch: int,
        fold_id: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float]
    ) -> None:
    """
    Log training and validation metrics per fold to MLflow

    Params:
        epoch: Current epoch index (0-based)
        fold_id: Fold identifier
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
    Returns:
        None
    """
    for metric_name, value in train_metrics.items():
        mlflow.log_metric(
            f"fold_{fold_id}_train_{metric_name}", value, step=epoch+1
        )
    for metric_name, value in val_metrics.items():
        mlflow.log_metric(
            f"fold_{fold_id}_val_{metric_name}", value, step=epoch+1
        )
