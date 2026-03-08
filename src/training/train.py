# THIS IS A TEMPLATE TRAINING SCRIPT

# Standard Python library imports
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Type

# Third party imports
import mlflow
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Local imports
from src.datasets import BrainDataset, load_checkpoint
from src.training.logging import log_metrics_fold
from src.training.predict import get_prediction, predict
from src.utils.core_utils import get_device
from src.utils.metrics import get_metrics, precompute_gt_centralities
from src.utils.model_args import BaseModelArgs
from src.utils.submission_utils import (
    generate_submission,
    plot_cv_metrics,
    save_checkpoint
)


DEVICE, PIN_MEMORY = get_device()

TrainerFn = Callable[
    [
        torch.nn.Module,
        Dataset,
        BaseModelArgs,
        Dataset,
        int,
        bool,
        torch.nn.Module
    ],
    list[tuple[torch.Tensor, torch.Tensor]]
]


def run_3_fold_cross_validation(
    trainer_fn: TrainerFn,
    model_cls: Type[torch.nn.Module],
    model_args: BaseModelArgs,
    lr_data: np.ndarray,
    hr_data: np.ndarray,
    random_state: int = 42,
    get_final_metrics: bool = True,
    log_to_mlflow: bool = True,
    loss_fn: torch.nn.Module | None = None,
    use_checkpoint: bool = False,
    start_fold_idx: int = 0
) -> None:
    """
    Perform 3-fold cross-validation training and evaluation.

    Params:
        model_cls: PyTorch model class (not instance)
        lr_data: Low-resolution inputs, shape (N, H_in, W_in)
        hr_data: High-resolution targets, shape (N, H_out, W_out)
        trainer_fn: A function for training the model_cls on each fold
        random_state: Seed for reproducibility and KFold splitting
        get_final_metrics: If True, compute centrality-based metrics using
            precomputed GT
        log_to_mlflow: Whether or not to trigger mlflow logging
        make_model_args_fn: A function to return a dataclass with model
            hyperparameters
        loss_fn: The loss function to minimize
        use_checkpoint: If True, trains from start_fold_idx onwards, loads saved
            fold indices and fold_metrics_list to work with. If False, starts
            from scratch - assumes the SAME random_state is used across runs!!
        start_fold_idx: The fold index to start training from - only used if
            use_checkpoint is True
    Returns:
        None
    """
    lr_tensor = torch.tensor(lr_data, dtype=torch.float32)
    hr_tensor = torch.tensor(hr_data, dtype=torch.float32)

    learning_rate = model_args.lr
    batch_size = model_args.batch_size
    weight_decay = model_args.weight_decay

    if get_final_metrics:
        gtc_cache_path = Path("./checkpoint/all_gt_centralities.npy")
        if use_checkpoint and gtc_cache_path.exists():
            print(f"Loading precomputed GT centralities from {gtc_cache_path}")
            all_gt_centralities = (
                load_checkpoint(str(gtc_cache_path))
            )
        else:
            print("Precomputing GT centralities for all HR samples...")
            all_gt_centralities = precompute_gt_centralities(hr_data)

            print(f"Saving precomputed GT centralities to {gtc_cache_path}")
            save_checkpoint(
                data_to_save=all_gt_centralities,
                output_path=str(gtc_cache_path)
            )
    else:
        all_gt_centralities = None

    # Set up mlflow
    if log_to_mlflow:
        mlflow.set_experiment("DGBL-coursework")
        name = model_cls.__name__
        if name == "AGSRNet":
            title = (
                f"{name}_Lr-{learning_rate}_"
                f"lmbda-{model_args.lmbda}_"
                f"K-{model_args.K}"
            )
        else:
            title = (
                f"{name}_bs-{batch_size}_"
                f"Lr-{learning_rate}_"
                f"Wd-{weight_decay}"
            )
        mlflow.start_run(run_name=title)
        mlflow.log_params(asdict(model_args))

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    all_folds = list(kf.split(lr_tensor))

    metrics_cache_path = Path("./checkpoint/fold_metrics_list.npy")
    if use_checkpoint and metrics_cache_path.exists():
        # Use provided start_fold_idx value to truncate saved data
        fold_metrics_list = load_checkpoint(str(metrics_cache_path))
        fold_metrics_list = list(fold_metrics_list[:start_fold_idx])
    else:
        fold_metrics_list = []
        start_fold_idx = 0  # Make sure to start from the beginning

    for fold_idx in range(start_fold_idx, len(all_folds)):
        train_idx, val_idx = all_folds[fold_idx]
        print(f"\n===== Fold {fold_idx} =====")
        # split the data
        X_train, X_val = lr_tensor[train_idx], lr_tensor[val_idx]
        Y_train, Y_val = hr_tensor[train_idx], hr_tensor[val_idx]

        # initialize model per fold
        model = model_cls(model_args) if model_args is not None else model_cls()
        model = model.to(DEVICE)

        train_dataset = BrainDataset(X_train, Y_train)
        val_dataset = BrainDataset(X_val, Y_val)

        # train model and collect metrics
        output_mats_arr = trainer_fn(
            model,
            train_dataset,
            model_args,
            val_dataset,
            fold_idx,
            log_to_mlflow,
            loss_fn
        )

        # Remove padding if used
        padding = getattr(model_args, "padding", 0)
        if padding > 0:
            output_mats_arr = [
                (
                    pred[:, padding:-padding, padding:-padding],
                    gt[:, padding:-padding, padding:-padding],
                )
                for pred, gt in output_mats_arr
            ]

        # Extracting the fold predictions & putting them into a single
        # np.ndarray for submission
        pred_tensors = [tup[0] for tup in output_mats_arr]
        all_preds_tensor = torch.cat(pred_tensors, dim=0)
        hr_predictions = all_preds_tensor.detach().cpu().numpy()

        # Save fold predictions
        generate_submission(
            hr_predictions,
            output_path=f"./results/predictions_fold_{fold_idx + 1}.csv"
        )

        if get_final_metrics:
            fold_gt_cache = [all_gt_centralities[i] for i in val_idx]
        else:
            fold_gt_cache = None

        print(f"Calculating metrics for fold-{fold_idx}")
        fold_metrics = get_metrics(
            output_mats_arr,
            gt_centralities_cache=fold_gt_cache,
            final_metrics=get_final_metrics
        )
        fold_metrics_list.append(fold_metrics)

        save_checkpoint(
            data_to_save=fold_metrics_list,
            output_path=str(metrics_cache_path)
        )

        if log_to_mlflow:
            # Log per-fold metrics
            for metric_name, value in fold_metrics.items():
                mlflow.log_metric(f"fold{fold_idx}_{metric_name}", value)

    plot_cv_metrics(
        fold_metrics_list, extra_metrics=["mae_glob_eff", "mae_modularity"]
    )

    # Aggregate mean ± std metrics across folds
    for key in fold_metrics_list[0].keys():
        values = np.array([m[key] for m in fold_metrics_list])
        mean_val, std_val = np.mean(values), np.std(values)
        print(f"CV {key}: {mean_val:.5f} ± {std_val:.5f}")
        if log_to_mlflow:
            mlflow.log_metric(f"mean_{key}", mean_val)
            mlflow.log_metric(f"std_{key}", std_val)
            mlflow.end_run()


def _train_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module
) -> float:
    """
    Perform one training step for a single sample, including model updates

    Params:
        model: A generic model
        input_t: Input tensor, shape (LR_dim, LR_dim)
        optimizer: Optimizer for the model
        loss_fn: The loss function to minimize
    Returns:
        float: Total loss for this step
    """
    if loss_fn is None:
        raise ValueError("Please provide a loss function")

    x, y = x.to(DEVICE), y.to(DEVICE)  # ensure device consistency
    optimizer.zero_grad()
    outputs = get_prediction(model(x))
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    return outputs.detach(), loss.item()

# TODO: Consider implementing early stopping or scheduling using val_dataset


def train_model(
    model: torch.nn.Module,
    train_dataset: BrainDataset,
    model_args: BaseModelArgs,
    val_dataset: BrainDataset | None = None,
    fold_id: int | None = None,
    log_to_mlflow: bool = False,
    loss_fn: torch.nn.Module | None = None,
) -> torch.nn.Module:
    """
    Train a generic model with optional validation and MLflow logging

    Params:
        model: torch.nn.Module instance to train
        train_dataset: Training BrainDataset
        model_args: BaseModelArgs hyperparameters (or child class)
        val_dataset: Optional validation BrainDataset
        fold_id: Optional fold ID for logging
        log_to_mlflow: If True, log per-epoch metrics to MLflow
        loss_fn: The loss function to minimize
    Returns:
        torch.nn.Module: Trained model
    """
    model.to(DEVICE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_args.batch_size,
        shuffle=True,
        pin_memory=PIN_MEMORY
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay
    )

    for epoch in range(model_args.epochs):
        # train
        model.train()
        train_loss_running = 0.0
        train_mats_arr = []

        for x_train, y_train in tqdm(
            train_loader, desc=f"Train epoch {epoch + 1}"
        ):
            batch_pred, batch_loss = _train_step(
                model,
                x_train,
                y_train,
                optimizer,
                loss_fn
            )
            train_loss_running += batch_loss
            train_mats_arr.append(
                (batch_pred, y_train.detach())
            )

        avg_train = train_loss_running / len(train_loader)

        train_metrics = get_metrics(train_mats_arr, final_metrics=False)
        val_metrics = {}
        if val_dataset is not None:
            val_mats_arr = predict(model, val_dataset, batch_size=model_args.batch_size)
            val_metrics = get_metrics(val_mats_arr, final_metrics=False)

        if log_to_mlflow:
            log_metrics_fold(
                epoch,
                fold_id if fold_id is not None else -1,
                train_metrics,
                val_metrics
            )
            mlflow.log_metric(
                f"fold_{fold_id}_train_loss",
                avg_train,
                step=epoch + 1
            )

        print(
            f"Epoch {epoch + 1}: train_L1={avg_train:.5f} "
        )
        print(f"Train metrics: {train_metrics}")
        print(f"Validation metrics: {val_metrics}")

    return model


def train_fold(
    model: torch.nn.Module,
    train_dataset: BrainDataset,
    model_args: BaseModelArgs,
    val_dataset: BrainDataset,
    fold_id: int,
    log_to_mlflow: bool = True,
    loss_fn: torch.nn.Module | None = None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Train a model for one cross-validation fold.

    Params:
        model: PyTorch model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        hyperparams: a dataclass containing model & training hyperparameters
        fold_id: Current fold index
        seed: Random seed for DataLoader generator
        log_to_mlflow: Whether or not to trigger mlflow logging
        loss_fn: The loss function to minimize
    Returns:
        val_mats_arr: List of (prediction_batch, ground_truth_batch)
            from the final validation epoch
    """
    model = train_model(
        model,
        train_dataset,
        model_args,
        val_dataset=val_dataset,
        fold_id=fold_id,
        log_to_mlflow=log_to_mlflow,
        loss_fn=loss_fn
    )
    val_preds = predict(model, val_dataset)

    return val_preds
