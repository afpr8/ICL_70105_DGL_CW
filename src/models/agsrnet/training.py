# File to train the agsrnet model architecture as a baseline for evaluation

# Third party imports
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
from src.datasets import BrainDataset
from src.models.agsrnet.config import AGSRArgs
from src.models.agsrnet.model import (
    AGSRNet,
    Discriminator,
    gaussian_noise_layer
)
from src.models.agsrnet.preprocessing import (
    prepare_agsr_inputs,
    prepare_tensors
)
from src.training.logging import log_metrics_fold
from src.training.predict import predict
from src.utils.core_utils import get_device
from src.utils.metrics import get_metrics

DEVICE, PIN_MEMORY = get_device()


def _train_agsr_step(
    model: torch.nn.Module,
    netD: torch.nn.Module,
    lr_t: torch.Tensor,
    padded_hr: torch.Tensor,
    optimizerG: torch.optim.Optimizer,
    optimizerD: torch.optim.Optimizer,
    args: AGSRArgs,
    loss_fn: torch.nn.Module | None = None,
) -> float:
    """
    Perform one training step for a single LR-HR sample, including
    generator and discriminator updates

    Params:
        model: AGSRNet generator model
        netD: Discriminator model
        lr_t: Low-resolution input tensor, shape (LR_dim, LR_dim)
        padded_hr: Padded high-resolution target tensor, shape (HR_dim, HR_dim)
        optimizerG: Optimizer for the generator
        optimizerD: Optimizer for the discriminator
        args: AGSRArgs dataclass containing hyperparameters
        loss_fn: The intended loss function to minimize
            Expect None for AGSRNet since the objective is hardcoded
    Returns:
        float: Total generator + reconstruction loss for this step
    """
    if loss_fn is not None:
        raise ValueError("No loss_fn parameter is expected for AGSRNet")

    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCELoss()

    # Make sure all tensors are on the same device for the forward pass
    lr_t = lr_t.to(DEVICE)
    padded_hr = padded_hr.to(DEVICE)

    # Forward pass
    model_outputs, net_outs, start_gcn_outs, _ = model(lr_t)

    device = padded_hr.device
    _, eig_vecs = torch.linalg.eigh(padded_hr.cpu())
    eig_vecs = eig_vecs.to(device)

    weights = model.layer.weights.to(device)

    mse_loss = (
        args.lmbda * mse_loss_fn(net_outs, start_gcn_outs)
        + mse_loss_fn(model_outputs, padded_hr)
        + mse_loss_fn(weights, eig_vecs)
    )

    # Discriminator
    optimizerD.zero_grad()
    d_real = netD(model_outputs.detach())
    d_fake = netD(gaussian_noise_layer(padded_hr, args))
    dc_loss = (
        bce_loss_fn(d_real, torch.ones_like(d_real))
        + bce_loss_fn(d_fake, torch.zeros_like(d_fake))
    )
    dc_loss.backward()
    optimizerD.step()

    # Generator
    optimizerG.zero_grad()
    d_fake_gen = netD(gaussian_noise_layer(padded_hr, args))
    gen_loss = bce_loss_fn(d_fake_gen, torch.ones_like(d_fake_gen))
    (gen_loss + mse_loss).backward()
    optimizerG.step()

    return (gen_loss + mse_loss).item()


# TODO: Consider implementing early stopping or scheduling using val_dataset
def train_agsr(
    model: torch.nn.Module,
    train_dataset: BrainDataset,
    model_args: AGSRArgs,
    val_dataset: BrainDataset | None = None,
    fold_id: int | None = None,
    log_to_mlflow: bool = False,
    loss_fn: torch.nn.Module | None = None,
) -> torch.nn.Module:
    """
    Train AGSR-Net generator with optional validation and MLflow logging

    Params:
        model: AGSRNet instance to train
        train_dataset: Training BrainDataset
        model_args: AGSRArgs hyperparameters
        val_dataset: Optional validation BrainDataset
        fold_id: Optional fold ID for logging
        log_to_mlflow: If True, log per-epoch metrics to MLflow
        loss_fn: The intended loss function to minimize
            Expect None for AGSRNet since the objective is hardcoded
    Returns:
        torch.nn.Module: Trained model
    """
    model.to(DEVICE)

    optimizerG = torch.optim.Adam(
        model.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay
    )
    netD = Discriminator(model_args).to(DEVICE)
    optimizerD = torch.optim.Adam(
        netD.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_args.batch_size,
        shuffle=True,
        pin_memory=PIN_MEMORY
    )

    for epoch in range(model_args.epochs):
        model.train()
        epoch_loss = []

        for lr_np, hr_np in train_loader:
            lr_t, padded_hr = prepare_tensors(
                lr_np.squeeze(0).numpy(),
                hr_np.squeeze(0).numpy(),
                model_args
            )

            epoch_loss.append(
                _train_agsr_step(
                    model,
                    netD,
                    lr_t,
                    padded_hr,
                    optimizerG,
                    optimizerD,
                    model_args,
                    loss_fn
                )
            )

        # Compute metrics per epoch
        train_metrics = compute_metrics(model, train_dataset, model_args)
        val_metrics = {}
        if val_dataset is not None:
            val_metrics = compute_metrics(model, val_dataset, model_args)

        if log_to_mlflow:    
            log_metrics_fold(
                epoch,
                fold_id if fold_id is not None else -1,
                train_metrics,
                val_metrics
            )

        # Consider removing printing if too verbose
        print(
            f"Epoch {epoch+1}/{model_args.epochs}, "
            f"Loss={np.mean(epoch_loss):.5f}"
        )
        print(f"Train metrics: {train_metrics}")
        if val_dataset:
            print(f"Val metrics: {val_metrics}")

    return model


def train_fold_agsr(
    model: torch.nn.Module,
    train_dataset: BrainDataset,
    model_args: AGSRArgs,
    val_dataset: BrainDataset | None = None,
    fold_id: int | None = None,
    log_to_mlflow: bool = True,
    loss_fn: torch.nn.Module | None = None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Train AGSR-Net on a cross-validation fold and return validation predictions

    Params:
        model: AGSRNet instance to train
        train_dataset: Training BrainDataset
        val_dataset: Validation BrainDataset
        model_args: AGSRArgs hyperparameters
        fold_id: Fold identifier for logging
        log_to_mlflow: If True, log metrics to MLflow
    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]:
            Validation predictions paired with ground truth
    """
    model = train_agsr(
        model,
        train_dataset,
        model_args,
        val_dataset=val_dataset,
        fold_id=fold_id,
        log_to_mlflow=log_to_mlflow,
        loss_fn=loss_fn,
    )
    val_dataset_prepared = prepare_agsr_inputs(val_dataset, model_args)
    val_preds = predict(model, val_dataset_prepared)

    return val_preds


def compute_metrics(
        model: torch.nn.Module,
        dataset: BrainDataset,
        args: AGSRArgs
    ) -> dict[str, float]:
    """
    Compute evaluation metrics for a dataset using the generator model

    Params:
        model: Trained AGSRNet generator
        dataset: BrainDataset containing LR-HR pairs
        args: AGSRArgs with padding, lr_dim, hr_dim
    Returns:
        dict[str, float]: Computed metrics (e.g., MSE, PSNR) for the dataset
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=PIN_MEMORY
    )
    model.eval()
    mats_arr = []
    with torch.no_grad():
        for lr_np, hr_np in loader:
            lr_t, padded_hr = prepare_tensors(
                lr_np.squeeze(0).numpy(),
                hr_np.squeeze(0).numpy(),
                args
            )
            preds, _, _, _ = model(lr_t)
            mats_arr.append((preds.unsqueeze(0), padded_hr.unsqueeze(0)))
    return get_metrics(mats_arr, final_metrics=False)


def predict_from_arrays(
    model: torch.nn.Module,
    lr_arrays: np.ndarray,
    model_args: AGSRArgs
) -> np.ndarray:
    """
    Predict high-resolution outputs from low-resolution arrays

    Params:
        model: Trained AGSRNet generator
        lr_arrays: NumPy array of LR inputs, shape (N, LR_dim, LR_dim)
        model_args: AGSRArgs containing hr_dim and padding info
    Returns:
        np.ndarray: Predicted HR arrays of shape (N, center_dim, center_dim)
    """
    model.eval()
    n_samples = len(lr_arrays)
    padding = model_args.padding
    hr_dim = model_args.hr_dim
    center_dim = hr_dim - 2 * padding  # e.g., 268

    hr_predictions = np.zeros(
        (n_samples, center_dim, center_dim),
        dtype=np.float32
    )

    with torch.no_grad():
        for i, lr_np in enumerate(lr_arrays):
            lr_t = torch.tensor(lr_np, dtype=torch.float32, device=DEVICE)
            preds, _, _, _ = model(lr_t)
            pred_np = preds.detach().cpu().numpy()
            hr_predictions[i] = pred_np[
                padding:padding + center_dim,
                padding:padding + center_dim
            ]

    return hr_predictions


def train_full_and_predict(
    lr_train: np.ndarray,
    hr_train: np.ndarray,
    lr_test: np.ndarray,
    model_args: AGSRArgs
) -> np.ndarray:
    """
    Train AGSR-Net on full training data, then predict HR for test LR arrays

    Params:
        lr_train: NumPy array of LR training samples,
            shape (N_train, LR_dim, LR_dim)
        hr_train: NumPy array of HR training samples,
            shape (N_train, center_dim, center_dim)
        lr_test: NumPy array of LR test samples,
            shape (N_test, LR_dim, LR_dim)
        model_args: AGSRArgs instance with training hyperparameters
    Returns:
        np.ndarray: Predicted HR arrays, shape (N_test, center_dim, center_dim)
    """
    full_train_dataset = BrainDataset(
        torch.tensor(lr_train, dtype=torch.float32),
        torch.tensor(hr_train, dtype=torch.float32)
    )

    model = AGSRNet(model_args)

    # Train model on full dataset (no validation, no fold_id)
    model = train_agsr(
        model,
        full_train_dataset,
        model_args,
        val_dataset=None,
        fold_id=None,
        log_to_mlflow=False,
        loss_fn=None # This will default to the hardcoded functions for AGSRNet
    )

    hr_predictions = predict_from_arrays(model, lr_test, model_args)

    return hr_predictions
