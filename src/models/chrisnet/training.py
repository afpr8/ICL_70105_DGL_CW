# Training utilities for ChrisNet (NeuroSRGAN)

# Third party imports
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
from src.datasets import BrainDataset
from src.models.chrisnet.config import ChrisNetArgs
from src.models.chrisnet.model import (
    ChrisNet,
    StandardDiscriminator,
    TopologyAwareDiscriminator,
    gaussian_noise_layer,
    to_networkx,
    compute_topo_features,
)
from src.models.chrisnet.preprocessing import (
    compute_community_masks,
    prepare_tensors,
)
from src.training.logging import log_metrics_fold
from src.utils.core_utils import get_device
from src.utils.metrics import get_metrics

DEVICE, PIN_MEMORY = get_device()


def _precompute_gt_topo(
        dataset: BrainDataset,
        args: ChrisNetArgs
    ) -> list[torch.Tensor]:
    """
    Pre-compute ground-truth topology features for every sample in the dataset.
    Caching avoids recomputing expensive shortest-path computations each epoch.

    Params:
        dataset: BrainDataset containing LR-HR pairs
        args: ChrisNetArgs with padding info
    Returns:
        List of topology tensors [SWI, GE, Q] of shape (3,), one per sample
    """
    gt_topos = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for _, hr_np in loader:
        _, padded_hr = prepare_tensors(
            hr_np.squeeze(0).numpy(),
            hr_np.squeeze(0).numpy(),
            args
        )
        G = to_networkx(padded_hr, threshold_pct=args.threshold_pct)
        swi, ge, q = compute_topo_features(G)
        gt_topos.append(
            torch.tensor([swi, ge, q], dtype=torch.float32, device=DEVICE)
        )

    return gt_topos


def _train_chrisnet_step(
    model: ChrisNet,
    netD: torch.nn.Module,
    lr_t: torch.Tensor,
    padded_hr: torch.Tensor,
    masks: list[torch.Tensor] | None,
    gt_topo: torch.Tensor | None,
    optimizerG: torch.optim.Optimizer,
    optimizerD: torch.optim.Optimizer,
    args: ChrisNetArgs,
    loss_fn: torch.nn.Module | None = None,
) -> float:
    """
    Perform one training step for a single LR-HR sample, including
    generator and discriminator updates

    Params:
        model: ChrisNet generator
        netD: Discriminator (StandardDiscriminator or TopologyAwareDiscriminator)
        lr_t: LR input tensor of shape (lr_dim, lr_dim)
        padded_hr: Padded HR target tensor of shape (hr_dim, hr_dim)
        masks: Per-sample community masks (K tensors of shape (hr_dim, hr_dim)),
            or None for variants that do not use community structure
        gt_topo: Pre-computed topology features [SWI, GE, Q] for the HR target,
            or None for variants that do not use the topology discriminator
        optimizerG: Optimizer for the generator
        optimizerD: Optimizer for the discriminator
        args: ChrisNetArgs hyperparameters
        loss_fn: Unused; present for API consistency. Must be None
    Returns:
        Total generator + reconstruction loss for this step
    """
    if loss_fn is not None:
        raise ValueError("No loss_fn parameter is expected for ChrisNet")

    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCELoss()

    lr_t = lr_t.to(DEVICE)
    padded_hr = padded_hr.to(DEVICE)

    # Forward pass
    model_outputs, net_outs, start_gcn_outs, _ = model(lr_t, masks)

    # Spectral alignment loss: align GSR weights with HR eigenvectors
    device = padded_hr.device
    _, eig_vecs = torch.linalg.eigh(padded_hr.cpu())
    eig_vecs = eig_vecs.to(device)

    # Access GSR weights through the correct sub-layer
    gsr_layer = model.layer.gsr if hasattr(model.layer, 'gsr') else model.layer
    weights = gsr_layer.weights.to(device)

    mse_loss = (
        args.lmbda * mse_loss_fn(net_outs, start_gcn_outs)
        + mse_loss_fn(model_outputs, padded_hr)
        + mse_loss_fn(weights, eig_vecs)
    )

    # Discriminator update
    optimizerD.zero_grad()
    noisy_hr = gaussian_noise_layer(padded_hr, args)

    if isinstance(netD, TopologyAwareDiscriminator):
        # Compute pred_topo once and reuse in the generator update below
        with torch.no_grad():
            G_pred = to_networkx(
                model_outputs,
                threshold_pct=netD.threshold_pct
            )
            swi, ge, q = compute_topo_features(G_pred)
            pred_topo = torch.tensor(
                [swi, ge, q],
                dtype=torch.float32,
                device=DEVICE
            )
        d_real = netD(model_outputs.detach(), topo=pred_topo)
        d_fake = netD(noisy_hr, topo=gt_topo)
    else:
        pred_topo = None
        d_real = netD(model_outputs.detach())
        d_fake = netD(noisy_hr)

    dc_loss = (
        bce_loss_fn(d_real, torch.ones_like(d_real))
        + bce_loss_fn(d_fake, torch.zeros_like(d_fake))
    )
    dc_loss.backward()
    torch.nn.utils.clip_grad_norm_(netD.parameters(), args.grad_clip)
    optimizerD.step()

    # Generator update — pass model_outputs so gradients flow back to the
    #   generator
    # Reuse pred_topo computed above to avoid a second shortest-path computation
    optimizerG.zero_grad()

    if isinstance(netD, TopologyAwareDiscriminator):
        d_fake_gen = netD(model_outputs, topo=pred_topo)
    else:
        d_fake_gen = netD(model_outputs)

    gen_loss = bce_loss_fn(d_fake_gen, torch.ones_like(d_fake_gen))
    (args.gen_loss_weight * gen_loss + mse_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizerG.step()

    return (gen_loss + mse_loss).item()


def train_chrisnet(
    model: ChrisNet,
    train_dataset: BrainDataset,
    model_args: ChrisNetArgs,
    val_dataset: BrainDataset | None = None,
    fold_id: int | None = None,
    log_to_mlflow: bool = False,
    loss_fn: torch.nn.Module | None = None,
) -> ChrisNet:
    """
    Train ChrisNet generator with optional validation and MLflow logging.

    Community masks and (for topology variants) GT topology features are
    precomputed once before training begins and cached for all epochs.

    Params:
        model: ChrisNet instance to train
        train_dataset: Training BrainDataset
        model_args: ChrisNetArgs hyperparameters
        val_dataset: Optional validation BrainDataset
        fold_id: Optional fold ID for logging
        log_to_mlflow: If True, log per-epoch metrics to MLflow
        loss_fn: Unused; must be None
    Returns:
        Trained ChrisNet model
    """
    model.to(DEVICE)

    optimizerG = torch.optim.Adam(
        model.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay
    )

    # Choose discriminator based on variant
    use_topo_disc = model_args.variant in ('full', 'topology_only')
    use_community = model_args.variant in ('full', 'community_only')

    if use_topo_disc:
        netD = TopologyAwareDiscriminator(model_args).to(DEVICE)
    else:
        netD = StandardDiscriminator(model_args).to(DEVICE)

    optimizerD = torch.optim.Adam(
        netD.parameters(),
        lr=model_args.lr,
        weight_decay=model_args.weight_decay
    )

    # Pre-compute community masks (one list of K tensors per training sample)
    train_masks: list[list[torch.Tensor] | None] = []
    train_loader_raw = DataLoader(
        train_dataset, batch_size=1, shuffle=False, pin_memory=PIN_MEMORY
    )
    for lr_np, _ in train_loader_raw:
        if use_community:
            train_masks.append(
                compute_community_masks(lr_np.squeeze(0).numpy(), model_args)
            )
        else:
            train_masks.append(None)

    # Pre-compute GT topology features for topology-aware discriminator
    gt_topos: list[torch.Tensor | None]
    if use_topo_disc:
        gt_topos = _precompute_gt_topo(train_dataset, model_args)
    else:
        gt_topos = [None] * len(train_dataset)

    sample_indices = list(range(len(train_dataset)))

    for epoch in range(model_args.epochs):
        model.train()
        epoch_loss = []

        np.random.shuffle(sample_indices)

        for sample_idx in sample_indices:
            lr_np, hr_np = train_dataset[sample_idx]
            lr_t, padded_hr = prepare_tensors(
                lr_np.numpy(),
                hr_np.numpy(),
                model_args
            )

            epoch_loss.append(
                _train_chrisnet_step(
                    model,
                    netD,
                    lr_t,
                    padded_hr,
                    masks=train_masks[sample_idx],
                    gt_topo=gt_topos[sample_idx],
                    optimizerG=optimizerG,
                    optimizerD=optimizerD,
                    args=model_args,
                    loss_fn=loss_fn,
                )
            )

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

        print(
            f"Epoch {epoch+1}/{model_args.epochs}, "
            f"Loss={np.mean(epoch_loss):.5f}"
        )
        print(f"Train metrics: {train_metrics}")
        if val_dataset:
            print(f"Val metrics: {val_metrics}")

    return model


def train_fold_chrisnet(
        model: ChrisNet,
        train_dataset: BrainDataset,
        model_args: ChrisNetArgs,
        val_dataset: BrainDataset | None = None,
        fold_id: int | None = None,
        log_to_mlflow: bool = True,
        loss_fn: torch.nn.Module | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Train ChrisNet on a cross-validation fold and return validation predictions

    Params:
        model: ChrisNet instance to train
        train_dataset: Training BrainDataset
        model_args: ChrisNetArgs hyperparameters
        val_dataset: Validation BrainDataset
        fold_id: Fold identifier for logging
        log_to_mlflow: If True, log metrics to MLflow
        loss_fn: Unused; must be None
    Returns:
        Validation predictions paired with ground truth,
            list of (pred, target) tensor tuples
    """
    model = train_chrisnet(
        model,
        train_dataset,
        model_args,
        val_dataset=val_dataset,
        fold_id=fold_id,
        log_to_mlflow=log_to_mlflow,
        loss_fn=loss_fn,
    )

    use_community = model_args.variant in ('full', 'community_only')

    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            lr_np, hr_np = val_dataset[i]
            lr_arr = lr_np.numpy()
            lr_t, padded_hr = prepare_tensors(lr_arr, hr_np.numpy(), model_args)
            masks = (
                compute_community_masks(lr_arr, model_args)
                if use_community else None
            )
            pred, _, _, _ = model(lr_t, masks)
            val_preds.append(
                (pred.unsqueeze(0).detach(), padded_hr.unsqueeze(0).detach())
            )

    return val_preds


def compute_metrics(
        model: ChrisNet,
        dataset: BrainDataset,
        args: ChrisNetArgs
    ) -> dict[str, float]:
    """
    Compute evaluation metrics for a dataset using the generator model

    Params:
        model: Trained ChrisNet generator
        dataset: BrainDataset containing LR-HR pairs
        args: ChrisNetArgs with padding, lr_dim, hr_dim, and community params
    Returns:
        Computed metrics (e.g., MAE, PCC) for the dataset
    """
    use_community = args.variant in ('full', 'community_only')

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
            lr_arr = lr_np.squeeze(0).numpy()
            lr_t, padded_hr = prepare_tensors(
                lr_arr,
                hr_np.squeeze(0).numpy(),
                args
            )

            masks = (
                compute_community_masks(lr_arr, args) if use_community else None
            )
            preds, _, _, _ = model(lr_t, masks)
            mats_arr.append((preds.unsqueeze(0), padded_hr.unsqueeze(0)))

    return get_metrics(mats_arr, final_metrics=False)


def predict_from_arrays(
        model: ChrisNet,
        lr_arrays: np.ndarray,
        model_args: ChrisNetArgs
    ) -> np.ndarray:
    """
    Predict high-resolution outputs from low-resolution arrays

    Params:
        model: Trained ChrisNet generator
        lr_arrays: NumPy array of LR inputs, shape (N, lr_dim, lr_dim)
        model_args: ChrisNetArgs with hr_dim, padding, and community params
    Returns:
        Predicted HR arrays of shape (N, center_dim, center_dim)
    """
    use_community = model_args.variant in ('full', 'community_only')

    model.eval()
    n_samples = len(lr_arrays)
    padding = model_args.padding
    hr_dim = model_args.hr_dim
    center_dim = hr_dim - 2 * padding  # e.g. 268

    hr_predictions = np.zeros(
        (n_samples, center_dim, center_dim), dtype=np.float32
    )

    with torch.no_grad():
        for i, lr_np in enumerate(lr_arrays):
            lr_t = torch.tensor(lr_np, dtype=torch.float32, device=DEVICE)
            masks = (
                compute_community_masks(lr_np, model_args)
                if use_community else None
            )
            preds, _, _, _ = model(lr_t, masks)
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
        model_args: ChrisNetArgs
    ) -> np.ndarray:
    """
    Train ChrisNet on full training data, then predict HR for test LR arrays

    Params:
        lr_train: NumPy array of LR training samples
            shape (N_train, lr_dim, lr_dim)
        hr_train: NumPy array of HR training samples
            shape (N_train, center_dim, center_dim)
        lr_test: NumPy array of LR test samples, shape (N_test, lr_dim, lr_dim)
        model_args: ChrisNetArgs instance with training hyperparameters
    Returns:
        Predicted HR arrays, shape (N_test, center_dim, center_dim)
    """
    full_train_dataset = BrainDataset(
        torch.tensor(lr_train, dtype=torch.float32),
        torch.tensor(hr_train, dtype=torch.float32)
    )

    model = ChrisNet(model_args)

    model = train_chrisnet(
        model,
        full_train_dataset,
        model_args,
        val_dataset=None,
        fold_id=None,
        log_to_mlflow=False,
        loss_fn=None,
    )

    return predict_from_arrays(model, lr_test, model_args)
