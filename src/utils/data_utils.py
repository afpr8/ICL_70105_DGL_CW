# Utilities for preparing data

# Third party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from src.utils.model_args import BaseModelArgs
from src.utils.core_utils import get_device

DEVICE, PIN_MEMORY = get_device()


def check_for_negatives(df: pd.DataFrame) -> None:
    """
    This function checks if any column contains negative values

    Params:
        df: Input DataFrame to inspect
    Returns:
        None
    """
    has_negatives = (df < 0).any().any()

    if not has_negatives:
        print("Dataset has no negatives")
    else:
        print("Dataset contains negative values")


def check_for_nan(df: pd.DataFrame) -> None:
    """
    This function checks if any column contains NaN values

    Params:
        df: Input DataFrame to inspect
    Returns:
        None
    """
    has_nan = df.isna().any().any()

    if not has_nan:
        print("Dataset has no NaN values")
    else:
        print("Dataset contains NaN values")


def prepare_tensors(
        lr_np: np.ndarray | torch.Tensor,
        hr_np: np.ndarray | torch.Tensor,
        args: BaseModelArgs
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy LR and HR arrays to PyTorch tensors on the correct device
        (if not already) and pad the HR tensor

    Params:
        lr_np: Low-resolution input array of shape (LR_dim, LR_dim)
        hr_np: High-resolution target array of shape (center_dim, center_dim)
        args: AGSRArgs containing padding and device info
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            lr_t: LR tensor, shape (LR_dim, LR_dim), float32, on DEVICE
            padded_hr: Padded HR tensor, shape (HR_dim, HR_dim), float32,
                on DEVICE
    """
    if isinstance(lr_np, np.ndarray):
        lr_t = torch.tensor(lr_np, dtype=torch.float32, device=DEVICE)
    else:
        lr_t = lr_np

    if isinstance(hr_np, torch.Tensor):
        hr_np = hr_np.cpu().numpy()

    padded_hr = torch.tensor(
        pad_HR_adj(hr_np, args.padding),
        dtype=torch.float32,
        device=DEVICE
    )

    return lr_t, padded_hr


def pad_HR_adj(label: np.ndarray, padding: int) -> np. ndarray:
    """
    Pad a high-resolution adjacency matrix with zeros and set diagonal to 1

    Params:
        label: HR adjacency matrix, shape (center_dim, center_dim)
        padding: Number of rows/columns to pad on each side
    Returns:
        Padded adjacency matrix of shape (hr_dim, hr_dim)
    """
    padded = np.pad(
        label,
        ((padding, padding), (padding, padding)),
        mode="constant"
    )
    np.fill_diagonal(padded, 1)

    return padded
