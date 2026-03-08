# The necessary classes and functions for loading the datasets in data/ to use
#   for testing our GNNs

# Standard imports
from typing import Any

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Local imports
from src.matrix_vectorizer import MatrixVectorizer


class BrainDataset(Dataset):
    """
    PyTorch Dataset for paired brain connectivity matrices
    Each sample consists of:
        - A low-resolution adjacency matrix (input)
        - A high-resolution adjacency matrix (target)

    Params:
        data: Input tensor of shape (N, H, W)
        labels: Target tensor of shape (N, H_out, W_out)
    Raises:
        ValueError: If data and labels have mismatched lengths.
    """
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        if len(data) != len(labels):
            raise ValueError("Data & labels must have the same # of samples")
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns:
            length: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample.

        Params:
            index: Index of the sample.
        Returns:
            matrices: (input_matrix, target_matrix)
        """
        return self.data[index], self.labels[index]


def load_data(
        hr_path: str | None = "data/hr_train.csv",
        lr_path: str | None = "data/lr_train.csv",
        hr_dim: int = 268, # Defaults are set to match the provided CSVs
        lr_dim: int = 160
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load and reconstruct low-resolution and high-resolution matrices
        from CSV vectorized format

    Returns:
        training_data:
            lr_train: Array of shape (N, lr_dim, lr_dim)
            hr_train: Array of shape (N, hr_dim, hr_dim)
    """
    lr_data = None
    hr_data = None

    if lr_path is not None:
        df_lr = pd.read_csv(lr_path)
        lr_vec = df_lr.to_numpy()
        lr_data = _reconstruct_matrices(lr_vec, lr_dim)
    
    if hr_path is not None:
        df_hr = pd.read_csv(hr_path)
        hr_vec = df_hr.to_numpy()
        hr_data = _reconstruct_matrices(hr_vec, hr_dim)

    return lr_data, hr_data


def _reconstruct_matrices(
    vectors: np.ndarray,
    dim: int,
) -> np.ndarray:
    """
    Reconstruct adjacency matrices from vectorized representation

    Params:
        vectors: Array of shape (N, num_features)
        dim: Target matrix dimension
    Returns:
        matrices: Array of shape (N, dim, dim)
    """
    n_samples = vectors.shape[0]
    matrices = np.zeros((n_samples, dim, dim), dtype=np.float32)

    for i in range(n_samples):
        matrices[i] = MatrixVectorizer.anti_vectorize(
            vectors[i],
            dim,
            False,
        )

    return matrices

def load_checkpoint(input_path: str) -> Any:
    """
    Loads an object from a NumPy .npy file
    """
    # We must set allow_pickle=True to load lists of dicts
    data = np.load(input_path, allow_pickle=True)
    
    # If it's a 0-d array, it's holding our object; [()] extracts it
    if data.ndim == 0:
        return data.item()

    return data
