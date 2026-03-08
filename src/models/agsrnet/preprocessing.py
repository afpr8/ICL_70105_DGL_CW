# Data loading and preprocessing utilities for AGSRNet

# Third party imports
import numpy as np
import os
import scipy.io
import torch

# Local imports
from src.datasets import BrainDataset
from src.models.agsrnet.config import AGSRArgs
from src.utils.core_utils import get_device
from src.utils.data_utils import prepare_tensors

DEVICE, PIN_MEMORY = get_device()


def normalize_adj_torch(mx: torch.Tensor) -> torch.Tensor:
    """
    Symmetric normalization of an adjacency matrix: D^(-1/2) A D^(-1/2)

    Params:
        mx: Adjacency matrix, shape (N, N)
    Returns:
        Normalized adjacency matrix
    """
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = r_mat_inv_sqrt @ mx @ r_mat_inv_sqrt

    return mx


def unpad(data: np.ndarray, padding: np.ndarray) -> np.ndarray:
    """
    Remove padding from a matrix

    Params:
        data: Padded matrix, shape (hr_dim, hr_dim)
        padding: Number of rows/columns that were padded on each side
    Returns:
        Unpadded matrix of shape (center_dim, center_dim)
    """
    return data[padding:data.shape[0]-padding, padding:data.shape[1]-padding]


def extract_data(
        subject: int,
        session_str: str,
        parcellation_str: str,
        subjects_roi: np.ndarray,
        data_root: str = "drive/My Drive/BRAIN_DATASET",
        roi_file: str = "ROI_FC.mat"
    ) -> np.ndarray:
    """
    Load and preprocess ROI data for a single subject/session/parcellation

    Params:
        subject: Subject ID
        session_str: Session folder name
        parcellation_str: Parcellation type, e.g., "shen_268"
        subjects_roi: Existing stacked ROI array to append to
        data_root: Root directory of dataset
        roi_file: Name of .mat file containing ROI
    Returns:
        Updated subjects_roi array
    """
    folder_path = os.path.join(
        data_root, str(subject), session_str, parcellation_str)
    roi_data = scipy.io.loadmat(os.path.join(folder_path, roi_file))
    roi = roi_data['r']

    # Replacing NaN values
    roi[np.isnan(roi)] = 1
    roi = np.abs(roi, dtype=np.float32)

    # Taking the absolute values of the matrix
    roi = np.absolute(roi, dtype=np.float32)

    if parcellation_str == 'shen_268':
        roi = np.reshape(roi, (1, 268, 268))
    else:
        roi = np.reshape(roi, (1, 160, 160))

    if subjects_roi.shape[0] == 1 and subjects_roi.sum() == 0:
        return roi

    return np.concatenate((subjects_roi, roi), axis=0)


def load_data(
        start_value: int,
        end_value: int,
        data_root: str = "drive/My Drive/BRAIN_DATASET"
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load adjacency and label matrices for a range of subjects

    Params:
        start_value: First subject ID
        end_value: Last subject ID (exclusive)
        data_root: Dataset root folder
    Returns:
        subjects_adj: LR adjacency matrices, shape (N, lr_dim, lr_dim)
        subjects_label: HR label matrices, shape (N, hr_dim, hr_dim)
    """

    subjects_label = np.zeros((1, 268, 268), dtype=np.float32)
    subjects_adj = np.zeros((1, 160, 160), dtype=np.float32)

    for subject in range(start_value, end_value):
        subject_path = os.path.join(data_root, str(subject))

        if 'session_1' in os.listdir(subject_path):
            subjects_label = extract_data(
                subject, 'session_1', 'shen_268', subjects_label, data_root
            )
            subjects_adj = extract_data(
                subject, 'session_1', 'Dosenbach_160', subjects_adj, data_root
            )

    return subjects_adj, subjects_label


def data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all training and test adjacency and label matrices

    Returns:
        subjects_adj: LR training arrays
        subjects_labels: HR training arrays
        test_adj: LR test arrays
        test_labels: HR test arrays
    """
    subjects_adj, subjects_labels = load_data(25629, 25830)
    test_adj_1, test_labels_1 = load_data(25831, 25863)
    test_adj_2, test_labels_2 = load_data(30701, 30757)

    test_adj = np.concatenate((test_adj_1, test_adj_2), axis=0)
    test_labels = np.concatenate((test_labels_1, test_labels_2), axis=0)

    return subjects_adj, subjects_labels, test_adj, test_labels


def prepare_agsr_inputs(
    dataset: BrainDataset,
    args: AGSRArgs
) -> BrainDataset:
    """
    Prepare a BrainDataset for AGSRNet inference

    Params:
        dataset: Dataset containing LR-HR adjacency matrix pairs
        args: Model configuration containing lr_dim, hr_dim, and padding

    Returns:
        BrainDataset: Dataset with tensors prepared for AGSRNet inference
    """
    prepared_data = []

    for lr_np, hr_np in dataset:
        lr_t, padded_hr = prepare_tensors(lr_np, hr_np, args)

        prepared_data.append((lr_t.squeeze(0), padded_hr.squeeze(0)))

    inputs, targets = zip(*prepared_data)    

    return BrainDataset(torch.stack(inputs), torch.stack(targets))
