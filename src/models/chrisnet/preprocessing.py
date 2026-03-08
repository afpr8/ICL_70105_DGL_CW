# Data loading and preprocessing utilities for ChrisNet (NeuroSRGAN)

# Standard library imports
import os

# Third party imports
import community as community_louvain  # python-louvain
import networkx as nx
import numpy as np
import scipy.io
import torch

# Local imports
from src.datasets import BrainDataset
from src.models.chrisnet.config import ChrisNetArgs
from src.utils.core_utils import get_device

DEVICE, PIN_MEMORY = get_device()


def pad_HR_adj(label: np.ndarray, padding: int) -> np.ndarray:
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


def unpad(data: np.ndarray, padding: int) -> np.ndarray:
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
    roi = np.abs(roi).astype(np.float32)

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


def compute_community_masks(
        lr_adj: np.ndarray,
        args: ChrisNetArgs
    ) -> list[torch.Tensor]:
    """
    Compute binary HR community masks from a LR adjacency matrix

    Steps:
      1. Threshold the LR graph to the top-{threshold_pct} percentile edges
      2. Run Louvain community detection to find K_communities communities
      3. Map LR community assignments to HR nodes via
             hr_node -> LR node round(hr_node * (lr_dim-1) / (center_dim-1))
      4. Build outer-product binary masks v_k @ v_k.T of shape (center_dim, center_dim)
      5. Zero-pad each mask from (center_dim, center_dim) to (hr_dim, hr_dim)

    Params:
        lr_adj: LR adjacency matrix of shape (lr_dim, lr_dim)
        args: ChrisNetArgs containing K_communities, threshold_pct, hr_dim, padding
    Returns:
        List of K_communities binary mask tensors, each of shape (hr_dim, hr_dim),
            on DEVICE
    """
    lr_dim = lr_adj.shape[0]
    center_dim = args.hr_dim - 2 * args.padding   # e.g. 268
    k = args.K_communities
    padding = args.padding

    # Threshold to top-pct percentile edges
    threshold = np.percentile(lr_adj, args.threshold_pct)
    thresholded = (lr_adj >= threshold).astype(float)
    np.fill_diagonal(thresholded, 0)

    G_lr = nx.from_numpy_array(thresholded)
    partition = community_louvain.best_partition(G_lr, random_state=42)

    # Relabel communities so the K largest are 0..K-1; merge the rest into K-1
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    top_k_comms = sorted(community_sizes, key=community_sizes.get, reverse=True)[:k]
    comm_remap = {c: i for i, c in enumerate(top_k_comms)}

    lr_labels = np.array([
        comm_remap.get(partition[n], k - 1) for n in range(lr_dim)
    ])

    # Map LR community labels to HR nodes
    hr_labels = np.array([
        lr_labels[round(i * (lr_dim - 1) / (center_dim - 1))]
        for i in range(center_dim)
    ])

    # Build outer-product masks and zero-pad to hr_dim
    masks = []
    for community_id in range(k):
        v = (hr_labels == community_id).astype(np.float32)          # (center_dim,)
        mask_core = np.outer(v, v)                                   # (center_dim, center_dim)
        mask_padded = np.pad(
            mask_core,
            ((padding, padding), (padding, padding)),
            mode='constant'
        )                                                             # (hr_dim, hr_dim)
        masks.append(
            torch.tensor(mask_padded, dtype=torch.float32, device=DEVICE)
        )

    return masks


def prepare_tensors(
        lr_np: np.ndarray | torch.Tensor,
        hr_np: np.ndarray | torch.Tensor,
        args: ChrisNetArgs
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert numpy LR and HR arrays to PyTorch tensors on the correct device
        (if not already) and pad the HR tensor

    Params:
        lr_np: Low-resolution input array of shape (lr_dim, lr_dim)
        hr_np: High-resolution target array of shape (center_dim, center_dim)
        args: ChrisNetArgs containing padding and device info
    Returns:
        lr_t: LR tensor, shape (lr_dim, lr_dim), float32, on DEVICE
        padded_hr: Padded HR tensor, shape (hr_dim, hr_dim), float32, on DEVICE
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


def prepare_chrisnet_inputs(
        dataset: BrainDataset,
        args: ChrisNetArgs
    ) -> tuple[BrainDataset, list[list[torch.Tensor]]]:
    """
    Prepare a BrainDataset for ChrisNet inference, including community masks

    Params:
        dataset: Dataset containing LR-HR adjacency matrix pairs
        args: ChrisNetArgs containing lr_dim, hr_dim, padding, and community params
    Returns:
        prepared_dataset: BrainDataset with tensors prepared for ChrisNet inference
        all_masks: List of per-sample community mask lists, each of length K_communities
    """
    prepared_data = []
    all_masks = []

    for lr_np, hr_np in dataset:
        if isinstance(lr_np, torch.Tensor):
            lr_arr = lr_np.squeeze(0).numpy()
        else:
            lr_arr = lr_np.squeeze(0)

        lr_t, padded_hr = prepare_tensors(lr_arr, hr_np, args)
        masks = compute_community_masks(lr_arr, args)

        prepared_data.append((lr_t.cpu(), padded_hr.cpu()))
        all_masks.append(masks)

    inputs, targets = zip(*prepared_data)

    return BrainDataset(torch.stack(inputs), torch.stack(targets)), list(all_masks)
