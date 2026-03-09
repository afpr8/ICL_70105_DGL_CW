import numpy as np
import torch

from src.datasets import BrainDataset
from src.utils.model_args import BaseModelArgs
from src.models.NeuroSRGAN.preprocessing import prepare_model_inputs
from src.utils.core_utils import get_device
from src.utils.data_utils import pad_HR_adj, prepare_tensors

DEVICE, PIN_MEMORY = get_device()

# ----------------------------
# Helper function for dummy data
# ----------------------------

def make_dummy_arrays(lr_dim=4, hr_dim=8, padding=2):
    lr = np.random.rand(lr_dim, lr_dim).astype(np.float32)
    center_dim = hr_dim - 2 * padding
    hr = np.random.rand(center_dim, center_dim).astype(np.float32)
    return lr, hr

# ----------------------------
# Tests for prepare_tensors
# ----------------------------

def test_prepare_tensors_returns_correct_types_and_device():
    args = BaseModelArgs(lr_dim=4, hr_dim=8, padding=2)

    lr_np = np.random.rand(args.lr_dim, args.lr_dim).astype(np.float32)
    hr_np = np.random.rand(
        args.hr_dim - 2*args.padding,
        args.hr_dim - 2*args.padding
    ).astype(np.float32)

    lr_t, padded_hr = prepare_tensors(lr_np, hr_np, args)

    # Check types and device
    assert isinstance(lr_t, torch.Tensor)
    assert isinstance(padded_hr, torch.Tensor)
    assert lr_t.device.type == DEVICE.type
    assert padded_hr.device.type == DEVICE.type

    # Check shapes
    assert lr_t.shape == (args.lr_dim, args.lr_dim)
    assert padded_hr.shape == (args.hr_dim, args.hr_dim)

    # Check center off-diagonal matches HR
    center_hr = padded_hr[args.padding:args.padding + hr_np.shape[0],
                          args.padding:args.padding + hr_np.shape[1]]
    hr_tensor = torch.tensor(hr_np, dtype=torch.float32, device=DEVICE)
    # mask out diagonal
    mask = ~torch.eye(center_hr.shape[0], dtype=bool, device=DEVICE)
    assert torch.allclose(center_hr[mask], hr_tensor[mask])

    # Check all diagonal entries are 1
    diag = padded_hr.diagonal()
    assert torch.all(diag == 1)

# ----------------------------
# Tests for prepare_model_inputs
# ----------------------------

def test_prepare_model_inputs_returns_BrainDataset():
    args = BaseModelArgs(lr_dim=4, hr_dim=8, padding=2)

    # Create dummy BrainDataset with torch tensors
    dataset = []
    for _ in range(3):
        lr_np = np.random.rand(args.lr_dim, args.lr_dim).astype(np.float32)
        hr_np = np.random.rand(
            args.hr_dim - 2*args.padding,
            args.hr_dim - 2*args.padding
        ).astype(np.float32)
        dataset.append((torch.tensor(lr_np), torch.tensor(hr_np)))

    ds = BrainDataset(
        torch.stack([x[0] for x in dataset]),
        torch.stack([x[1] for x in dataset])
    )

    prepared_ds = prepare_model_inputs(ds, args)
    assert isinstance(prepared_ds, BrainDataset)
    assert len(prepared_ds) == len(ds)

    for i in range(len(prepared_ds)):
        lr_t, padded_hr = prepared_ds[i]

        # LR/HR shapes
        assert lr_t.shape == (args.lr_dim, args.lr_dim)
        assert padded_hr.shape == (args.hr_dim, args.hr_dim)

        # Center values (mask out diagonal)
        hr_np = ds[i][1].numpy()
        center_hr = padded_hr[
            args.padding:args.padding + hr_np.shape[0],
            args.padding:args.padding + hr_np.shape[1]
        ]

        # Create a boolean mask to exclude diagonal
        mask = ~torch.eye(center_hr.shape[0], dtype=bool, device=DEVICE)

        # Compare only off-diagonal entries
        assert torch.allclose(center_hr[mask], torch.tensor(
            hr_np,
            dtype=torch.float32,
            device=DEVICE)[mask]
        )

        # Check diagonal is all ones
        diag = padded_hr.diagonal()
        assert torch.all(diag == 1)

# ----------------------------
# Test padding function independently
# ----------------------------

def test_pad_HR_adj_padding_and_diagonal():
    hr = np.ones((4, 4), dtype=np.float32)
    padding = 2
    padded = pad_HR_adj(hr, padding)

    # Check shape
    expected_shape = (hr.shape[0] + 2*padding, hr.shape[1] + 2*padding)
    assert padded.shape == expected_shape

    # Check center matches original HR
    center_hr = padded[padding:padding + hr.shape[0],
                       padding:padding + hr.shape[1]]
    assert np.all(center_hr == hr)

    # Check that all diagonal entries are 1
    assert np.all(np.diag(padded) == 1)

    # Check that non-diagonal padded area is zeros
    padded_mask = np.ones_like(padded, dtype=bool)
    padded_mask[
        padding:padding + hr.shape[0], padding:padding + hr.shape[1]
    ] = False
    np.fill_diagonal(padded_mask, False)  # exclude diagonal
    assert np.all(padded[padded_mask] == 0)
