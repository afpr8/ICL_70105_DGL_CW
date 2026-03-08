import numpy as np
import pytest
import random
import torch

from src.utils.core_utils import get_device, set_seed


def test_set_seed_reproducibility():
    """
    Test that set_seed makes random, numpy, and torch reproducible
    """
    seed = 42
    set_seed(seed)

    # Test Python random
    rand_val_1 = random.randint(0, 100)
    set_seed(seed)
    rand_val_2 = random.randint(0, 100)
    assert rand_val_1 == rand_val_2, "Python random not reproducible"

    # Test NumPy random
    np_val_1 = np.random.rand()
    set_seed(seed)
    np_val_2 = np.random.rand()
    assert np.isclose(np_val_1, np_val_2), "NumPy random not reproducible"

    # Test PyTorch CPU random
    torch_val_1 = torch.rand(1).item()
    set_seed(seed)
    torch_val_2 = torch.rand(1).item()
    assert np.isclose(torch_val_1, torch_val_2), \
        "PyTorch CPU random not reproducible"

    # Test PyTorch GPU random (if available)
    if torch.cuda.is_available():
        torch_val_1_gpu = torch.rand(1, device="cuda").item()
        set_seed(seed)
        torch_val_2_gpu = torch.rand(1, device="cuda").item()
        assert np.isclose(torch_val_1_gpu, torch_val_2_gpu), \
            "PyTorch CUDA random not reproducible"


@pytest.mark.parametrize("verbose", [True, False])
def test_get_device(verbose):
    """
    Test that get_device returns a valid torch.device.
    """
    device, pin_memory = get_device(verbose=verbose)
    assert isinstance(device, torch.device)
    # Should be one of the allowed devices
    assert device.type in ("cuda", "cpu", "mps")
    assert isinstance(pin_memory, bool)
