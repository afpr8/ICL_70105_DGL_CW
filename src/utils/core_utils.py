# Generic utilities

import numpy as np
import random
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, & PyTorch to ensure reproducibility

    Params:
        seed: Random seed value to use.
    Returns:
        None
    """
    # Python, NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch (CPU & GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make PyTorch fully deterministic (PyTorch >= 1.8)
    torch.use_deterministic_algorithms(True)


def get_device(verbose: bool = True) -> torch.device:
    """
    Determine the best available PyTorch device.

    Params:
        verbose: If True, prints device information.
    Returns:
        device: torch.device object ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        # For Apple Silicon
        device = torch.device("mps")
        pin_memory = False
        if verbose:
            print("Using Apple Silicon MPS backend")
    else:
        device = torch.device("cpu")
        pin_memory = False
        if verbose:
            print("Using CPU")

    return device, pin_memory
