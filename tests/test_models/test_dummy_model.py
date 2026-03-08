import pytest
import torch

from src.models.dummy_model import DummyModel
from src.utils.core_utils import get_device

DEVICE, PIN_MEMORY = get_device()


def test_dummy_model_output_shape():
    """
    Test that DummyModel outputs correct shape
    """
    model = DummyModel(target_size=16)
    x = torch.rand((2, 8, 8))  # batch_size=2, 8x8 matrices
    y = model(x)
    assert y.shape == (2, 16, 16)


@pytest.mark.parametrize("device", [torch.device("cpu"), DEVICE])
def test_dummy_model_device(device):
    """
    Test that DummyModel runs on CPU or DEVICE correctly
    """
    model = DummyModel(target_size=16).to(device)
    x = torch.rand((2, 8, 8), device=device)
    y = model(x)
    assert y.device.type == device.type
    # Optional: check index if you care
    if device.index is not None:
        assert y.device.index == device.index


def test_dummy_model_symmetry():
    """
    Test that DummyModel output is symmetric
    """
    model = DummyModel(target_size=8)
    x = torch.rand((1, 4, 4))
    y = model(x)
    # Check symmetry: y == y^T along last two dims
    diff = (y - y.transpose(1, 2)).abs().max()
    assert diff < 1e-6


def test_dummy_model_relu():
    """
    Test that DummyModel output is non-negative (ReLU applied)
    """
    model = DummyModel(target_size=8)
    x = torch.rand((1, 4, 4)) - 0.5  # include negative values
    y = model(x)
    assert (y >= 0).all()


def test_dummy_model_input_validation():
    """
    Test that DummyModel raises ValueError if input shape is wrong
    """
    model = DummyModel(target_size=8)
    x = torch.rand((4, 4))  # missing batch dim
    with pytest.raises(ValueError):
        model(x)


def test_dummy_model_batch_size_one():
    """
    Test that DummyModel works with batch_size=1
    """
    model = DummyModel(target_size=8)
    x = torch.rand((1, 5, 5))
    y = model(x)
    assert y.shape == (1, 8, 8)
    assert (y >= 0).all()
