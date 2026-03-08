import pytest
import torch
from torch.utils.data import Dataset

# Local imports
from src.utils.core_utils import get_device
from src.training.predict import get_prediction, predict

DEVICE, PIN_MEMORY = get_device()


class DummyDataset(Dataset):
    def __init__(self, n_samples=3, H=4, W=4):
        self.data = [torch.ones((H, W)) * i for i in range(n_samples)]
        self.targets = [torch.ones((H, W)) * i for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class DummyModel(torch.nn.Module):
    """Returns input unchanged, shape preserved"""
    def forward(self, x):
        return x

class DummyModelTuple(torch.nn.Module):
    """Returns tuple (prediction, extra)"""
    def forward(self, x):
        return (x, x*2)

class DummyModelObj(torch.nn.Module):
    """Returns object with .prediction attribute"""
    class Output:
        def __init__(self, x):
            self.prediction = x
    def forward(self, x):
        return self.Output(x)

# -------------------
# Tests for get_prediction
# -------------------

def test_get_prediction_tensor():
    x = torch.tensor([1.0])
    assert torch.equal(get_prediction(x), x)


def test_get_prediction_tuple_list():
    x = (torch.tensor([2.0]), "extra")
    y = [torch.tensor([3.0]), "extra"]
    assert torch.equal(get_prediction(x), torch.tensor([2.0]))
    assert torch.equal(get_prediction(y), torch.tensor([3.0]))


def test_get_prediction_object():
    class Obj:
        def __init__(self):
            self.prediction = torch.tensor([5.0])
    obj = Obj()
    assert torch.equal(get_prediction(obj), torch.tensor([5.0]))


def test_get_prediction_unsupported():
    with pytest.raises(ValueError):
        get_prediction("unsupported")

# -------------------
# Tests for predict
# -------------------

def test_predict_basic():
    dataset = DummyDataset(n_samples=3, H=4, W=4)
    model = DummyModel().to(DEVICE)
    preds = predict(model, dataset, batch_size=2)

    # All outputs should match inputs
    for i, (pred, gt) in enumerate(preds):
        assert torch.allclose(pred, gt)
        assert pred.device.type == DEVICE.type
