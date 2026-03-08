import numpy as np
import torch

# Local imports
from src.datasets import BrainDataset
from src.models.agsrnet.config import AGSRArgs
from src.models.agsrnet.training import (
    predict_from_arrays,
    train_full_and_predict,
    _train_agsr_step,
    prepare_agsr_inputs,
    train_fold_agsr
)
from src.models.agsrnet.model import Discriminator
from src.utils.core_utils import get_device
from src.utils.metrics import compute_metrics

DEVICE, PIN_MEMORY = get_device()

class DummyModel(torch.nn.Module):
    """Fake AGSRNet for testing that matches required attributes."""
    def __init__(self, hr_dim):
        super().__init__()
        self.hr_dim = hr_dim
        self.param = torch.nn.Parameter(torch.ones(1))
        # Create a dummy 'layer' object with a 'weights' attribute
        class DummyLayer:
            def __init__(self, hr_dim):
                self.weights = torch.ones(hr_dim, hr_dim)
        self.layer = DummyLayer(hr_dim)

    def forward(self, x):
        # Return prediction, net_outs, start_gcn_outs, _ (dummy)
        pred = torch.ones(self.hr_dim, self.hr_dim, device=x.device)
        return pred, pred, pred, None


def make_dummy_dataset(n=2, lr_dim=4, hr_dim=8, padding=2):
    center_dim = hr_dim - 2 * padding
    lr = torch.rand(n, lr_dim, lr_dim)
    hr = torch.rand(n, center_dim, center_dim)
    return BrainDataset(lr, hr)


def test_compute_metrics_runs():
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)

    dataset = make_dummy_dataset(
        n=2,
        lr_dim=args.lr_dim,
        hr_dim=args.hr_dim,
        padding=args.padding
    )

    model = DummyModel(args.hr_dim)

    metrics = compute_metrics(model, dataset, args)

    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert all(isinstance(v, (float, np.floating)) for v in metrics.values())


def test_predict_from_arrays_shape():
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)

    model = DummyModel(args.hr_dim)

    lr_arrays = np.random.rand(3, args.lr_dim, args.lr_dim).astype(np.float32)

    preds = predict_from_arrays(model, lr_arrays, args)

    center_dim = args.hr_dim - 2 * args.padding

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3, center_dim, center_dim)


def test_predict_from_arrays_values():
    """Ensure cropping works correctly."""
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)

    class OnesModel(torch.nn.Module):
        def __init__(self, hr_dim):
            super().__init__()
            self.hr_dim = hr_dim

        def forward(self, x):
            return torch.ones(self.hr_dim, self.hr_dim), None, None, None

    model = OnesModel(args.hr_dim)

    lr_arrays = np.random.rand(2, args.lr_dim, args.lr_dim).astype(np.float32)

    preds = predict_from_arrays(model, lr_arrays, args)

    assert np.allclose(preds, 1.0)


# ---- mock training ----
def fake_train(
    model,
    train_dataset,
    model_args,
    val_dataset=None,
    fold_id=None,
    log_to_mlflow=False,
    loss_fn=None
):
    return model


def test_train_full_and_predict(monkeypatch):
    """Test full pipeline without real training."""

    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)

    lr_train = np.random.rand(2, args.lr_dim, args.lr_dim).astype(np.float32)
    hr_train = np.random.rand(
        2,
        args.hr_dim - 2 * args.padding,
        args.hr_dim - 2 * args.padding
    ).astype(np.float32)

    lr_test = np.random.rand(3, args.lr_dim, args.lr_dim).astype(np.float32)

    monkeypatch.setattr(
        "src.models.agsrnet.training.train_agsr",
        fake_train
    )
    class Dummy(torch.nn.Module):
        def __init__(self, hr_dim):
            super().__init__()
            self.hr_dim = hr_dim
        def forward(self, x):
            return torch.ones(self.hr_dim, self.hr_dim), None, None, None
    monkeypatch.setattr(
        "src.models.agsrnet.training.AGSRNet",
        lambda args: Dummy(args.hr_dim)
    )

    preds = train_full_and_predict(lr_train, hr_train, lr_test, args)
    center_dim = args.hr_dim - 2 * args.padding
    assert preds.shape == (3, center_dim, center_dim)
    assert np.allclose(preds, 1.0)


def test_train_agsr_step_returns_float():
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)
    model = DummyModel(args.hr_dim).to(DEVICE)
    netD = Discriminator(args).to(DEVICE)
    lr_t = torch.rand(args.lr_dim, args.lr_dim)
    padded_hr = torch.rand(args.hr_dim, args.hr_dim)
    optimizerG = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.01)

    loss = _train_agsr_step(
        model,
        netD,
        lr_t,
        padded_hr,
        optimizerG,
        optimizerD,
        args
    )
    assert isinstance(loss, float)
    assert loss > 0


def test_prepare_agsr_inputs_returns_dataset():
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)
    dataset = make_dummy_dataset(
        n=2,
        lr_dim=args.lr_dim,
        hr_dim=args.hr_dim,
        padding=args.padding
    )
    prepared_dataset = prepare_agsr_inputs(dataset, args)

    assert isinstance(prepared_dataset, BrainDataset)
    for lr_t, hr_t in prepared_dataset:
        assert isinstance(lr_t, torch.Tensor)
        assert isinstance(hr_t, torch.Tensor)
        assert lr_t.shape == (args.lr_dim, args.lr_dim)
        assert hr_t.shape == (args.hr_dim, args.hr_dim)


def test_train_fold_agsr(monkeypatch):
    args = AGSRArgs(lr_dim=4, hr_dim=8, padding=2)
    train_dataset = make_dummy_dataset(
        n=2,
        lr_dim=args.lr_dim,
        hr_dim=args.hr_dim,
        padding=args.padding
    )
    val_dataset = make_dummy_dataset(
        n=1,
        lr_dim=args.lr_dim,
        hr_dim=args.hr_dim,
        padding=args.padding
    )

    # mock train_agsr
    monkeypatch.setattr("src.models.agsrnet.training.train_agsr", fake_train)
    # mock predict
    monkeypatch.setattr(
        "src.models.agsrnet.training.predict",
        lambda model, ds: [
            (
                torch.ones(args.hr_dim, args.hr_dim),
                torch.ones(args.hr_dim, args.hr_dim)
            ) for _ in range(len(ds))
            ]
    )
    # mock AGSRNet
    class Dummy(torch.nn.Module):
        def forward(self, x, lr_dim, hr_dim):
            return torch.ones(hr_dim, hr_dim), None, None, None
    monkeypatch.setattr(
        "src.models.agsrnet.training.AGSRNet",
        lambda ks, args: Dummy()
    )

    val_preds = train_fold_agsr(
        Dummy(),
        train_dataset,
        args,
        val_dataset=val_dataset,
        fold_id=0,
        log_to_mlflow=False
    )
    assert isinstance(val_preds, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in val_preds)
