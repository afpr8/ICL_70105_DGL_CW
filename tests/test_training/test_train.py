# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# from src.datasets import BrainDataset
# from src.models.dummy_model import DummyModel
# from src.training.train import (
#     run_3_fold_cross_validation,
#     train_fold,
#     train_model,
#     _train_step
# )
# from src.utils.core_utils import get_device
# from src.utils.metrics import get_metrics
# from src.utils.model_args import BaseModelArgs

# DEVICE, PIN_MEMORY = get_device()

# # ------------------------
# # Helpers
# # ------------------------

# def make_dummy_data(n_samples=6, lr_dim=160, hr_dim=268):
#     """Generate small dummy LR/HR datasets for testing."""
#     lr = torch.rand((n_samples, lr_dim, lr_dim))
#     hr = torch.rand((n_samples, hr_dim, hr_dim))
#     return lr, hr


# def make_dummy_model_args(
#     lr=1e-3,
#     batch_size=1,
#     weight_decay=0,
#     epochs=1,
#     padding=0
# ):
#     """Create a minimal BaseModelArgs for testing."""
#     return BaseModelArgs(
#         lr=lr,
#         batch_size=batch_size,
#         weight_decay=weight_decay,
#         epochs=epochs,
#         padding=padding
#     )

# # ------------------------
# # train_fold tests
# # ------------------------

# def test_train_fold_runs_and_returns_correct_type():
#     lr, hr = make_dummy_data(n_samples=4, lr_dim=16, hr_dim=16)
#     train_dataset = BrainDataset(lr, hr)
#     val_dataset = BrainDataset(lr, hr)
#     model = DummyModel(target_size=16).to(DEVICE)
#     model_args = make_dummy_model_args()

#     val_mats = train_fold(
#         model,
#         train_dataset,
#         model_args,
#         val_dataset,
#         fold_id=0,
#         log_to_mlflow=False,
#         loss_fn=torch.nn.L1Loss()
#     )

#     # Validate return type
#     assert isinstance(val_mats, list)
#     assert all(isinstance(tup, tuple) and len(tup) == 2 for tup in val_mats)
#     # Validate shapes
#     for pred, gt in val_mats:
#         assert pred.shape == gt.shape

# # ------------------------
# # run_3_fold_cross_validation tests
# # ------------------------

# def test_run_3_fold_cross_validation_runs(monkeypatch):
#     """Smoke test: ensure the 3-fold CV loop executes without error."""
#     lr, hr = make_dummy_data(n_samples=6, lr_dim=16, hr_dim=16)
#     model_args = make_dummy_model_args()

#     # Monkeypatch train_fold to avoid real training
#     from src.training import train as train_module
#     def fake_train_fold(
#             model,
#             train_dataset,
#             model_args,
#             val_dataset,
#             fold_id,
#             log_to_mlflow=True,
#             loss_fn=None
#         ):
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=model_args.batch_size,
#             pin_memory=PIN_MEMORY
#         )
#         return [(x, y) for x, y in val_loader]  # just return input/output
    
#     monkeypatch.setattr(train_module, "train_fold", fake_train_fold)

#     # Run CV with DummyModel
#     run_3_fold_cross_validation(
#         fake_train_fold,
#         DummyModel,
#         model_args,
#         lr.numpy(),
#         hr.numpy(),
#         random_state=42,
#         get_final_metrics=False,
#         log_to_mlflow=False,
#         loss_fn=None
#     )

#     # If it reaches here, the test passes
#     assert True

# # ------------------------
# # Integration test with final_metrics
# # ------------------------

# def test_train_fold_with_final_metrics_centralities():
#     """
#     Check that train_fold can run with precomputed centralities (no crash)
#     """
#     lr, hr = make_dummy_data(n_samples=4, lr_dim=16, hr_dim=16)
#     train_dataset = BrainDataset(lr, hr)
#     val_dataset = BrainDataset(lr, hr)
#     model = DummyModel(target_size=16).to(DEVICE)
#     model_args = make_dummy_model_args()

#     val_mats = train_fold(
#         model,
#         train_dataset,
#         model_args,
#         val_dataset,
#         fold_id=0,
#         log_to_mlflow=False,
#         loss_fn=torch.nn.L1Loss()
#     )

#     # Precompute dummy centralities
#     dummy_centralities = [
#         {
#             "bc": [0]*16,
#             "ec": [0]*16,
#             "pc": [0]*16,
#             "glob_eff": 0,
#             "modularity": 0
#         } for _ in range(len(hr))
#     ]

#     # Compute metrics with centralities
#     metrics_res = get_metrics(
#         val_mats,
#         gt_centralities_cache=dummy_centralities,
#         final_metrics=True
#     )
#     assert isinstance(metrics_res, dict)
#     for k, v in metrics_res.items():
#         assert isinstance(v, (float, np.floating))


# def test_train_step_basic():
#     """Ensure _train_step runs and returns correct types and shapes."""
#     x = torch.rand((2, 16, 16), device=DEVICE)
#     y = torch.rand((2, 16, 16), device=DEVICE)
#     model = DummyModel(target_size=16).to(DEVICE)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#     loss_fn = torch.nn.MSELoss()

#     pred, loss_val = _train_step(model, x, y, optimizer, loss_fn)

#     assert isinstance(pred, torch.Tensor)
#     assert pred.shape == y.shape
#     assert isinstance(loss_val, float)


# def test_train_model_runs_with_and_without_validation():
#     """Smoke test for train_model with optional validation."""
#     lr, hr = make_dummy_data(n_samples=4, lr_dim=16, hr_dim=16)
#     train_dataset = BrainDataset(lr, hr)
#     val_dataset = BrainDataset(lr, hr)
#     model_args = make_dummy_model_args(epochs=2)  # run 2 epochs for test

#     # Model without validation
#     model = DummyModel(target_size=16)
#     trained_model = train_model(
#         model,
#         train_dataset,
#         model_args,
#         val_dataset=None,
#         fold_id=0,
#         log_to_mlflow=False,
#         loss_fn=torch.nn.MSELoss()
#     )
#     assert isinstance(trained_model, DummyModel)

#     # Model with validation
#     model2 = DummyModel(target_size=16)
#     trained_model2 = train_model(
#         model2,
#         train_dataset,
#         model_args,
#         val_dataset=val_dataset,
#         fold_id=0,
#         log_to_mlflow=False,
#         loss_fn=torch.nn.MSELoss()
#     )
#     assert isinstance(trained_model2, DummyModel)
