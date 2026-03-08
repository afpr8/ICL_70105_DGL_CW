import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets import BrainDataset, load_data, _reconstruct_matrices
from src.matrix_vectorizer import MatrixVectorizer


def test_brain_dataset_length_and_getitem():
    data = torch.randn(5, 160, 160)
    labels = torch.randn(5, 268, 268)
    dataset = BrainDataset(data, labels)

    # __len__
    assert len(dataset) == 5

    # __getitem__
    sample_data, sample_label = dataset[2]
    assert sample_data.shape == (160, 160)
    assert sample_label.shape == (268, 268)


def test_brain_dataset_mismatched_lengths_raises():
    data = torch.randn(5, 160, 160)
    labels = torch.randn(4, 268, 268)
    with pytest.raises(ValueError):
        BrainDataset(data, labels)


def test_brain_dataset_invalid_type_raises():
    data = np.random.randn(5, 160, 160)  # not a tensor
    labels = torch.randn(5, 268, 268)
    with pytest.raises(TypeError):
        BrainDataset(data, labels)


def test_reconstruct_matrices_calls_anti_vectorize(monkeypatch):
    vectors = np.random.randn(2, 10)
    dim = 3

    called = []

    def fake_anti_vectorize(vec, dim_in, flag):
        called.append((vec.shape[0], dim_in, flag))
        return np.ones((dim, dim), dtype=np.float32)

    monkeypatch.setattr(MatrixVectorizer, "anti_vectorize", fake_anti_vectorize)

    matrices = _reconstruct_matrices(vectors, dim)
    assert matrices.shape == (2, dim, dim)
    assert len(called) == 2  # called for each sample


def test_load_data_reads_csv(tmp_path, monkeypatch):
    # create dummy CSVs
    hr_csv = tmp_path / "hr.csv"
    lr_csv = tmp_path / "lr.csv"

    pd.DataFrame(np.random.randn(2, 4)).to_csv(hr_csv, index=False)
    pd.DataFrame(np.random.randn(2, 4)).to_csv(lr_csv, index=False)

    called_vectors = []

    def fake_anti_vectorize(vec, dim, flag):
        called_vectors.append(vec.shape[0])
        return np.ones((dim, dim), dtype=np.float32)

    monkeypatch.setattr(MatrixVectorizer, "anti_vectorize", fake_anti_vectorize)

    lr_train, hr_train = load_data(
        hr_path=str(hr_csv),
        lr_path=str(lr_csv),
        hr_dim=3,
        lr_dim=2
    )

    assert lr_train.shape == (2, 2, 2)
    assert hr_train.shape == (2, 3, 3)
    assert called_vectors == [4, 4, 4, 4]  # each CSV row called once
