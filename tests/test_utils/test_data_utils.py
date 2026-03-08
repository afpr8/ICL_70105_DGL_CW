import pandas as pd

from src.utils.data_utils import check_for_nan, check_for_negatives


def test_check_for_negatives_no_negatives(capsys):
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [0, 5, 10]
    })
    check_for_negatives(df)
    captured = capsys.readouterr()
    assert "Dataset has no negatives" in captured.out


def test_check_for_negatives_with_negatives(capsys):
    df = pd.DataFrame({
        "a": [-1, 2, 3],
        "b": [0, 5, 10]
    })
    check_for_negatives(df)
    captured = capsys.readouterr()
    assert "Dataset contains negative values" in captured.out


def test_check_for_nan_no_nan(capsys):
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [0, 5, 10]
    })
    check_for_nan(df)
    captured = capsys.readouterr()
    assert "Dataset has no NaN values" in captured.out


def test_check_for_nan_with_nan(capsys):
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [0, 5, 10]
    })
    check_for_nan(df)
    captured = capsys.readouterr()
    assert "Dataset contains NaN values" in captured.out
