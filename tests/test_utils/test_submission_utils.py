import numpy as np
import pandas as pd
import pytest

# Local imports
from src.utils.submission_utils import generate_submission

# -----------------------------
# Helpers
# -----------------------------

def make_dummy_hr(n_samples=3, n_roi=4):
    """Generate symmetric matrices with zero diagonal."""
    mats = np.random.rand(n_samples, n_roi, n_roi)
    for i in range(n_samples):
        mats[i] = (mats[i] + mats[i].T) / 2
        np.fill_diagonal(mats[i], 0)
    return mats


# -----------------------------
# Tests
# -----------------------------

def test_generate_submission_creates_csv(tmp_path):
    """Basic smoke test that CSV is written."""
    hr = make_dummy_hr(n_samples=2, n_roi=4)

    output_file = tmp_path / "submission.csv"

    generate_submission(hr, output_file)

    assert output_file.exists()


def test_generate_submission_csv_structure(tmp_path):
    """Ensure CSV has correct columns and length."""
    n_samples = 2
    n_roi = 4
    hr = make_dummy_hr(n_samples=n_samples, n_roi=n_roi)

    output_file = tmp_path / "submission.csv"

    generate_submission(hr, output_file)

    df = pd.read_csv(output_file)

    vec_len = n_roi * (n_roi - 1) // 2
    expected_rows = n_samples * vec_len

    assert list(df.columns) == ["ID", "Predicted"]
    assert len(df) == expected_rows
    assert df["ID"].iloc[0] == 1
    assert df["ID"].iloc[-1] == expected_rows


def test_generate_submission_values_not_nan(tmp_path):
    """Check predictions contain numeric values."""
    hr = make_dummy_hr(n_samples=2, n_roi=4)

    output_file = tmp_path / "submission.csv"

    generate_submission(hr, output_file)

    df = pd.read_csv(output_file)

    assert df["Predicted"].notnull().all()


def test_generate_submission_invalid_shape_raises(tmp_path):
    """Passing non-square matrices should fail."""
    bad_hr = np.random.rand(2, 4, 5)

    output_file = tmp_path / "submission.csv"

    with pytest.raises(Exception):
        generate_submission(bad_hr, output_file)
