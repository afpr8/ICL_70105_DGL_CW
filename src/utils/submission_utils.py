# Utilities for submitting model results

# Standard imports 
from pathlib import Path
from typing import Any

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from src.matrix_vectorizer import MatrixVectorizer


def generate_submission(hr_predictions, output_path="./submission.csv"):
    """
    hr_predictions: numpy array of shape (n_samples, 268, 268)
    Vectorizes each matrix, flattens all, and writes submission CSV.

    Params:
        hr_predictions: numpy array of shape (n_samples, n_roi, n_roi)
        output_path: path to save the CSV
    Returns:
        None
    """
    hr_predictions = np.asarray(hr_predictions) # Make sure it is an array
    n_samples, n_roi, n_roi2 = hr_predictions.shape
    if n_roi != n_roi2:
        raise ValueError("Matrices must be square")
    vec_len = n_roi * (n_roi - 1) // 2  # 35778

    pred_vecs = np.stack(
        [MatrixVectorizer.vectorize(mat) for mat in hr_predictions]
    )

    # Flatten to 1D
    flattened = pred_vecs.ravel()
    expected_len = n_samples * vec_len
    if len(flattened) != expected_len:
        raise ValueError(f"Flattened array length {len(flattened)} != expected {expected_len}")
    print(f"Submission array length: {len(flattened)}")

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "ID": np.arange(1, len(flattened) + 1),
        "Predicted": flattened,
    })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file.resolve()}")


def plot_cv_metrics(fold_metrics_list, extra_metrics=None):
    """
    Produces the required 2x2 grid of bar plots for 3-fold CV metrics.
    """
    # Mapping your dict keys to the display labels for the x-axis
    metric_mapping = {
        "mae": "MAE",
        "pcc": "PCC",
        "js_distance": "JSD",
        "mae_pc": "MAE (PC)",
        "mae_ec": "MAE (EC)",
        "mae_bc": "MAE (BC)"
    }

    # Add your 2 extra measures if they exist in the dict
    if extra_metrics:
        for m in extra_metrics:
            metric_mapping[m] = m.upper()

    display_labels = list(metric_mapping.values())
    internal_keys = list(metric_mapping.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    colors = ['#f07167', '#81b271', '#6a67f0', '#f6c878', '#98f5ff', '#90ee90', '#c18fde', '#de8f8f']

    # 1. Plot Folds 1, 2, and 3
    for i in range(3):
        ax = axes[i]
        values = [fold_metrics_list[i].get(k, 0) for k in internal_keys]
        ax.bar(display_labels, values, color=colors[:len(display_labels)])
        ax.set_title(f"Fold {i+1}")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Plot Average Across Folds with Standard Deviation
    ax_avg = axes[3]
    all_fold_values = np.array([[m.get(k, 0) for k in internal_keys] for m in fold_metrics_list])

    means = np.mean(all_fold_values, axis=0)
    stds = np.std(all_fold_values, axis=0)

    ax_avg.bar(display_labels, means, yerr=stds, capsize=5, color=colors[:len(display_labels)], alpha=0.8)
    ax_avg.set_title("Avg. Across Folds")
    ax_avg.tick_params(axis='x', rotation=45)
    ax_avg.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("results/cv_metrics_report.png")

def save_checkpoint(data_to_save: Any, output_path: str) -> None:
    """
    Saves any object to disk using NumPy's binary format
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists
    
    np.save(output_file, data_to_save, allow_pickle=True)
    print(f"Checkpoint saved to {output_file.resolve()}")
