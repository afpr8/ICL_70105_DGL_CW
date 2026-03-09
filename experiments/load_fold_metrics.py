# Adhoc script to load fold_metrics for the final report

# Standard library imports
from pathlib import Path

# Third party imports
import numpy as np

# Local imports
from src.datasets import load_checkpoint

def print_fold_metrics(metrics_cache_path: str) -> None:
    """
        Load npy file and print mean of dict values across list entries

        Params:
            metrics_cache_path: The filepath from the cd
        Returns:
            None
    """
    fold_metrics_list = load_checkpoint(str(metrics_cache_path)).tolist()

    for key in fold_metrics_list[0].keys():
        values = np.array([m[key] for m in fold_metrics_list])
        mean_val, std_val = np.mean(values), np.std(values)
        print(f"CV {key}: {mean_val:.5f} ± {std_val:.5f}")

if __name__ == "__main__":
    metrics_cache_path = Path("./checkpoint/neurosrgan_community_fold_metrics_list.npy")

    print_fold_metrics(metrics_cache_path)
