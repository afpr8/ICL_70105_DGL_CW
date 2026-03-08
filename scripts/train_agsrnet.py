# The main entry point to train AGSRNet and generate predictions and save them
# to submission.csv for evaluation

from src.datasets import load_data
from src.models.agsrnet.config import AGSRArgs
from src.models.agsrnet.model import AGSRNet
from src.models.agsrnet.training import (
    train_agsr,
    train_fold_agsr,
    train_full_and_predict
)
from src.training.train import run_3_fold_cross_validation
from src.utils.core_utils import set_seed
from src.utils.submission_utils import generate_submission

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)

    lr_train, hr_train = load_data()

    model_args = AGSRArgs()

    run_3_fold_cross_validation(
        train_fold_agsr,
        AGSRNet,
        model_args,
        lr_train,
        hr_train,
        random_state=SEED,
        get_final_metrics=True,
        use_checkpoint=True,
        start_fold_idx=0
    )

    # --- Generate test submission ---
    print("\n===== Generating Test Submission =====")
    # _ should be None since hr_path is None
    lr_test, _ = load_data(hr_path=None, lr_path="data/lr_test.csv")

    hr_predictions = train_full_and_predict(
        lr_train,
        hr_train,
        lr_test,
        AGSRNet,
        model_args,
        train_agsr
    )

    generate_submission(hr_predictions, output_path="./results/submission.csv")
