# The main entry point to train ChrisNet and generate predictions and save them
# to submission.csv for evaluation

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from src.datasets import load_data
from src.models.chrisnet.config import ChrisNetArgs
from src.models.chrisnet.model import ChrisNet
from src.models.chrisnet.training import (
    train_chrisnet,
    train_fold_chrisnet,
    train_full_and_predict
)
from src.training.train import run_3_fold_cross_validation
from src.utils.core_utils import set_seed
from src.utils.submission_utils import generate_submission

if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)

    lr_train, hr_train = load_data()

    model_args = ChrisNetArgs()  # variant='full' by default

    run_3_fold_cross_validation(
        train_fold_chrisnet,
        ChrisNet,
        model_args,
        lr_train,
        hr_train,
        random_state=SEED,
        get_final_metrics=True,
        use_checkpoint=True,
        start_fold_idx=0
    )

    print("\n===== Generating Test Submission =====")
    lr_test, _ = load_data(hr_path=None, lr_path="data/lr_test.csv")

    hr_predictions = train_full_and_predict(
        lr_train,
        hr_train,
        lr_test,
        ChrisNet,
        model_args,
        train_chrisnet
    )

    generate_submission(hr_predictions, output_path="./results/submission.csv")
