# Utilities for preparing data

import pandas as pd


def check_for_negatives(df: pd.DataFrame) -> None:
    """
    This function checks if any column contains negative values

    Params:
        df: Input DataFrame to inspect
    Returns:
        None
    """
    has_negatives = (df < 0).any().any()

    if not has_negatives:
        print("Dataset has no negatives")
    else:
        print("Dataset contains negative values")


def check_for_nan(df: pd.DataFrame) -> None:
    """
    This function checks if any column contains NaN values

    Params:
        df: Input DataFrame to inspect
    Returns:
        None
    """
    has_nan = df.isna().any().any()

    if not has_nan:
        print("Dataset has no NaN values")
    else:
        print("Dataset contains NaN values")
