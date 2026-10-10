import numpy as np
import pandas as pd


def create_features(df):
    """
    Create the same engineered features used during model training.
    """

    df = df.copy()

    # Age
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # Handle DAYS_EMPLOYED anomaly
    df["EMPLOYED_ANOMALY"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    df.loc[
        df["DAYS_EMPLOYED"] == 365243,
        "DAYS_EMPLOYED"
    ] = np.nan

    # Employment years
    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED"] / 365
    )

    # Credit / income ratio
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"]
    )

    # Annuity / income ratio
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"]
    )

    # Credit / goods price ratio
    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_GOODS_PRICE"]
    )

    # Income per family member
    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] /
        df["CNT_FAM_MEMBERS"]
    )

    # Children ratio
    df["CHILDREN_RATIO"] = (
        df["CNT_CHILDREN"] /
        df["CNT_FAM_MEMBERS"]
    )

    # Replace infinite values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df