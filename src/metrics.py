import pandas as pd
import numpy as np

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with mean, median, std, min, and max for the selected columns
    """
    cols = ["age", "height", "weight", "systolic_bp", "cholesterol"]

    stats_df = df[cols].agg(["mean", "median", "min", "max"])
    return stats_df


def simulated_disease_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with the proportion of simulated disease cases
    """

    disease_count = df["disease"].mean()

    simulated_count = (np.random.random(1000) < disease_count).mean()

    diff = disease_count - simulated_count

    return { "disease_count": disease_count,
         "simulated_count": simulated_count,
         "difference": diff }


def sbp_mean(df: pd.DataFrame) -> float:
    """
    Returns the mean of systolic blood pressure
    """
    return float(np.mean(df["systolic_bp"]))
