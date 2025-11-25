import pandas as pd
import numpy as np

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with mean, median, std, min, and max for the selected columns
    """

    cols = ["age", "height", "weight", "systolic_bp", "cholesterol"]

    stats_df = df[cols].agg(["mean", "median", "min", "max"])
    return stats_df
