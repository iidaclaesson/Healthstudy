import pandas as pd
import numpy as np

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with mean, median, std, min, and max for the selected columns
    """

    stats = []
    for col in ["age", "weight", "height", "systolic_bp", "cholesterol"]:
        columns = {
            "variable": col,
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max()
        }
        stats.append(columns)
    return pd.DataFrame(stats)
