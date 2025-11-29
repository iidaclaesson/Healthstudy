import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


def ci_mean_normal(x, confidence: float = 0.95):
    """
    Returns the confidence interval for systolic blood pressure
    """
    x = np.asarray(x, dtype=float)
    mean_x = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    n = len(x)
    z_critical = 1.96
    half_width = z_critical * std / sqrt(n)
    lo, hi = mean_x - half_width, mean_x + half_width
    return lo, hi, mean_x, std, n


def bootstrap_mean(smokers, nonsmokers, n_boot=10_000): 
    """
    Returns the bootstrap results for the difference in mean systolic blood pressure
    between smokers and non-smokers.
    """
    obs_diff = smokers.mean() - nonsmokers.mean()
    
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        smokers_star = np.random.choice(smokers, size=len(smokers), replace=True)
        nonsmokers_star = np.random.choice(nonsmokers, size=len(nonsmokers), replace=True)
        boot_diffs[i] = smokers_star.mean() - nonsmokers_star.mean()
    p_boot = np.mean(boot_diffs >= obs_diff)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return float(obs_diff), float(p_boot), (float(ci_low), float(ci_high))


def linear_regression(df):
    """
    Returns the coefficients of a linear 
    regression predicting systolic blood pressure
    from age and weight
    """

    y = df["systolic_bp"].values.reshape(-1, 1)
    X = df[["age", "weight"]].values

    X = np.column_stack((np.ones(len(X)), X))

    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)

    y_pred = X @ beta_hat   

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return beta_hat.flatten(), float(r_squared)


class HealthAnalyzer:
    """
    A class which analyzes health data
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def sbp_mean(self) -> float:
        return float(self.df["systolic_bp"].mean())
    
    def smoker_diff(self) -> float:
        smokers = self.df[self.df["smoker"] == "Yes"]["systolic_bp"]
        nonsmokers = self.df[self.df["smoker"] == "No"]["systolic_bp"]
        return float(smokers.mean() - nonsmokers.mean())