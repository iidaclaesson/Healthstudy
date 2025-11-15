import pandas as pd

REQUIRED = [
    "id", "age", "sex", "height", "weight",
    "systolic_bp", "cholesterol", "smoker", "disease"  
]

def load_data(path: str) -> pd.DataFrame:
    """
    Reads csv.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures numeric columns are of the numeric type
    """
    out = df.copy()
    for c in ["age", "height", "weight", "systolic_bp", "cholesterol", "disease"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        return out