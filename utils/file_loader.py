# utils/file_loader.py
import pandas as pd

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file at {path}: {e}")
