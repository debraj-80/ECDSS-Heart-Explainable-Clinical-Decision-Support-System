"""Load and merge UCI Heart Disease datasets."""

from pathlib import Path

import pandas as pd

from src.data.download_data import COLUMNS, FILES, download_datasets


def load_and_merge(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all raw datasets, assign column names, and merge them.

    Returns a single clean DataFrame with all rows combined.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists() or not any(raw_path.glob("*.data")):
        download_datasets(raw_dir)

    frames = []
    for file_name in FILES:
        file_path = raw_path / file_name
        if file_path.exists():
            frames.append(pd.read_csv(file_path, names=COLUMNS, na_values="?", header=None))

    df = pd.concat(frames, ignore_index=True)
    return df