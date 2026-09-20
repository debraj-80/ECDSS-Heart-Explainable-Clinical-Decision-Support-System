"""Download UCI Heart Disease datasets."""

from pathlib import Path

import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_datasets(raw_dir: str = "data/raw") -> list[Path]:
    """Download the four UCI datasets and return their paths.

    The UCI source files are raw .data files with no header row. We must read them
    as data, not as a header row, so we preserve every patient record.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for file_name in FILES:
        file_path = raw_path / file_name
        url = f"{BASE_URL}{file_name}"

        try:
            df = pd.read_csv(url, names=COLUMNS, na_values="?", header=None)
            df.to_csv(file_path, index=False, header=False)
            downloaded.append(file_path)
        except Exception as exc:  # pragma: no cover - network/download failure path
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return downloaded


def load_raw_datasets(raw_dir: str = "data/raw") -> list[pd.DataFrame]:
    """Load all raw datasets from CSV files."""
    raw_path = Path(raw_dir)
    dfs = []
    for file_name in FILES:
        file_path = raw_path / file_name
        if file_path.exists():
            dfs.append(pd.read_csv(file_path, names=COLUMNS, na_values="?", header=None))
    return dfs