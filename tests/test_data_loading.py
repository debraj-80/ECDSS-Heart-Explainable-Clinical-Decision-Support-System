"""Regression tests for dataset loading and preprocessing flow."""

import pandas as pd
import numpy as np

from src.data.load_data import load_and_merge
from src.data.preprocess import preprocess_dataframe
from app.utils import prepare_patient_input


def test_load_and_merge_keeps_all_ucl_rows():
    """The merged UCI dataset should retain all patient records from all four sources."""
    df = load_and_merge("data/raw")
    assert len(df) == 920
    assert set(df.columns) >= {"age", "sex", "cp", "trestbps", "chol", "target"}


def test_preprocess_dataframe_does_not_impute_before_split():
    """Preprocessing should only coerce types and target values, not learn from the full dataset."""
    df = pd.DataFrame({
        "age": [40, 50, np.nan],
        "sex": [1, 0, 1],
        "cp": [1, 2, 3],
        "trestbps": [120, 130, np.nan],
        "chol": [200, 210, 220],
        "fbs": [0, 1, 0],
        "restecg": [0, 1, 2],
        "thalach": [150, 160, 170],
        "exang": [0, 1, 0],
        "oldpeak": [1.2, np.nan, 2.0],
        "slope": [1, 2, 3],
        "ca": [0, 1, 2],
        "thal": [3, 6, 7],
        "target": [0, 1, 2],
    })

    result = preprocess_dataframe(df)

    assert pd.isna(result.loc[2, "age"]) or result.loc[2, "age"] != result.loc[0, "age"]
    assert pd.isna(result.loc[2, "trestbps"]) or result.loc[2, "trestbps"] != result.loc[0, "trestbps"]
    assert pd.isna(result.loc[1, "oldpeak"]) or result.loc[1, "oldpeak"] != result.loc[0, "oldpeak"]


def test_prepare_patient_input_uses_ucl_encodings():
    """The app should preserve the original UCI category encodings for patient input."""
    patient = prepare_patient_input(
        age=55,
        sex=1,
        cp=4,
        trestbps=140,
        chol=250,
        fbs=0,
        restecg=2,
        thalach=160,
        exang=1,
        oldpeak=1.8,
        slope=3,
        ca=3,
        thal=7,
    )

    assert patient.loc[0, "cp"] == 4
    assert patient.loc[0, "restecg"] == 2
    assert patient.loc[0, "slope"] == 3
    assert patient.loc[0, "ca"] == 3
    assert patient.loc[0, "thal"] == 7
