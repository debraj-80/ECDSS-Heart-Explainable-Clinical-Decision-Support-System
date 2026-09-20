"""Utility functions for the ECDSS-Heart web application."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def load_model(model_path: str = "models/stacking_model.pkl"):
    """Load a trained model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_preprocessing_pipeline(
    pipeline_path: str = "outputs/preprocessing_pipeline.pkl",
):
    """Load the fitted preprocessing pipeline."""
    path = Path(pipeline_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    return joblib.load(path)


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

ENGINEERED_FEATURE_NAMES = ["RPP", "Thalach_Age_Ratio", "Log_Chol"]

ALL_FEATURE_NAMES = FEATURE_NAMES + ENGINEERED_FEATURE_NAMES


def prepare_patient_input(
    age: float,
    sex: int,
    cp: int,
    trestbps: float,
    chol: float,
    fbs: int,
    restecg: int,
    thalach: float,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
) -> pd.DataFrame:
    """Prepare patient input data with engineered features.
    
    Args:
        age: Age in years.
        sex: Sex (1=male, 0=female).
        cp: Chest pain type (1-4).
        trestbps: Resting blood pressure (mm Hg).
        chol: Serum cholesterol (mg/dl).
        fbs: Fasting blood sugar > 120 mg/dl (1=true, 0=false).
        restecg: Resting ECG results (0-2).
        thalach: Maximum heart rate achieved.
        exang: Exercise-induced angina (1=yes, 0=no).
        oldpeak: ST depression induced by exercise relative to rest.
        slope: Slope of peak exercise ST segment (1-3).
        ca: Number of major vessels colored by fluoroscopy (0-3).
        thal: Thalassemia (3=normal, 6=fixed defect, 7=reversible defect).
        
    Returns:
        DataFrame with raw features and engineered features.
    """
    raw_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    
    df = pd.DataFrame([raw_data])
    
    # Apply feature engineering
    df["RPP"] = df["thalach"] * df["trestbps"]
    age_val = df["age"].replace(0, np.nan)
    df["Thalach_Age_Ratio"] = df["thalach"] / age_val
    df["Thalach_Age_Ratio"] = df["Thalach_Age_Ratio"].fillna(0)
    df["Log_Chol"] = np.log(df["chol"] + 1)
    
    return df


def predict_risk(
    model,
    patient_data: pd.DataFrame,
    preprocessing_pipeline=None,
) -> Dict[str, Any]:
    """Predict heart disease risk for a patient.
    
    Args:
        model: Fitted stacking model.
        patient_data: Patient DataFrame with raw + engineered features.
        preprocessing_pipeline: Fitted preprocessing pipeline (optional).
        
    Returns:
        Dictionary with prediction, probabilities, and risk level.
    """
    if preprocessing_pipeline is not None:
        patient_processed = preprocessing_pipeline.transform(patient_data)
    else:
        patient_processed = patient_data
    
    prediction = int(model.predict(patient_processed)[0])
    probabilities = model.predict_proba(patient_processed)[0]
    
    risk_level = "lower predicted risk" if prediction == 0 else "higher predicted risk"
    
    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "probability_no_disease": float(probabilities[0]),
        "probability_disease": float(probabilities[1]),
    }