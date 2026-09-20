"""Feature engineering module for ECDSS-Heart.

Contains reusable functions for creating engineered features.
Each function is documented with its clinical/mathematical rationale.
"""

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for heart disease prediction.
    
    Applies the following feature engineering steps:
    - RPP (Rate Pressure Product): heart rate × systolic BP
    - Thalach_Age_Ratio: maximum heart rate / age
    - Log_Chol: natural log of cholesterol
    
    Pulse_Pressure: NOT included because the UCI Heart Disease
    dataset does not contain diastolic blood pressure measurements.
    True pulse pressure = systolic - diastolic cannot be calculated
    from the available features. Including a false pulse pressure
    calculation would be mathematically invalid.
    
    Args:
        df: Input DataFrame with raw UCI features.
        
    Returns:
        DataFrame with original features plus engineered features.
    """
    df = df.copy()
    
    # RPP: Rate Pressure Product = heart rate × systolic blood pressure
    # Clinical significance: indicator of myocardial oxygen consumption
    if "thalach" in df.columns and "trestbps" in df.columns:
        df["RPP"] = df["thalach"] * df["trestbps"]
    else:
        raise ValueError("Required columns 'thalach' and 'trestbps' missing for RPP.")
    
    # Thalach_Age_Ratio: maximum heart rate / age
    # Clinical significance: relative cardiac capacity normalized by age
    if "thalach" in df.columns and "age" in df.columns:
        # Avoid division by zero; replace age=0 with NaN then fill
        age = df["age"].replace(0, np.nan)
        df["Thalach_Age_Ratio"] = df["thalach"] / age
        df["Thalach_Age_Ratio"] = df["Thalach_Age_Ratio"].fillna(0)
    else:
        raise ValueError("Required columns 'thalach' and 'age' missing for Thalach_Age_Ratio.")
    
    # Log_Chol: natural log of cholesterol
    # Clinical significance: reduces right-skew of cholesterol distribution
    if "chol" in df.columns:
        df["Log_Chol"] = np.log(df["chol"] + 1)  # +1 to handle chol=0
    else:
        raise ValueError("Required column 'chol' missing for Log_Chol.")
    
    # Pulse_Pressure: omitted - UCI dataset lacks diastolic BP.
    # True pulse pressure = systolic - diastolic cannot be computed.
    # The original calculation (trestbps - trestbps.min()) was a false
    # proxy that subtracted the minimum trestbps value across the dataset,
    # not a valid physiological pulse pressure. This feature is therefore
    # excluded to prevent misinformation.
    # If future data includes diastolic BP, this can be added as:
    # df["Pulse_Pressure"] = df["trestbps"] - df["diastolic_bp"]
    
    return df