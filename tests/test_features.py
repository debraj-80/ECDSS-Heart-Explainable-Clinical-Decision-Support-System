"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import create_features


def create_test_df():
    """Create a test DataFrame with raw features."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, 100),
        "sex": np.random.choice([0, 1], 100),
        "cp": np.random.randint(0, 4, 100),
        "trestbps": np.random.randint(90, 180, 100),
        "chol": np.random.randint(150, 400, 100),
        "fbs": np.random.choice([0, 1], 100),
        "restecg": np.random.randint(0, 3, 100),
        "thalach": np.random.randint(100, 200, 100),
        "exang": np.random.choice([0, 1], 100),
        "oldpeak": np.random.uniform(0, 6, 100),
        "slope": np.random.randint(0, 3, 100),
        "ca": np.random.randint(0, 4, 100),
        "thal": np.random.choice([0, 1, 2], 100),
        "target": np.random.choice([0, 1], 100),
    })
    return df


def test_create_features_adds_engineered_columns():
    """Test that feature engineering adds the expected columns."""
    df = create_test_df()
    result = create_features(df)
    
    assert "RPP" in result.columns
    assert "Thalach_Age_Ratio" in result.columns
    assert "Log_Chol" in result.columns


def test_create_features_preserves_original_columns():
    """Test that original features are preserved."""
    df = create_test_df()
    original_cols = set(df.columns)
    result = create_features(df)
    
    assert original_cols.issubset(set(result.columns))


def test_create_features_rpp_calculation():
    """Test that RPP is calculated correctly."""
    df = create_test_df()
    result = create_features(df)
    
    expected_rpp = df["thalach"] * df["trestbps"]
    pd.testing.assert_series_equal(
        result["RPP"], expected_rpp, check_names=False
    )


def test_create_features_no_nan_inf():
    """Test that engineered features have no NaN or inf values."""
    df = create_test_df()
    result = create_features(df)
    
    assert not result["RPP"].isna().any()
    assert not np.isinf(result["RPP"]).any()
    assert not result["Thalach_Age_Ratio"].isna().any()
    assert not np.isinf(result["Thalach_Age_Ratio"]).any()
    assert not result["Log_Chol"].isna().any()
    assert not np.isinf(result["Log_Chol"]).any()


def test_log_chol_values_positive():
    """Test that Log_Chol values are positive."""
    df = create_test_df()
    result = create_features(df)
    
    assert (result["Log_Chol"] >= 0).all()


def test_thalach_age_ratio_positive():
    """Test that Thalach_Age_Ratio is positive."""
    df = create_test_df()
    result = create_features(df)
    
    assert (result["Thalach_Age_Ratio"] > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])