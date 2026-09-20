"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import (
    preprocess_dataframe,
    binary_target,
    split_data,
)
from src.data.download_data import COLUMNS


def create_test_df(n_rows=100):
    """Create a test DataFrame with synthetic data."""
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(30, 80, n_rows),
        "sex": np.random.choice([0, 1], n_rows),
        "cp": np.random.randint(1, 5, n_rows),
        "trestbps": np.random.randint(90, 180, n_rows),
        "chol": np.random.randint(150, 400, n_rows),
        "fbs": np.random.choice([0, 1], n_rows),
        "restecg": np.random.randint(0, 3, n_rows),
        "thalach": np.random.randint(100, 200, n_rows),
        "exang": np.random.choice([0, 1], n_rows),
        "oldpeak": np.random.uniform(0, 6, n_rows),
        "slope": np.random.randint(1, 4, n_rows),
        "ca": np.random.randint(0, 4, n_rows),
        "thal": np.random.choice([3, 6, 7], n_rows),
        "target": np.random.choice([0, 1, 2, 3, 4], n_rows),
    })
    return df


def test_preprocess_dataframe_preserves_missing_values_until_imputation():
    """Test that preprocessing leaves missing feature values for the fitted imputer."""
    df = create_test_df()
    df.loc[0, "age"] = np.nan
    df.loc[1, "chol"] = np.nan
    
    result = preprocess_dataframe(df)
    
    assert pd.isna(result.loc[0, "age"])
    assert pd.isna(result.loc[1, "chol"])


def test_preprocess_dataframe_sex_conversion():
    """Test that numeric sex values remain numeric."""
    df = create_test_df()
    result = preprocess_dataframe(df)
    
    assert np.issubdtype(result["sex"].dtype, np.number)
    assert set(result["sex"].unique()).issubset({0, 1, 0.0, 1.0})


def test_binary_target_conversion():
    """Test that target is converted to binary correctly."""
    df = create_test_df()
    
    result = binary_target(df)
    
    assert set(result["target"].unique()).issubset({0, 1})
    assert result["target"].dtype == int


def test_split_data():
    """Test that train/test split works correctly."""
    df = create_test_df()
    df = preprocess_dataframe(df)
    df = binary_target(df)
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns


def test_split_data_preserves_ratio():
    """Test that split preserves class ratio."""
    df = create_test_df()
    df = preprocess_dataframe(df)
    df = binary_target(df)
    
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)
    
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    
    assert abs(train_ratio - test_ratio) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])