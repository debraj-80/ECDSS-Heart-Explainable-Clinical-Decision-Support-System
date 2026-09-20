"""Tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
from src.models.base_models import get_all_base_models
from src.models.stacking import get_stacking_model


def create_test_data(n_samples=200):
    """Create synthetic training/test data."""
    np.random.seed(42)
    X = pd.DataFrame({
        "age": np.random.randint(30, 80, n_samples),
        "sex": np.random.choice([0, 1], n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.randint(90, 180, n_samples),
        "chol": np.random.randint(150, 400, n_samples),
        "fbs": np.random.choice([0, 1], n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(100, 200, n_samples),
        "exang": np.random.choice([0, 1], n_samples),
        "oldpeak": np.random.uniform(0, 6, n_samples),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.choice([0, 1, 2], n_samples),
        "RPP": np.random.randint(10000, 30000, n_samples),
        "Thalach_Age_Ratio": np.random.uniform(1.5, 5, n_samples),
        "Log_Chol": np.random.uniform(5, 6.5, n_samples),
    })
    y = pd.Series(np.random.choice([0, 1], n_samples))
    return X, y


def test_base_models_can_train():
    """Test that all base models can be trained."""
    X, y = create_test_data()
    models = get_all_base_models()
    
    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})


def test_base_models_can_predict_proba():
    """Test that base models can predict probabilities."""
    X, y = create_test_data()
    models = get_all_base_models()
    
    for name, model in models.items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all()
        assert (proba <= 1).all()


def test_stacking_model_can_train():
    """Test that stacking model can be trained."""
    X, y = create_test_data()
    stack_model = get_stacking_model()
    stack_model.fit(X, y)
    
    predictions = stack_model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_stacking_model_can_predict_proba():
    """Test that stacking model can predict probabilities."""
    X, y = create_test_data()
    stack_model = get_stacking_model()
    stack_model.fit(X, y)
    
    proba = stack_model.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_stacking_has_base_estimators():
    """Test that stacking model has the expected base estimators."""
    stack_model = get_stacking_model()
    
    assert hasattr(stack_model, "estimators")
    assert len(stack_model.estimators) == 3


def test_predictions_are_binary():
    """Test that predictions are binary (0 or 1)."""
    X, y = create_test_data()
    models = get_all_base_models()
    models["stacking"] = get_stacking_model()
    
    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)
        assert all(p in [0, 1] for p in predictions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])