"""Training module for ECDSS-Heart.

Handles the complete training pipeline:
1. Load data
2. Convert data types and target
3. Apply feature engineering
4. Split train/test
5. Fit preprocessing on training only
6. Apply resampling on training only
7. Train base models
8. Train stacking ensemble
9. Save trained models using joblib
"""

from pathlib import Path
from typing import Any, Dict

import joblib

from src.data.load_data import load_and_merge
from src.data.preprocess import run_full_preprocessing_pipeline
from src.models.stacking import get_stacking_model
from src.models.base_models import get_all_base_models


def train_pipeline(
    data_path: str = "data/raw",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Complete training pipeline with leakage-safe preprocessing."""
    print("--- [Status] Loading UCI datasets... ---")
    df = load_and_merge(data_path)
    print(f"Loaded dataset shape: {df.shape}")

    print("--- [Status] Running preprocessing pipeline... ---")
    (
        X_train_resampled,
        X_test,
        y_train_resampled,
        y_test,
        preprocessing_pipeline,
        X_train_preproc,
        y_train_resampled_check,
    ) = run_full_preprocessing_pipeline(df, test_size=test_size, random_state=random_state)

    feature_names = X_train_resampled.columns.tolist()
    print(f"Feature names: {feature_names}")

    print("--- [Status] Training Stacking Ensemble... ---")
    stacking_model = get_stacking_model(random_state=random_state)
    stacking_model.fit(X_train_resampled, y_train_resampled)

    base_models = get_all_base_models(random_state=random_state)
    fitted_base_models = {}
    for name, model in base_models.items():
        model.fit(X_train_resampled, y_train_resampled)
        fitted_base_models[name] = model

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(stacking_model, models_dir / "stacking_model.pkl")
    joblib.dump(fitted_base_models["random_forest"], models_dir / "random_forest_model.pkl")
    joblib.dump(fitted_base_models["xgboost"], models_dir / "xgboost_model.pkl")
    joblib.dump(fitted_base_models["catboost"], models_dir / "catboost_model.pkl")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing_pipeline, outputs_dir / "preprocessing_pipeline.pkl")

    print("--- [Status] Models saved successfully. ---")

    return {
        "stack_model": stacking_model,
        "base_models": fitted_base_models,
        "X_train_resampled": X_train_resampled,
        "X_test": X_test,
        "y_train_resampled": y_train_resampled,
        "y_test": y_test,
        "preprocessing_pipeline": preprocessing_pipeline,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    results = train_pipeline()
    print("\nTraining complete!")
    print(
        f"Stacking model accuracy (test): "
        f"{results['stack_model'].score(results['X_test'], results['y_test']):.4f}"
    )