"""DiCE counterfactual explanation module for ECDSS-Heart.

Generates hypothetical counterfactual examples showing which features
would need to change for a different prediction.

IMPORTANT: Counterfactuals are model-generated hypothetical changes,
NOT medical advice. They do not imply that changing a feature will
actually improve a patient's health.
"""

import numpy as np
import pandas as pd
import dice_ml
from typing import Dict, Any, Optional

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


class _PreprocessedModel:
    """Adapt raw-feature DiCE inputs to the representation used by the model."""

    def __init__(self, model, preprocessing_pipeline):
        self.model = model
        self.preprocessing_pipeline = preprocessing_pipeline

    def _transform(self, features: pd.DataFrame):
        return self.preprocessing_pipeline.transform(features)

    def predict(self, features: pd.DataFrame):
        return self.model.predict(self._transform(features))

    def predict_proba(self, features: pd.DataFrame):
        return self.model.predict_proba(self._transform(features))


def prepare_dice_training_data(
    X_train: pd.DataFrame,
    categorical_features: Optional[list] = None,
) -> pd.DataFrame:
    """Prepare representative raw training data for valid DiCE categories."""
    categorical_features = categorical_features or CATEGORICAL_FEATURES
    prepared = X_train.copy()

    for column in prepared.columns:
        if column in categorical_features:
            mode = prepared[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else 0
            prepared[column] = (
                pd.to_numeric(prepared[column], errors="coerce")
                .fillna(fill_value)
                .round()
                .astype(int)
            )
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = prepared[column].fillna(prepared[column].median())

    return prepared


def setup_dice(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,
    continuous_features: Optional[list] = None,
    categorical_features: Optional[list] = None,
    preprocessing_pipeline=None,
) -> dice_ml.Dice:
    """Set up the DiCE counterfactual explainer.
    
    Args:
        X_train: Training features (used for DiCE data setup).
        y_train: Training labels.
        model: Fitted sklearn-compatible model.
        continuous_features: List of continuous feature names.
        categorical_features: Feature names whose values must remain categorical.
        preprocessing_pipeline: Fitted pipeline for raw training features.
        
    Returns:
        Configured DiCE Dice object.
    """
    categorical_features = [
        feature for feature in (categorical_features or CATEGORICAL_FEATURES)
        if feature in X_train.columns
    ]
    X_train = prepare_dice_training_data(X_train, categorical_features)

    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    train_df.columns = list(X_train.columns) + ["target"]
    
    if continuous_features is None:
        continuous_features = [
            feature for feature in X_train.columns
            if feature not in categorical_features
        ]
    
    # DiCE data setup
    data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name="target",
    )
    
    model_for_dice = model
    if preprocessing_pipeline is not None:
        model_for_dice = _PreprocessedModel(model, preprocessing_pipeline)
    ml_model = dice_ml.Model(model=model_for_dice, backend="sklearn")
    
    # Create Dice explainer
    explainer = dice_ml.Dice(data, ml_model, method="random")
    
    return explainer


def generate_counterfactuals(
    explainer: dice_ml.Dice,
    patient: pd.DataFrame,
    num_counterfactuals: int = 3,
    desired_class: int = 0,
) -> dice_ml.Dice:
    """Generate counterfactual explanations for a patient.
    
    Args:
        explainer: Configured DiCE explainer.
        patient: Patient features (1 row DataFrame).
        num_counterfactuals: Number of counterfactual examples to generate.
        desired_class: Target class for counterfactuals.
        
    Returns:
        DiCE counterfactual explanation object.
    """
    counterfactuals = explainer.generate_counterfactuals(
        patient,
        total_CFs=num_counterfactuals,
        desired_class=desired_class,
    )
    return counterfactuals


def get_patient_prediction(model, patient: pd.DataFrame) -> Dict[str, Any]:
    """Get the model prediction for a patient.
    
    Args:
        model: Fitted classifier.
        patient: Patient features (1 row DataFrame).
        
    Returns:
        Dictionary with prediction class and probability.
    """
    prediction = int(model.predict(patient)[0])
    probabilities = model.predict_proba(patient)[0]
    
    return {
        "prediction": prediction,
        "probability_no_disease": float(probabilities[0]),
        "probability_disease": float(probabilities[1]),
    }


def analyze_counterfactual_changes(
    patient: pd.DataFrame,
    counterfactuals: dice_ml.Dice,
) -> pd.DataFrame:
    """Analyze which features changed between original and counterfactual.
    
    Args:
        patient: Original patient data.
        counterfactuals: DiCE counterfactual explanation object.
        
    Returns:
        DataFrame showing original vs counterfactual values and changes.
    """
    try:
        cf_df = counterfactuals.visualize_as_dataframe(show_only_changes=True)
        return cf_df
    except (ImportError, ModuleNotFoundError):
        # Fallback when IPython is not available
        print("[Info] IPython not available. Returning raw counterfactual data.")
        cf_examples = counterfactuals.cf_examples_list[0]
        return cf_examples.final_cfs_df