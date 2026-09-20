"""SHAP-based explainability for ECDSS-Heart.

Explains feature importance using SHAP values.
Clearly documents which model is being explained.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path


def generate_shap_explanation(
    model,
    X_test: pd.DataFrame,
    model_name: str = "model",
    output_dir: str = "outputs/figures",
) -> None:
    """Generate SHAP summary plot for a given model.
    
    Args:
        model: Fitted classifier with SHAP-compatible interface.
        X_test: Test features.
        model_name: Name of the model being explained.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Feature Importance ({model_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_summary_{model_name.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - Bar ({model_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_importance_{model_name.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"SHAP plots saved for {model_name}")


def explain_xgboost_from_stack(
    stack_model,
    X_test: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> None:
    """Explain the XGBoost base learner from the stacking ensemble.
    
    Note: SHAP is applied to the XGBoost component, NOT the full
    stacking ensemble. The docstring for generated plots should
    reflect this accurately.
    """
    # Extract XGBoost model from the stacking ensemble
    xgb_model = stack_model.named_estimators_["xgb"]
    
    generate_shap_explanation(
        xgb_model, X_test,
        model_name="XGBoost (Base Learner)",
        output_dir=output_dir,
    )
