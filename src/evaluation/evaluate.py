"""Evaluate trained models on the untouched test set.

All metrics are calculated from actual model predictions.
No manual/estimated metrics are used.
"""

from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_data import load_and_merge
from src.data.preprocess import run_full_preprocessing_pipeline
from src.evaluation.metrics import (
    calculate_accuracy,
    calculate_auc_from_curve,
    calculate_classification_report,
    calculate_confusion_matrix,
    calculate_f1,
    calculate_precision,
    calculate_recall,
    calculate_roc_auc,
    calculate_roc_curve,
)

matplotlib.use("Agg")


def evaluate_single_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Evaluate a single model on the test set and save its plots and report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": calculate_accuracy(y_test, y_pred),
        "Precision": calculate_precision(y_test, y_pred),
        "Recall": calculate_recall(y_test, y_pred),
        "F1-Score": calculate_f1(y_test, y_pred),
        "ROC-AUC": calculate_roc_auc(y_test, y_prob),
    }

    cm = calculate_confusion_matrix(y_test, y_pred)
    report = calculate_classification_report(y_test, y_pred, labels=[0, 1], output_dict=True)
    fpr, tpr, _ = calculate_roc_curve(y_test, y_prob)
    auc_value = calculate_auc_from_curve(fpr, tpr)

    base_path = Path(output_dir)
    figures_dir = base_path / "figures"
    tables_dir = base_path / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace(" ", "_").lower()
    pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).to_csv(
        tables_dir / f"{safe_name}_confusion_matrix.csv"
    )
    pd.DataFrame(report).transpose().to_csv(tables_dir / f"{safe_name}_classification_report.csv")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{safe_name}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--- {model_name} ---")
    for key, value in metrics.items():
        if key != "Model":
            print(f"  {key}: {value:.4f}")

    return metrics


def evaluate_all_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Evaluate all models and save results."""
    results = []
    for name, model in models.items():
        metrics = evaluate_single_model(model, X_test, y_test, name, output_dir=output_dir)
        results.append(metrics)

    df_results = pd.DataFrame(results)
    output_path = Path(output_dir) / "tables"
    output_path.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path / "model_results.csv", index=False)

    print("\n=== MODEL COMPARISON TABLE ===")
    print(df_results.to_string(index=False))

    return df_results


def evaluate_classification_reports(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs",
) -> Dict[str, Dict[str, Any]]:
    """Generate and save classification reports for all models."""
    reports = {}
    output_path = Path(output_dir) / "tables"
    output_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = calculate_classification_report(y_test, y_pred, labels=[0, 1], output_dict=True)
        reports[name] = report
        report_df = pd.DataFrame(report).transpose()
        safe_name = name.replace(" ", "_").lower()
        report_df.to_csv(output_path / f"{safe_name}_classification_report.csv")

    return reports


def run_evaluation(
    raw_dir: str = "data/raw",
    output_dir: str = "outputs",
):
    """Load the trained models and evaluate them on the untouched hold-out test set."""
    df = load_and_merge(raw_dir)
    _, X_test, _, y_test, _, _, _ = run_full_preprocessing_pipeline(df)

    model_files = {
        "random_forest": "models/random_forest_model.pkl",
        "xgboost": "models/xgboost_model.pkl",
        "catboost": "models/catboost_model.pkl",
        "stacking": "models/stacking_model.pkl",
    }

    models = {name: joblib.load(path) for name, path in model_files.items() if Path(path).exists()}
    if not models:
        raise FileNotFoundError("No trained model artifacts were found. Run python -m src.models.train first.")

    evaluate_all_models(models, X_test, y_test, output_dir=output_dir)
    evaluate_classification_reports(models, X_test, y_test, output_dir=output_dir)
    return models


if __name__ == "__main__":
    run_evaluation()
