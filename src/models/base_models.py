"""Base model definitions for ECDSS-Heart.

Random Forest, XGBoost, and CatBoost classifiers with
reproducible random states.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Return a configured RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def get_xgboost(random_state: int = 42) -> XGBClassifier:
    """Return a configured XGBoost classifier."""
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state,
        eval_metric="logloss",
    )


def get_catboost(random_state: int = 42) -> CatBoostClassifier:
    """Return a configured CatBoost classifier."""
    return CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_state=random_state,
        verbose=0,
    )


def get_all_base_models(
    random_state: int = 42,
) -> Dict[str, Any]:
    """Return a dictionary of all base models.
    
    Keys are model names, values are instantiated classifiers.
    """
    return {
        "random_forest": get_random_forest(random_state),
        "xgboost": get_xgboost(random_state),
        "catboost": get_catboost(random_state),
    }