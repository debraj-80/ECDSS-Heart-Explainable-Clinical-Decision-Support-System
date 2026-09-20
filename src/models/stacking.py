"""Stacking ensemble model for ECDSS-Heart.

Builds a StackingClassifier with Random Forest, XGBoost, and CatBoost
as base learners and Logistic Regression as the meta learner.
"""

from typing import Dict, Any
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.base_models import get_all_base_models


def get_stacking_model(
    random_state: int = 42,
) -> StackingClassifier:
    """Build and return a StackingClassifier.
    
    Base learners:
    - Random Forest
    - XGBoost
    - CatBoost
    
    Meta learner: Logistic Regression
    
    Returns:
        Configured but unfitted StackingClassifier.
    """
    base_learners = get_all_base_models(random_state=random_state)
    meta_learner = LogisticRegression(random_state=random_state)
    
    stack_model = StackingClassifier(
        estimators=list(base_learners.items()),
        final_estimator=meta_learner,
        cv=5,
    )
    return stack_model