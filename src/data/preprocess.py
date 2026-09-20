"""Preprocessing pipeline with proper train/test separation.

CRITICAL: Resampling and preprocessing are fitted only on training data
to avoid data leakage. Uses imblearn.pipeline.Pipeline for correctness.
"""

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import create_features

TARGET_COL = "target"
RANDOM_STATE = 42


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert feature and target columns to numeric values without imputing."""
    df = df.copy()

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for col in df.columns:
        if col == TARGET_COL:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def binary_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Convert target to binary: 0 = no disease, 1 = disease."""
    df = df.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets before any learning-based preprocessing."""
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline() -> ImbPipeline:
    """Build preprocessing pipeline (imputer + scaler) fitted on training data only."""
    pipeline = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


def apply_resampling(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE-Tomek resampling only on training data."""
    smt = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    return pd.DataFrame(X_resampled), y_resampled


def run_full_preprocessing_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ImbPipeline, pd.DataFrame, pd.Series]:
    """Complete leakage-safe pipeline for the heart disease model.

    Workflow:
    1. Clean dataframe
    2. Convert target to binary
    3. Perform feature engineering
    4. Split into train/test
    5. Fit imputer + scaler on training only
    6. Transform both train and test
    7. Apply SMOTE-Tomek only on training data
    """
    df = preprocess_dataframe(df)
    df = binary_target(df)
    df = create_features(df)

    X_train, X_test, y_train, y_test = split_data(
        df, test_size=test_size, random_state=random_state
    )

    preprocessing_pipeline = build_preprocessing_pipeline()
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    feature_names = X_train.columns.tolist()
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=feature_names)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=feature_names)

    X_train_resampled, y_train_resampled = apply_resampling(
        X_train_preprocessed_df, y_train, random_state=random_state
    )

    return (
        X_train_resampled,
        X_test_preprocessed_df,
        y_train_resampled,
        y_test,
        preprocessing_pipeline,
        X_train_preprocessed_df,
        y_train,
    )