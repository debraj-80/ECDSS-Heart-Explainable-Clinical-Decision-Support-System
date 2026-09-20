"""ECDSS-Heart Streamlit Application.

A research/educational clinical decision-support prototype for
heart disease risk prediction.

DISCLAIMER:
This application is a research/educational machine-learning prototype
and is NOT a medical diagnostic tool. Predictions should not be used
as a substitute for professional medical advice.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from utils import (
    load_model,
    load_preprocessing_pipeline,
    prepare_patient_input,
    predict_risk,
)

from src.data.load_data import load_and_merge
from src.data.preprocess import (
    binary_target,
    preprocess_dataframe,
    split_data,
)
from src.explainability.dice_explainer import (
    CATEGORICAL_FEATURES,
    generate_counterfactuals as create_counterfactuals,
    prepare_dice_training_data,
    setup_dice,
)
from src.features.feature_engineering import create_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ECDSS-Heart",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DISCLAIMER
# ============================================================

st.sidebar.markdown(
    """
    ---
    ### Disclaimer

    This application is a **research/educational machine-learning
    prototype** and is **NOT a medical diagnostic tool**.

    Predictions should not be used as a substitute for professional
    medical advice.

    ---
    """
)


# ============================================================
# LOAD MODEL AND PREPROCESSING PIPELINE
# ============================================================

@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing pipeline."""
    try:
        model = load_model("models/stacking_model.pkl")
        pipeline = load_preprocessing_pipeline(
            "outputs/preprocessing_pipeline.pkl"
        )
        return model, pipeline

    except FileNotFoundError:
        st.error(
            "Model files not found. Please run training first:\n\n"
            "`python -m src.models.train`"
        )
        st.stop()


# ============================================================
# PATIENT INPUT FORM
# ============================================================

def patient_input_form():
    """Create the patient input form using the exact UCI encodings."""

    st.header("Patient Information")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=55,
        )

        sex = st.selectbox(
            "Sex",
            options=[1, 0],
            format_func=lambda x: (
                "Male" if x == 1 else "Female"
            ),
        )

        cp = st.selectbox(
            "Chest pain type (cp)",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic",
            }[x],
            help="Exact UCI encoding used during training: 1, 2, 3, 4.",
        )

        trestbps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=80,
            max_value=250,
            value=130,
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        chol = st.number_input(
            "Serum cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=250,
        )

        fbs = st.selectbox(
            "Fasting blood sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: (
                "Yes" if x == 1 else "No"
            ),
        )

        restecg = st.selectbox(
            "Resting ECG result (restecg)",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy",
            }[x],
            help="Exact UCI encoding used during training: 0, 1, 2.",
        )

        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=60,
            max_value=220,
            value=150,
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        exang = st.selectbox(
            "Exercise-induced angina",
            options=[0, 1],
            format_func=lambda x: (
                "Yes" if x == 1 else "No"
            ),
        )

        oldpeak = st.number_input(
            "ST depression (oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        slope = st.selectbox(
            "Slope of peak exercise ST segment (slope)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping",
            }[x],
            help="Exact UCI encoding used during training: 1, 2, 3.",
        )

        ca = st.selectbox(
            "Number of major vessels colored by fluoroscopy (ca)",
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{x} vessel(s)",
            help="Exact UCI encoding used during training: 0, 1, 2, 3.",
        )

        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed defect",
                7: "Reversible defect",
            }[x],
            help="Exact UCI encoding used during training: 3, 6, 7.",
        )

    return (
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

def display_prediction(result: dict):
    """Display model prediction and probabilities."""

    st.header("Model Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Risk Level")

        if result["prediction"] == 0:

            st.success(
                f"Model prediction: **{result['risk_level']}**"
            )

        else:

            st.warning(
                f"Model prediction: **{result['risk_level']}**"
            )

        st.metric(
            "Probability (No Disease)",
            f"{result['probability_no_disease']:.1%}",
        )

        st.metric(
            "Probability (Disease)",
            f"{result['probability_disease']:.1%}",
        )

    with col2:

        st.subheader("Prediction Confidence")

        confidence = max(
            result["probability_no_disease"],
            result["probability_disease"],
        )

        st.progress(confidence)

        st.write(
            f"Confidence: **{confidence:.1%}**"
        )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def display_shap_explanation(
    model,
    patient_data,
    preprocessing_pipeline,
):
    """Display bounded SHAP explanation for the XGBoost base learner."""

    st.header("Feature Importance (SHAP)")

    st.info(
        "SHAP explanation of the XGBoost base learner on the same "
        "transformed feature representation used for the model prediction. "
        "Red indicates features pushing toward higher risk; "
        "blue indicates features pushing toward lower risk."
    )

    try:

        # ----------------------------------------------------
        # Extract XGBoost base learner
        # ----------------------------------------------------

        xgb_model = model.named_estimators_["xgboost"]

        # ----------------------------------------------------
        # Apply the same preprocessing used for prediction
        # ----------------------------------------------------

        processed = preprocessing_pipeline.transform(
            patient_data
        )

        processed_df = pd.DataFrame(
            processed,
            columns=patient_data.columns,
        )

        # ----------------------------------------------------
        # Create TreeExplainer
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(xgb_model)

        shap_values = explainer(processed_df)

        # ----------------------------------------------------
        # Extract SHAP values for current patient
        # ----------------------------------------------------

        values = shap_values.values[0]

        # Handle possible multi-output SHAP format
        if values.ndim > 1:

            values = values[:, 1]

        feature_names = processed_df.columns.tolist()

        # ----------------------------------------------------
        # Create SHAP dataframe
        # ----------------------------------------------------

        shap_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": values,
            }
        )

        shap_df["abs_value"] = (
            shap_df["shap_value"].abs()
        )

        # Show only the most influential features
        shap_df = (
            shap_df
            .sort_values(
                "abs_value",
                ascending=False,
            )
            .head(12)
            .sort_values(
                "shap_value"
            )
        )

        # ----------------------------------------------------
        # Create controlled-size plot
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        bars = ax.barh(
            shap_df["feature"],
            shap_df["shap_value"],
        )

        # Positive = higher risk contribution
        # Negative = lower risk contribution

        for bar, value in zip(
            bars,
            shap_df["shap_value"],
        ):

            if value >= 0:

                bar.set_color("red")

            else:

                bar.set_color("blue")

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_xlabel(
            "SHAP value"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "SHAP Explanation - XGBoost Base Learner"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

        plt.close(fig)

    except Exception as e:

        st.warning(
            f"Could not generate SHAP explanation: {e}"
        )


# ============================================================
# DICE COUNTERFACTUAL EXPLANATION
# ============================================================

def generate_counterfactuals(
    model,
    patient_data,
    preprocessing_pipeline,
):
    """Generate DiCE counterfactual explanations."""

    st.header("Counterfactual Explanation (DiCE)")

    st.caption(
        "Model-generated hypothetical counterfactuals. "
        "Not a medical recommendation."
    )

    try:

        # ----------------------------------------------------
        # Load and prepare training data
        # ----------------------------------------------------

        training_df = create_features(
            binary_target(
                preprocess_dataframe(
                    load_and_merge("data/raw")
                )
            )
        )

        X_train, _, y_train, _ = split_data(
            training_df
        )

        X_train = prepare_dice_training_data(
            X_train,
            CATEGORICAL_FEATURES,
        )

        # ----------------------------------------------------
        # Setup DiCE
        # ----------------------------------------------------

        explainer = setup_dice(
            X_train,
            y_train,
            model,
            categorical_features=CATEGORICAL_FEATURES,
            preprocessing_pipeline=preprocessing_pipeline,
        )

        # ----------------------------------------------------
        # Current prediction
        # ----------------------------------------------------

        processed_patient = (
            preprocessing_pipeline.transform(
                patient_data
            )
        )

        prediction = int(
            model.predict(
                processed_patient
            )[0]
        )

        # ----------------------------------------------------
        # Generate counterfactuals
        # ----------------------------------------------------

        result = create_counterfactuals(
            explainer,
            patient_data,
            num_counterfactuals=2,
            desired_class=1 - prediction,
        )

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        cf_df = (
            result
            .cf_examples_list[0]
            .final_cfs_df
        )

        if cf_df.empty:

            st.info(
                "No counterfactuals were generated "
                "for this patient."
            )

            return

        # Restore integer representation
        # for categorical features

        for column in CATEGORICAL_FEATURES:

            if column in cf_df.columns:

                cf_df[column] = (
                    cf_df[column]
                    .round()
                    .astype(int)
                )

        st.dataframe(
            cf_df,
            use_container_width=True,
        )

        st.caption(
            "These counterfactuals use the original feature scale "
            "and represent model-generated hypothetical changes, "
            "not medical advice."
        )

    except Exception as e:

        st.warning(
            f"Could not generate counterfactual explanation: {e}"
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point."""

    st.title("ECDSS-Heart")

    st.subheader(
        "Explainable Clinical Decision Support System "
        "for Heart Disease Prediction"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, pipeline = load_artifacts()

    # --------------------------------------------------------
    # Patient form
    # --------------------------------------------------------

    (
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal,
    ) = patient_input_form()

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "Predict Risk",
        type="primary",
    ):

        # ----------------------------------------------------
        # Prepare patient data
        # ----------------------------------------------------

        patient_data = prepare_patient_input(
            age=age,
            sex=sex,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal,
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        result = predict_risk(
            model,
            patient_data,
            pipeline,
        )

        display_prediction(result)

        # ----------------------------------------------------
        # Engineered features
        # ----------------------------------------------------

        st.subheader(
            "Engineered Features"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "RPP",
            f"{patient_data['RPP'].values[0]:.0f}",
        )

        col2.metric(
            "Thalach/Age Ratio",
            f"{patient_data['Thalach_Age_Ratio'].values[0]:.2f}",
        )

        col3.metric(
            "Log Cholesterol",
            f"{patient_data['Log_Chol'].values[0]:.2f}",
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        display_shap_explanation(
            model,
            patient_data,
            pipeline,
        )

        # ----------------------------------------------------
        # DiCE
        # ----------------------------------------------------

        generate_counterfactuals(
            model,
            patient_data,
            pipeline,
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()