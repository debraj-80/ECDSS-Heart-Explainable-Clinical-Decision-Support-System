# ECDSS-Heart

**Explainable Clinical Decision Support System for Heart Disease Prediction**

A research/educational machine-learning prototype that predicts heart disease risk using the UCI Heart Disease dataset, with SHAP and DiCE explainability.

---

## Disclaimer

This application is a **research/educational machine-learning prototype** and is **NOT** a medical diagnostic tool. Predictions should not be used as a substitute for professional medical advice. The model outputs are model-generated predictions and do not constitute clinical recommendations.

---

## Project Overview

ECDSS-Heart is an explainable clinical decision support system for heart disease risk prediction. It combines multiple machine learning models in a stacking ensemble architecture and provides both global (SHAP) and local (DiCE) explainability.

### Research Objective

To develop an interpretable machine learning system that can predict heart disease risk while providing transparent explanations for its predictions, suitable for educational and research purposes.

### Dataset

This project uses the **UCI Heart Disease Dataset**, which combines four databases:

| Database | Source |
|----------|--------|
| Cleveland | Cleveland Clinic Foundation |
| Hungarian | Hungarian Institute of Cardiology |
| Switzerland | University Hospital, Zurich |
| VA Long Beach | VA Medical Center, Long Beach |

**Features (13):** age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope of peak exercise ST segment, number of major vessels, thalassemia.

**Target:** Binary classification (0 = no heart disease, 1 = heart disease).

---

## Methodology

```
UCI Heart Disease Datasets (4 sources)
            ↓
     Data Cleaning
     (type conversion, fixed missing-value placeholders)
            ↓
     Target Conversion
     (binary disease/no-disease target)
            ↓
     Feature Engineering
     (RPP, Thalach/Age Ratio, Log Cholesterol)
            ↓
     Train/Test Split (80/20, stratified)
            ↓
     Preprocessing Pipeline
     (median imputation + standard scaling)
     [Fitted ONLY on training data]
            ↓
     SMOTE-Tomek Resampling
     [Applied ONLY on training data]
            ↓
     Base Learners
     (Random Forest, XGBoost, CatBoost)
            ↓
     Stacking Ensemble
     (Meta-learner: Logistic Regression)
            ↓
     Prediction
            ↓
     Evaluation
     (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve)
            ↓
     Explainability
     (SHAP + DiCE)
```

### Critical Fix: Data Leakage Prevention

The original prototype had a data leakage problem: statistical preprocessing and resampling were applied before the train/test split. This allowed information from the test set to influence model fitting and produced optimistic results.

**How it is fixed in the final pipeline:**

1. Raw UCI data is loaded.
2. Type conversion is applied.
3. The binary target is created.
4. Engineered features are added.
5. Train/test split occurs before any learned preprocessing step.
6. Imputation and scaling are fit only on training data.
7. SMOTE-Tomek resampling is applied only to the training split.
8. The untouched test set is used only for final evaluation.

This preserves the scientific integrity of the evaluation procedure.

---

## Models

| Model | Type | Description |
|-------|------|-------------|
| Random Forest | Base Learner | Ensemble of decision trees with bagging |
| XGBoost | Base Learner | Gradient-boosted decision trees |
| CatBoost | Base Learner | Gradient boosting with categorical feature handling |
| Stacking Ensemble | Meta-Learner | Combines base learners via Logistic Regression |

All models use `random_state=42` for reproducibility.

---

## Explainability

### SHAP (SHapley Additive exPlanations)

- Provides global feature importance
- Explains individual predictions
- Applied to the XGBoost base learner (not the full ensemble) for cleaner visualization
- **Note:** SHAP analysis is performed on the XGBoost component, not the full stacking ensemble

### DiCE (Diverse Counterfactual Explanations)

- Generates counterfactual examples for individual predictions
- Shows which features would need to change for a different prediction
- Counterfactuals are model-generated hypothetical changes, NOT medical advice

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions |
| Precision | Proportion of positive predictions that were correct |
| Recall | Proportion of actual positives correctly identified |
| F1-Score | Harmonic mean of precision and recall |
| ROC-AUC | Area under the Receiver Operating Characteristic curve |

All metrics are calculated from actual model predictions on the untouched test set.

---

## Project Structure

```
ECDSS-Heart/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/                    # Downloaded UCI datasets
│   └── processed/              # Processed data (gitignored)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_models.py
│   │   ├── stacking.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── evaluate.py
│   └── explainability/
│       ├── __init__.py
│       ├── shap_explainer.py
│       └── dice_explainer.py
│
├── models/                     # Saved model files (gitignored)
│   └── .gitkeep
│
├── outputs/
│   ├── figures/                # Generated plots (gitignored)
│   ├── tables/                 # Generated tables (gitignored)
│   └── reports/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   └── templates/
│       └── index.html
│
└── tests/
    ├── test_preprocessing.py
    ├── test_features.py
    └── test_model.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ECDSS-Heart.git
cd ECDSS-Heart

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python -m src.models.train
```

This will:
- Download the UCI Heart Disease datasets
- Apply feature engineering
- Run the preprocessing pipeline (with leakage prevention)
- Train the stacking ensemble
- Save trained models to `models/`

### 2. Run the Streamlit Application

```bash
streamlit run app/app.py
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Run Notebooks

Open the notebooks in the `notebooks/` directory in Jupyter or VS Code.

---

## Limitations

- **Dataset Size:** The combined UCI dataset contains ~920 samples, which is small for deep learning approaches.
- **Dataset Heterogeneity:** The four databases have different collection protocols and may introduce batch effects.
- **Model Limitations:** This is a traditional ML approach, not a deep learning model. Feature engineering is critical.
- **No Clinical Validation:** This prototype has not been validated on external clinical datasets.
- **No Real-World Deployment:** This system has not been deployed in a clinical setting.
- **Pulse Pressure:** True pulse pressure (systolic - diastolic) cannot be calculated from the UCI dataset as diastolic blood pressure is not included. The original prototype's "pulse pressure" calculation was mathematically invalid and has been removed.

---

## Feature Engineering

| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| RPP (Rate Pressure Product) | thalach × trestbps | Indicator of myocardial oxygen consumption |
| Thalach/Age Ratio | thalach / age | Relative cardiac capacity normalized by age |
| Log Cholesterol | ln(chol + 1) | Reduces right-skew of cholesterol distribution |

**Pulse Pressure:** Excluded from the final implementation. The UCI Heart Disease dataset does not contain diastolic blood pressure measurements, so true pulse pressure (systolic - diastolic) cannot be calculated. The original calculation (`trestbps - trestbps.min()`) was a false proxy that subtracted the minimum trestbps value across the dataset, not a valid physiological measure.

---

## Changes from Original Colab Prototype

| Issue | Original | Fixed |
|-------|----------|-------|
| Data Leakage | SMOTE-Tomek applied before train/test split | Resampling applied only to training data after split |
| Pulse Pressure | `trestbps - trestbps.min()` (invalid) | Removed; true pulse pressure requires diastolic BP |
| Manual Metrics | Hardcoded comparison values | All metrics calculated from actual model predictions |
| Simulated ROC | Hand-drawn ROC curves | All ROC curves generated from `predict_proba` |
| Code Structure | Single monolithic Colab notebook | Modular Python package with proper imports |
| Pipeline | No reusable preprocessing pipeline | `sklearn`/`imblearn` pipeline with proper fit/transform |
| Model Serialization | Models not saved | Models saved with `joblib` |
| Testing | No tests | Basic pytest suite |
| Web Application | None | Streamlit app with explainability |

---

## Methodological Problems Discovered

1. **Data Leakage (Critical):** SMOTE-Tomek was applied to the entire dataset before train/test split, causing synthetic samples from the test set to influence training. Fixed by splitting first, then resampling only training data.

2. **Invalid Pulse Pressure:** The calculation `trestbps - trestbps.min()` does not represent pulse pressure. Pulse pressure = systolic - diastolic. Since the UCI dataset lacks diastolic BP, this feature was removed.

3. **Fabricated Metrics:** The model comparison section used hardcoded values (`Accuracy = 0.8895`, `Recall = 0.95`, etc.) and simulated ROC curves rather than computing from actual predictions. Fixed by computing all metrics from model output.

---

## License

MIT License

---

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease Dataset
- scikit-learn, XGBoost, CatBoost, SHAP, DiCE-ML libraries
- Built as a B.Tech CSE/AI-ML final year project