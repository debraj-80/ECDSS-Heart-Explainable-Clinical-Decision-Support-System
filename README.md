# 🩺 ECDSS-Heart: Explainable Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Clinical Safety](https://img.shields.io/badge/Recall-0.95-red.svg)](#-performance-metrics)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20%2F%20DiCE-success.svg)](#-explainability-layer)

## 📌 Executive Summary
**ECDSS-Heart** is a next-generation clinical framework designed to bridge the gap between "black-box" machine learning and medical trust. While traditional models often prioritize accuracy over interpretability, this system utilizes a **Heterogeneous Stacking Ensemble** integrated with **Explainable AI (XAI)** to provide transparent, high-sensitivity heart disease predictions.

---

## 🚀 Performance Metrics
This project was engineered with a **"safety-first"** clinical mindset. In medical screening, a false negative is significantly more dangerous than a false positive; hence, this model maximizes **Recall**.

| Metric | Result | Industry Context |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **0.95** | Successfully identifies 95% of high-risk heart disease patients. |
| **Accuracy** | **88.95%** | High overall diagnostic reliability across 4 international cohorts. |
| **ROC AUC** | **0.95** | Superior diagnostic sensitivity and class separation capability. |

### **Full Performance Benchmarking**
The following chart illustrates how the Stacking Ensemble consistently outperforms individual base learners across all critical metrics:

<img width="846" height="521" alt="comparision" src="https://github.com/user-attachments/assets/b53f8f69-37f9-4698-9874-24a237583d75" />

---

## 🏗️ Architecture & Engineering Complexity
The system employs a sophisticated multi-tier pipeline to ensure data robustness and model generalization.

### 1. Stacking Ensemble Strategy
The core of the system is a two-tier architecture using a heterogeneous mix of **Random Forest**, **XGBoost**, and **CatBoost** feeding into a **Logistic Regression Meta-Learner**.

<img width="1187" height="690" alt="a4d5c4f9-1861-431b-b2ee-e50280b89afd" src="https://github.com/user-attachments/assets/a05e8c7b-0d41-44af-be61-888526427e75" />


### 2. Diagnostic Reliability (ROC Curve)
The ROC analysis confirms the ensemble's superior capability in discriminating between healthy and high-risk states compared to standard models.

<img width="686" height="542" alt="796482b8-2028-412b-b26e-5aaa431ecece" src="https://github.com/user-attachments/assets/5b08fdb7-3f62-41d5-a095-68414e064734" />


---

## 📊 Individual Parameter Deep-Dive
To satisfy clinical requirements, each parameter was analyzed separately to ensure the model does not sacrifice one metric for another.


|<img width="678" height="484" alt="accuracy" src="https://github.com/user-attachments/assets/23f62abf-05e4-43e1-b5fb-c289cf7e7ee4" />
|<img width="678" height="484" alt="recall" src="https://github.com/user-attachments/assets/31788854-c45a-4199-be59-d11edf9251b0" />
|<img width="678" height="484" alt="precision" src="https://github.com/user-attachments/assets/1ecb6b31-52aa-46df-867d-3db1ee49dba6" />
|<img width="678" height="484" alt="F1score" src="https://github.com/user-attachments/assets/6798658a-e332-42f4-9529-63e9b1d5a427" />


---

## 🔍 Explainability Layer (XAI)
In the 2026 healthcare landscape, interpretability is a regulatory requirement. ECDSS-Heart moves from purely predictive to **prescriptive**.

* **Global Interpretability (SHAP)**: Provides a transparent view of which clinical features are driving global decisions.
* **Actionable Insights (DiCE)**: Implements **Diverse Counterfactual Explanations** to tell clinicians exactly what minimal physiological changes would shift a patient from "High Risk" to "Safe".

---

## 🛠️ Technical Stack
* **Languages**: Python 3.9+
* **ML Frameworks**: Scikit-Learn, XGBoost, CatBoost
* **XAI Frameworks**: SHAP, DiCE (Diverse Counterfactual Explanations)
* **Data Processing**: Pandas, NumPy, SMOTE-Tomek
---

## 📂 Dataset Architecture & Diversity
To ensure the framework is globally applicable and to mitigate population bias often found in single-center studies, this project utilizes an integrated repository from the **UCI Machine Learning Repository**.

### **Data Integration Pipeline**
* **Multi-National Cohorts**: Aggregated four primary heart disease datasets—Cleveland, Hungarian, Switzerland, and VA Long Beach.
* **Cohort Representation**: The merged dataset provides a diverse view of cardiovascular health across international boundaries.
* **Missing Data Strategy**: Utilized **Median Imputation** for physiological measurements to preserve the central tendency and distribution integrity of the clinical data.

| Dataset Source | Focus Area | Access Link |
| :--- | :--- | :--- |
| **Cleveland** | Primary Clinical Data | [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Hungarian** | Regional Diagnostics | [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Switzerland** | European Cohort | [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **VA Long Beach** | US Veteran Data | [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+disease) |

### **Feature Engineering & Selection**
* **Domain-Driven Feature**: Introduced the **Rate Pressure Product (RPP)** as a clinical proxy for myocardial stress.
* **Predictor Validation**: SHAP analysis confirmed the engineered RPP feature as a **top-5 predictor** of cardiac risk.



---

## ⚙️ Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/yourusername/ECDSS-Heart.git](https://github.com/yourusername/ECDSS-Heart.git)

# Install dependencies
pip install -r requirements.txt

# Run the training and explanation pipeline
python main.py



## ⚖️ License & Copyright
Copyright (c) 2026 [Your Name]

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for academic and research purposes, provided that proper credit is given to the original author and the ECDSS-Heart framework.

---

## ✍️ How to Cite
If you use this framework or the engineered features (RPP) in your research, please cite:

> [Your Name], "ECDSS-Heart: An Explainable Clinical Decision Support System for Heart Disease Prediction Using Heterogeneous Stacking Ensembles," 2026.
