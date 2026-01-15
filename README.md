# 🩺 ECDSS-Heart: Explainable Clinical Decision Support System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Clinical Safety](https://img.shields.io/badge/Recall-0.95-red.svg)](#-performance-metrics)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20%2F%20DiCE-success.svg)](#-explainability-layer)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

<img width="846" height="521" alt="comparision" src="https://github.com/user-attachments/assets/9a762354-10c1-4039-b030-aaac20297395" />


---

## 🏗️ Architecture & Engineering Complexity
The system employs a sophisticated multi-tier pipeline to ensure data robustness and model generalization.

### 1. Stacking Ensemble Strategy
The core of the system is a two-tier architecture using a heterogeneous mix of **Random Forest**, **XGBoost**, and **CatBoost** feeding into a **Logistic Regression Meta-Learner**.

<img width="1187" height="690" alt="a4d5c4f9-1861-431b-b2ee-e50280b89afd" src="https://github.com/user-attachments/assets/f6f93a76-7914-4061-bdc6-6b575399f699" />

### 2. Diagnostic Reliability (ROC Curve)
The ROC analysis confirms the ensemble's superior capability in discriminating between healthy and high-risk states compared to standard models.

<img width="686" height="542" alt="796482b8-2028-412b-b26e-5aaa431ecece" src="https://github.com/user-attachments/assets/130e5233-a065-4a07-b83e-cad042bbc0aa" />



---

## 📂 Dataset Architecture
To ensure the framework is globally applicable, this project utilizes an integrated repository from the **UCI Machine Learning Repository**, aggregating four primary cohorts: **Cleveland**, **Hungarian**, **Switzerland**, and **VA Long Beach**.



---

## ⚖️ License & Copyright
Copyright (c) 2026 **Debraj Saha**

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software for academic and research purposes, provided that proper credit is given to the original author and the **ECDSS-Heart** framework.

---

## ✍️ How to Cite
If you utilize this framework or the engineered **Rate Pressure Product (RPP)** features in your research, please use the following citation:

> **Saha, D.** (2026). *ECDSS-Heart: An Explainable Clinical Decision Support System for Heart Disease Prediction Using Heterogeneous Stacking Ensembles*. GitHub Repository: [https://github.com/debraj-80/ECDSS-Heart](https://github.com/debraj-80/ECDSS-Heart-Explainable-Clinical-Decision-Support-System)
