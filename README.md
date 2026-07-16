# Fruit-Intelligent-system
Fruit Waste Intelligence System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

**Food Waste Intelligence System (FWIS)** is an AI-powered diagnostic engine designed to intercept and audit inbound agricultural supply chain manifests before they hit distribution bays. By evaluating environmental transit telemetry, the system dynamically predicts **expected batch waste percentage**, estimates **remaining shelf life (RSL)**, and classifies the **primary physiological degradation risk**, triggering real-time automated operational routing logic.

---

## 📊 Core Engine Performance

The system utilizes an ensemble of optimized Scikit-Learn models trained on complex simulated physiological strain vectors (incorporating thermal stress, desiccation, and packaging vulnerabilities).

### Model Metrics:
*   **Batch Waste Regression Engine:** `MAE: 3.14%` | `R²: 0.9678`
*   **Remaining Shelf Life (RSL) Engine:** `MAE: 0.63 Days` | `R²: 0.9968`
*   **Degradation Classification Engine:** `91% Global Accuracy`

### 🧬 Classification Suite Precision Breakdown:
| Degradation Risk Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Pathogen / Rot (Heat Stress)** | 0.93 | 0.96 | 0.94 |
| **Chilling Injury (Cold Stress)** | 0.95 | 0.91 | 0.93 |
| **Mechanical Bruising** | 0.92 | 0.92 | 0.92 |
| **Dehydration / Shrinkage** | 0.88 | 0.83 | 0.86 |
| **Normal Senescence** | 0.84 | 0.80 | 0.82 |

---

## 🛠️ Feature Engineering Pipeline

The model expands raw transit metadata into advanced bio-meteorological indicators:
*   **`Transit_Days`**: Calculates underlying dynamic transit durations: $\text{Distance} / 450.0 + \text{Unplanned Delays}$.
*   **`Degree_Days`**: Reflects cumulative thermal energy and respiratory acceleration exposure ($\text{Avg Temp} \times \text{Transit Days}$).
*   **`VPD_Proxy`**: Acts as a simplified Vapor Pressure Deficit indicator capturing moisture loss gradients ($(100 - \text{RH}\%) \times \text{Avg Temp}$).

---

## 🚀 Repository Structure

```text
├── app.py                  # Streamlit Interactive Web Application
├── model_waste.pkl         # Trained Random Forest Regressor (Waste %)
├── model_rsl.pkl           # Trained Random Forest Regressor (Shelf Life)
├── model_reason.pkl        # Trained Random Forest Classifier (Risk Reason)
├── columns_layout.pkl      # Saved OHE training column schema pipeline
├── requirements.txt        # Package dependencies
└── README.md               # Repository Documentation
