# 💼 SalaryIQ — Enterprise Employee Salary Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Visualizations](https://img.shields.io/badge/Charts-Plotly-purple.svg)](https://plotly.com/)

A machine learning application that predicts annual employee compensation in Indian Rupees (INR) based on demographic, educational, professional, and regional factors. Featuring an ensemble **Random Forest Regressor** alongside an Ordinary Least Squares (OLS) **Linear Regression baseline**, packaged in an executive SaaS dashboard.

---

## 🌟 Key Features

* **Multi-Variable ML Architecture**:
  * **Age** (Numerical)
  * **Gender** (`Female`, `Male`, `Other`)
  * **Education Level** (`High School`, `Bachelor's`, `Master's`, `PhD`)
  * **Job Title** (10 Technical, Analytical, & Managerial roles)
  * **Years of Experience** (Numerical)
  * **Area / Location** (7 Metropolitan & Tech Hub locations across India)
* **High-Precision Predictive Models**:
  * **Primary Model:** Random Forest Regressor (150 Decision Trees, $R^2 \approx 0.9025$)
  * **Baseline Model:** Linear Regression ($R^2 \approx 0.9385$)
  * **Evaluation Metrics:** Honest holdout evaluation ($R^2$, MAE, RMSE) calculated on a 20% test split (`random_state=42`).
* **SaaS Analytics Dashboard**:
  * **Interactive Salary Estimator**: Real-time compensation forecasts, estimated monthly take-home, and market percentile standing.
  * **Comparative Mode**: Live side-by-side delta between Random Forest and Linear Regression.
  * **Feature Importance Analytics**: Visual breakdown of relative feature contributions across trees.
  * **Executive Market Analytics**: Regional pay distributions, role medians, and experience progression curves.
  * **Dataset Explorer**: Filterable records viewer with 1-click CSV export.
* **Dual Interface**: Full web application via Streamlit + crash-proof Command Line Interface (`predict_cli.py`).

---

## 📊 Model Architecture & Performance

Data preprocessing utilizes `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` for categorical dimensions and passthrough for numerical dimensions:

| Metric | Random Forest Regressor | Linear Regression (Baseline) |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9025** | **0.9385** |
| **Mean Absolute Error (MAE)** | **INR 2,47,820** | **INR 2,00,563** |
| **Root Mean Squared Error (RMSE)**| **INR 3,05,972** | **INR 2,42,884** |
| **Model Type** | 150 Decision Trees (`max_depth=14`) | Parametric Ordinary Least Squares |

### Relative Feature Contribution (Random Forest)
1. **Years of Experience:** 65.47%
2. **Age:** 24.07%
3. **Job Title:** 6.54%
4. **Education Level:** 1.97%
5. **Area (Location):** 1.52%
6. **Gender:** 0.44%

---

## 📁 Repository Structure

```text
├── app.py                      # Enterprise Streamlit web application
├── train_model.py              # Synthetic data generator & model training pipeline
├── predict_cli.py              # Standalone Command Line Interface (CLI)
├── requirements.txt            # Python dependencies
├── salary_data.csv             # 1,000-sample demonstration dataset
├── salary_model.joblib         # Serialized Scikit-Learn Random Forest pipeline
├── salary_lr_model.joblib      # Serialized Scikit-Learn Linear Regression pipeline
├── model_metrics.json          # Precomputed metrics and feature importances
├── test_evaluation.csv         # Holdout test set actual vs predicted values
├── run_app.bat                 # One-click Windows runner script
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 4. Run via Command Line Interface (CLI)
```bash
# Specific inputs:
python predict_cli.py --age 28 --gender Female --education "Bachelor's" --title "Machine Learning Engineer" --experience 5 --area "Bangalore"

# Interactive prompt:
python predict_cli.py --interactive
```

---

## ⚠️ Academic & Demonstration Notice
This project uses a synthetic demonstration dataset and machine learning pipelines developed for college / academic demonstration purposes. All predictions and evaluations reflect this demonstration dataset.
