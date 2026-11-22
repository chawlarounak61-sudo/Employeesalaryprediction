"""
Employee Salary Prediction - Enhanced Model Training & Dataset Generator
Includes: Age, Gender, Education Level, Job Title, Years of Experience, Area
Models: Random Forest Regressor (Ensemble) & Linear Regression (Baseline)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def generate_synthetic_data(num_samples=1000, random_state=42):
    """
    Generate realistic synthetic employee salary dataset in Indian Rupees (INR).
    
    Variables:
    - Age (Numeric, 21 - 62)
    - Gender (Categorical: Male, Female, Other)
    - Education Level (Categorical: High School, Bachelor's, Master's, PhD)
    - Job Title (Categorical: 10 roles across Engineering, Data, Management, & Operations)
    - Years of Experience (Numeric, constrained realistically by age and entry age)
    - Area (Categorical: Bangalore, Mumbai, Delhi-NCR, Hyderabad, Pune, Chennai, Remote / Tier 2)
    - Salary (Target: Annual Salary in INR)
    """
    np.random.seed(random_state)

    job_titles = [
        "Software Engineer",
        "Senior Software Engineer",
        "Data Scientist",
        "Machine Learning Engineer",
        "Product Manager",
        "DevOps Engineer",
        "Data Analyst",
        "Financial Analyst",
        "Engineering Manager",
        "HR Specialist"
    ]
    job_weights = [0.18, 0.14, 0.10, 0.08, 0.08, 0.10, 0.10, 0.08, 0.06, 0.08]

    areas = [
        "Bangalore",
        "Mumbai",
        "Delhi-NCR",
        "Hyderabad",
        "Pune",
        "Chennai",
        "Remote / Tier 2 Cities"
    ]
    area_weights = [0.26, 0.18, 0.18, 0.14, 0.10, 0.08, 0.06]

    genders = ["Female", "Male", "Other"]
    gender_weights = [0.46, 0.50, 0.04]

    education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
    edu_weights = [0.08, 0.54, 0.30, 0.08]

    # Salary baselines by Job Title
    job_salary_bonus = {
        "Engineering Manager": 1250000,
        "Product Manager": 880000,
        "Machine Learning Engineer": 780000,
        "Senior Software Engineer": 680000,
        "Data Scientist": 640000,
        "DevOps Engineer": 460000,
        "Software Engineer": 360000,
        "Financial Analyst": 290000,
        "Data Analyst": 230000,
        "HR Specialist": 150000
    }

    # Minimum experience requirement bias for senior roles
    min_exp_for_role = {
        "Engineering Manager": 8,
        "Senior Software Engineer": 4,
        "Product Manager": 3,
        "Data Scientist": 2,
        "Machine Learning Engineer": 2,
        "DevOps Engineer": 1,
        "Software Engineer": 0,
        "Financial Analyst": 0,
        "Data Analyst": 0,
        "HR Specialist": 0
    }

    # Area location-based market allowances
    area_salary_bonus = {
        "Bangalore": 300000,
        "Mumbai": 280000,
        "Delhi-NCR": 260000,
        "Hyderabad": 190000,
        "Pune": 160000,
        "Chennai": 140000,
        "Remote / Tier 2 Cities": 0
    }

    edu_salary_bonus = {
        "High School": 0,
        "Bachelor's": 180000,
        "Master's": 420000,
        "PhD": 750000
    }

    entry_ages = {
        "High School": 18,
        "Bachelor's": 22,
        "Master's": 24,
        "PhD": 27
    }

    records = []

    for _ in range(num_samples):
        edu = np.random.choice(education_levels, p=edu_weights)
        role = np.random.choice(job_titles, p=job_weights)
        area = np.random.choice(areas, p=area_weights)
        gender = np.random.choice(genders, p=gender_weights)

        min_entry = entry_ages[edu]
        min_exp_role = min_exp_for_role[role]
        
        # Calculate realistic age
        min_age = min_entry + min_exp_role
        age = int(np.random.randint(min_age, 63))
        
        # Calculate realistic experience
        max_exp = max(0, age - min_entry)
        exp = int(np.clip(
            np.random.normal(loc=max(min_exp_role, max_exp * 0.75), scale=2.5),
            min_exp_role,
            max_exp
        ))

        # Base salary + role tier + experience + location + education + variance
        base_salary = 320000
        role_comp = job_salary_bonus[role]
        exp_comp = exp * 92000
        area_comp = area_salary_bonus[area]
        edu_comp = edu_salary_bonus[edu]
        age_maturity = max(0, age - min_age) * 8000
        
        # Realistic market variance (negotiation, company tier, performance)
        market_variance = np.random.normal(0, 240000)

        salary = base_salary + role_comp + exp_comp + area_comp + edu_comp + age_maturity + market_variance
        salary = max(240000, round(salary / 1000.0) * 1000)

        records.append({
            "Age": age,
            "Gender": gender,
            "Education Level": edu,
            "Job Title": role,
            "Years of Experience": exp,
            "Area": area,
            "Salary": salary,
            "Data_Label": "Synthetic Demonstration Data"
        })

    df = pd.DataFrame(records)
    return df


def train_and_evaluate(df, output_dir="."):
    """
    Train Random Forest Regressor & Linear Regression pipelines with all 6 features.
    """
    feature_cols = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Area"]
    X = df[feature_cols]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ["Gender", "Education Level", "Job Title", "Area"]
    numeric_features = ["Age", "Years of Experience"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    # 1. Random Forest Regressor Pipeline
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=150,
                max_depth=14,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)

    rf_r2 = float(r2_score(y_test, rf_pred))
    rf_mae = float(mean_absolute_error(y_test, rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

    # 2. Linear Regression Pipeline
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)

    lr_r2 = float(r2_score(y_test, lr_pred))
    lr_mae = float(mean_absolute_error(y_test, lr_pred))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))

    print("=" * 70)
    print("ENHANCED MODEL EVALUATION ON TEST SET (20% Holdout):")
    print(f"Features Used: {', '.join(feature_cols)}")
    print("-" * 70)
    print(f"Random Forest Regressor (Ensemble):")
    print(f"  * R² Score : {rf_r2:.4f}")
    print(f"  * MAE      : INR {rf_mae:,.2f}")
    print(f"  * RMSE     : INR {rf_rmse:,.2f}")
    print("-" * 70)
    print(f"Linear Regression (Baseline):")
    print(f"  * R² Score : {lr_r2:.4f}")
    print(f"  * MAE      : INR {lr_mae:,.2f}")
    print(f"  * RMSE     : INR {lr_rmse:,.2f}")
    print("=" * 70)

    # Feature Importances extraction
    rf_model = rf_pipeline.named_steps["regressor"]
    preproc = rf_pipeline.named_steps["preprocessor"]
    cat_encoder = preproc.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = cat_feature_names + numeric_features

    raw_importances = rf_model.feature_importances_

    # Detailed one-hot feature importances
    detailed_importance = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(all_feature_names, raw_importances), key=lambda x: x[1], reverse=True)
    }

    # Grouped parent feature importance (Job Title, Experience, Area, Age, Education, Gender)
    grouped_importance = {col: 0.0 for col in feature_cols}
    for name, imp in zip(all_feature_names, raw_importances):
        matched = False
        for parent in categorical_features:
            if name.startswith(parent + "_"):
                grouped_importance[parent] += float(imp)
                matched = True
                break
        if not matched and name in numeric_features:
            grouped_importance[name] += float(imp)

    grouped_importance = {
        k: round(v * 100, 2)
        for k, v in sorted(grouped_importance.items(), key=lambda x: x[1], reverse=True)
    }

    # Save models
    main_model_path = os.path.join(output_dir, "salary_model.joblib")
    joblib.dump(rf_pipeline, main_model_path)
    print(f"Primary model (Random Forest) saved to: {main_model_path}")

    rf_model_path = os.path.join(output_dir, "salary_rf_model.joblib")
    joblib.dump(rf_pipeline, rf_model_path)

    lr_model_path = os.path.join(output_dir, "salary_lr_model.joblib")
    joblib.dump(lr_pipeline, lr_model_path)

    # Metadata & options for UI dropdowns
    metadata = {
        "primary_model": "Random Forest Regressor",
        "r2_score": rf_r2,
        "mae": rf_mae,
        "rmse": rf_rmse,
        "test_samples": len(y_test),
        "random_forest": {
            "r2_score": rf_r2,
            "mae": rf_mae,
            "rmse": rf_rmse
        },
        "linear_regression": {
            "r2_score": lr_r2,
            "mae": lr_mae,
            "rmse": lr_rmse
        },
        "grouped_feature_importances": grouped_importance,
        "detailed_feature_importances": detailed_importance,
        "features": {
            "Age": {"type": "numeric", "min": int(df["Age"].min()), "max": int(df["Age"].max())},
            "Gender": {"type": "categorical", "options": sorted(df["Gender"].unique().tolist())},
            "Education Level": {"type": "categorical", "options": ["High School", "Bachelor's", "Master's", "PhD"]},
            "Job Title": {"type": "categorical", "options": sorted(df["Job Title"].unique().tolist())},
            "Years of Experience": {"type": "numeric", "min": int(df["Years of Experience"].min()), "max": int(df["Years of Experience"].max())},
            "Area": {"type": "categorical", "options": sorted(df["Area"].unique().tolist())}
        }
    }

    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metrics & options saved to: {metrics_path}")

    # Test evaluation dataframe
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Salary"] = y_test
    test_eval_df["Predicted_Salary"] = np.round(rf_pred, 2)
    test_eval_df["LR_Predicted_Salary"] = np.round(lr_pred, 2)
    test_eval_path = os.path.join(output_dir, "test_evaluation.csv")
    test_eval_df.to_csv(test_eval_path, index=False)
    print(f"Test evaluation data saved to: {test_eval_path}")

    # Verification sample
    sample_input = pd.DataFrame([{
        "Age": 28,
        "Gender": "Female",
        "Education Level": "Bachelor's",
        "Job Title": "Machine Learning Engineer",
        "Years of Experience": 5,
        "Area": "Bangalore"
    }])
    sample_rf = rf_pipeline.predict(sample_input)[0]
    sample_lr = lr_pipeline.predict(sample_input)[0]
    print(f"\n[Verification Sample]: Age=28, Female, ML Engineer, 5 yrs exp, Bangalore")
    print(f"-> RF Predicted Salary: INR {sample_rf:,.2f}")
    print(f"-> LR Predicted Salary: INR {sample_lr:,.2f}")

    return rf_pipeline, metadata


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "salary_data.csv")

    print(f"Generating enhanced synthetic dataset (1,000 rows)...")
    dataset = generate_synthetic_data(num_samples=1000, random_state=42)
    dataset.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")

    print("\nTraining Enhanced Random Forest & Linear Regression Pipelines...")
    train_and_evaluate(dataset, output_dir=current_dir)


if __name__ == "__main__":
    main()
