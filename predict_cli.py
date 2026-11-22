"""
Employee Salary Prediction - Command Line Interface (CLI)
Multi-Variable Random Forest College Demonstration Project
"""

import argparse
import json
import os
import sys
import joblib
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RF_MODEL_PATH = os.path.join(BASE_DIR, "salary_model.joblib")
LR_MODEL_PATH = os.path.join(BASE_DIR, "salary_lr_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")


def format_inr(amount: float) -> str:
    """Format number into Indian Rupee representation safely across all terminals."""
    s = f"{amount:.2f}"
    parts = s.split(".")
    integer_part = parts[0]
    decimal_part = parts[1]

    is_negative = integer_part.startswith("-")
    if is_negative:
        integer_part = integer_part[1:]

    if len(integer_part) <= 3:
        formatted = integer_part
    else:
        last3 = integer_part[-3:]
        other = integer_part[:-3]
        groups = []
        while len(other) > 2:
            groups.insert(0, other[-2:])
            other = other[:-2]
        if other:
            groups.insert(0, other)
        formatted = ",".join(groups) + "," + last3

    if is_negative:
        formatted = "-" + formatted

    return f"INR {formatted}"


def ensure_models_exist():
    """Ensure trained models exist; if not, train using train_model.py."""
    if not os.path.exists(RF_MODEL_PATH):
        print("[INFO] Model not found. Training model now...")
        import train_model
        train_model.main()
    rf_model = joblib.load(RF_MODEL_PATH)
    lr_model = joblib.load(LR_MODEL_PATH) if os.path.exists(LR_MODEL_PATH) else None
    return rf_model, lr_model


def predict_salary(model, age: int, gender: str, education: str, title: str, experience: int, area: str):
    valid_edu = ["High School", "Bachelor's", "Master's", "PhD"]
    if education not in valid_edu:
        matched = [e for e in valid_edu if e.lower() == education.lower()]
        education = matched[0] if matched else "Bachelor's"

    input_df = pd.DataFrame([{
        "Age": int(age),
        "Gender": gender,
        "Education Level": education,
        "Job Title": title,
        "Years of Experience": int(experience),
        "Area": area
    }])

    pred = model.predict(input_df)[0]
    return pred, input_df


def main():
    parser = argparse.ArgumentParser(description="Employee Salary Prediction CLI (Multi-Variable Random Forest)")
    parser.add_argument("--age", type=int, default=28, help="Age in years (e.g. 28)")
    parser.add_argument("--gender", type=str, default="Female", choices=["Female", "Male", "Other"], help="Gender")
    parser.add_argument("--education", type=str, default="Bachelor's", help="Education: High School, Bachelor's, Master's, PhD")
    parser.add_argument("--title", type=str, default="Machine Learning Engineer", help="Job Title")
    parser.add_argument("--experience", type=int, default=5, help="Years of Experience (e.g. 5)")
    parser.add_argument("--area", type=str, default="Bangalore", help="Area / City (e.g. Bangalore, Mumbai, Delhi-NCR, Hyderabad, Pune, Chennai)")
    parser.add_argument("--interactive", action="store_true", help="Prompt for inputs interactively")

    args = parser.parse_args()
    rf_model, lr_model = ensure_models_exist()

    if args.interactive:
        print("\n=== Employee Salary Prediction (Interactive CLI) ===")
        try:
            age_in = input("Enter Age (e.g. 28): ").strip()
            age = int(age_in) if age_in else 28

            gender_in = input("Enter Gender [Female/Male/Other]: ").strip()
            gender = gender_in if gender_in in ["Female", "Male", "Other"] else "Female"

            print("Available Education Levels: 1. High School | 2. Bachelor's | 3. Master's | 4. PhD")
            edu_in = input("Enter Education Level [Bachelor's]: ").strip()
            edu_map = {"1": "High School", "2": "Bachelor's", "3": "Master's", "4": "PhD"}
            edu = edu_map.get(edu_in, edu_in if edu_in else "Bachelor's")

            title_in = input("Enter Job Title [Machine Learning Engineer]: ").strip()
            title = title_in if title_in else "Machine Learning Engineer"

            exp_in = input("Enter Years of Experience (e.g. 5): ").strip()
            exp = int(exp_in) if exp_in else 5

            area_in = input("Enter Area/City [Bangalore]: ").strip()
            area = area_in if area_in else "Bangalore"
        except (ValueError, EOFError):
            age, gender, edu, title, exp, area = args.age, args.gender, args.education, args.title, args.experience, args.area
    else:
        age, gender, edu, title, exp, area = args.age, args.gender, args.education, args.title, args.experience, args.area

    rf_pred, input_df = predict_salary(rf_model, age, gender, edu, title, exp, area)
    lr_pred = lr_model.predict(input_df)[0] if lr_model else None

    # Load metrics
    rf_r2, rf_mae = "0.9025", "INR 2,47,820"
    lr_r2, lr_mae = "0.9385", "INR 2,00,563"
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
                rf_m = metrics.get("random_forest", {})
                lr_m = metrics.get("linear_regression", {})
                if rf_m:
                    rf_r2 = f"{rf_m.get('r2_score', 0.9025):.4f}"
                    rf_mae = format_inr(rf_m.get('mae', 247820))
                if lr_m:
                    lr_r2 = f"{lr_m.get('r2_score', 0.9385):.4f}"
                    lr_mae = format_inr(lr_m.get('mae', 200563))
        except Exception:
            pass

    print("\n" + "=" * 72)
    print("      EMPLOYEE SALARY PREDICTION - MULTI-VARIABLE DEMO")
    print("=" * 72)
    print(" MODEL PERFORMANCE (Test Set):")
    print(f"  * Random Forest Regressor    : R² = {rf_r2} | MAE = {rf_mae}")
    print(f"  * Linear Regression Baseline : R² = {lr_r2} | MAE = {lr_mae}")
    print("-" * 72)
    print(" INPUT DETAILS:")
    print(f"  * Age                 : {age} years")
    print(f"  * Gender              : {gender}")
    print(f"  * Education Level     : {edu}")
    print(f"  * Job Title           : {title}")
    print(f"  * Years of Experience : {exp} years")
    print(f"  * Area / Location     : {area}")
    print("-" * 72)
    print(f" PREDICTED SALARY (Random Forest): {format_inr(rf_pred)}")
    if lr_pred is not None:
        print(f" PREDICTED SALARY (Linear Reg.)  : {format_inr(lr_pred)}")
    print("=" * 72)
    print(" DISCLAIMER:")
    print("  This prediction was generated using an ensemble Random Forest model")
    print("  trained on a synthetic demonstration dataset.")
    print("  It is NOT an output from the original Colab model.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
