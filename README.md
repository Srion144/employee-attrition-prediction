# Employee Attrition Prediction System

Predicts whether an employee is likely to leave an organization using HR analytics data, and deploys the prediction as an interactive web app.

## Problem
Employee turnover is costly. This project builds a classification model to flag employees at risk of leaving, so HR can intervene proactively.

## Dataset
IBM HR Analytics Employee Attrition & Performance (Kaggle) — 1,470 employee records, 35 attributes.

## Approach
1. Data preprocessing (missing values, duplicates, outlier capping, encoding)
2. Exploratory Data Analysis
3. Feature engineering (TenureRatio)
4. Class balancing using SMOTE
5. Trained 3 models: Logistic Regression, Random Forest, XGBoost
6. Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.789 | 0.394 | 0.596 | 0.475 | 0.797 |
| Random Forest | 0.847 | 0.542 | 0.277 | 0.366 | 0.800 |
| **XGBoost (Final Model)** | **0.864** | **0.652** | 0.319 | **0.429** | **0.812** |

## How to Run
1. Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost streamlit joblib

2. Launch the app:
cd Streamlit_App
streamlit run app.py


3. Fill in employee details in the form and click **Predict Attrition Risk**.

## Project Structure

EmployeeAttritionPrediction_YourName/
├── Dataset/            # Raw CSV data
├── Notebook/            # EDA, preprocessing, model training
├── Model/               # Saved model, scaler, feature columns (.pkl)
├── Streamlit_App/        # Deployed prediction app
├── Documentation/        # Report, PPT, saved charts
└── README.md

## Author
[Your Name]