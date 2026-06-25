# Task 2: Credit Risk Prediction Report

## Overview
The goal of this project was to develop a predictive model to identify potential loan defaults. The project was divided into two main phases: data cleaning/EDA, and model training/evaluation.

## Methodology

### Phase 1: Cleaning and Exploratory Data Analysis (EDA)
1.  **Imputation**: Addressed missing values in the dataset by filling categorical columns with the mode and numerical columns with the median.
2.  **Feature Engineering**: Created new metrics, including `TotalIncome` and `Income_to_Loan_Ratio`, to provide better predictive features.
3.  **Visualization**: Migrated visualizations to **Plotly** for interactive data exploration, including 3D plots to analyze multidimensional relationships between income, loan amount, and loan terms.

### Phase 2: Modeling and Addressing Class Imbalance
1.  **Class Imbalance Mitigation**:
    *   Implemented `class_weight='balanced'` for Logistic Regression and Decision Tree models to improve sensitivity to the minority 'Risk/Default' class.
    *   Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data.
2.  **Model Selection**: Introduced a **Random Forest Classifier** as a robust ensemble model to optimize overall performance and class recall.
3.  **Evaluation**: Evaluated models using comprehensive classification reports focusing on precision, recall, and F1-score to ensure proper identification of risky loans.

## Key Insights
*   **Credit History** is the most significant predictor of loan approval.
*   Feature engineering (`TotalIncome`, `Income_to_Loan_Ratio`) improved model capability.
*   Addressing class imbalance significantly improved the recall for the 'Default' class, ensuring the bank identifies more potential high-risk applicants.

## Conclusion
The modeling pipeline successfully balances high accuracy with the essential goal of maximizing the detection of loan defaults, providing a more reliable tool for credit risk assessment.
