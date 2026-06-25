<div align="center">

# 📊 Data Science & Machine Learning Portfolio

### by Faizan Toheed

[![GitHub](https://img.shields.io/badge/GitHub-faizantoheed456-181717?style=for-the-badge&logo=github)](https://github.com/faizantoheed456)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

*A collection of three end-to-end Data Science projects — from exploratory analysis to classification and regression modeling.*

</div>

---

## 📑 Table of Contents

| # | Project | Type | Key Result |
|---|---------|------|-------------|
| 1 | [Iris Dataset — Exploratory Data Analysis](#-1-iris-dataset-exploratory-data-analysis) | EDA / Visualization | Clear species separation via petal dimensions |
| 2 | [Credit Risk Prediction](#-2-credit-risk-prediction) | Classification | Improved recall on defaults via SMOTE + Random Forest |
| 3 | [Medical Insurance Claim Prediction](#-3-medical-insurance-claim-amount-prediction) | Regression | **R² = 0.8595** with Gradient Boosting |

---

## 🌸 1. Iris Dataset — Exploratory Data Analysis

**Goal:** Explore the classic Iris dataset (150 flowers, 3 species: *setosa*, *versicolor*, *virginica*) to understand feature distributions and relationships ahead of future classification modeling.

### 🔧 Methodology
- **Data Loading** — Loaded and structured the dataset using `pandas`.
- **Interactive Visualization** — Used `Plotly` for fully interactive exploration:
  - 📊 Histograms to study individual feature distributions
  - 🔵 Scatter plots comparing sepal vs. petal dimensions
  - 🧊 3D scatter plots to visualize species separation across multiple dimensions simultaneously

### 💡 Key Insights
- 🌿 **Petal length & width** are by far the strongest features for separating the three species.
- 🎯 Species clusters are **highly distinct** when petal dimensions are plotted together.
- ✅ The dataset is clean and well-structured — an ideal baseline for classification benchmarking.

### ✅ Conclusion
The EDA pinpointed the most discriminative features and produced clear visual evidence of species clustering, laying a solid foundation for predictive modeling.

<details>
<summary><strong>🛠️ Tech Stack</strong></summary>

`Python` · `Pandas` · `Plotly`

</details>

---

## 💳 2. Credit Risk Prediction

**Goal:** Build a predictive model to flag potential loan defaults, balancing overall accuracy with the critical need to catch high-risk applicants.

### 🔧 Methodology

**Phase 1 — Cleaning & EDA**
- 🧹 **Imputation** — Filled missing categorical values with the mode and numerical values with the median.
- ⚙️ **Feature Engineering** — Engineered `TotalIncome` and `Income_to_Loan_Ratio` to strengthen predictive signal.
- 📊 **Visualization** — Migrated to interactive `Plotly` visuals, including 3D plots relating income, loan amount, and loan term.

**Phase 2 — Modeling & Class Imbalance**
- ⚖️ **Class Imbalance Mitigation**
  - Applied `class_weight='balanced'` for Logistic Regression and Decision Tree models.
  - Used **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data.
- 🌲 **Model Selection** — Introduced a **Random Forest Classifier** as a robust ensemble approach.
- 📐 **Evaluation** — Assessed models via precision, recall, and F1-score, with a focus on correctly identifying risky loans.

### 💡 Key Insights
- 🏦 **Credit History** is the single most significant predictor of loan approval.
- 📈 Engineered features (`TotalIncome`, `Income_to_Loan_Ratio`) measurably boosted model performance.
- 🎯 Addressing class imbalance **significantly improved recall** on the minority "Default" class — directly helping the bank catch more high-risk applicants.

### ✅ Conclusion
The final pipeline balances strong overall accuracy with the core business goal: maximizing detection of loan defaults, resulting in a more dependable credit risk assessment tool.

<details>
<summary><strong>🛠️ Tech Stack</strong></summary>

`Python` · `Pandas` · `Scikit-learn` · `imbalanced-learn (SMOTE)` · `Plotly`

</details>

---

## 🏥 3. Medical Insurance Claim Amount Prediction

**Goal:** Predict individual medical insurance claim amounts (`charges`) from personal attributes, using the Medical Cost Personal Dataset (1,338 records), to support actuarial risk assessment and premium pricing.

### 🔧 Methodology

**Phase 1 — Cleaning, Exploration & Visualization**
- 🔍 **Data Inspection** — 1,338 records, 7 columns, **zero missing values**. Categorical: `sex`, `smoker`, `region`. Numerical: `age`, `bmi`, `children`, `charges`.
- 🎯 **Target Analysis** — `charges` is strongly right-skewed: **median ≈ $9,382**, **mean ≈ $13,270**, **max ≈ $63,770**.
- 📊 **Bivariate & Multidimensional Visualization:**
  - 🚬 **Smoking Impact:** Non-smokers average **$8,434**, smokers average **$32,050** — nearly a **4× increase**.
  - 📈 **Age vs. Charges:** Clear upward trend, split into three distinct bands tied to smoking status.
  - ⚖️ **BMI vs. Charges:** Minor effect for non-smokers, but for smokers with BMI ≥ 30, charges regularly exceed **$35,000**.
  - 🧊 **Multidimensional Joint Plot:** Age × BMI, colored by smoker status, sized by claim amount — revealing compounding risk.

**Phase 2 — Preprocessing & Baseline Modeling**
- 🔢 **Encoding** — Binary-mapped `sex`/`smoker`; one-hot encoded `region` (dropped `northeast` to avoid the dummy variable trap).
- 🔗 **Correlation Analysis** — `smoker_yes` is the strongest linear correlate (**r = 0.79**), followed by `age` (r = 0.30) and `bmi` (r = 0.20).
- ✂️ **Train/Test Split** — 80/20 split (1,070 / 268 records), `random_state=42` for reproducibility.
- 📏 **Baseline Model** — Ordinary Least Squares (OLS) Linear Regression with residual diagnostics.

**Phase 3 — Advanced Modeling & Optimization**

Residual diagnostics on the baseline exposed two OLS violations: a **non-linear BMI–smoking interaction** and **heteroscedasticity**. To address these:
- ✖️ **Interaction Feature** — Created `bmi_x_smoker`, collapsing to 0 for non-smokers and capturing the amplified BMI effect for smokers.
- 📉 **Log-Transformed Target** — Applied `np.log1p` to compress the right tail and stabilize residual variance; predictions inverted via `np.expm1` before evaluation.
- 🌳 **Gradient Boosting Regressor** — A non-linear benchmark model that naturally captures interaction effects without manual transformation.

### 📊 Model Performance Comparison

| Model | MAE ($) | RMSE ($) | R² Score |
|---|---|---|---|
| Baseline OLS (no interaction, raw target) | $4,181.19 | $5,796.28 | 0.7836 |
| Improved OLS (`bmi_x_smoker` + log target) | $3,974.49 | $8,559.98 | 0.5280 |
| **🏆 Gradient Boosting (benchmark)** | **$2,590.12** | **$4,670.90** | **0.8595** |

> **Note on the log-transform trade-off:** `np.log1p` reduced MAE by compressing the right tail, but the `np.expm1` inversion amplified absolute errors on extreme high-value predictions — inflating RMSE and depressing R² for the Improved OLS model. Gradient Boosting handles these skewed, interaction-heavy relationships natively, delivering the best overall balance of low error and high explained variance.

### 💡 Key Insights
- 🚬 **Smoking is the dominant driver** — being a smoker adds an average of **$23,651.13** to baseline charges, holding all else constant.
- 📐 **Demographic progression matters:** +$337.09 per BMI unit, +$256.98 per year of age, +$425.28 per additional child.
- 🔥 **The BMI × Smoking interaction is critical** — `bmi_x_smoker` accounts for **74.56%** of Gradient Boosting's total feature importance.
- 🚻 **Gender & region are nearly negligible** — male vs. female differs by only -$18.59; regional adjustments range from -$370.68 to -$809.80 vs. the northeast baseline.
- 🌳 **Non-linear models win** — Gradient Boosting achieved **R² = 85.95%** and **MAE = $2,590.12**, confirming tree-based models suit complex healthcare cost structures far better than linear approaches.

### ✅ Conclusion
A complete predictive pipeline for medical insurance claims was developed, progressing from a solid OLS baseline (R² = 0.7836) to a final **Gradient Boosting Regressor** that explains **85.95% of variance** with a low MAE of **$2,590.12** — by correctly modeling the non-linear interaction between smoking and body weight. The result is a robust, data-backed tool for premium calculation and medical cost estimation.

<details>
<summary><strong>🛠️ Tech Stack</strong></summary>

`Python` · `Pandas` · `NumPy` · `Statsmodels / Scikit-learn` · `Gradient Boosting` · `Plotly`

</details>

---

## 🧰 Overall Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

## 📂 Repository Structure

```
.
├── Task_1_Iris_EDA/
│   └── Task_1_Iris_EDA_Report.docx
├── Task_2_Credit_Risk_Prediction/
│   └── Task_2_Credit_Risk_Prediction_Report.docx
├── Task_3_Insurance_Claim_Prediction/
│   └── Task_3_Insurance_Claim_Prediction_Report.docx
└── README.md
```

> 💡 *Update the folder/file names above to match your actual repository layout (e.g., if you add notebooks or `.py` scripts alongside the reports).*

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/faizantoheed456/<your-repo-name>.git
cd <your-repo-name>

# Install common dependencies
pip install pandas numpy scikit-learn plotly imbalanced-learn statsmodels
```

## 📬 Contact

<div align="center">

**Faizan Toheed**

[![GitHub](https://img.shields.io/badge/GitHub-faizantoheed456-181717?style=flat-square&logo=github)](https://github.com/faizantoheed456)

⭐ If you found this portfolio useful or interesting, consider giving it a star!

</div>
