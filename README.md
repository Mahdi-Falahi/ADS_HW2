# Introduction to Data Science (HW2) — Ames Housing Analytics

This repository contains the complete end-to-end data science pipeline developed for Introduction to Data Science Homework 2. Utilizing the **Ames Housing Dataset**, the project progresses through systematic data cleaning, outlier mitigation, high-dimensional feature engineering, and automated feature selection via Mutual Information. The final processed data is used to evaluate a comprehensive suite of machine learning algorithms across three primary paradigms: Continuous Regression, Binary Classification, and Multiclass Classification.

---

## Author Information

* **Name:** Mahdi Falahi
* **Student ID:** 402102238
* **Course:** Introduction to Data Science (HW2)

---

## Repository Structure

```text
├── AmesHousing.csv              # Pristine raw dataset from Kaggle
├── new cleaner.py               # Automated data cleaning, feature engineering, and selection script
├── AmesHousing_MI_applied.csv   # Post-processed dataset containing selected features
├── Assignment 2.ipynb           # Core Jupyter notebook with full modeling experiments and theoretical answers
└── README.md                    # Project documentation

```

---

## Workflow Breakdown

The project architecture is structured into two main decoupled operational stages:

### 1. Data Cleaning & Feature Engineering (`new cleaner.py`)

Before passing data to any learning model, this script processes the raw dataset to ensure stability and mathematical consistency:

* **Imputation:** Missing categorical entries are designated as `'None'`. Numeric missingness is filled with `0`, while spatial gaps (such as `Lot Frontage`) are dynamically imputed using the median values of their respective neighborhoods.
* **Feature Generation:** Constructs core domain-specific attributes including total square footage (`Total_SF`), geometric interaction items (`OverallQual2`, `GrLivArea2`), asset aging indicators (`House_Age`, `Remodel_Age`, `Garage_Age`), total structural bathrooms (`Total_Bathrooms`), and parking space per vehicle.
* **Outlier Mitigation & Transformations:** Continuous features with heavy right-hand tails are Winsorized at the 99th percentile. Highly skewed target variables (such as `SalePrice`) undergo a stabilizing log-transformation ($log(1+x)$) to ensure compliance with regression assumptions.
* **Encoding & Scaling:** Nominal variables are expanded via low-dimensional one-hot encoding, and ordinal criteria are mapped directly to zero-indexed integer systems. Continuous and engineered elements are normalized using a standard scaling technique.
* **Feature Selection:** Computes **Mutual Information (MI) Regression** scores across the entire feature space. Elements failing to exceed a variance relevancy threshold ($\text{MI} > 0.05$) are dropped, optimizing the workspace down to the highest-scoring features saved in `AmesHousing_MI_applied.csv`.

### 2. Modeling & Evaluation Pipeline (`Assignment 2.ipynb`)

The processed data is ingested by the primary notebook environment, divided into four dedicated analytical sections:

#### Part 1: Advanced Regression Methods

* **Objective Function Optimization:** Models are optimized and monitored via a custom evaluation scorer measuring **Mean Absolute Percentage Error (MAPE)** transformed back onto the original dollar scale ($np.expm1$) during K-Fold Cross-Validation.
* **Algorithms Evaluated:** Evaluates an Ordinary Least Squares (OLS) baseline against regularized continuous linear frameworks (Ridge, Lasso, ElasticNet) and non-linear Support Vector Regression (SVR with Radial Basis Function kernel).
* **Parsimony Verification:** Investigates structural shrinkage parameters inside Lasso to analyze automatic feature pruning and identify key predictors like total space metrics and physical material scores.

#### Part 2: Binary Classification Frameworks

* **Target Transformation:** The continuous housing cost profile is split at its exact sample median, creating a balanced classification objective: `Is_Expensive`.
* **Algorithms Evaluated:** Trains and bench-tests Logistic Regression, K-Nearest Neighbors (KNN), Linear Support Vector Machines (Linear SVM), Kernel Support Vector Machines (RBF SVM), Decision Trees, and Random Forest Classifiers.

#### Part 3: Multiclass Classification & Advanced Ensembles

* **Target Discretization:** The pricing architecture is segmented into four distinct categorical tiers based on population quartiles: `Budget`, `Standard`, `Premium`, and `Luxury`.
* **Feature Enhancements:** To handle boundary blending between adjacent categories, cross-interaction variables are introduced (`SF_per_Qual`, `Age_Condition_Ratio`, `Location_Prestige_Index`).
* **Algorithms Evaluated:** Evaluates One-Vs-Rest (OVR) and Multinomial Logistic Regressions, Decision Trees, Random Forests, XGBoost, LightGBM, and CatBoost models.
* **Ensemble Synthesis:** Combines the top individual classifiers into a **Blended Soft-Voting Meta-Ensemble** (fusing RBF SVM and CatBoost probabilities) to minimize classification tier errors.

#### Part 4: Theoretical Explanations

* Contains comprehensive academic explanations addressing fundamental machine learning principles, including the bias-variance trade-off, the mathematics of the kernel trick, structural variances between $L_1$ and $L_2$ regularizations, multi-class operational differences (OVR vs. Multinomial), decision boundaries across algorithmic families, tree pruning mechanics, and loss metric optimization under class imbalance scenarios.

---

## Performance Summary

### 1. Regression Leaderboard (Sorted by Cross-Validated Test MAPE)

Regularized linear models demonstrated strong generalization, while unconstrained non-linear SVR exhibited overfitting trends.

| Model Hierarchy | Train $R^2$ (Log) | Test $R^2$ (Log) | Test MAE ($) | Test MAPE (%) |
| --- | --- | --- | --- | --- |
| **ElasticNet** | 0.8979 | 0.9248 | $15,688 | **8.33%** |
| **Ridge ($L_2$)** | 0.8984 | 0.9248 | $15,605 | **8.35%** |
| **Lasso ($L_1$)** | 0.8982 | 0.9243 | $15,769 | **8.36%** |
| OLS Baseline | 0.9012 | 0.9201 | $16,482 | 8.62% |
| SVR (RBF Kernel) | 0.9816 | 0.9079 | $16,382 | 8.74% |

### 2. Binary Classification Metrics

Models evaluated on the balanced median split yielded highly accurate classification boundaries.

| Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Random Forest** | **93.13%** | 95.06% | 90.91% | **92.94%** | **0.9810** |
| Kernel SVM (RBF) | 92.59% | 95.35% | 89.45% | 92.31% | 0.9799 |
| Logistic Regression | 92.04% | 95.29% | 88.36% | 91.70% | 0.9787 |
| Linear SVM | 91.86% | 94.23% | 89.09% | 91.59% | 0.9773 |
| K-Nearest Neighbors | 90.60% | 94.42% | 86.18% | 90.11% | 0.9716 |
| Decision Tree | 89.51% | 90.04% | 88.73% | 89.38% | 0.9526 |

### 3. Multiclass Classification Metrics (Sorted by Out-of-Sample Accuracy)

The ensemble strategy achieved the highest classification performance across the four pricing categories.

| Model Architecture | Test Accuracy | F1 Micro | F1 Macro | F1 Weighted | Avg Tier Shift |
| --- | --- | --- | --- | --- | --- |
| **Blended Meta-Ensemble** | **79.57%** | **79.57%** | **79.69%** | **79.71%** | **0.2061** |
| Kernel SVM (RBF) | 79.02% | 79.02% | 79.11% | 79.12% | 0.2152 |
| XGBoost Classifier | 78.66% | 78.66% | 78.77% | 78.79% | 0.2188 |
| LightGBM Classifier | 78.48% | 78.48% | 78.63% | 78.65% | 0.2206 |
| Linear SVM (OVR) | 77.22% | 77.22% | 77.31% | 77.33% | 0.2333 |
| CatBoost Classifier | 77.22% | 77.22% | 77.31% | 77.32% | 0.2297 |
| Logistic Regression | 75.95% | 75.95% | 76.03% | 76.05% | 0.2477 |
| Decision Tree | 73.24% | 73.24% | 73.63% | 73.63% | 0.2785 |

---

## Setup and Execution Guidelines

### Environment Requirements

Verify that your local system contains the necessary dependency packages prior to execution:

```bash
pip install pandas numpy scipy scikit-learn xgboost lightgbm catboost matplotlib seaborn plotly

```

### Execution Steps

1. **Data Prep Generation Pipeline:** Run the preprocessing cleaner to convert the raw file into the optimized dataset.
```bash
python "new cleaner.py"

```


This generates `AmesHousing_MI_applied.csv` inside your designated working root directory.
2. **Analysis and Model Training:** Launch the notebook environment to execute cross-validations, generate diagnostic plots, and review performance benchmarks.
```bash
jupyter notebook "Assignment 2.ipynb"

```
