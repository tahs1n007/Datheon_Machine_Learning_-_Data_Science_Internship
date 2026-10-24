# Datheon_Machine_Learning_-_Data_Science_Internship
# Datheon Machine Learning & Data Science Internship

This repository contains the end-to-end machine learning pipelines, executed Jupyter Notebooks (`.ipynb`), standalone Python scripts (`.py`), datasets (`.csv`), and diagnostic visualizations completed for the Datheon Machine Learning & Data Science Internship program.

## Repository Folder Structure

```text
Datheon_Machine_Learning_&_Data_Science_Internship/
├── README.md
├── requirements.txt
├── Task_1_Iris_Classification/
│   ├── Task_1_Iris_Classification_Pipeline.ipynb
│   ├── task1_iris_classification.py
│   ├── iris.csv
│   ├── iris_pairplot.png
│   ├── confusion_matrices.png
│   └── decision_boundaries.png
└── Task_2_Housing_Regression/
    ├── Task_2_Housing_Price_Prediction.ipynb
    ├── task2_housing_regression.py
    ├── housing.csv
    ├── housing_correlation_heatmap.png
    ├── housing_predicted_vs_actual.png
    └── housing_residuals_analysis.png
```

## Completed Tasks & Benchmark Results

### Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)
- **Objective:** Implement a supervised multi-class classification pipeline on the Iris dataset ($n = 150$, 3 species: *setosa*, *versicolor*, *virginica*) with stratified 80/20 train/test splitting, `StandardScaler` normalization, exploratory pairplots, confusion matrix heatmaps, and 2D PCA decision boundary visualizations.
- **Key Benchmark Results (Hold-Out Test Set, $n = 30$):**

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | **100.00%** (30/30) | **1.0000** | **1.0000** | **1.0000** | 95.83% ± 2.64% |
| **Decision Tree (`max_depth=4`)** | **93.33%** (28/30) | **0.9333** | **0.9333** | **0.9333** | 94.17% ± 3.33% |

### Task 2: Real Estate Housing Price Prediction (OLS, Ridge & Lasso Regression)
- **Objective:** Predict `median_house_value` across California census block groups ($n = 20,640$, 80/20 train/test split) by imputing missing `total_bedrooms` records (`SimpleImputer(strategy="median")`), engineering domain interaction/ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`, `income_x_rooms`), handling severe multicollinearity among raw count attributes ($r = 0.86\text{--}0.96$), one-hot encoding `ocean_proximity`, standardizing continuous variables, and tuning **Ridge (`L2`)** and **Lasso (`L1`)** via 5-fold `GridSearchCV`.
- **Key Benchmark Results (Hold-Out Test Set, $n = 4,128$):**

| Regression Model | Optimal $\alpha$ (5-Fold CV) | Test RMSE ($) | Test MAE ($) | Test $R^2$ Score | Regularization Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ordinary Least Squares (OLS)** | None | $39,140.53 | $30,861.43 | 0.9019 | Unpenalized baseline; large opposing weights on collinear counts |
| **Ridge Regression (L2)** | $\alpha = 0.01$ | **$39,139.66** | **$30,861.20** | **0.9019** | Smooth L2 shrinkage across correlated count features |
| **Lasso Regression (L1)** | $\alpha = 1.0$ | $39,132.39 | $30,859.11 | 0.9020 | Sparse L1 penalty; zeroes redundant count features at higher $\alpha$ |

## Setup & Reproduction Instructions

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/<your-username>/Datheon_Machine_Learning_&_Data_Science_Internship.git
   cd "Datheon_Machine_Learning_&_Data_Science_Internship"
   pip install -r requirements.txt
   ```
2. **Run Task 1 (Iris Classification):**
   ```bash
   cd Task_1_Iris_Classification
   python task1_iris_classification.py
   ```
3. **Run Task 2 (California Housing Regression):**
   ```bash
   cd ../Task_2_Housing_Regression
   python task2_housing_regression.py
   ```
