#!/usr/bin/env python3
"""
Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)
Datheon Machine Learning & Data Science Internship

End-to-end continuous regression pipeline on the California Housing dataset:
1. Loads 'housing.csv' and performs median imputation on 'total_bedrooms'.
2. Engineers domain interaction & ratio features ('rooms_per_household',
   'bedrooms_per_room', 'population_per_household', 'income_x_rooms') and
   exports the correlation & multicollinearity heatmap.
3. One-hot encodes 'ocean_proximity', performs an 80/20 train/test split,
   and standardizes continuous predictors with StandardScaler.
4. Tunes Ridge (L2) and Lasso (L1) across alpha hyperparameters via 5-fold
   GridSearchCV and benchmarks against Ordinary Least Squares (OLS) using
   RMSE, MAE, and R^2 scores.
5. Generates high-resolution Predicted vs. Actual and Residual Error plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def main():
    # 1. Load California Housing Dataset
    df = pd.read_csv("housing.csv")
    print(f"Loaded California Housing Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("Missing values before imputation:\n", df.isnull().sum(), "\n")

    # 2. Impute Missing Values & Engineer Interaction/Ratio Features
    imputer = SimpleImputer(strategy="median")
    df_imputed = df.copy()
    df_imputed[["total_bedrooms"]] = imputer.fit_transform(df[["total_bedrooms"]])

    df_imputed["rooms_per_household"] = df_imputed["total_rooms"] / df_imputed["households"]
    df_imputed["bedrooms_per_room"] = df_imputed["total_bedrooms"] / df_imputed["total_rooms"]
    df_imputed["population_per_household"] = df_imputed["population"] / df_imputed["households"]
    df_imputed["income_x_rooms"] = df_imputed["median_income"] * df_imputed["rooms_per_household"]

    # Plot Correlation Heatmap
    sns.set_theme(style="whitegrid", context="notebook")
    numeric_cols = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population", "households",
        "median_income", "rooms_per_household", "bedrooms_per_room",
        "population_per_household", "income_x_rooms", "median_house_value"
    ]
    corr_matrix = df_imputed[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9.5), dpi=200)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1.0, vmax=1.0, center=0.0, linewidths=0.6, ax=ax
    )
    ax.set_title(
        "California Housing Dataset: Feature Correlation & Multicollinearity Heatmap",
        fontsize=13, fontweight="bold", pad=14
    )
    plt.tight_layout()
    plt.savefig("housing_correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. Encode Categorical Attributes & Standardize Continuous Variables
    df_encoded = pd.get_dummies(df_imputed, columns=["ocean_proximity"], drop_first=True, dtype=float)
    feature_cols = [c for c in df_encoded.columns if c != "median_house_value"]
    continuous_cols = [c for c in numeric_cols if c != "median_house_value"]

    X = df_encoded[feature_cols].copy()
    y = df_encoded["median_house_value"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    # 4. Fit OLS, Ridge (L2), and Lasso (L1) via 5-Fold GridSearchCV
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    y_pred_ols = ols_model.predict(X_test)

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 250.0]}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_grid = GridSearchCV(
        Ridge(random_state=42),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    ridge_grid.fit(X_train, y_train)
    best_ridge = ridge_grid.best_estimator_
    y_pred_ridge = best_ridge.predict(X_test)

    lasso_grid = GridSearchCV(
        Lasso(max_iter=5000, random_state=42),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    lasso_grid.fit(X_train, y_train)
    best_lasso = lasso_grid.best_estimator_
    y_pred_lasso = best_lasso.predict(X_test)

    print("=" * 78)
    print(f"{'Model':<32} | {'Test RMSE':<14} | {'Test MAE':<14} | {'Test R²':<10}")
    print("=" * 78)
    for name, preds in [
        ("Ordinary Least Squares (OLS)", y_pred_ols),
        (f"Ridge Regression (L2, alpha={ridge_grid.best_params_['alpha']})", y_pred_ridge),
        (f"Lasso Regression (L1, alpha={lasso_grid.best_params_['alpha']})", y_pred_lasso),
    ]:
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name:<32} | ${rmse:>12,.2f} | ${mae:>12,.2f} | {r2:>8.4f}")
    print("=" * 78)

    # 5. Plot Predicted vs. Actual Values & Residual Distribution Curves
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=200, sharey=True)
    for ax, (title, preds, color) in zip(axes, [
        ("Ordinary Least Squares (OLS)", y_pred_ols, "#2b5c8f"),
        (f"Ridge Regression (L2, α={ridge_grid.best_params_['alpha']})", y_pred_ridge, "#1b7837"),
        (f"Lasso Regression (L1, α={lasso_grid.best_params_['alpha']})", y_pred_lasso, "#b35806"),
    ]):
        ax.scatter(y_test / 1000.0, preds / 1000.0, alpha=0.3, s=16, color=color)
        ax.plot([0, 520], [0, 520], "r--", lw=2.0, label="Ideal Fit (y = x)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Actual Median House Value ($1,000s)")
        if ax == axes[0]:
            ax.set_ylabel("Predicted Median House Value ($1,000s)")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("housing_predicted_vs_actual.png", dpi=200)
    plt.close(fig)

    res_ols = (y_test - y_pred_ols) / 1000.0
    res_ridge = (y_test - y_pred_ridge) / 1000.0
    res_lasso = (y_test - y_pred_lasso) / 1000.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=200)
    sns.histplot(res_ridge, bins=45, stat="density", color="#1b7837", alpha=0.22, ax=axes[0])
    sns.kdeplot(res_ols, color="#2b5c8f", lw=2.4, ax=axes[0], label="OLS KDE")
    sns.kdeplot(res_ridge, color="#1b7837", lw=2.2, linestyle="--", ax=axes[0], label="Ridge KDE")
    sns.kdeplot(res_lasso, color="#b35806", lw=2.0, linestyle=":", ax=axes[0], label="Lasso KDE")
    axes[0].axvline(0.0, color="red", linestyle="--", lw=1.5)
    axes[0].set_title("Residual Error Distribution Curves", fontweight="bold")
    axes[0].set_xlabel("Residual Error ($1,000s)")
    axes[0].legend()

    axes[1].scatter(y_pred_ridge / 1000.0, res_ridge, alpha=0.3, s=15, color="#1b7837", label="Ridge")
    axes[1].axhline(0.0, color="red", linestyle="--", lw=1.8)
    axes[1].set_title("Residuals vs. Predicted Values", fontweight="bold")
    axes[1].set_xlabel("Predicted Median House Value ($1,000s)")
    axes[1].set_ylabel("Residual Error ($1,000s)")
    axes[1].legend()

    lasso_sparse = Lasso(alpha=250.0, max_iter=5000, random_state=42).fit(X_train, y_train)
    top_idx = np.argsort(np.abs(ols_model.coef_))[-10:]
    y_pos = np.arange(len(top_idx))
    axes[2].barh(y_pos + 0.26, ols_model.coef_[top_idx] / 1000.0, height=0.26, color="#2b5c8f", label="OLS")
    axes[2].barh(y_pos, best_ridge.coef_[top_idx] / 1000.0, height=0.26, color="#1b7837", label="Ridge")
    axes[2].barh(y_pos - 0.26, lasso_sparse.coef_[top_idx] / 1000.0, height=0.26, color="#b35806", label="Lasso (α=250)")
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([feature_cols[i] for i in top_idx])
    axes[2].set_title("Top 10 Standardized Coefficients ($1,000s)", fontweight="bold")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("housing_residuals_analysis.png", dpi=200)
    plt.close(fig)
    print("Saved all diagnostic plots successfully.")


if __name__ == "__main__":
    main()