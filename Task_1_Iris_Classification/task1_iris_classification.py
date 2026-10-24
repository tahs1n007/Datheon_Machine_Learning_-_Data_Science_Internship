"""
Task 1: Multi-Class Benchmark Classification Pipeline (Iris Dataset)
Datheon Machine Learning & Data Science Internship

Workflow:
1. Load the benchmark Iris dataset using Pandas.
2. Perform exploratory pairplots with Seaborn.
3. Split into 80/20 stratified train/test sets and normalize features using StandardScaler().
4. Train LogisticRegression and DecisionTreeClassifier models and benchmark Accuracy, Precision, Recall, and F1-Score.
5. Plot Confusion Matrix Heatmaps and 2D Decision Boundary Visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def main():
    # Set visual theme
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.05)

    # 1. Load dataset
    df = pd.read_csv("iris.csv")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nClass Distribution:\n", df["species"].value_counts())
    print("\nFirst 5 Rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe().round(3))

    # 2. Exploratory Pairplot
    palette_map = {"setosa": "#2b5c8f", "versicolor": "#2ca02c", "virginica": "#d95f02"}
    pair_grid = sns.pairplot(
        df,
        hue="species",
        palette=palette_map,
        diag_kind="kde",
        markers=["o", "s", "D"],
        plot_kws={"alpha": 0.85, "s": 55, "edgecolor": "white", "linewidth": 0.6},
        height=2.2,
    )
    pair_grid.fig.suptitle(
        "Exploratory Feature Pairplot of Iris Species",
        y=1.02,
        fontsize=14,
        fontweight="bold",
    )
    pair_grid.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close(pair_grid.fig)

    # 3. 80/20 Train/Test Split & StandardScaler Normalization
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].values

    le = LabelEncoder()
    y = le.fit_transform(df["species"].values)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(
        f"\nTotal Samples: {len(X)} | Training Set (80%): {X_train.shape[0]} | Test Set (20%): {X_test.shape[0]}"
    )

    # 4. Train Models & Benchmark Classification Metrics
    log_reg = LogisticRegression(random_state=42, max_iter=200)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)

    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=4)
    dt_clf.fit(X_train_scaled, y_train)
    y_pred_dt = dt_clf.predict(X_test_scaled)

    models_results = {}
    for name, y_pred in [
        ("Logistic Regression", y_pred_lr),
        ("Decision Tree Classifier", y_pred_dt),
    ]:
        models_results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (Macro)": precision_score(y_test, y_pred, average="macro"),
            "Recall (Macro)": recall_score(y_test, y_pred, average="macro"),
            "F1-Score (Macro)": f1_score(y_test, y_pred, average="macro"),
            "F1-Score (Weighted)": f1_score(y_test, y_pred, average="weighted"),
        }

    benchmark_df = pd.DataFrame(models_results).T.round(4)
    print("\n=== MODEL BENCHMARK SUMMARY (TEST SET: n=30) ===")
    print(benchmark_df.to_string())
    print("\n=== LOGISTIC REGRESSION CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred_lr, target_names=class_names, digits=4))
    print("=== DECISION TREE CLASSIFIER CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred_dt, target_names=class_names, digits=4))

    # 5A. Plot Confusion Matrix Heatmaps
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_dt = confusion_matrix(y_test, y_pred_dt)

    fig_cm, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 13, "weight": "bold"},
        ax=axes[0],
        linewidths=1,
        linecolor="white",
    )
    axes[0].set_title(
        f"Logistic Regression\n(Test Accuracy: {models_results['Logistic Regression']['Accuracy']*100:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Predicted Species", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Actual Species", fontsize=11, fontweight="bold")

    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 13, "weight": "bold"},
        ax=axes[1],
        linewidths=1,
        linecolor="white",
    )
    axes[1].set_title(
        f"Decision Tree Classifier\n(Test Accuracy: {models_results['Decision Tree Classifier']['Accuracy']*100:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_xlabel("Predicted Species", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Actual Species", fontsize=11, fontweight="bold")

    fig_cm.suptitle(
        "Confusion Matrix Comparison on Hold-Out Test Set (80/20 Split)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig_cm.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    # 5B. Plot 2D Decision Boundary Visualization
    X_petal = df[["petal_length", "petal_width"]].values
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_petal, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler_2d = StandardScaler()
    Xp_train_sc = scaler_2d.fit_transform(Xp_train)
    Xp_test_sc = scaler_2d.transform(Xp_test)
    Xp_all_sc = np.vstack([Xp_train_sc, Xp_test_sc])

    lr_2d = LogisticRegression(random_state=42).fit(Xp_train_sc, yp_train)
    acc_lr_2d = accuracy_score(yp_test, lr_2d.predict(Xp_test_sc))

    dt_2d = DecisionTreeClassifier(random_state=42, max_depth=4).fit(Xp_train_sc, yp_train)
    acc_dt_2d = accuracy_score(yp_test, dt_2d.predict(Xp_test_sc))

    x_min, x_max = Xp_all_sc[:, 0].min() - 0.5, Xp_all_sc[:, 0].max() + 0.5
    y_min, y_max = Xp_all_sc[:, 1].min() - 0.5, Xp_all_sc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )

    cmap_bg = ListedColormap(["#d0e1f2", "#d9f0d3", "#fdd0a2"])
    point_colors = ["#1f4e78", "#238b45", "#d94801"]
    markers = ["o", "s", "D"]

    fig_db, axes_db = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, model_2d, title, acc_2d in [
        (axes_db[0], lr_2d, "Logistic Regression Decision Boundaries", acc_lr_2d),
        (axes_db[1], dt_2d, "Decision Tree Decision Boundaries", acc_dt_2d),
    ]:
        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)
        ax.contour(xx, yy, Z, colors="k", linewidths=0.7, alpha=0.5)
        for idx, cls_name in enumerate(class_names):
            mask_tr = yp_train == idx
            mask_te = yp_test == idx
            ax.scatter(
                Xp_train_sc[mask_tr, 0],
                Xp_train_sc[mask_tr, 1],
                c=point_colors[idx],
                label=f"{cls_name} (Train)",
                marker=markers[idx],
                s=45,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.85,
            )
            ax.scatter(
                Xp_test_sc[mask_te, 0],
                Xp_test_sc[mask_te, 1],
                c=point_colors[idx],
                label=f"{cls_name} (Test)",
                marker=markers[idx],
                s=85,
                edgecolor="black",
                linewidth=1.2,
                alpha=1.0,
            )
        ax.set_title(
            f"{title}\n(2D Test Accuracy: {acc_2d*100:.2f}%)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Standardized Petal Length", fontsize=11, fontweight="bold")
        ax.set_ylabel("Standardized Petal Width", fontsize=11, fontweight="bold")
        ax.legend(
            loc="upper left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9
        )

    fig_db.suptitle(
        "Decision Boundary Visualization (Standardized Petal Length vs. Petal Width)",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    fig_db.savefig("decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig_db)


if __name__ == "__main__":
    main()