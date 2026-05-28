# ============================================================
# CodeAlpha Internship  -  Task 1: Iris Flower Classification
# Intern : Moussa Guindo
# ID     : CA/DF1/100834
# Domain : Data Science
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

RANDOM_STATE  = 42
TEST_SIZE     = 0.2
COLORS        = {
    "setosa"    : "#e74c3c",
    "versicolor": "#2ecc71",
    "virginica" : "#3498db",
}
OUTPUT_EXPLORATION  = "iris_exploration.png"
OUTPUT_CONFUSION    = "iris_confusion_matrix.png"
OUTPUT_COMPARISON   = "iris_model_comparison.png"
OUTPUT_MODEL        = "iris_best_model.pkl"
OUTPUT_SCALER       = "iris_scaler.pkl"


# ─────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, object]:
    """Load the Iris dataset and return a DataFrame with a species column."""
    iris = load_iris()
    df   = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris


def print_summary(df: pd.DataFrame) -> None:
    """Print a structured summary of the dataset."""
    separator = "=" * 60
    print(separator)
    print("  CodeAlpha  -  Iris Flower Classification")
    print(separator)
    print("\nDataset overview (first 10 rows):")
    print(df.head(10).to_string())
    print(f"\nShape : {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nDescriptive statistics:")
    print(df.describe().round(2).to_string())
    print("\nMissing values:")
    print(df.isnull().sum().to_string())
    print("\nClass distribution:")
    print(df["species"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────
# 2. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_exploration(df: pd.DataFrame) -> None:
    """Generate and save a 2x2 exploratory visualization panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Iris Flower - Exploratory Data Analysis",
        fontsize=16, fontweight="bold"
    )

    # Sepal scatter
    for species, color in COLORS.items():
        subset = df[df["species"] == species]
        axes[0, 0].scatter(
            subset["sepal length (cm)"], subset["sepal width (cm)"],
            label=species, color=color, alpha=0.7, edgecolors="white", s=60
        )
    axes[0, 0].set_xlabel("Sepal Length (cm)")
    axes[0, 0].set_ylabel("Sepal Width (cm)")
    axes[0, 0].set_title("Sepal : Length vs Width")
    axes[0, 0].legend()

    # Petal scatter
    for species, color in COLORS.items():
        subset = df[df["species"] == species]
        axes[0, 1].scatter(
            subset["petal length (cm)"], subset["petal width (cm)"],
            label=species, color=color, alpha=0.7, edgecolors="white", s=60
        )
    axes[0, 1].set_xlabel("Petal Length (cm)")
    axes[0, 1].set_ylabel("Petal Width (cm)")
    axes[0, 1].set_title("Petal : Length vs Width")
    axes[0, 1].legend()

    # Petal length histogram
    for species, color in COLORS.items():
        subset = df[df["species"] == species]
        axes[1, 0].hist(
            subset["petal length (cm)"],
            bins=15, label=species, color=color, alpha=0.6
        )
    axes[1, 0].set_xlabel("Petal Length (cm)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution : Petal Length")
    axes[1, 0].legend()

    # Petal width boxplot
    species_list = df["species"].unique()
    bp_data      = [df[df["species"] == s]["petal width (cm)"].values for s in species_list]
    axes[1, 1].boxplot(
        bp_data, labels=species_list, patch_artist=True,
        boxprops=dict(facecolor="#3498db", alpha=0.6)
    )
    axes[1, 1].set_title("Boxplot : Petal Width by Species")
    axes[1, 1].set_ylabel("Petal Width (cm)")

    plt.tight_layout()
    plt.savefig(OUTPUT_EXPLORATION, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nExploration chart saved : {OUTPUT_EXPLORATION}")


# ─────────────────────────────────────────────────────────────
# 3. DATA PREPARATION
# ─────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, iris) -> tuple:
    """Split and scale the dataset. Returns train/test arrays and the fitted scaler."""
    X = df.drop("species", axis=1)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    print(f"\nTraining set : {X_train.shape[0]} samples")
    print(f"Test set     : {X_test.shape[0]} samples")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────

def build_models() -> dict:
    """Return a dictionary of untrained classifiers."""
    return {
        "KNN (k=5)"     : KNeighborsClassifier(n_neighbors=5),
        "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM (RBF)"     : SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test, iris) -> dict:
    """Train each model, evaluate it, and return a results dictionary."""
    results   = {}
    separator = "=" * 60

    print(f"\n{separator}")
    print("  Model Results")
    print(separator)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)

        results[name] = {
            "model"      : model,
            "accuracy"   : acc,
            "predictions": y_pred,
        }

        print(f"\nModel    : {name}")
        print(f"Accuracy : {acc * 100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

    return results


# ─────────────────────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str, iris) -> None:
    """Plot and save the confusion matrix of the best model."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Confusion Matrix  -  {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_CONFUSION, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved : {OUTPUT_CONFUSION}")


# ─────────────────────────────────────────────────────────────
# 6. MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing model accuracies."""
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, accs,
        color=["#e74c3c", "#2ecc71", "#3498db"],
        edgecolor="white", linewidth=1.2
    )
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 3,
            f"{acc:.1f}%",
            ha="center", va="top",
            fontsize=12, color="white", fontweight="bold"
        )
    ax.set_ylim(80, 105)
    ax.set_title("Model Comparison  -  Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_COMPARISON, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Model comparison chart saved : {OUTPUT_COMPARISON}")


# ─────────────────────────────────────────────────────────────
# 7. MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────

def save_artifacts(model, scaler) -> None:
    """Persist the best model and its scaler to disk."""
    joblib.dump(model,  OUTPUT_MODEL)
    joblib.dump(scaler, OUTPUT_SCALER)
    print(f"\nModel saved  : {OUTPUT_MODEL}")
    print(f"Scaler saved : {OUTPUT_SCALER}")


# ─────────────────────────────────────────────────────────────
# 8. PREDICTION DEMO
# ─────────────────────────────────────────────────────────────

def predict_sample(model, scaler, iris) -> None:
    """Run a single-sample prediction and display the result."""
    sample    = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_sc = scaler.transform(sample)
    pred      = model.predict(sample_sc)

    separator = "=" * 60
    print(f"\n{separator}")
    print("  Prediction Demo  -  New Sample")
    print(separator)
    print("Input  : sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2")
    print(f"Output : {iris.target_names[pred[0]].upper()}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # Load & summarize
    df, iris = load_data()
    print_summary(df)

    # Explore visually
    plot_exploration(df)

    # Prepare
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, iris)

    # Train & evaluate
    models  = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test, iris)

    # Select best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]
    print(f"\nBest model : {best_name}  -  Accuracy : {best['accuracy'] * 100:.2f}%")

    # Visualize
    plot_confusion_matrix(y_test, best["predictions"], best_name, iris)
    plot_model_comparison(results)

    # Persist
    save_artifacts(best["model"], scaler)

    # Demo
    predict_sample(best["model"], scaler, iris)

    print("\nTask 1 completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()