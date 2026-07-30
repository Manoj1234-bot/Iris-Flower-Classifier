"""
Day 20 - Iris Flower Classifier
Core Skills: Classification basics (scikit-learn)
Author: Manoj S

The classic "Hello World" of machine learning classification. The Iris
dataset (built into scikit-learn, no download needed) contains 150
flower measurements across 3 species: Setosa, Versicolor, Virginica.

This script:
    - Trains and compares 3 different classifiers (KNN, Decision Tree,
      Logistic Regression) to show how model choice affects accuracy
    - Evaluates each with accuracy, confusion matrix, classification report
    - Visualizes feature relationships and decision boundaries
    - Lets you input custom flower measurements for a live prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the Iris dataset as a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))
    return df, iris.feature_names, iris.target_names


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train 3 different classifiers on the same data and compare accuracy.
    This shows that model choice matters, even for a "simple" dataset.
    """
    models = {
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200),
    }

    results = {}
    print("\n" + "-" * 55)
    print("Comparing 3 classifiers:")
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": accuracy, "predictions": y_pred}
        print(f"  {name:28s} Accuracy: {accuracy * 100:.2f}%")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest performing model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)")

    return results, best_name


def show_detailed_report(y_test, y_pred, target_names, model_name):
    """Print confusion matrix and classification report for one model."""
    print(f"\nDetailed Report for: {model_name}")
    print("\nConfusion Matrix:")
    print("(rows = actual species, columns = predicted species)")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


def plot_feature_relationships(df, feature_names):
    """Save a pairwise scatter plot showing how features separate species."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"setosa": "#e74c3c", "versicolor": "#3498db", "virginica": "#2ecc71"}

    for species, color in colors.items():
        subset = df[df["species_name"] == species]
        axes[0].scatter(subset[feature_names[0]], subset[feature_names[1]],
                         label=species, color=color, alpha=0.7)
        axes[1].scatter(subset[feature_names[2]], subset[feature_names[3]],
                         label=species, color=color, alpha=0.7)

    axes[0].set_xlabel(feature_names[0])
    axes[0].set_ylabel(feature_names[1])
    axes[0].set_title("Sepal Length vs Sepal Width")
    axes[0].legend()

    axes[1].set_xlabel(feature_names[2])
    axes[1].set_ylabel(feature_names[3])
    axes[1].set_title("Petal Length vs Petal Width")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("iris_feature_plot.png")
    print("\nSaved feature relationship plot to iris_feature_plot.png")
    print("(Notice how Petal measurements separate species much more cleanly than Sepal ones!)")


def predict_custom_flower(model, scaler, feature_names, target_names, use_scaling):
    """Let the user input custom measurements and get a live prediction."""
    print("\n" + "-" * 55)
    print("Try predicting a custom flower's species!")
    print("-" * 55)

    defaults = [5.8, 3.0, 3.8, 1.2]  # roughly average values across the dataset

    values = []
    for feature, default in zip(feature_names, defaults):
        while True:
            user_input = input(f"{feature} (default {default}): ").strip()

            if not user_input:
                values.append(default)
                break

            try:
                values.append(float(user_input))
                break
            except ValueError:
                print(f"  '{user_input}' isn't a valid number. Please enter a number (e.g. 5.8), or press Enter for default.")

    X_custom = np.array([values])
    if use_scaling:
        X_custom = scaler.transform(X_custom)

    prediction = model.predict(X_custom)[0]
    probabilities = model.predict_proba(X_custom)[0] if hasattr(model, "predict_proba") else None

    print(f"\nPredicted species: {target_names[prediction]}")
    if probabilities is not None:
        print("Confidence breakdown:")
        for name, prob in zip(target_names, probabilities):
            print(f"  {name}: {prob * 100:.1f}%")


def main():
    print("=" * 55)
    print("      🌸  IRIS FLOWER CLASSIFIER")
    print("=" * 55)

    df, feature_names, target_names = load_data()
    print(f"\nLoaded {len(df)} samples across {len(target_names)} species: {', '.join(target_names)}")
    print(f"Features: {', '.join(feature_names)}")

    X = df[feature_names].values
    y = df["species"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features - helps KNN and Logistic Regression, doesn't hurt Decision Tree
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, best_name = train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test)

    show_detailed_report(y_test, results[best_name]["predictions"], target_names, best_name)

    plot_feature_relationships(df, feature_names)

    best_model = results[best_name]["model"]

    while True:
        choice = input("\nPredict a custom flower's species? (y/n): ").strip().lower()
        if choice == 'y':
            predict_custom_flower(best_model, scaler, feature_names, target_names, use_scaling=True)
        else:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()