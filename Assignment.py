import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

# Load datasets
# Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# MNIST (using load_digits as the standard scikit-learn MNIST-like digits dataset)
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Define the classifier
clf = SVC(kernel='linear', C=1.0, random_state=42)

# Configuration for folds
folds_list = [5, 10]

datasets = {
    'Iris': (X_iris, y_iris),
    'MNIST (Digits)': (X_digits, y_digits)
}

# Run Cross-Validation
for dataset_name, (X, y) in datasets.items():
    print(f"=== {dataset_name} Dataset ===")
    for k in folds_list:
        # K-Fold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        kf_scores = cross_val_score(clf, X, y, cv=kf)

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        skf_scores = cross_val_score(clf, X, y, cv=skf)

        print(f"{k}-Fold CV Accuracy: {kf_scores.mean():.4f} (+/- {kf_scores.std() * 2:.4f})")
        print(f"Stratified {k}-Fold CV Accuracy: {skf_scores.mean():.4f} (+/- {skf_scores.std() * 2:.4f})")
    print()