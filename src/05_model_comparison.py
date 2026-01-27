"""
05_model_comparison.py

Purpose:
Systematic comparison of multiple ML models
on fixed NOMIC embeddings for AI vs Human detection
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


def evaluate(model, X_train, X_val, y_train, y_val, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    print(f"{name:<30} | Acc: {acc:.4f} | F1: {f1:.4f}")
    return acc, f1


def main():
    print("Loading embeddings...")
    with open("data/embeddings/X_train.pkl", "rb") as f:
        X = pickle.load(f)

    with open("data/embeddings/y_train.pkl", "rb") as f:
        y = pickle.load(f)

    print(f"Total samples: {len(y)}")

    print("Splitting train/validation...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nMODEL COMPARISON RESULTS")
    print("-" * 75)

    # ----- GROUP A: LINEAR MODELS -----
    evaluate(
        LogisticRegression(max_iter=1000),
        X_tr, X_val, y_tr, y_val,
        "Logistic Regression"
    )

    evaluate(
        LinearSVC(),
        X_tr, X_val, y_tr, y_val,
        "Linear SVM"
    )

    # ----- GROUP B: TREE-BASED MODELS -----
    evaluate(
        DecisionTreeClassifier(random_state=42),
        X_tr, X_val, y_tr, y_val,
        "Decision Tree"
    )

    evaluate(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_tr, X_val, y_tr, y_val,
        "Random Forest"
    )

    evaluate(
        ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X_tr, X_val, y_tr, y_val,
        "Extra Trees"
    )

    # ----- GROUP C: BOOSTING MODELS -----
    evaluate(
        GradientBoostingClassifier(random_state=42),
        X_tr, X_val, y_tr, y_val,
        "Gradient Boosting"
    )

    if xgb_available:
        evaluate(
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ),
            X_tr, X_val, y_tr, y_val,
            "XGBoost"
        )
    else:
        print("XGBoost not installed — skipped")

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
