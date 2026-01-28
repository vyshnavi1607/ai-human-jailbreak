"""
import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

def evaluate_model_cv(model, X, y, name, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores, f1_scores = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds))

    print(
        f"{name:<25} | "
        f"Acc: {np.mean(acc_scores):.4f} | "
        f"F1: {np.mean(f1_scores):.4f}"
    )

def main():
    print("Loading full-dataset embeddings...")
    with open("data/embeddings/X_all.pkl", "rb") as f:
        X = pickle.load(f)

    with open("data/embeddings/y_all.pkl", "rb") as f:
        y = pickle.load(f)

    X = np.array(X)
    y = np.array(y)

    print(f"Total samples: {len(y)}")
    print("\nMODEL COMPARISON (Stratified K-Fold CV)")
    print("-" * 75)

    # -------- STAGE 1: BASELINE MODELS --------
    evaluate_model_cv(
        LogisticRegression(max_iter=2000),
        X, y, "Logistic Regression"
    )

    evaluate_model_cv(
        LinearSVC(),
        X, y, "Linear SVM"
    )

    evaluate_model_cv(
        DecisionTreeClassifier(random_state=42),
        X, y, "Decision Tree"
    )

    evaluate_model_cv(
        KNeighborsClassifier(n_neighbors=5),
        X, y, "k-NN"
    )

    evaluate_model_cv(
        GaussianNB(),
        X, y, "Naive Bayes"
    )

    # -------- STAGE 2: ENSEMBLE MODELS --------
    evaluate_model_cv(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X, y, "Random Forest"
    )

    evaluate_model_cv(
        ExtraTreesClassifier(n_estimators=200, random_state=42),
        X, y, "Extra Trees"
    )

    evaluate_model_cv(
        GradientBoostingClassifier(random_state=42),
        X, y, "Gradient Boosting"
    )

    evaluate_model_cv(
        AdaBoostClassifier(random_state=42),
        X, y, "AdaBoost"
    )

if __name__ == "__main__":
    main()


"""



import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Classical ML
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Ensemble / Boosting
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)

# Neural (on embeddings)
from sklearn.neural_network import MLPClassifier

# Optional advanced boosters
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except:
    HAS_CAT = False


def evaluate_cv(model, X, y, name, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accs, f1s = [], []

    for tr_idx, val_idx in skf.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
        f1s.append(f1_score(y[val_idx], preds))

    print(f"{name:<30} | Acc: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f}")


def main():
    with open("data/embeddings/X_all.pkl", "rb") as f:
        X = np.array(pickle.load(f))
    with open("data/embeddings/y_all.pkl", "rb") as f:
        y = np.array(pickle.load(f))

    print(f"Total samples: {len(y)}")
    print("-" * 80)

    # Linear & Margin-based
    evaluate_cv(LogisticRegression(max_iter=3000), X, y, "Logistic Regression")
    evaluate_cv(LinearSVC(), X, y, "Linear SVM")
    evaluate_cv(SVC(kernel="rbf"), X, y, "SVM (RBF)")
    evaluate_cv(SVC(kernel="poly"), X, y, "SVM (Polynomial)")
    evaluate_cv(SVC(kernel="sigmoid"), X, y, "SVM (Sigmoid)")

    # Instance / Probabilistic
    evaluate_cv(KNeighborsClassifier(n_neighbors=5), X, y, "k-NN")
    evaluate_cv(GaussianNB(), X, y, "Naive Bayes (Gaussian)")

    # Tree-based
    evaluate_cv(DecisionTreeClassifier(random_state=42), X, y, "Decision Tree")

    # Ensemble / Boosting
    evaluate_cv(RandomForestClassifier(n_estimators=200), X, y, "Random Forest")
    evaluate_cv(ExtraTreesClassifier(n_estimators=200), X, y, "Extra Trees")
    evaluate_cv(GradientBoostingClassifier(), X, y, "Gradient Boosting")
    evaluate_cv(AdaBoostClassifier(), X, y, "AdaBoost")
    evaluate_cv(HistGradientBoostingClassifier(), X, y, "HistGBM")

    # Advanced Boosters (if available)
    if HAS_XGB:
        evaluate_cv(XGBClassifier(eval_metric="logloss"), X, y, "XGBoost")
    else:
        print("XGBoost skipped (not installed)")

    if HAS_LGB:
        evaluate_cv(lgb.LGBMClassifier(), X, y, "LightGBM")
    else:
        print("LightGBM skipped (not installed)")

    if HAS_CAT:
        evaluate_cv(CatBoostClassifier(verbose=0), X, y, "CatBoost")
    else:
        print("CatBoost skipped (not installed)")

    # Neural baseline
    evaluate_cv(
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300),
        X, y, "MLP (Feedforward NN)"
    )


if __name__ == "__main__":
    main()
