import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def evaluate(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        print(f"{name} | Fold {fold}")
        model.fit(X[tr], y[tr])
        preds = model.predict(X[val])

        accs.append(accuracy_score(y[val], preds))
        f1s.append(f1_score(y[val], preds))

    print(f"{name:<30} | Acc: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f}")
    print("-" * 70)


def main():
    print("Loading jailbreak embeddings...")

    with open("data/embeddings/X_jailbreak.pkl", "rb") as f:
        X = np.array(pickle.load(f))

    with open("data/embeddings/y_jailbreak.pkl", "rb") as f:
        y = np.array(pickle.load(f))

    print(f"Total samples: {len(y)}")
    print("-" * 70)

    # -------- BASELINE --------
    evaluate(
        LogisticRegression(max_iter=1000),
        X, y,
        "Logistic Regression"
    )

    # -------- SVM MODELS --------
    evaluate(
        SVC(kernel="rbf", C=1.0),
        X, y,
        "SVM (RBF)"
    )

    evaluate(
        SVC(kernel="poly", degree=3, C=1.0),
        X, y,
        "SVM (Polynomial)"
    )

    # -------- NEURAL MODEL --------
    evaluate(
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            random_state=42
        ),
        X, y,
        "MLP (Feedforward NN)"
    )

    # -------- OPTIONAL TREE MODEL --------
    evaluate(
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        X, y,
        "Random Forest"
    )

    print("\nJailbreak model evaluation completed successfully ✅")


if __name__ == "__main__":
    main()
