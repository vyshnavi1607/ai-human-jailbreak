import pickle
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Try imports safely
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


def evaluate_model(model, X, y, name):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs, f1s = [], []

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        print(f"{name} | Fold {fold}")
        model.fit(X[tr], y[tr])
        preds = model.predict(X[val])
        accs.append(accuracy_score(y[val], preds))
        f1s.append(f1_score(y[val], preds))

    print(f"{name:<15} | Acc: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f}")
    print("-" * 60)


def main():
    print("Loading embeddings...")

    with open("data/embeddings/X_all.pkl", "rb") as f:
        X = np.array(pickle.load(f))

    with open("data/embeddings/y_all.pkl", "rb") as f:
        y = np.array(pickle.load(f))

    print(f"Total samples: {len(y)}")
    print("=" * 60)

    if HAS_XGB:
        evaluate_model(
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            ),
            X, y, "XGBoost"
        )
    else:
        print("XGBoost skipped (not installed)")

    if HAS_LGB:
        evaluate_model(
            lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            ),
            X, y, "LightGBM"
        )
    else:
        print("LightGBM skipped (not installed)")

    if HAS_CAT:
        evaluate_model(
            CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                verbose=0,
                random_state=42
            ),
            X, y, "CatBoost"
        )
    else:
        print("CatBoost skipped (not installed)")


if __name__ == "__main__":
    main()
