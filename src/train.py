import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train disease prediction models.")
    p.add_argument("--data", type=str, default="data/improved_disease_dataset.csv")
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cv",
        action="store_true",
        help="Run StratifiedKFold cross-validation (prints mean accuracy).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Put your CSV at data/ or pass --data."
        )

    data = pd.read_csv(data_path)

    if "disease" not in data.columns:
        # Fallback to last column if the dataset doesn't name it explicitly.
        data = data.copy()
        data = data.rename(columns={data.columns[-1]: "disease"})

    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])

    X = data.drop(columns=["disease"])
    y = data["disease"]

    
    if "gender" in X.columns:
        le = LabelEncoder()
        X["gender"] = le.fit_transform(X["gender"])

    X = X.fillna(0)

    ros = RandomOverSampler(random_state=args.seed)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    if args.cv:
        models = {
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        for name, model in models.items():
            scores = cross_val_score(
                model,
                X_resampled,
                y_resampled,
                cv=skf,
                scoring="accuracy",
                n_jobs=-1,
                error_score="raise",
            )
            print("=" * 50)
            print(f"Model: {name}")
            print(f"Scores: {scores}")
            print(f"Mean Accuracy: {scores.mean():.4f}")

    svm_model = SVC()
    nb_model = GaussianNB()
    rf_model = RandomForestClassifier()

    svm_model.fit(X_resampled, y_resampled)
    nb_model.fit(X_resampled, y_resampled)
    rf_model.fit(X_resampled, y_resampled)

    
    svm_acc = accuracy_score(y_resampled, svm_model.predict(X_resampled))
    nb_acc = accuracy_score(y_resampled, nb_model.predict(X_resampled))
    rf_acc = accuracy_score(y_resampled, rf_model.predict(X_resampled))
    print(f"SVM train accuracy: {svm_acc * 100:.2f}%")
    print(f"NB train accuracy:  {nb_acc * 100:.2f}%")
    print(f"RF train accuracy:  {rf_acc * 100:.2f}%")

    joblib.dump(encoder, outdir / "encoder.joblib")
    joblib.dump(list(X.columns), outdir / "symptoms.joblib")
    joblib.dump(svm_model, outdir / "svm_model.joblib")
    joblib.dump(nb_model, outdir / "nb_model.joblib")
    joblib.dump(rf_model, outdir / "rf_model.joblib")

    print(f"Saved artifacts to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
