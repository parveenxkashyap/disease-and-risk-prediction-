import argparse
from pathlib import Path
from statistics import mode

import joblib
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict disease from symptoms.")
    p.add_argument("--models", type=str, default="models", help="Models folder path.")
    p.add_argument(
        "--symptoms",
        type=str,
        required=True,
        help='Comma-separated symptoms (e.g. "Itching,Skin Rash,Nodal Skin Eruptions")',
    )
    return p.parse_args()


def build_input_vector(all_symptoms: list[str], user_symptoms: list[str]) -> pd.DataFrame:
    symptom_index = {s: i for i, s in enumerate(all_symptoms)}
    vec = np.zeros(len(all_symptoms), dtype=int)

    for s in user_symptoms:
        s_clean = s.strip()
        if s_clean in symptom_index:
            vec[symptom_index[s_clean]] = 1

    # DataFrame keeps feature names, avoids sklearn "invalid feature names" warnings
    return pd.DataFrame([vec], columns=all_symptoms)


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models)

    encoder = joblib.load(models_dir / "encoder.joblib")
    all_symptoms = joblib.load(models_dir / "symptoms.joblib")
    svm_model = joblib.load(models_dir / "svm_model.joblib")
    nb_model = joblib.load(models_dir / "nb_model.joblib")
    rf_model = joblib.load(models_dir / "rf_model.joblib")

    user_symptoms = [s.strip() for s in args.symptoms.split(",") if s.strip()]
    X_in = build_input_vector(all_symptoms, user_symptoms)

    rf_pred = encoder.classes_[rf_model.predict(X_in)[0]]
    nb_pred = encoder.classes_[nb_model.predict(X_in)[0]]
    svm_pred = encoder.classes_[svm_model.predict(X_in)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])

    print(f"Random Forest Prediction: {rf_pred}")
    print(f"Naive Bayes Prediction:  {nb_pred}")
    print(f"SVM Prediction:         {svm_pred}")
    print(f"Final Prediction:       {final_pred}")


if __name__ == "__main__":
    main()
