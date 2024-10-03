# Intelligent Disease and Risk Prediction

A machine-learning based disease prediction project that trains multiple classifiers and returns an ensemble (majority-vote) prediction from user-entered symptoms.

## What the model does
- Trains SVM, Gaussian Naive Bayes, and Random Forest classifiers.
- Handles class imbalance using RandomOverSampler.
- Evaluates models with Stratified K-Fold cross-validation.
- Predicts disease from a comma-separated list of symptoms and returns:
  - Individual model predictions
  - Final prediction using majority vote

## Project structure
- `Intelligent_Disease_and_Risk_Prediction.ipynb` — Original notebook (Colab-style).
- `src/train.py` — Train models from CSV and save artifacts.
- `src/predict.py` — CLI prediction using saved artifacts.
- `data/` — Put the dataset here (ignored by git).
- `models/` — Saved models + encoder + feature list (ignored by git).

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
