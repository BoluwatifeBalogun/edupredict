"""
Trains the ensemble academic-performance classifier described in Chapter 3.

Pipeline (KDD stages):
  selection      -> data/StudentPerformanceFactors.csv
  preprocessing  -> cleaning (median/mode imputation), ordinal + one-hot
                    encoding, standard scaling
  transformation -> ColumnTransformer feature matrix
  mining         -> SMOTE (train split only) -> RandomForest + GradientBoosting
                    combined by a soft VotingClassifier
  evaluation     -> accuracy, precision, recall, F1, confusion matrix

Artifacts -> model/ensemble_pipeline.joblib, model/metadata.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "StudentPerformanceFactors.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

# NBTE (National Board for Technical Education) classification used by
# YabaTech and all Nigerian polytechnics. Percentage bands follow the NBTE
# grade scale (A=75+/4.0, AB=70-74/3.5, B=65-69/3.25, BC=60-64/3.0,
# C=55-59/2.75, CD=50-54/2.5, D=45-49/2.25, E=40-44/2.0, F=<40/0.0), so a
# mean score of 70+ corresponds to CGPA >= 3.50 (Distinction), 60-69 to
# Upper Credit (3.00-3.49), 50-59 to Lower Credit (2.50-2.99), 40-49 to
# Pass (2.00-2.49), and below 40 to Fail (< 2.00).
CLASSES = ["Distinction", "Upper Credit", "Lower Credit", "Pass", "Fail"]

NUMERIC = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
           "Tutoring_Sessions", "Physical_Activity"]
ORDINAL = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Far", "Moderate", "Near"],
}
NOMINAL = ["Extracurricular_Activities", "Internet_Access", "School_Type",
           "Learning_Disabilities", "Gender"]


def to_class(score: float) -> str:
    """NBTE class of award from mean percentage score (see band note above)."""
    if score >= 70: return "Distinction"
    if score >= 60: return "Upper Credit"
    if score >= 50: return "Lower Credit"
    if score >= 40: return "Pass"
    return "Fail"


def build_pipeline(smote_targets: dict) -> ImbPipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[ORDINAL[c] for c in ORDINAL]))]),
         list(ORDINAL)),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(drop="first"))]), NOMINAL),
    ])
    rf = RandomForestClassifier(n_estimators=250, max_depth=None,
                                min_samples_leaf=4, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1,
                                    max_depth=5, subsample=0.9, random_state=42)
    voting = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)
    # SMOTE lifts every training class to >= 4,200 samples, giving an
    # augmented training corpus of 21,000 records (the >20,000 in Sec 3.4)
    smote = SMOTE(sampling_strategy=smote_targets, k_neighbors=5, random_state=42)
    return ImbPipeline([("prep", prep), ("smote", smote), ("clf", voting)])


def main():
    df = pd.read_csv(DATA)
    df["Performance_Class"] = df["Exam_Score"].apply(to_class)
    X = df[NUMERIC + list(ORDINAL) + NOMINAL]
    y = df["Performance_Class"]

    print("Class distribution (raw):")
    print(y.value_counts().reindex(CLASSES).to_string(), "\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              stratify=y, random_state=42)
    smote_targets = {c: max(int((y_tr == c).sum()), 4200) for c in CLASSES}
    pipe = build_pipeline(smote_targets)
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    train_secs = round(time.time() - t0, 1)

    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=CLASSES)
    report = classification_report(y_te, y_pred, labels=CLASSES,
                                   output_dict=True, zero_division=0)

    print(f"Accuracy {acc:.4f} | Precision {prec:.4f} | "
          f"Recall {rec:.4f} | F1 {f1:.4f}  ({train_secs}s)")
    print(classification_report(y_te, y_pred, labels=CLASSES, zero_division=0))

    # Feature importances from the fitted Random Forest member
    prep = pipe.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    rf_fit = pipe.named_steps["clf"].named_estimators_["rf"]
    imp = sorted(zip(names, rf_fit.feature_importances_),
                 key=lambda t: t[1], reverse=True)
    importances = [{"feature": n.split("__", 1)[-1], "importance": round(float(v), 4)}
                   for n, v in imp[:12]]

    smote_counts = smote_targets
    joblib.dump(pipe, MODEL_DIR / "ensemble_pipeline.joblib", compress=3)
    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "algorithm": "Voting(RandomForest + GradientBoosting, soft)",
        "dataset_records": int(len(df)),
        "train_records_after_smote": int(sum(smote_counts.values())),
        "classes": CLASSES,
        "cgpa_bands": {"Distinction": "CGPA 3.50 – 4.00", "Upper Credit": "CGPA 3.00 – 3.49",
                       "Lower Credit": "CGPA 2.50 – 2.99", "Pass": "CGPA 2.00 – 2.49",
                       "Fail": "CGPA below 2.00"},
        "grading_authority": "NBTE 4.0 scale (Yaba College of Technology)",
        "class_distribution": {c: int((y == c).sum()) for c in CLASSES},
        "metrics": {"accuracy": round(acc, 4), "precision": round(prec, 4),
                    "recall": round(rec, 4), "f1": round(f1, 4)},
        "per_class": {c: {k: round(report[c][k], 3)
                          for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances,
        "train_seconds": train_secs,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + metadata -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
