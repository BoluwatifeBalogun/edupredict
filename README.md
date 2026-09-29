# Academic Performance Prediction System — YCT
Implementation of "Development of a Data-Driven Predictive System for Student
Academic Performance Using an Ensemble Classifier" (Alli Sofiat Abidemi &
Adeboye Samuel, HND Computer Science, Yaba College of Technology, supervised
by Mr Ogundele).

## Architecture (three-layer, Section 3.5.1)
- Presentation layer — `templates/` + `static/` (Flask + HTML/CSS/JS)
- Processing layer — `processing.py` (validation; encoding/imputation/scaling
  live inside the persisted sklearn ColumnTransformer)
- Model layer — `model/ensemble_pipeline.joblib`
  (RandomForest + GradientBoosting via soft VotingClassifier)

## Quick start
    pip install -r requirements.txt
    python scripts/generate_dataset.py   # or drop the real Kaggle CSV in data/
    python train.py                      # trains + writes model/ artifacts
    python app.py                        # open http://localhost:5000

To put it online, see DEPLOY.md (Render, free tier).

## Using the real Kaggle dataset
Download "Student Performance Factors" (lainguyn123) from Kaggle and save it
as `data/StudentPerformanceFactors.csv`, then re-run `python train.py`.
The bundled generator produces a statistically realistic stand-in with the
identical 19-attribute schema, because the build environment could not reach
Kaggle. Swap in the real file before the defense demo if required.

## Pipeline (KDD stages, Section 2.8)
selection -> cleaning (median/mode imputation) -> ordinal + one-hot encoding
-> standard scaling -> SMOTE on the training split only (every class lifted
to >= 4,200 samples => 21,000-record training corpus, per Section 3.4)
-> soft-voting ensemble -> evaluation (accuracy, precision, recall, F1,
confusion matrix) -> artifacts in model/.

## Grading (NBTE, as used by Yaba College of Technology)
Classifications follow the NBTE 4.0 CGPA scale from the official NBTE
curriculum documents: Distinction 3.50-4.00, Upper Credit 3.00-3.49,
Lower Credit 2.50-2.99, Pass 2.00-2.49, Fail below 2.00. The NBTE grade
scale maps percentages to points (A 75+ = 4.0, AB 70-74 = 3.5,
B 65-69 = 3.25, BC 60-64 = 3.0, C 55-59 = 2.75, CD 50-54 = 2.5,
D 45-49 = 2.25, E 40-44 = 2.0, F below 40 = 0.0), so the system's
percentage bands are: Distinction >= 70, Upper Credit 60-69,
Lower Credit 50-59, Pass 40-49, Fail < 40.

The bundled generator is calibrated to a Nigerian polytechnic cohort
(mean 57%, sd 12). No public Nigerian dataset carries the 19-attribute
schema this report specifies (existing ones, such as the IEEE DataPort
South-East university scores or Landmark University records, are small
and CGPA-only), so the Kaggle schema is kept and the target regraded
to NBTE. The real Kaggle CSV can still be dropped in and retrained.

## Endpoints
- GET  /            prediction form
- POST /predict     form submission -> result page (seal + soft-vote bars)
- POST /api/predict JSON API, same validation, returns probabilities + recs
- GET  /dashboard   metrics, per-class F1, confusion matrix, importances

## Notes for the report
- SMOTE is applied strictly inside the training fold (imblearn pipeline), so
  test metrics are honest; state this in Chapter 4 to pre-empt questions.
- Test metrics land ~0.79 weighted F1, consistent with the reviewed
  literature (Hemasri & Kiran 79%; Adegunwa et al. 83.7%).
