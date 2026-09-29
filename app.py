"""
Presentation layer entry point (Flask). Three-layer mapping per Sec 3.5.1:
  templates/ + static/  -> Presentation Layer
  processing.py         -> Processing Layer
  model/                -> Model Layer (persisted ensemble pipeline)
"""
import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

import processing as proc

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)

PIPELINE = joblib.load(ROOT / "model" / "ensemble_pipeline.joblib")
META = json.loads((ROOT / "model" / "metadata.json").read_text())
CLASSES = META["classes"]


def run_prediction(record: dict) -> dict:
    model_record = proc.to_model_record(record)
    frame = proc.to_frame(model_record)
    predicted = PIPELINE.predict(frame)[0]
    proba = PIPELINE.predict_proba(frame)[0]
    order = list(PIPELINE.classes_)
    probs = {c: round(float(proba[order.index(c)]) * 100, 1) for c in CLASSES}
    return {
        "predicted": predicted,
        "confidence": probs[predicted],
        "probabilities": probs,
        "at_risk": predicted in proc.RISK_CLASSES,
        "note": proc.CLASS_NOTES[predicted],
        "recommendations": proc.recommendations(model_record, predicted),
    }


@app.route("/")
def index():
    return render_template("index.html",
                           numeric=proc.NUMERIC_FIELDS,
                           selects=proc.SELECT_FIELDS,
                           meta=META, form={}, errors=[])


@app.route("/predict", methods=["POST"])
def predict():
    record, errors = proc.validate(request.form)
    if errors:
        return render_template("index.html",
                               numeric=proc.NUMERIC_FIELDS,
                               selects=proc.SELECT_FIELDS,
                               meta=META, form=request.form, errors=errors), 400
    result = run_prediction(record)
    return render_template("result.html", result=result,
                           record=record, meta=META,
                           numeric=proc.NUMERIC_FIELDS,
                           selects=proc.SELECT_FIELDS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    record, errors = proc.validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400
    return jsonify({"ok": True, **run_prediction(record)})


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", meta=META)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
