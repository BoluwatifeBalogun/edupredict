"""Processing layer: validates Nigerian-native inputs (CGPA on the NBTE 4.0
scale, class days of the last 20) and maps them to the model's feature space.
Physical activity and family income are omitted from the form and imputed to
dataset baselines by the pipeline's imputers (NaN passthrough)."""
import numpy as np
import pandas as pd

NUMERIC_FIELDS = {
    "Previous_CGPA": ("Previous CGPA (ND / last level)", 0, 4, 0.01, "On the NBTE 4.0 scale, e.g. 2.75"),
    "Days_Present": ("Class days attended (of last 20)", 0, 20, 1, "Days present out of the last 20 class days"),
    "Hours_Studied": ("Hours studied per week", 0, 44, 1, "Average private study hours in a week"),
    "Tutoring_Sessions": ("Tutoring sessions per week", 0, 8, 1, "Extra academic support sessions"),
    "Sleep_Hours": ("Sleep hours per night", 3, 12, 1, "Average nightly sleep"),
}
SELECT_FIELDS = {
    "Access_to_Resources": ("Access to learning resources", ["Low", "Medium", "High"]),
    "Internet_Access": ("Internet access", ["No", "Yes"]),
    "Teacher_Quality": ("Teacher quality", ["Low", "Medium", "High"]),
    "School_Type": ("School type", ["Public", "Private"]),
    "Peer_Influence": ("Peer influence", ["Negative", "Neutral", "Positive"]),
    "Distance_from_Home": ("Distance from home", ["Near", "Moderate", "Far"]),
    "Extracurricular_Activities": ("Extracurricular activities", ["No", "Yes"]),
    "Motivation_Level": ("Motivation level", ["Low", "Medium", "High"]),
    "Parental_Involvement": ("Parental involvement", ["Low", "Medium", "High"]),
    "Parental_Education_Level": ("Parental education level", ["High School", "College", "Postgraduate"]),
    "Learning_Disabilities": ("Learning disabilities", ["No", "Yes"]),
    "Gender": ("Gender", ["Male", "Female"]),
}

def validate(form):
    record, errors = {}, []
    for name, (label, lo, hi, _s, _h) in NUMERIC_FIELDS.items():
        raw = form.get(name)
        raw = raw.strip() if isinstance(raw, str) else raw
        if raw in (None, ""):
            errors.append(f"{label} is required."); continue
        try: val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number."); continue
        if not lo <= val <= hi:
            errors.append(f"{label} must be between {lo} and {hi}."); continue
        record[name] = val
    for name, (label, options) in SELECT_FIELDS.items():
        raw = form.get(name)
        raw = raw.strip() if isinstance(raw, str) else raw
        if raw not in options:
            errors.append(f"Choose a valid option for {label.lower()}."); continue
        record[name] = raw
    return record, errors

def cgpa_to_percent(g: float) -> float:
    """Inverse of the NBTE grade-point bands (A 75+/4.0 ... F <40/0)."""
    g = max(0.0, min(4.0, float(g)))
    if g >= 3.5: p = 70 + (g - 3.5) / 0.5 * 15
    elif g >= 3.0: p = 60 + (g - 3.0) / 0.5 * 10
    elif g >= 2.5: p = 50 + (g - 2.5) / 0.5 * 10
    elif g >= 2.0: p = 40 + (g - 2.0) / 0.5 * 10
    else: p = 20 + g / 2.0 * 20
    return round(p, 1)

def to_model_record(record: dict) -> dict:
    m = dict(record)
    m["Previous_Scores"] = cgpa_to_percent(m.pop("Previous_CGPA"))
    m["Attendance"] = round(max(0.0, min(20.0, m.pop("Days_Present"))) / 20 * 100, 1)
    m["Physical_Activity"] = np.nan   # imputed to dataset median
    m["Family_Income"] = np.nan       # imputed to dataset mode
    return m

def to_frame(model_record: dict) -> pd.DataFrame:
    return pd.DataFrame([model_record])

RISK_CLASSES = {"Pass", "Fail"}
CLASS_NOTES = {
    "Distinction": "Trajectory is excellent and Direct Entry eligible. Sustain current study habits and consider peer-tutoring roles to consolidate mastery.",
    "Upper Credit": "Strong trajectory; Upper Credit keeps HND and Direct Entry doors open. Targeted improvement below could reach Distinction.",
    "Lower Credit": "Stable but improvable. The factors below are holding this profile back and respond well to early intervention.",
    "Pass": "This profile is academically at risk. Early support this semester is strongly advised, focusing on the factors below.",
    "Fail": "This profile is at high risk of carry-overs and failure. Immediate intervention is advised: structured tutoring, counselling and a monitored study plan.",
}

def recommendations(m: dict, predicted: str) -> list[str]:
    r = []
    if m["Hours_Studied"] < 15: r.append("Raise weekly private study toward 18–25 hours with a fixed timetable.")
    if m["Attendance"] < 75: r.append("Attending fewer than 15 of every 20 class days; consistent lecture attendance is the single strongest lever.")
    if m["Previous_Scores"] < 50: r.append("Prior CGPA is below the Lower Credit range; schedule remedial sessions on foundational courses.")
    if m["Tutoring_Sessions"] == 0 and predicted in RISK_CLASSES: r.append("Enrol in at least one departmental tutorial session per week.")
    if m["Motivation_Level"] == "Low": r.append("Pair with an academic mentor or study group to build motivation.")
    if m["Access_to_Resources"] == "Low": r.append("Connect the student to library e-resources and the departmental past-question bank.")
    if m["Internet_Access"] == "No": r.append("Arrange campus ICT-centre access for online learning materials.")
    if m["Sleep_Hours"] < 6: r.append("Under 6 hours of sleep impairs retention; target 7–8 hours.")
    if m["Peer_Influence"] == "Negative": r.append("Encourage a change of study circle; peer influence is currently negative.")
    if m["Learning_Disabilities"] == "Yes": r.append("Refer to the student support unit for learning-accommodation options.")
    if not r and predicted in RISK_CLASSES:
        r.append("No single weak factor stands out, yet the profile is at risk; assign an academic adviser and review progress at mid-semester.")
    elif not r:
        r.append("No weak factors detected in this profile; maintain current habits.")
    return r[:5]
