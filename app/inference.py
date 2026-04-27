import joblib
import numpy as np
from app.config import *
from datetime import datetime

import os

print("CWD:", os.getcwd())
print("MODEL_DIR:", MODEL_DIR)
print("FILES:", list(MODEL_DIR.glob("*")))


try:
    grade_model = joblib.load(GRADE_MODEL_PATH)
    stress_model = joblib.load(STRESS_MODEL_PATH)
    scaler_g = joblib.load(SCALER_G_PATH)
    scaler_s = joblib.load(SCALER_S_PATH)
    balance_cap = joblib.load(BALANCE_CAP_PATH)

    MODELS_LOADED = True
    print("Models loaded successfully")
except Exception as e:
    MODELS_LOADED = False
    print("Model loading failed:", e)


def sanitize(f):
    # Only clamp pathological outliers — do NOT cap OnlineCourses at 10
    # The scaler was fit on the full uncapped distribution
    f.Discussions = min(f.Discussions, 50)   # hard outlier guard only
    return f

def engineer(f):
    # ── Grade model composites (must match notebook cell 10 exactly) ──────
    engagement    = f.Discussions + f.AssignmentCompletion   # OnlineCourses excluded
    consistency   = f.Attendance * f.StudyHours
    tech_intensity = f.EduTech * f.OnlineCourses

    # ── Stress model composites (must match notebook cell 12 exactly) ─────
    workload   = f.StudyHours + f.AssignmentCompletion
    social_load = f.Discussions                              # OnlineCourses excluded

    raw_balance = f.StudyHours / (f.OnlineCourses + 1)
    balance     = min(raw_balance, balance_cap)              # 99th-pct clip from training

    overload    = workload / (f.Attendance / 100 + 0.1)     # ratio, not product

    grade_vec = [
        f.StudyHours, f.Attendance, f.Resources,
        f.OnlineCourses, f.Discussions,
        f.AssignmentCompletion, f.EduTech,
        engagement, consistency, tech_intensity              # 10 features
    ]
    stress_vec = [
        f.StudyHours, f.Attendance,
        f.OnlineCourses, f.Discussions,
        f.AssignmentCompletion, f.EduTech,
        f.Extracurricular,
        workload, social_load, balance, overload             # 11 features
    ]
    return grade_vec, stress_vec

def bam(grade, stress):
    if grade >= 2 and stress == 0: return "Optimal"
    if grade >= 2 and stress == 1: return "Monitor"
    if grade >= 2 and stress == 2: return "Burnout Risk"
    if grade < 2 and stress == 0: return "Underperforming"
    if grade < 2 and stress == 1: return "At Risk"
    return "Critical"

def risk(grade, stress):
    return ((1 - grade/3) + (stress/2)) / 2

def predict(req):
    f = sanitize(req.features)
    g, s = engineer(f)

    Xg = scaler_g.transform([g])
    Xs = scaler_s.transform([s])

    # Let the ML models do their job!
    gp = int(grade_model.predict(Xg)[0])
    sp = int(stress_model.predict(Xs)[0])
    
    assert scaler_g.n_features_in_ == 10, \
    f"Grade scaler expects 10 features, got {scaler_g.n_features_in_}"
    assert scaler_s.n_features_in_ == 11, \
    f"Stress scaler expects 11 features, got {scaler_s.n_features_in_}"
    print("Feature count assertions passed.")

    return {
        "student_id": req.student_id,
        "grade": gp,
        "stress": sp,
        "bam": bam(gp, sp),
        "risk_score": round(risk(gp, sp), 4),
        "timestamp": datetime.utcnow().isoformat()
    }