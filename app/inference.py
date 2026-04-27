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
    MODELS_LOADED = True
    print("Models loaded successfully")
except Exception as e:
    MODELS_LOADED = False
    print("Model loading failed:", e)


def sanitize(f):
    f.OnlineCourses = min(f.OnlineCourses, 10)
    f.Discussions = min(f.Discussions, 10)
    return f

def engineer(f):
    engagement = f.OnlineCourses + f.Discussions + f.AssignmentCompletion
    consistency = f.Attendance * f.StudyHours
    tech_intensity = f.EduTech * f.OnlineCourses
    workload = f.StudyHours + f.AssignmentCompletion
    social_load = f.Discussions + f.OnlineCourses
    balance = f.StudyHours / (f.OnlineCourses + 1)
    overload = engagement * f.StudyHours

    grade_vec = [
        f.StudyHours, f.Attendance, f.Resources,
        f.OnlineCourses, f.Discussions,
        f.AssignmentCompletion, f.EduTech,
        engagement, consistency, tech_intensity
    ]

    stress_vec = [
        f.StudyHours, f.Attendance,
        f.OnlineCourses, f.Discussions,
        f.AssignmentCompletion, f.EduTech,
        f.Extracurricular,
        workload, social_load, balance, overload
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

    return {
        "student_id": req.student_id,
        "grade": gp,
        "stress": sp,
        "bam": bam(gp, sp),
        "risk_score": round(risk(gp, sp), 4),
        "timestamp": datetime.utcnow().isoformat()
    }
