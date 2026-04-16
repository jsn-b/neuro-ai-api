import os
from pathlib import Path

MODEL_DIR = os.getenv("MODEL_DIR")

if not MODEL_DIR:
    raise RuntimeError("MODEL_DIR must be explicitly set")

MODEL_DIR = Path(MODEL_DIR)

GRADE_MODEL_PATH = MODEL_DIR / "grade_model.pkl"
STRESS_MODEL_PATH = MODEL_DIR / "stress_model.pkl"
SCALER_G_PATH = MODEL_DIR / "scaler_g.pkl"
SCALER_S_PATH = MODEL_DIR / "scaler_s.pkl"