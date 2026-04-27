from fastapi import FastAPI, HTTPException
from app.schemas import InferenceRequest
from app.inference import predict, MODELS_LOADED

app = FastAPI()


@app.get("/")
def root():
    return "<html><h1>ML API version 3.0</h1><h3>You are good to go</h3></html>"

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/predict")
def run_prediction(req: InferenceRequest):
    if not MODELS_LOADED:
        raise RuntimeError("Models not loaded")
    try:
        return predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
