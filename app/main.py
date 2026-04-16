from fastapi import FastAPI, HTTPException
from app.schemas import InferenceRequest
from app.inference import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Student Inference API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def run_prediction(req: InferenceRequest):
    try:
        return predict(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))