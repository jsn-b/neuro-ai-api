# Student Performance Inference API

## Overview
Dual-model ML system predicting student academic performance and stress.

## Features
- Random Forest models
- Feature engineering pipeline
- BAM risk system
- FastAPI deployment

## Run Locally
```bash
export MODEL_DIR=./artifacts
uvicorn app.main:app --reload