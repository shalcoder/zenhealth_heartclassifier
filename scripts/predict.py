# scripts/predict.py - FIXED VERSION
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Cardiac Disease Predictor API", version="1.0")

# Load model and scaler at startup
try:
    model = joblib.load('models/cardiac_disease_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # Load feature names
    with open('models/feature_names.txt', 'r') as f:
        FEATURE_NAMES = f.read().strip().split(',')

    print(f"✓ Model loaded successfully")
    print(f"✓ Scaler loaded successfully")
    print(f"✓ Feature names: {FEATURE_NAMES}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

class PatientInput(BaseModel):
    st_slope: conint(ge=0, le=3)
    exercise_angina: conint(ge=0, le=1)
    chest_pain_type: conint(ge=1, le=4)
    max_heart_rate: conint(ge=60, le=202)
    oldpeak: confloat(ge=-2.6, le=6.2)
    sex: conint(ge=0, le=1)
    age: conint(ge=28, le=77)

    class Config:
        json_schema_extra = {
            "example": {
                "st_slope": 2,
                "exercise_angina": 1,
                "chest_pain_type": 4,
                "max_heart_rate": 120,
                "oldpeak": 1.5,
                "sex": 1,
                "age": 60
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Cardiac Disease Predictor API",
        "version": "1.0",
        "model_accuracy": "90.76%",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post('/predict')
async def predict(input: PatientInput):
    try:
        # FIX: Create DataFrame with proper feature names
        input_df = pd.DataFrame([{
            'ST slope': input.st_slope,
            'exercise angina': input.exercise_angina,
            'chest pain type': input.chest_pain_type,
            'max heart rate': input.max_heart_rate,
            'oldpeak': input.oldpeak,
            'sex': input.sex,
            'age': input.age
        }])

        # Scale using DataFrame (preserves feature names)
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        probability = float(probabilities[prediction])

        # Determine risk level
        disease_prob = float(probabilities[1])
        if disease_prob < 0.4:
            risk_level = "Low"
        elif disease_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "prediction": prediction,
            "prediction_label": "Disease Detected" if prediction == 1 else "No Disease",
            "probability": round(probability, 4),
            "disease_probability": round(disease_prob, 4),
            "no_disease_probability": round(float(probabilities[0]), 4),
            "risk_level": risk_level,
            "confidence": f"{probability*100:.2f}%"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get('/features')
async def get_features():
    return {
        "features": FEATURE_NAMES,
        "count": len(FEATURE_NAMES)
    }
