from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Medical Diagnostics API", description="API для анализа медицинских метрик")

# Загружаем обученную модель
model = joblib.load('model.joblib')


class MedicalRecord(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float

@app.get("/")
def read_root():
    return {"message": "Medical API is running. Ready for diagnostics!"}


@app.post("/predict")
def predict_diagnosis(record: MedicalRecord):
    data = np.array([[
        record.mean_radius,
        record.mean_texture,
        record.mean_perimeter,
        record.mean_area,
        record.mean_smoothness
    ]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        diagnosis = "Benign (Доброкачественная - Низкий риск)"
    else:
        diagnosis = "Malignant (Злокачественная - Высокий риск)"

    return {
        "diagnosis": diagnosis,
        "analyzed_features": 5
    }