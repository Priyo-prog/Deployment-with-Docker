from fastapi import FastAPI, Query, Body
from pydantic import BaseModel,Field

import pandas as pd

from prometheus_client import Gauge, Histogram, Summary, Counter
from prometheus_fastapi_instrumentator import Instrumentator

import joblib
import uvicorn

app = FastAPI()

model = joblib.load("saved_models/regression_model.pkl")

prediction_count = Counter("model_prediction_total", "Total No. of Model Predictions")
prediction_latency = Histogram("model_prediction_latency_seconds", "Latency of Prediction in Seconds")
prediction_errors = Counter("model_predictions_errors_total", "Total No model predictions errors")

class House(BaseModel):
    Id : int
    LotArea : int
    OverallQual : int
    TotalBsmtSF : int
    SndFlrSF : int
    GrLivArea : int
    BsmtHalfBath : int
    HalfBath : int
    GarageCars : int
    MiscVal : int

@app.post("/predict/")
async def predict(house: House):
    data = pd.DataFrame(house.model_dump())
    y_pred = model.predict(data)[0]    

    return {"Prediction": int(y_pred)}

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000)