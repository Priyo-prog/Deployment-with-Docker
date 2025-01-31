from fastapi import FastAPI, Query, Body
from pydantic import BaseModel,Field, create_model
from typing import List, Tuple, Dict, Any

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

# class House(BaseModel):
#     MSSubClass : int
#     OverallQual : int
#     BsmtFinSF1  : int
#     TotalBsmtSF : int
#     SndFlrSF  : int
#     LowQualFinSF   : int
#     GrLivArea  : int
#     HalfBath : int
#     BedroomAbvGr : int
#     WoodDeckSF : int

# Function to create a Pydantic model dynamically from a text file
def create_pydantic_model_from_file(file_path: str) -> BaseModel:
    fields = {}
    with open(file_path, "r") as file:
        for line in file:
            name, type_hint = line.strip().split(":")
            fields[name] = (eval(type_hint), ...)
    
    return create_model("House", **fields)    

House = create_pydantic_model_from_file("saved_models/features.txt")

@app.post("/predict/")
async def predict(houses: List[House]):
    data = pd.DataFrame([house.model_dump() for house in houses])
    y_pred = model.predict(data)[0]    

    return {"Prediction": int(y_pred)}

Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000)