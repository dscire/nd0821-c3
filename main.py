# Put the code for your API here.
#from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from starter.ml.model import *
#from starter.ml.data import *
from starter.ml import model, data
import pandas as pd
import joblib


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

class WorkerInfo(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example":
                {
                    "age": 42,
                    "workclass": "Private",
                    "fnlgt": 159449,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 5178,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                }
        }


app = FastAPI(
    title="My salary prediction API",
    description="This API predict whether income exceeds $50K/yr based on a RandomForest classifier trained on census data.",
    version="0.1.0"
)

rfcmodel, encoder, lb = model.load_models(naming='_deploy')

# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my old API!"}


@app.post("/inference")
async def inference(worker_info: WorkerInfo):
    worker_dict = worker_info.dict()
    worker_dict = {k:[v] for k,v in worker_dict.items()}

    df = pd.DataFrame.from_dict(worker_dict)
    df.rename(columns=lambda x: x.replace("_","-"), inplace=True)

    X, _, _, _ = data.process_data(
    df, categorical_features=cat_features,
    encoder=encoder, lb=lb, training=False)
    pred = model.inference(rfcmodel, X)
    y = lb.inverse_transform(pred)[0]
    return {"prediction": y}
