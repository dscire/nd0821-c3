# Put the code for your API here.
#from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starter.ml.model import *
import joblib


app = FastAPI(
    title="My salary prediction API",
    description="This API predict whether income exceeds $50K/yr based on a RandomForest classifier trained on census data.",
    version="0.1.0"
)

# model = joblib.load('./model/rfc_model.pkl')
# encoder= joblib.load('./model/encoder.pkl')
# lb = joblib.load('./model/lb.pkl')


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my API!"}
