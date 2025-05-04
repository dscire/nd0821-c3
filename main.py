# Put the code for your API here.
#from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starter.ml.model import *
import joblib


app = FastAPI(
    title="My model API",
    description="TO DO",
    version="0.1.0"
)

# model = joblib.load('./model/rfc_model.pkl')
# encoder= joblib.load('./model/encoder.pkl')
# lb = joblib.load('./model/lb.pkl')


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my API!"}
