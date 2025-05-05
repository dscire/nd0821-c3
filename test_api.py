"""_summary_
"""
import json
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to my updated API!"}

def test_api_inference_below():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_api_inference_above():
    data = {
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
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"
