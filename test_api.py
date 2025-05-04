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
    assert r.json() == {"greeting": "Welcome to my API!"}
