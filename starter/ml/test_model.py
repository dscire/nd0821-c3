import pytest
import joblib
import collections
import numpy as np
import pandas as pd
from starter.ml.model import *
from starter.ml.data import *


# Create fixture to load mode and sample data for testing
@pytest.fixture(autouse=True)
def sample_data():
    data = pd.read_csv('./data/census_clean.csv')
    encoder = joblib.load('./model/encoder.pkl')
    lb = joblib.load('./model/lb.pkl')

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

    X, y, _, _ = process_data(data, categorical_features=cat_features,
                              label="salary", training=True,
                              encoder=encoder, lb=lb)
    #ret_val = dataset(X, y)
    #return ret_val
    return X, y


@pytest.fixture(autouse=True)
def foo():
    return 1, 2

@pytest.fixture
def sample_model():
    model = joblib.load('./model/rfc_model.pkl')
    return model

# Gets fixture automagically through autouse
#def test_breaks():
#    arg1, arg2 = foo
#    assert arg1 <= arg2

# Explicit request for fixture foo
def test_works(foo):
    arg1, arg2 = foo
    assert arg1 <= arg2


def test_train_model(sample_data):
    X, y = sample_data
    assert len(y) == len(X)


def test_compute_model_metrics(sample_data, sample_model):
    X, y = sample_data
    pred = inference(sample_model, X)
    precision, recall, fbeta = compute_model_metrics(y, pred)

    assert isinstance(precision, float)
    assert precision >= 0.0
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(sample_data, sample_model):
    X, y = sample_data
    pred = inference(sample_model, X)

    assert len(pred) == len(y)
    assert isinstance(pred, np.ndarray)


# test_compute_model_metrics(sample_data, sample_model)
