import os
import pytest
import joblib
import collections
import numpy as np
import pandas as pd
from starter.ml.model import *
from starter.ml.data import *


@pytest.fixture(autouse=True)
def sample_models():
    model = joblib.load('./model/rfc_model.pkl')
    encoder = joblib.load('./model/encoder.pkl')
    lb = joblib.load('./model/lb.pkl')
    return model, encoder, lb


# Create fixture to load mode and sample data for testing
@pytest.fixture(autouse=True)
def sample_data(sample_models):
    data = pd.read_csv('./data/census_clean.csv')

    _, encoder, lb = sample_models

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


def test_train_model(sample_data):
    X, y = sample_data
    assert len(y) == len(X)


def test_compute_model_metrics(sample_data, sample_models):
    X, y = sample_data
    model, _, _ = sample_models
    pred = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, pred)

    assert isinstance(precision, float)
    assert precision >= 0.0
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(sample_data, sample_models):
    X, y = sample_data
    model, _, _ = sample_models
    pred = inference(model, X)

    assert len(pred) == len(y)
    assert isinstance(pred, np.ndarray)


def test_save_models(sample_models):
    model, encoder, lb = sample_models
    test_naming = '_ptst'
    save_models(model, encoder, lb, naming=test_naming)
    assert os.path.isfile('./model/rfc_model'+test_naming+'.pkl')
    assert os.path.isfile('./model/encoder'+test_naming+'.pkl')
    assert os.path.isfile('./model/lb'+test_naming+'.pkl')


def test_load_models():
    model, encoder, lb = load_models(naming='')
    assert model != None
    assert encoder != None
    assert lb != None
