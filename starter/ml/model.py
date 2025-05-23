"""This package provides functionalities to load, train and save a machine
learning model that predicts salary range of an individual based on the
relevant "census" features.

Author: Daniele Sciretti
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import logging
import joblib

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    param_grid = {
        'n_estimators': [100],  # [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [5],  # [4,5,100],
        'criterion': ['gini'],  # ['gini', 'entropy']
    }

    logging.info(f"Training RandomForest with grid search:{param_grid}")

    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    rfc = cv_rfc.best_estimator_

    logging.info(f"RandomForest best params are:{rfc.get_params()}")

    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Type may vary
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_models(model, encoder=None, lb=None, naming=""):
    """Save models and encoders to pkl files

    Inputs
    ------
    model (Type may vary):
        Trained sklearn model file
    encoder (Type may vary, optional):
        Defined sklearn encoder. Defaults to None.
    lb (Type may vary, optional):
        Defined sklearn label binarizer. Defaults to None.
    naming (str, optional):
        Naming modifier for saved files. Defaults to ''
    """
    joblib.dump(model, './model/rfc_model'+naming+'.pkl')
    joblib.dump(encoder, './model/encoder'+naming+'.pkl')
    joblib.dump(lb, './model/lb'+naming+'.pkl')


def load_models(naming=""):
    """Load models and encoders from files

    Inputs
    ------
    naming (str, optional):
        Naming modifier for loading files. Defaults to ''
    Returns
    -------
    model (Type may vary):
        Trained sklearn model
    encoder (Type may vary):
        Sklearn encoder.
    lb (Type may vary):
        Sklearn label binarizer.
    """
    model = joblib.load('./model/rfc_model'+naming+'.pkl')
    encoder = joblib.load('./model/encoder'+naming+'.pkl')
    lb = joblib.load('./model/lb'+naming+'.pkl')

    return model, encoder, lb
