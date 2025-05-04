"""This script uses the packages ml/data.py and ml/models.py to train a
ML model that predicts that salary of a worker based on a set
of features.

Returns:
    _type_: _description_
"""

# Script to train machine learning model.

import logging

from sklearn.model_selection import train_test_split

import pandas as pd
from ml.data import *
from  ml.model import *
import joblib


logging.basicConfig(level=logging.INFO,
                    filename='./logs/train_model.log',
                    filemode='w',
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add the necessary imports for the starter code.

# Add code to load in the data.

def load_and_preprocess_data(dataset_name='./data/census_clean.csv'):
    logger.info(f"Loading data from {dataset_name}")
    data = pd.read_csv(dataset_name)
    data = data.drop_duplicates()
    return data

def model_slices(model, data, categorical_features, encoder, lb):

    for cat in categorical_features:
        for val in data[cat].unique():
            data_tmp = data[data[cat] == val]
            X, y, _, _ = process_data(
                data_tmp, categorical_features=categorical_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            y_pred = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, y_pred)
            msg = f" == category {cat}/{val}: "
            msg += f"prec={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}"
            logging.info(msg)


if __name__ == "__main__":

    data = load_and_preprocess_data()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True,
        encoder=None, lb=None
    )

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )

    # Save preprocessing, e.g. encoder
    joblib.dump(encoder, './model/encoder_test.pkl')
    joblib.dump(lb, './model/lb_test.pkl')

    # Train and save a model.
    logging.info("Training ML model")
    model = train_model(X_train, y_train)

    model_name = './model/rfc_model_test.pkl'
    logging.info(f"Saving ML model to file {model_name}")
    joblib.dump(model, model_name)

    logging.info("Computing model metrics")

    y_train_pred = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, y_train_pred)

    logging.info(" Metrics on training data:")
    logging.info(f" prec={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}")

    y_test_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_pred)

    logging.info(" Metrics on test data:")
    logging.info(f" prec={precision:.2f}, recall={recall:.2f}, fbeta={fbeta:.2f}")

    logging.info(" Metrics on slices:")
    model_slices(model, data, cat_features, encoder, lb)
