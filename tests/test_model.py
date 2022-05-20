import joblib
import pandas as pd
import numpy as np
import pytest
import sys

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "../")

from train_model.ml.data import process_data
from train_model.ml.model import train_model
from train_model.ml.model import inference

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

@pytest.fixture
def data():
    return pd.read_csv("data/clean.csv")

@pytest.fixture
def test():
    return pd.read_csv("data/test.csv")

def test_process_data(data):

    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    
    assert X_train.shape[0] > 0
    assert X_train.shape[1] > 0
    assert isinstance(y_train, np.ndarray) 
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

def test_train_model(data):

    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference(test):

    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")

    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=CAT_FEATURES, 
        label=None, training=False,
        encoder=encoder, lb=lb
    )

    model = joblib.load("model/model.pkl")
    
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)