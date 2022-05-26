import sys
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_root_path():
    response = client.get("/")
    assert response.status_code == 200

def test_post_path_zero_class():
    data = {"age": 43,
              "workclass": "Federal-gov",
              "fnlgt": 77516,
              "education": "Bachelors",
              "education-num": 13,
              "marital-status": "Divorced",
              "occupation": "Adm-clerical",
              "relationship": "Wife",
              "race": "Black",
              "sex": "Male",
              "capital-gain": 0.0,
              "capital-loss": 0.0,
              "hours-per-week": 40.0,
              "native-country": "United-States"}

    response = client.post(
        "/inference",
        json=data
    )

    assert response.status_code == 200
    assert response.json() == {'prediction': '<=50K'}


def test_post_path_one_class():
    data = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 187720,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = client.post(
        "/inference",
        json=data
    )

    assert response.status_code == 200
    assert response.json() == {'prediction': '>50K'}