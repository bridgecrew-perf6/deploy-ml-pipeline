import requests
import json

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
        "native-country": "United-States"
}

response = requests.post("http://127.0.0.1:8000/inference/", data=json.dumps(data))

print(response.json())