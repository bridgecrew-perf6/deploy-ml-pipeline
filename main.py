import joblib
import logging
import pandas as pd
import constants as constans_project

from typing import Union
from webbrowser import BaseBrowser
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.ml.data import process_data
from src.ml.model import inference

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

ENCODER = joblib.load(constans_project.MODEL_PATH +
                      constans_project.ENCODER_FILE)
LB = joblib.load(constans_project.MODEL_PATH +
                 constans_project.LB_FILE)
MODEL = joblib.load(constans_project.MODEL_PATH +
                    constans_project.MODEL_FILE)

CAT_FEATURES = constans_project.CAT_FEATURES

class CensusInputItem(BaseModel):
    age: int
    workclass: str
    fnlgt: float
    education: str
    education_num: int = Field(None, alias="education-num")
    marital_status: str = Field(None, alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(None, alias="capital-gain")
    capital_loss: float = Field(None, alias="capital-loss")
    hours_per_week: int = Field(None, alias="hours-per-week")
    native_country: str = Field(None, alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 43,
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
        }

app = FastAPI()

# GET endpoint
@app.get("/")
async def say_greeting():
    return {"greeting": "Welcome to the deploy-ml-pipeline project!"}

@app.post("/inference")
async def model_inference(item: CensusInputItem):

    df_preds = pd.DataFrame(jsonable_encoder(item),
                            index=["value"])
    
    data, _, _, _ = process_data(
        df_preds, 
        categorical_features=CAT_FEATURES, 
        label=None, training=False,
        encoder=ENCODER, lb=LB
    )
    
    preds = inference(MODEL, data)


    if preds[0] == 0:
        item_prediction = '<=50K'
    else:
        item_prediction = '>50K'

    return {"prediction": item_prediction}