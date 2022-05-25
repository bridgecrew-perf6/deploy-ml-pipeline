from typing import Union
from webbrowser import BaseBrowser
from fastapi import FastAPI
from pydantic import BaseModel, Field

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
    return item