# deploy-ml-pipeline

This Machine Learning pipeline trains a Random Forest model to classify the salary of individuals (<=50K and >50K) using Census data. It also deploys the best model using FastAPI and Heroku.

## Contents of the project

This project contains the following:

* **data:** Data for training and testing the model.

## Running the project

Before y ou run this project, make sure you satisfy the following requirements:

* Python >= 3.8
* Conda


1. Create a conda environment:
   
   To create a conda environment using yml file:

   ```
   conda env create --name deploy-ml-pipeline --file=requirements.yml
   conda activate deploy-ml-pipeline
   ```

   To create a virtual environment using requirements.txt:

   ```
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. Train the Random Forest model:

   ```
   python train_model/run.py
   ```

3. Run the REST API that serves the model locally:

   ```
   uvicorn main:app --reload
   ```

>Note: To check the docs, go to http://127.0.0.1:8000/docs.

## Running the tests

To run the tests:

```
python -m pytest
```

