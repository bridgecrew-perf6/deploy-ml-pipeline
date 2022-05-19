# deploy-ml-pipeline

This Machine Learning pipeline trains a Random Forest model on Census data and deploys the best model using FastAPI and Heroku.

## Contents of the project

## Running the project

Before y ou run this project, make sure you satisfy the following requirements:

* Python >= 3.8
* Conda


1. Create a conda environment:
   
   To create a conda environment using yml file:

   ```
   conda env create --name deploy-ml-pipeline --file=requirements.yml
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

