# Script to train machine learning model.

import logging
import pandas as pd

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml import process_data

# Configure logging
logging.basicConfig(
    filename='../logs/ml-pipeline.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Add code to load in the data.
data = pd.read_csv("../data/clean.csv")
logging.info("Load clean data")

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
