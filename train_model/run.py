# Script to train machine learning model.
import joblib
import logging
import pandas as pd

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference

# Configure logging
logging.basicConfig(
    filename='../logs/ml-pipeline.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Add code to load in the data.
try:
    data = pd.read_csv("../data/clean.csv")
    logging.info("Load clean data")
except FileNotFoundError:
    logging.info("Clean data CSV file not found")

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
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)

fbeta, precision, recall = compute_model_metrics(y_test, preds)

logging.info("Model metrics: ")
logging.info("Fbeta score: %s", fbeta)
logging.info("Precision: %s", precision)
logging.info("Recall: %s", recall)

# Saving the model and the Onehotencoder
try:
    joblib.dump(
            model,
            "../model/" +
            'model.pkl')
    logging.info("Saved best model.")

    joblib.dump(
            encoder,
            "../model/" +
            'encoder.pkl')
    logging.info("Saved Onehotencoder.")

except Exception as err:
    logging.error("Error while saving best model and encoder: %s ", 
                  err)

