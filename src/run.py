# Script to train machine learning model.
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
from ml.metrics_slices import compute_metrics_slices

# Add code to load in the data.
try:
    data = pd.read_csv("../data/clean.csv")
    print("Load clean data")
except FileNotFoundError:
    print("Clean data CSV file not found")

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

X_train, y_train, encoder_train, lb_train = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder_train, lb=lb_train
)

# Train and save a model.
model = train_model(X_train, y_train)

preds = inference(model, X_test)

# Compute metrics for all of the test data
fbeta, precision, recall = compute_model_metrics(y_test, preds)

print("Model metrics: ")
print("Fbeta score: %s", fbeta)
print("Precision: %s", precision)
print("Recall: %s", recall)

#Compute slices metrics for a given categorical feature
cat_feature = "education"
metric_slices = compute_metrics_slices(model, encoder_train, 
                                       lb_train, test, cat_feature)

print("Model metrics for categorical feature %s", cat_feature)
print(metric_slices)

# Save the slices metrics to text file
try:
    with open("../slice_output.txt", "a") as file_object:
        file_object.write(str(metric_slices))
    print("Saving slice metrics to slice_output.txt")
except FileNotFoundError:
    print("slice_output.txt file not found")

# Save the model and the Onehotencoder
try:
    joblib.dump(
            model,
            "../model/" +
            'model.pkl')
    print("Saved best model.")

    joblib.dump(
            lb_train,
            "../model/" +
            'lb.pkl')
    print("Saved LabelBinarizer.")

    joblib.dump(
            encoder_train,
            "../model/" +
            'encoder.pkl')
    print("Saved Onehotencoder.")

except Exception as err:
    print("Error while saving best model and encoder: %s ", 
                  err)
