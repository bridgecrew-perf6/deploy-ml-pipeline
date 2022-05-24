from ml.data import process_data
from ml.model import compute_model_metrics
from ml.model import inference

def compute_metrics_slices(model, encoder, lb, test, cat_feature):
    """ Run model inferences on slices of categorical data.
    Inputs
    ------
    model : 
        Trained Random Forest model with the 
        best parameters after performing grid search.
    test : pandas DataFrame
        Data used for testing the model.
    cat_feature :
        Categorical variable for computing the slice metrics
    Returns
    -------
    metrics : Dictionary
        Metrics for the slice of data.
    """

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

    model_slices = {}

    for cls_class in test[cat_feature].unique():
        data_test = test[test[cat_feature] == cls_class].copy()

        X_test, y_test, _, _ = process_data(
        data_test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
        )

        preds = inference(model, X_test)

        metrics = {}
        fbeta, precision, recall = compute_model_metrics(y_test, preds)
        metrics["fbeta"] = fbeta
        metrics["precision"] = precision
        metrics["recall"] = recall

        model_slices[cls_class] = metrics

    return model_slices