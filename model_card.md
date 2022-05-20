# Model Card

Last updated: May 2022

## Model Details

This is a RandomForestClassifier with the following hyperparameters:

{'criterion': 'gini', 'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 20} 

## Intended Use

This model is used to classify the salary of individuals (<=50K and >50K) using Census data.

## Training Data

The data used for training comes from [Census data](https://archive.ics.uci.edu/ml/datasets/census+income). It has the following features:

* age 
* workclass
* fnlgt
* education
* education-num
* marital-status
* occupation
* relationship
* race
* sex
* capital-gain
* capital-loss
* hours-per-week
* native-country 
* salary

## Evaluation Data

The Census data was split into training and test with test_size = 20%

## Metrics
The model achived the following metrics:

Fbeta score=0.7744593202883625
Precision=0.48360128617363346
Recall=0.5954077593032463

## Ethical Considerations

The data that was used for training has bias since it has more data about male individuals from the USA.

## Caveats and Recommendations

The can model can be improved by adding other sources of data such as country (GDP). Also, we could try other training algorithms such as XGBoost.