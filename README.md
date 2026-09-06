# Machine Failure Prediction

This project demonstrates an end-to-end machine learning and MLOps
workflow for predicting industrial machine failure from operational
and sensor data.

## Project objectives

The project is designed to:

- validate and register the source dataset;
- prepare stratified training and test datasets;
- preprocess numerical and categorical variables;
- train and tune an XGBoost classification model;
- evaluate failure-detection performance;
- automate the machine learning pipeline with GitHub Actions; and
- deploy an interactive prediction application using Streamlit
  Community Cloud.

## Project structure

```text
Machine_Failure_Prediction/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── data/
│   └── machine-failure-prediction.csv
├── deployment/
│   ├── app.py
│   ├── best_machine_failure_model_v1.joblib
│   └── requirements.txt
├── model_building/
│   ├── register.py
│   ├── prep.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Machine learning workflow

The automated pipeline runs the following jobs in sequence:

1. **Dataset registration** — loads the dataset and validates its
   structure.
2. **Data preparation** — removes the identifier, separates the
   predictors and target, and creates stratified train/test splits.
3. **Model training** — preprocesses the variables, performs
   hyperparameter tuning, evaluates the model and saves the trained
   pipeline.

## Model

The saved model is a Scikit-learn pipeline containing:

- standard scaling for numerical variables;
- one-hot encoding for machine type; and
- an XGBoost classifier with class-imbalance handling.

The model prioritizes recall for the failure class so that potentially
failing machines are less likely to be missed.

## Application

The Streamlit application accepts the following inputs:

- machine type;
- air temperature;
- process temperature;
- rotational speed;
- torque; and
- tool wear.

It returns a machine-failure prediction and the estimated probability
of failure.

## Run the application locally

Install the application dependencies:

```bash
pip install -r deployment/requirements.txt
```

Start Streamlit from the repository root:

```bash
streamlit run deployment/app.py
```

## Automation and deployment

GitHub Actions executes the machine learning pipeline and commits the
resulting trained model to the repository. Streamlit Community Cloud
serves the application from:

```text
deployment/app.py
```

## Author

**Dr. Fredrick Edward Kitoogo**
