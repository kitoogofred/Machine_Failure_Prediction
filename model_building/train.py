"""
Train and evaluate the machine-failure prediction model.

The complete fitted pipeline pipeline, including preprocessing and the
XGBoost classifier, is saved for use by the Streamlit application.
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Resolve paths relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

XTRAIN_PATH = PROJECT_ROOT / "Xtrain.csv"
XTEST_PATH = PROJECT_ROOT / "Xtest.csv"
YTRAIN_PATH = PROJECT_ROOT / "ytrain.csv"
YTEST_PATH = PROJECT_ROOT / "ytest.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "best_machine_failure_model_v1.joblib"
)

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]
TARGET_COLUMN = "Failure"


def load_prepared_data():
    """
    Load and validate the prepared training and test datasets.
    """

    required_files = [
        XTRAIN_PATH,
        XTEST_PATH,
        YTRAIN_PATH,
        YTEST_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Prepared dataset files are missing: {missing_files}"
        )

    Xtrain = pd.read_csv(XTRAIN_PATH)
    Xtest = pd.read_csv(XTEST_PATH)

    ytrain_frame = pd.read_csv(YTRAIN_PATH)
    ytest_frame = pd.read_csv(YTEST_PATH)

    if TARGET_COLUMN not in ytrain_frame.columns:
        raise ValueError(
            f"{YTRAIN_PATH.name} does not contain "
            f"'{TARGET_COLUMN}'."
        )

    if TARGET_COLUMN not in ytest_frame.columns:
        raise ValueError(
            f"{YTEST_PATH.name} does not contain "
            f"'{TARGET_COLUMN}'."
        )

    ytrain = ytrain_frame[TARGET_COLUMN]
    ytest = ytest_frame[TARGET_COLUMN]

    expected_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_training_features = [
        feature
        for feature in expected_features
        if feature not in Xtrain.columns
    ]

    missing_test_features = [
        feature
        for feature in expected_features
        if feature not in Xtest.columns
    ]

    if missing_training_features:
        raise ValueError(
            "Training data is missing features: "
            f"{missing_training_features}"
        )

    if missing_test_features:
        raise ValueError(
            "Test data is missing features: "
            f"{missing_test_features}"
        )

    if len(Xtrain) != len(ytrain):
        raise ValueError(
            "Training features and target are not aligned."
        )

    if len(Xtest) != len(ytest):
        raise ValueError(
            "Test features and target are not aligned."
        )

    return Xtrain, Xtest, ytrain, ytest


def train_model():
    """
    Tune, evaluate and save the machine-failure classifier.
    """

    Xtrain, Xtest, ytrain, ytest = load_prepared_data()

    # Confirm that both target classes are present
    class_counts = ytrain.value_counts().sort_index()

    if 0 not in class_counts.index or 1 not in class_counts.index:
        raise ValueError(
            "Training target must contain classes 0 and 1."
        )

    # Give the minority failure class greater training importance
    scale_pos_weight = (
        class_counts.loc[0]
        / class_counts.loc[1]
    )

    print("Training class distribution:")
    print(class_counts)
    print(
        f"Calculated scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # Preprocess numerical and categorical variables consistently
    preprocessor = make_column_transformer(
        (
            StandardScaler(),
            NUMERIC_FEATURES,
        ),
        (
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES,
        ),
        remainder="drop",
    )

    # Limit XGBoost to one thread because GridSearchCV manages
    # parallelism across candidate models
    classifier = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        eval_metric="logloss",
    )

    pipeline = make_pipeline(
        preprocessor,
        classifier,
    )

    # Pipeline-generated estimator name is xgbclassifier
    parameter_grid = {
        "xgbclassifier__n_estimators": [50, 100],
        "xgbclassifier__max_depth": [2, 3],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=cross_validation,
        scoring="recall",
        n_jobs=-1,
        refit=True,
        error_score="raise",
        verbose=1,
    )

    print("\nStarting hyperparameter search...")
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # Generate class predictions and failure probabilities
    predictions = best_model.predict(Xtest)
    failure_probabilities = (
        best_model.predict_proba(Xtest)[:, 1]
    )

    # Calculate evaluation metrics
    test_precision = precision_score(
        ytest,
        predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        ytest,
        predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        ytest,
        predictions,
        zero_division=0,
    )

    test_roc_auc = roc_auc_score(
        ytest,
        failure_probabilities,
    )

    test_pr_auc = average_precision_score(
        ytest,
        failure_probabilities,
    )

    test_confusion_matrix = confusion_matrix(
        ytest,
        predictions,
    )

    print("\n✅ Hyperparameter search completed.")
    print("Best parameters:", grid_search.best_params_)
    print(
        "Best cross-validation recall: "
        f"{grid_search.best_score_:.4f}"
    )

    print("\nTest classification report:")
    print(
        classification_report(
            ytest,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Test confusion matrix:")
    print(test_confusion_matrix)

    print("\nTest metrics for failure prediction:")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC: {test_pr_auc:.4f}")

    # Ensure that the deployment directory exists
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete preprocessing-and-model pipeline
    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model was not saved at: {MODEL_PATH}"
        )

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(
        f"Model file size: "
        f"{MODEL_PATH.stat().st_size:,} bytes"
    )

    return best_model


if __name__ == "__main__":
    train_model()
