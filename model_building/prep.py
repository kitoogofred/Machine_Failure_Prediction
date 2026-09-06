"""
Prepare the machine-failure dataset for model training.

The Type variable is intentionally retained as raw H/L/M strings.
Categorical encoding will be handled inside the modelling pipeline,
ensuring identical preprocessing during training and prediction.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Resolve the repository root independently of the working directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "machine-failure-prediction.csv"
)

TARGET_COLUMN = "Failure"
IDENTIFIER_COLUMN = "UDI"

# Define output paths expected by the model-training stage
XTRAIN_PATH = PROJECT_ROOT / "Xtrain.csv"
XTEST_PATH = PROJECT_ROOT / "Xtest.csv"
YTRAIN_PATH = PROJECT_ROOT / "ytrain.csv"
YTEST_PATH = PROJECT_ROOT / "ytest.csv"


def prepare_data() -> None:
    """
    Load, validate, split and save the machine-failure dataset.

    The target is stratified to preserve the failure-class ratio in
    both the training and test datasets.
    """

    # Confirm that the raw dataset exists
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}"
        )

    # Load the raw dataset
    dataframe = pd.read_csv(RAW_DATA_PATH)

    if dataframe.empty:
        raise ValueError(
            "The machine-failure dataset contains no records."
        )

    # Validate the columns needed for preparation
    required_columns = {
        IDENTIFIER_COLUMN,
        "Type",
        TARGET_COLUMN,
    }

    missing_columns = sorted(
        required_columns.difference(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    # Remove the record identifier; it is not a predictive feature
    dataframe = dataframe.drop(
        columns=[IDENTIFIER_COLUMN]
    )

    # Ensure the target contains no missing values
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"The target column '{TARGET_COLUMN}' contains "
            "missing values."
        )

    # Ensure the categorical predictor contains no missing values
    if dataframe["Type"].isna().any():
        raise ValueError(
            "The Type column contains missing values."
        )

    # Separate predictors and target
    features = dataframe.drop(
        columns=[TARGET_COLUMN]
    )
    target = dataframe[TARGET_COLUMN]

    # Confirm that stratification is possible
    class_counts = target.value_counts()

    if class_counts.shape[0] < 2:
        raise ValueError(
            "The target must contain at least two classes."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class must contain at least two records "
            "for stratified splitting."
        )

    # Create reproducible stratified training and test datasets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Save prepared datasets in the repository root
    Xtrain.to_csv(
        XTRAIN_PATH,
        index=False
    )

    Xtest.to_csv(
        XTEST_PATH,
        index=False
    )

    ytrain.to_csv(
        YTRAIN_PATH,
        index=False
    )

    ytest.to_csv(
        YTEST_PATH,
        index=False
    )

    print("✅ Data preparation completed successfully.")
    print(f"Training records: {len(Xtrain):,}")
    print(f"Test records: {len(Xtest):,}")
    print(
        "Type values retained as:",
        sorted(features["Type"].unique())
    )

    print("\nOverall failure distribution (%):")
    print(
        target.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTraining failure distribution (%):")
    print(
        ytrain.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTest failure distribution (%):")
    print(
        ytest.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nPrepared files:")
    for output_path in [
        XTRAIN_PATH,
        XTEST_PATH,
        YTRAIN_PATH,
        YTEST_PATH,
    ]:
        print(f"- {output_path}")


if __name__ == "__main__":
    prepare_data()
