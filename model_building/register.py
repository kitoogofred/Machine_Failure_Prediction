"""
Load, validate and profile the machine-failure dataset.

This script:
1. Locates the dataset relative to the repository root.
2. Loads the CSV file.
3. Confirms that the required variables are present.
4. Checks that the dataset contains records.
5. Displays its dimensions, columns and target distribution.

The relative-path design allows the script to run locally, in Colab
and through GitHub Actions.
"""

from pathlib import Path

import pandas as pd


# Determine the repository root from this script's location:
# repository_root/model_building/register.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Construct a reliable path to the raw dataset
RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "machine-failure-prediction.csv"
)

# Define the expected dataset schema
EXPECTED_COLUMNS = [
    "UDI",
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Failure",
]


def register_dataset() -> pd.DataFrame:
    """
    Load and validate the machine-failure dataset.

    Returns
    -------
    pandas.DataFrame
        Validated machine-failure dataset.

    Raises
    ------
    FileNotFoundError
        If the expected CSV file cannot be found.
    ValueError
        If the dataset is empty or required columns are missing.
    """

    # Confirm that the dataset exists
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_PATH}"
        )

    # Load the raw dataset
    dataframe = pd.read_csv(RAW_PATH)

    # Confirm that the dataset contains records
    if dataframe.empty:
        raise ValueError(
            "The machine-failure dataset contains no records."
        )

    # Identify any required variables that are missing
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing expected columns: "
            f"{missing_columns}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    # Display registration and validation information
    print("✅ Dataset registered successfully.")
    print(f"Dataset path: {RAW_PATH}")
    print(
        f"Rows: {dataframe.shape[0]:,}, "
        f"Columns: {dataframe.shape[1]}"
    )
    print("Columns:", list(dataframe.columns))

    print("\nFailure distribution:")
    print(
        dataframe["Failure"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nFailure distribution (%):")
    print(
        dataframe["Failure"]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .sort_index()
        .mul(100)
        .round(2)
    )

    return dataframe


if __name__ == "__main__":
    register_dataset()
