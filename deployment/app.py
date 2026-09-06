"""
Streamlit application for machine-failure prediction.

The application collects operational parameters, passes them to the
fitted preprocessing-and-XGBoost pipeline, and displays the predicted
class together with the estimated failure probability.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Configure the browser tab and application layout
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered",
)

# Locate the trained model stored beside this application
APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = (
    APP_DIRECTORY
    / "best_machine_failure_model_v1.joblib"
)

FAILURE_THRESHOLD = 0.50


@st.cache_resource
def load_model():
    """
    Load and cache the fitted machine-failure prediction pipeline.

    Caching prevents the model from being reloaded from disk whenever
    a user changes an input control.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(
        "The prediction model could not be loaded. "
        "Please contact the application administrator."
    )

    with st.expander("Technical details"):
        st.code(str(error))

    st.stop()


st.title("⚙️ Machine Failure Prediction")

st.write(
    """
    This application estimates the likelihood of machine failure from
    selected configuration and operating parameters. Enter the current
    measurements and select **Predict failure risk**.
    """
)

st.info(
    "The model is designed to prioritise detecting potential failures. "
    "A high result should therefore trigger further technical "
    "inspection rather than be treated as a confirmed diagnosis."
)

# Use a form so prediction occurs only after all inputs are submitted
with st.form("machine_failure_form"):

    machine_type = st.selectbox(
        label="Machine type",
        options=["H", "L", "M"],
        help=(
            "Select the machine-quality category: "
            "H = high, L = low, or M = medium."
        ),
    )

    air_temperature = st.number_input(
        label="Air temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=298.0,
        step=0.1,
        format="%.1f",
    )

    process_temperature = st.number_input(
        label="Process temperature",
        min_value=250.0,
        max_value=500.0,
        value=324.0,
        step=0.1,
        format="%.1f",
    )

    rotational_speed = st.number_input(
        label="Rotational speed (RPM)",
        min_value=0,
        max_value=3_000,
        value=1_400,
        step=1,
    )

    torque = st.number_input(
        label="Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1,
        format="%.1f",
    )

    tool_wear = st.number_input(
        label="Tool wear (minutes)",
        min_value=0,
        max_value=300,
        value=10,
        step=1,
    )

    submitted = st.form_submit_button(
        "Predict failure risk",
        type="primary",
        use_container_width=True,
    )


if submitted:

    # Use the exact feature names expected by the fitted pipeline
    input_data = pd.DataFrame(
        [
            {
                "Air temperature": air_temperature,
                "Process temperature": process_temperature,
                "Rotational speed": rotational_speed,
                "Torque": torque,
                "Tool wear": tool_wear,
                "Type": machine_type,
            }
        ]
    )

    try:
        failure_probability = float(
            model.predict_proba(input_data)[0, 1]
        )

        prediction = int(
            failure_probability >= FAILURE_THRESHOLD
        )

        st.subheader("Prediction result")

        probability_percentage = (
            failure_probability * 100
        )

        st.metric(
            label="Estimated failure probability",
            value=f"{probability_percentage:.2f}%",
        )

        st.progress(
            min(
                max(failure_probability, 0.0),
                1.0,
            )
        )

        if prediction == 1:
            st.error(
                "⚠️ Potential machine failure detected"
            )

            st.write(
                "The machine should be referred for technical "
                "inspection and appropriate maintenance assessment."
            )

        else:
            st.success(
                "✅ No immediate machine failure detected"
            )

            st.write(
                "Continue routine monitoring and preventive "
                "maintenance. This result does not guarantee that "
                "a future failure will not occur."
            )

        with st.expander("Submitted machine information"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.error(
            "The application could not generate a prediction. "
            "Please review the inputs or contact the administrator."
        )

        with st.expander("Technical details"):
            st.code(str(error))


with st.expander("Model performance and limitations"):
    st.markdown(
        """
        The model achieved the following held-out test results:

        - **Failure recall:** 92.65%
        - **Failure precision:** 19.81%
        - **Failure F1-score:** 32.64%
        - **ROC-AUC:** 94.38%
        - **PR-AUC:** 43.82%

        The model is intentionally sensitive to possible failures and
        may therefore generate false alarms. Predictions should support,
        rather than replace, engineering inspection and professional
        maintenance decisions.
        """
    )

st.caption(
    "Machine Failure Prediction — MLOps demonstration application"
)
