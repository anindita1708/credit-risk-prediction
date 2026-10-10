
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
API_URL = "http://127.0.0.1:8000/predict"


# ==================================================
# 2. DASHBOARD STYLING
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. FEATURE LABELS
# ==================================================

FEATURE_LABELS = {
    "DAYS_BIRTH": "Age (days)",
    "AGE_YEARS": "Age (years)",
    "DAYS_EMPLOYED": "Employment duration (days)",
    "EMPLOYMENT_YEARS": "Employment duration (years)",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_GOODS_PRICE": "Goods price",
    "AMT_ANNUITY": "Annual loan payment",
    "AMT_CREDIT": "Credit amount",
    "AMT_INCOME_TOTAL": "Annual income",
    "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_GOODS_RATIO": "Credit-to-goods-price ratio",
    "INCOME_PER_FAMILY_MEMBER": "Income per family member",
    "EMPLOYED_ANOMALY": "Employment data anomaly",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family members",
    "CHILDREN_RATIO": "Children-to-family ratio",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Quarterly credit bureau requests",
    "DAYS_ID_PUBLISH": "Days since ID publication",
    "DEF_30_CNT_SOCIAL_CIRCLE": "30-day defaults among social circle",
    "DEF_60_CNT_SOCIAL_CIRCLE": "60-day defaults among social circle",
    "FLAG_OWN_CAR_N": "Does not own a car",
    "OWN_CAR_AGE": "Car age",
    "FLAG_DOCUMENT_3": "Document 3 indicator",
    "CODE_GENDER_F": "Gender: female",
    "CODE_GENDER_M": "Gender: male",
    "NAME_FAMILY_STATUS_Married": "Family status: married",
    "NAME_EDUCATION_TYPE_Higher education": "Education: higher education",
    "NAME_EDUCATION_TYPE_Secondary / secondary special":
        "Education: secondary education",

    # Labels already returned by src/predict.py
    "External Credit Score 1": "External credit score 1",
    "External Credit Score 2": "External credit score 2",
    "External Credit Score 3": "External credit score 3",
    "Goods Price": "Goods price",
    "Loan Annuity": "Annual loan payment",
    "Credit Amount": "Credit amount",
    "Total Income": "Annual income",
    "Credit-to-Income Ratio": "Credit-to-income ratio",
    "Annuity-to-Income Ratio": "Annuity-to-income ratio",
    "Credit-to-Goods Price Ratio": "Credit-to-goods-price ratio",
    "Income per Family Member": "Income per family member",
    "Employment Days": "Employment duration (days)",
    "Employment Duration (Years)": "Employment duration (years)",
    "Employment Data Anomaly": "Employment data anomaly",
    "Number of Children": "Number of children",
    "Number of Family Members": "Family members",
    "Children-to-Family Ratio": "Children-to-family ratio",
    "Credit Bureau Requests in Last Quarter":
        "Quarterly credit bureau requests",
    "Days Since ID Document Published": "Days since ID publication",
    "Car Age": "Car age",
    "Document 3 Provided": "Document 3 indicator",
    "Gender: Female": "Gender: female",
    "Gender: Male": "Gender: male",
    "Education: Higher Education": "Education: higher education",
    "Education: Secondary / Secondary Special":
        "Education: secondary education",
    "Days Since Birth": "Age (days)",
    "30-Day Defaults Among Social Circle":
        "30-day defaults among social circle",
}


def readable_feature_name(name):
    """Convert a model feature name to a readable dashboard label."""
    name = str(name)
    name = name.replace("num__", "").replace("cat__", "")

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    return name.replace("_", " ").strip().title()


# ==================================================
# 4. LOAD BASELINE APPLICANT
# ==================================================

@st.cache_data
def load_baseline():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, nrows=1)

    # The target and applicant ID must not be sent to the model.
    df = df.drop(
        columns=["TARGET", "SK_ID_CURR"],
        errors="ignore",
    )

    return df.iloc[0].to_dict()


def baseline_number(baseline, column, fallback):
    value = baseline.get(column, fallback)

    try:
        value = float(value)
        if pd.notna(value):
            return value
    except (TypeError, ValueError):
        pass

    return float(fallback)


# ==================================================
# 5. HEADER AND SIDEBAR
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>Credit Risk Intelligence</h1>
        <p>
            Home Credit Default Risk Prediction |
            XGBoost Machine Learning |
            SHAP Explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Machine-learning decision-support prototype. "
    "Predictions are estimates and require human review."
)

with st.sidebar:
    st.title("Project Overview")
    st.write("**Model:** XGBoost")
    st.write("**API:** FastAPI")
    st.write("**Explainability:** SHAP")
    st.divider()
    st.write("Enter applicant information to run a prediction.")
    st.warning(
        "Fields not shown in the form retain values from a "
        "baseline dataset record. This is a demonstration only."
    )


# ==================================================
# 6. APPLICANT INPUT FORM
# ==================================================

st.subheader("Applicant Information")
st.write("Enter the details below and select Analyze Applicant.")

try:
    baseline = load_baseline()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


with st.form("applicant_form"):

    st.markdown("#### Personal and Employment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        default_age = round(
            -baseline_number(baseline, "DAYS_BIRTH", -30 * 365) / 365
        )
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=max(18, min(100, default_age)),
        )

    with col2:
        default_employment = round(
            -baseline_number(baseline, "DAYS_EMPLOYED", -5 * 365) / 365,
            1,
        )
        employment_years = st.number_input(
            "Employment duration (years)",
            min_value=0.0,
            max_value=60.0,
            value=max(0.0, min(60.0, default_employment)),
            step=0.5,
        )

    with col3:
        income = st.number_input(
            "Annual income",
            min_value=1.0,
            value=max(
                1.0,
                baseline_number(baseline, "AMT_INCOME_TOTAL", 200000),
            ),
            step=10000.0,
        )

    st.markdown("#### Loan Details")

    col4, col5, col6 = st.columns(3)

    with col4:
        credit = st.number_input(
            "Credit amount",
            min_value=1.0,
            value=max(
                1.0,
                baseline_number(baseline, "AMT_CREDIT", 500000),
            ),
            step=10000.0,
        )

    with col5:
        annuity = st.number_input(
            "Annual loan payment",
            min_value=0.0,
            value=max(
                0.0,
                baseline_number(baseline, "AMT_ANNUITY", 25000),
            ),
            step=1000.0,
        )

    with col6:
        goods_price = st.number_input(
            "Goods price",
            min_value=1.0,
            value=max(
                1.0,
                baseline_number(baseline, "AMT_GOODS_PRICE", 400000),
            ),
            step=10000.0,
        )

    st.markdown("#### External Credit Scores")

    score_defaults = [
        baseline_number(baseline, "EXT_SOURCE_1", 0.5),
        baseline_number(baseline, "EXT_SOURCE_2", 0.5),
        baseline_number(baseline, "EXT_SOURCE_3", 0.5),
    ]

    score1, score2, score3 = st.columns(3)

    with score1:
        ext1 = st.slider(
            "External score 1",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, score_defaults[0])),
            step=0.01,
        )

    with score2:
        ext2 = st.slider(
            "External score 2",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, score_defaults[1])),
            step=0.01,
        )

    with score3:
        ext3 = st.slider(
            "External score 3",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, score_defaults[2])),
            step=0.01,
        )

    submitted = st.form_submit_button(
        "Analyze Applicant",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# 7. SEND APPLICANT TO FASTAPI
# ==================================================

if submitted:

    if annuity > income:
        st.warning(
            "The annual loan payment exceeds annual income. "
            "Please verify the entered values."
        )

    applicant = baseline.copy()

    # Convert user-entered values into the dataset's feature format.
    applicant["DAYS_BIRTH"] = -int(age * 365)
    applicant["DAYS_EMPLOYED"] = -int(employment_years * 365)

    applicant["AMT_INCOME_TOTAL"] = float(income)
    applicant["AMT_CREDIT"] = float(credit)
    applicant["AMT_ANNUITY"] = float(annuity)
    applicant["AMT_GOODS_PRICE"] = float(goods_price)

    applicant["EXT_SOURCE_1"] = float(ext1)
    applicant["EXT_SOURCE_2"] = float(ext2)
    applicant["EXT_SOURCE_3"] = float(ext3)

    applicant.pop("TARGET", None)
    applicant.pop("SK_ID_CURR", None)

    # Convert missing values to JSON null.
    applicant_series = pd.Series(applicant, dtype="object")
    applicant = applicant_series.where(
        pd.notna(applicant_series),
        None,
    ).to_dict()

    with st.spinner("Analyzing applicant..."):

        try:
            response = requests.post(
                API_URL,
                json={"applicant_data": applicant},
                timeout=120,
            )

            response.raise_for_status()
            result = response.json()["prediction"]

            st.session_state["prediction_result"] = result

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the API server "
                "in another terminal."
            )

        except requests.exceptions.Timeout:
            st.error("The API request timed out. Please try again.")

        except requests.exceptions.HTTPError:
            st.error(f"API error: {response.text}")

        except (KeyError, ValueError) as exc:
            st.error(f"Unexpected API response: {exc}")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ==================================================
# 8. DISPLAY PREDICTION RESULTS
# ==================================================

result = st.session_state.get("prediction_result")

if result:

    st.divider()
    st.subheader("Prediction Results")

    probability = float(result["probability"])
    threshold = float(result["threshold"])
    classification = result["risk_level"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Probability",
        f"{probability:.2%}",
    )

    col2.metric(
        "Decision Threshold",
        f"{threshold:.2f}",
    )

    col3.metric(
        "Model Classification",
        classification,
    )

    st.markdown("#### Predicted Probability")

    st.progress(
        max(0.0, min(1.0, probability)),
        text=f"Probability: {probability:.2%}",
    )

    st.caption(
        "The threshold is the model's configured classification cutoff, "
        "not a regulatory or lending standard."
    )

    if probability >= threshold:
        st.warning(
            "The model classified this record as Payment Difficulty. "
            "A qualified reviewer should assess the result."
        )
    else:
        st.success(
            "The model classified this record as No Difficulty. "
            "This does not guarantee repayment."
        )

    # ==================================================
    # 9. VISUAL SHAP CHART
    # ==================================================

    st.divider()
    st.subheader("What Influenced This Prediction?")

    risk_factors = pd.DataFrame(
        result.get("top_risk_factors", [])
    )

    protective_factors = pd.DataFrame(
        result.get("top_protective_factors", [])
    )

    factors = pd.concat(
        [risk_factors, protective_factors],
        ignore_index=True,
    )

    if not factors.empty and "shap_value" in factors.columns:

        factors["SHAP Value"] = pd.to_numeric(
            factors["shap_value"],
            errors="coerce",
        )

        factors = factors.dropna(subset=["SHAP Value"]).copy()

        factors["Feature"] = factors["feature"].apply(
            readable_feature_name
        )

        factors["Absolute Impact"] = factors["SHAP Value"].abs()

        factors = factors.sort_values(
            "Absolute Impact",
            ascending=True,
        )

        st.write(
            "Positive SHAP values push the model toward higher predicted "
            "risk. Negative values push it toward lower predicted risk. "
            "These values are not percentages or proof of causation."
        )

        st.markdown("#### SHAP Contribution by Feature")

        chart_data = factors.set_index("Feature")[["SHAP Value"]]

        st.bar_chart(
            chart_data,
            horizontal=True,
            height=460,
        )

        st.caption(
            "Longer bars indicate larger absolute contributions among "
            "the features displayed."
        )

        # Display readable feature labels in the tables.
        left, right = st.columns(2)

        with left:
            st.markdown("#### Factors Increasing Predicted Risk")

            if not risk_factors.empty:
                risk_display = risk_factors.copy()

                risk_display["Feature"] = risk_display["feature"].apply(
                    readable_feature_name
                )

                risk_display["SHAP Contribution"] = pd.to_numeric(
                    risk_display["shap_value"],
                    errors="coerce",
                ).round(4)

                st.dataframe(
                    risk_display[
                        ["Feature", "SHAP Contribution"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No risk-increasing factors were returned.")

        with right:
            st.markdown("#### Factors Decreasing Predicted Risk")

            if not protective_factors.empty:
                protective_display = protective_factors.copy()

                protective_display["Feature"] = (
                    protective_display["feature"].apply(
                        readable_feature_name
                    )
                )

                protective_display["SHAP Contribution"] = pd.to_numeric(
                    protective_display["shap_value"],
                    errors="coerce",
                ).round(4)

                st.dataframe(
                    protective_display[
                        ["Feature", "SHAP Contribution"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No risk-decreasing factors were returned.")

    else:
        st.info("No SHAP explanation factors were returned.")

    st.divider()

    st.caption(
        "Educational portfolio prototype. Some fields retain values "
        "from a baseline dataset record. Model contributions do not "
        "establish causation, and this output should not be the sole "
        "basis for a lending decision."
    )

else:
    st.info(
        "Enter applicant details and select Analyze Applicant "
        "to view the prediction and explanations."
    )