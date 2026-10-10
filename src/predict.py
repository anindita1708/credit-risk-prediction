import json
import joblib
import pandas as pd
import shap

from pathlib import Path

from features import create_features


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT /
    "models" /
    "xgboost.pkl"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT /
    "models" /
    "feature_preprocessor.pkl"
)

THRESHOLD_PATH = (
    PROJECT_ROOT /
    "models" /
    "threshold_config.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

with open(
    THRESHOLD_PATH,
    "r"
) as f:
    threshold_config = json.load(f)


THRESHOLD = round(
    threshold_config["threshold"],
    2
)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {

    # --------------------------------------------------------
    # External credit scores
    # --------------------------------------------------------

    "EXT_SOURCE_1":
        "External Credit Score 1",

    "EXT_SOURCE_2":
        "External Credit Score 2",

    "EXT_SOURCE_3":
        "External Credit Score 3",


    # --------------------------------------------------------
    # Financial features
    # --------------------------------------------------------

    "AMT_GOODS_PRICE":
        "Goods Price",

    "AMT_ANNUITY":
        "Loan Annuity",

    "AMT_CREDIT":
        "Credit Amount",

    "AMT_INCOME_TOTAL":
        "Total Income",

    "CREDIT_INCOME_RATIO":
        "Credit-to-Income Ratio",

    "ANNUITY_INCOME_RATIO":
        "Annuity-to-Income Ratio",

    "CREDIT_GOODS_RATIO":
        "Credit-to-Goods Price Ratio",

    "INCOME_PER_FAMILY_MEMBER":
        "Income per Family Member",


    # --------------------------------------------------------
    # Employment
    # --------------------------------------------------------

    "DAYS_EMPLOYED":
        "Employment Days",

    "EMPLOYMENT_YEARS":
        "Employment Duration (Years)",

    "EMPLOYED_ANOMALY":
        "Employment Data Anomaly",


    # --------------------------------------------------------
    # Family
    # --------------------------------------------------------

    "CNT_CHILDREN":
        "Number of Children",

    "CNT_FAM_MEMBERS":
        "Number of Family Members",

    "CHILDREN_RATIO":
        "Children-to-Family Ratio",


    # --------------------------------------------------------
    # Credit bureau
    # --------------------------------------------------------

    "AMT_REQ_CREDIT_BUREAU_QRT":
        "Credit Bureau Requests in Last Quarter",

    "DAYS_ID_PUBLISH":
        "Days Since ID Document Published",


    # --------------------------------------------------------
    # Vehicle ownership
    # --------------------------------------------------------

    "FLAG_OWN_CAR_N":
        "Does Not Own a Car",

    "OWN_CAR_AGE":
        "Car Age",


    # --------------------------------------------------------
    # Occupation
    # --------------------------------------------------------

    "OCCUPATION_TYPE_Core staff":
        "Occupation: Core Staff",


    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    "FLAG_DOCUMENT_3":
        "Document 3 Provided",


    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    "CODE_GENDER_F":
        "Gender: Female",

    "CODE_GENDER_M":
        "Gender: Male",


    # --------------------------------------------------------
    # Family status
    # --------------------------------------------------------

    "NAME_FAMILY_STATUS_Married":
        "Family Status: Married",


    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    "NAME_EDUCATION_TYPE_Higher education":
        "Education: Higher Education",

    "NAME_EDUCATION_TYPE_Secondary / secondary special":
        "Education: Secondary / Secondary Special"
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_risk(applicant_data):
    """
    Predict payment difficulty risk and explain
    the prediction using SHAP.

    Parameters
    ----------
    applicant_data : pandas.DataFrame
        Raw applicant data.

    Returns
    -------
    dict
        Prediction probability, risk level,
        risk factors and protective factors.
    """


    # --------------------------------------------------------
    # 1. COPY INPUT DATA
    # --------------------------------------------------------

    df = applicant_data.copy()


    # --------------------------------------------------------
    # 2. REMOVE TARGET IF PRESENT
    # --------------------------------------------------------

    if "TARGET" in df.columns:

        df = df.drop(
            columns=["TARGET"]
        )


    # --------------------------------------------------------
    # 3. REMOVE APPLICANT ID
    # --------------------------------------------------------

    if "SK_ID_CURR" in df.columns:

        df = df.drop(
            columns=["SK_ID_CURR"]
        )


    # --------------------------------------------------------
    # 4. FEATURE ENGINEERING
    # --------------------------------------------------------

    df = create_features(
        df
    )


    # --------------------------------------------------------
    # 5. PREPROCESS INPUT
    # --------------------------------------------------------

    X_processed = preprocessor.transform(
        df
    )


    # --------------------------------------------------------
    # 6. PREDICT PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_processed
    )[:, 1]

    probability = float(
        probability[0]
    )


    # --------------------------------------------------------
    # 7. APPLY DECISION THRESHOLD
    # --------------------------------------------------------

    prediction = int(
        probability >= THRESHOLD
    )


    if prediction == 1:

        risk_level = "Payment Difficulty"

    else:

        risk_level = "No Difficulty"


    # --------------------------------------------------------
    # 8. CALCULATE SHAP VALUES
    # --------------------------------------------------------

    shap_values = explainer(
        X_processed
    )


    # --------------------------------------------------------
    # 9. GET SHAP VALUES FOR APPLICANT
    # --------------------------------------------------------

    applicant_shap_values = (
        shap_values.values[0]
    )


    # --------------------------------------------------------
    # 10. CREATE SHAP EXPLANATION TABLE
    # --------------------------------------------------------

    explanation = pd.DataFrame({

        "Feature": feature_names,

        "SHAP_Value": applicant_shap_values

    })


    # --------------------------------------------------------
    # 11. CALCULATE ABSOLUTE IMPACT
    # --------------------------------------------------------

    explanation["Impact"] = (
        explanation["SHAP_Value"]
        .abs()
    )


    # --------------------------------------------------------
    # 12. CLEAN FEATURE NAMES
    # --------------------------------------------------------

    explanation["Feature"] = (
        explanation["Feature"]
        .str.replace(
            "num__",
            "",
            regex=False
        )
        .str.replace(
            "cat__",
            "",
            regex=False
        )
    )


    # --------------------------------------------------------
    # 13. CONVERT TO HUMAN-READABLE NAMES
    # --------------------------------------------------------

    explanation["Feature"] = (
        explanation["Feature"]
        .map(FEATURE_LABELS)
        .fillna(
            explanation["Feature"]
        )
    )


    # --------------------------------------------------------
    # 14. RISK-INCREASING FACTORS
    # --------------------------------------------------------

    risk_factors = (
        explanation[
            explanation["SHAP_Value"] > 0
        ]
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(5)
    )


    # --------------------------------------------------------
    # 15. PROTECTIVE / RISK-DECREASING FACTORS
    # --------------------------------------------------------

    protective_factors = (
        explanation[
            explanation["SHAP_Value"] < 0
        ]
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(5)
    )


    # --------------------------------------------------------
    # 16. CREATE RISK FACTOR OUTPUT
    # --------------------------------------------------------

    top_risk_factors = []

    for _, row in risk_factors.iterrows():

        top_risk_factors.append({

            "feature": row["Feature"],

            "shap_value": round(
                float(row["SHAP_Value"]),
                6
            ),

            "impact": round(
                float(row["Impact"]),
                6
            ),

            "direction":
                "Increases predicted risk"

        })


    # --------------------------------------------------------
    # 17. CREATE PROTECTIVE FACTOR OUTPUT
    # --------------------------------------------------------

    top_protective_factors = []

    for _, row in protective_factors.iterrows():

        top_protective_factors.append({

            "feature": row["Feature"],

            "shap_value": round(
                float(row["SHAP_Value"]),
                6
            ),

            "impact": round(
                float(row["Impact"]),
                6
            ),

            "direction":
                "Decreases predicted risk"

        })


    # --------------------------------------------------------
    # 18. RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {

        "probability": round(
            probability,
            4
        ),

        "threshold": THRESHOLD,

        "prediction": prediction,

        "risk_level": risk_level,

        "top_risk_factors":
            top_risk_factors,

        "top_protective_factors":
            top_protective_factors

    }