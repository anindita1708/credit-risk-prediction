
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_risk


@pytest.fixture
def sample_applicant():
    data_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "application_train.csv"
    )

    df = pd.read_csv(data_path, nrows=1)
    df = df.drop(columns=["TARGET"], errors="ignore")
    return df


def test_prediction_returns_expected_fields(sample_applicant):
    result = predict_risk(sample_applicant)

    expected_keys = {
        "probability",
        "threshold",
        "prediction",
        "risk_level",
        "top_risk_factors",
        "top_protective_factors",
    }

    assert expected_keys.issubset(result.keys())


def test_probability_is_valid(sample_applicant):
    result = predict_risk(sample_applicant)

    assert 0 <= result["probability"] <= 1


def test_threshold_is_valid(sample_applicant):
    result = predict_risk(sample_applicant)

    assert 0 <= result["threshold"] <= 1


def test_prediction_matches_threshold(sample_applicant):
    result = predict_risk(sample_applicant)

    expected_prediction = int(
        result["probability"] >= result["threshold"]
    )

    assert result["prediction"] == expected_prediction


def test_explanations_are_returned(sample_applicant):
    result = predict_risk(sample_applicant)

    assert isinstance(result["top_risk_factors"], list)
    assert isinstance(result["top_protective_factors"], list)

    for factor in result["top_risk_factors"]:
        assert factor["direction"] == "Increases predicted risk"

    for factor in result["top_protective_factors"]:
        assert factor["direction"] == "Decreases predicted risk"