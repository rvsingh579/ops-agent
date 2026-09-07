import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.anomaly_detector import (
    answer_query,
    fit_isolation_forest,
    get_top_contributing_sensors,
    score_anomalies,
)


@pytest.fixture
def scored_df_stub():
    """A small dataframe that already looks like clean() + score_anomalies()
    output - for testing answer_query / get_top_contributing_sensors in
    isolation from IsolationForest itself.
    """
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 2],
            "time_cycles": [10, 20, 10],
            "sensor_1": [0.2, -2.5, 0.1],
            "sensor_2": [1.0, 0.3, -0.4],
            "anomaly_score": [-0.05, 0.30, -0.02],
            "is_anomaly": [False, True, False],
        }
    )


# --- fit_isolation_forest ---


def test_fit_isolation_forest_returns_fitted_model():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    model = fit_isolation_forest(df, ["a", "b"], contamination=0.1, random_state=42)
    assert isinstance(model, IsolationForest)
    model.predict(df[["a", "b"]])  # raises if the model wasn't actually fitted


def test_fit_isolation_forest_is_reproducible_with_same_random_state():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    model_1 = fit_isolation_forest(df, ["a", "b"], random_state=7)
    model_2 = fit_isolation_forest(df, ["a", "b"], random_state=7)
    scores_1 = model_1.decision_function(df[["a", "b"]])
    scores_2 = model_2.decision_function(df[["a", "b"]])
    assert np.allclose(scores_1, scores_2)


# --- score_anomalies ---


def test_score_anomalies_sign_is_flipped_from_raw_decision_function():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    model = fit_isolation_forest(df, ["a", "b"], random_state=1)
    scored = score_anomalies(model, df, ["a", "b"])

    raw_decision = model.decision_function(df[["a", "b"]])
    assert np.allclose(scored["anomaly_score"], -raw_decision)


def test_score_anomalies_is_anomaly_is_boolean_and_matches_predict():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    model = fit_isolation_forest(df, ["a", "b"], random_state=1)
    scored = score_anomalies(model, df, ["a", "b"])

    assert scored["is_anomaly"].dtype == bool
    expected = model.predict(df[["a", "b"]]) == -1
    assert (scored["is_anomaly"].values == expected).all()


# --- get_top_contributing_sensors ---


def test_get_top_contributing_sensors_orders_by_absolute_value():
    row = pd.Series({"sensor_1": 0.5, "sensor_2": -2.5, "sensor_3": 1.0})
    top = get_top_contributing_sensors(row, ["sensor_1", "sensor_2", "sensor_3"], top_n=2)
    assert [name for name, _ in top] == ["sensor_2", "sensor_3"]
    assert top[0][1] == -2.5  # sign preserved, not just the absolute value


def test_get_top_contributing_sensors_respects_top_n():
    row = pd.Series({"sensor_1": 0.5, "sensor_2": -2.5, "sensor_3": 1.0})
    top = get_top_contributing_sensors(row, ["sensor_1", "sensor_2", "sensor_3"], top_n=1)
    assert len(top) == 1


# --- answer_query ---


def test_answer_query_returns_correct_row(domain_config_stub, scored_df_stub):
    result = answer_query(
        scored_df_stub,
        domain_config_stub,
        unit_number=1,
        time_cycle=20,
        feature_columns=["sensor_1", "sensor_2"],
    )
    assert result["unit_number"] == 1
    assert result["time_cycles"] == 20
    assert bool(result["is_anomaly"]) is True
    assert np.isclose(result["anomaly_score"], 0.30)
    assert result["top_contributing_sensors"][0][0] == "sensor_1"  # |-2.5| > |0.3|


def test_answer_query_raises_on_missing_row(domain_config_stub, scored_df_stub):
    with pytest.raises(ValueError, match="No row found"):
        answer_query(
            scored_df_stub,
            domain_config_stub,
            unit_number=99,
            time_cycle=1,
            feature_columns=["sensor_1", "sensor_2"],
        )


def test_answer_query_raises_on_multiple_matches(domain_config_stub, scored_df_stub):
    duplicated = pd.concat([scored_df_stub, scored_df_stub.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Multiple rows"):
        answer_query(
            duplicated,
            domain_config_stub,
            unit_number=1,
            time_cycle=10,
            feature_columns=["sensor_1", "sensor_2"],
        )


# --- integration test: encodes the manual Checkpoint 3 verification as a regression test ---


def test_anomaly_score_rises_near_failure_on_real_data(project_root):
    from src.data_loader import load_and_prepare, load_domain_config

    config = load_domain_config()
    filepath = project_root / "data" / "raw" / "train_FD001.txt"
    df, stats, feature_columns = load_and_prepare(filepath, config)

    model = fit_isolation_forest(df, feature_columns, contamination=0.05, random_state=42)
    scored = score_anomalies(model, df, feature_columns)

    unit1 = scored[scored["unit_number"] == 1].sort_values("time_cycles")
    first_20_mean = unit1.head(20)["anomaly_score"].mean()
    last_20_mean = unit1.tail(20)["anomaly_score"].mean()

    assert last_20_mean > first_20_mean
