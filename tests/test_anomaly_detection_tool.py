import pandas as pd

from src.tools.anomaly_detection_tool import (
    anomaly_detection_tool,
    detect_anomalies,
    summarize_anomalies,
)


def test_summarize_anomalies_reports_flagged_count_and_top_contributor(tool_domain_config_stub):
    df_subset = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [10, 11],
        "op_setting_1": [0.1, 0.2],
        "sensor_8": [1.0, 3.0],
        "sensor_3": [0.5, -0.5],
        "anomaly_score": [-0.05, 0.10],
        "is_anomaly": [False, True],
    })
    feature_columns = ["op_setting_1", "sensor_8", "sensor_3"]

    result = summarize_anomalies(df_subset, tool_domain_config_stub, feature_columns)

    assert "Unit 1, cycles 10-11 (2 readings):" in result
    assert "1/2 readings flagged anomalous (50.0%)" in result
    assert "Most recent cycle (11): anomaly_score=0.1, is_anomaly=True" in result
    assert "sensor_8 (Nf, Physical fan speed): z-score=3.0" in result


# --- detect_anomalies: real cached data, no LLM involved ---


def test_detect_anomalies_flags_more_near_failure_than_early_life():
    # The core Phase 1 regression property ("score/flag rate rises near
    # failure"), now verified through the tool's actual text interface -
    # these exact numbers were confirmed by hand when the tool was built.
    early = detect_anomalies("unit: 1, cycles: 1-20")
    late = detect_anomalies("unit: 1, cycles: last 20")

    assert "0/20 readings flagged anomalous (0.0%)" in early
    assert "8/20 readings flagged anomalous (40.0%)" in late


def test_detect_anomalies_returns_error_for_nonexistent_unit():
    result = detect_anomalies("unit: 500")

    assert result == "Unit 500 does not exist. Valid unit numbers range from 1 to 100."


def test_detect_anomalies_returns_error_for_nonexistent_cycle():
    result = detect_anomalies("unit: 1, cycles: 9999")

    assert "No matching cycles for unit 1" in result


def test_detect_anomalies_returns_error_for_unparseable_query():
    result = detect_anomalies("engine number seven")

    assert result == "Could not find a unit number in the query."


# --- the actual LangChain Tool ---


def test_anomaly_detection_tool_invoke_matches_detect_anomalies():
    result = anomaly_detection_tool.invoke("unit: 1, cycles: last 20")

    assert "8/20 readings flagged anomalous (40.0%)" in result