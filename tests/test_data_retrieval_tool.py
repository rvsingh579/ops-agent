import pandas as pd
import pytest

from src.tools.data_retrieval_tool import (
    data_retrieval_tool,
    resolve_cycle_range,
    retrieve_data,
    summarize_readings,
)


def test_resolve_cycle_range_handles_explicit_range():
    cycles_spec = "10-15"
    available_cycles = list(range(1, 22))

    result = resolve_cycle_range(cycles_spec, available_cycles)

    assert result == [10, 11, 12, 13, 14, 15]

def test_resolve_cycle_range_handles_all_range():
    cycles_spec = "all"
    available_cycles = list(range(1, 22))

    result = resolve_cycle_range(cycles_spec, available_cycles)

    assert result == available_cycles

def test_resolve_cycle_range_handles_last_n():
    cycles_spec = "last 7"
    available_cycles = list(range(1, 22))

    result = resolve_cycle_range(cycles_spec, available_cycles)

    assert result == [15, 16, 17, 18, 19, 20, 21]

def test_resolve_cycle_range_handles_single_cycle():
    cycles_spec = "75"
    available_cycles = list(range(1, 97))

    result = resolve_cycle_range(cycles_spec, available_cycles)

    assert result == [75]

def test_resolve_cycle_range_handles_unreal_spec():
    cycles_spec = "not a real spec"
    available_cycles = list(range(1, 31))

    result = resolve_cycle_range(cycles_spec, available_cycles)

    assert result == list(range(11, 31))

def test_summarize_readings_formats_sensor_and_op_setting_columns(tool_domain_config_stub):
    df_subset = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [10, 11],
        "op_setting_1": [0.5, 0.6],
        "sensor_8": [2388.0, 2389.0],
        "sensor_3": [1590.0, 1592.0],
    })
    feature_columns = ["op_setting_1", "sensor_8", "sensor_3"]

    result = summarize_readings(df_subset, tool_domain_config_stub, feature_columns)

    assert "Unit 1, cycles 10-11 (2 readings):" in result
    assert "sensor_8 (Nf, Physical fan speed, rpm)" in result


# --- retrieve_data: real cached data, no LLM involved. unit/cycles are now
# separate typed arguments - no more query-string parsing at this layer. ---


def test_retrieve_data_returns_summary_for_valid_query():
    result = retrieve_data(1, "last 5")

    assert "Unit 1, cycles 188-192 (5 readings):" in result

def test_retrieve_data_returns_error_for_nonexistent_unit():
    result = retrieve_data(500)

    assert result == "Unit 500 does not exist. Valid unit numbers range from 1 to 100."

def test_retrieve_data_returns_error_for_nonexistent_cycle():
    result = retrieve_data(1, "9999")

    assert "No matching cycles for unit 1" in result


# --- data_retrieval_tool: the actual LangChain Tool. .invoke() takes a dict
# of the named arguments now, not a single query string. ---


def test_data_retrieval_tool_invoke_matches_retrieve_data():
    result = data_retrieval_tool.invoke({"unit": 1, "cycles": "last 5"})

    assert "Unit 1, cycles 188-192 (5 readings):" in result

def test_data_retrieval_tool_invoke_raises_on_invalid_unit_type():
    # This is the NEW failure mode under native function calling: a bad
    # value fails at the schema boundary (pydantic), not inside our code -
    # confirmed for real when the model itself once tried exactly this.
    with pytest.raises(Exception):
        data_retrieval_tool.invoke({"unit": "engine number seven"})
