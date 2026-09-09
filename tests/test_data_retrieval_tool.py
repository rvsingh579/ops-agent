from src.tools.data_retrieval_tool import parse_query, resolve_cycle_range, retrieve_data, summarize_readings, data_retrieval_tool
import pandas as pd

def test_parse_query_extracts_unit_and_cycles():
    query = "unit: 3, cycles: 10-15"

    result = parse_query(query)

    assert result["unit_number"] == 3
    assert result["cycles_spec"] == "10-15"

def test_parse_query_returns_error_when_no_unit_found():
    query = "engine number seven"

    result = parse_query(query)

    assert "error" in result

def test_parse_query_defaults_cycles_when_not_specified():
    query = "unit 9"

    result = parse_query(query)

    assert result["unit_number"] == 9
    assert result["cycles_spec"] == "last 20"

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

def test_retrieve_data_returns_summary_for_valid_query():
    result = retrieve_data("unit: 1, cycles: last 5")

    assert "Unit 1, cycles 188-192 (5 readings):" in result

def test_retrieve_data_returns_error_for_nonexistent_unit():
    result = retrieve_data("unit: 500")

    assert result == "Unit 500 does not exist. Valid unit numbers range from 1 to 100."

def test_retrieve_data_returns_error_for_nonexistent_cycle():
    result = retrieve_data("unit: 1, cycles: 9999")

    assert "No matching cycles for unit 1" in result

def test_retrieve_data_returns_error_for_unparseable_query():
    result = retrieve_data("engine number seven")

    assert result == "Could not find a unit number in the query."

def test_data_retrieval_tool_invoke_matches_retrieve_data():
    result = data_retrieval_tool.invoke("unit: 1, cycles: last 5")

    assert "Unit 1, cycles 188-192 (5 readings):" in result