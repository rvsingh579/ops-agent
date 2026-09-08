from pathlib import Path

from langchain_core.tools import tool

from src.anomaly_detector import fit_isolation_forest, get_top_contributing_sensors, score_anomalies
from src.data_loader import load_and_prepare, load_domain_config
from src.tools.data_retrieval_tool import parse_query, resolve_cycle_range

FILEPATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "train_FD001.txt"


def summarize_anomalies(df_subset, domain_config, feature_columns, top_n=3) -> str:
    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]

    df_subset = df_subset.sort_values(time_column)

    unit_number = df_subset[id_column].iloc[0]
    first_cycle = df_subset[time_column].iloc[0]
    last_cycle = df_subset[time_column].iloc[-1]
    reading_count = len(df_subset)

    flagged_count = int(df_subset["is_anomaly"].sum())
    flagged_pct = round(100 * flagged_count / reading_count, 1)

    mean_score = round(df_subset["anomaly_score"].mean(), 4)
    min_score = round(df_subset["anomaly_score"].min(), 4)
    max_score = round(df_subset["anomaly_score"].max(), 4)

    last_row = df_subset.iloc[-1]
    last_score = round(last_row["anomaly_score"], 4)
    last_is_anomaly = bool(last_row["is_anomaly"])

    top_contributors = get_top_contributing_sensors(last_row, feature_columns, top_n=top_n)

    sensor_lookup = {entry["name"]: entry for entry in domain_config["sensors"]}

    def describe_column(column):
        if column in sensor_lookup:
            info = sensor_lookup[column]
            return f"{column} ({info['symbol']}, {info['description']})"
        return column

    contributor_lines = [
        f"  {describe_column(name)}: z-score={round(value, 3)}"
        for name, value in top_contributors
    ]

    lines = [
        f"Unit {unit_number}, cycles {first_cycle}-{last_cycle} ({reading_count} readings):",
        f"anomaly_score: mean={mean_score} min={min_score} max={max_score}",
        f"{flagged_count}/{reading_count} readings flagged anomalous ({flagged_pct}%)",
        f"Most recent cycle ({int(last_row[time_column])}): "
        f"anomaly_score={last_score}, is_anomaly={last_is_anomaly}",
        "Top contributing sensors at most recent cycle:",
    ] + contributor_lines

    return "\n".join(lines)


_CACHE = {}


def _get_cached_scores():
    """Fit Isolation Forest once, on the whole (normalized) training set,
    and score the whole set once - both expensive, both deterministic given
    a fixed random_state, so there is no reason to redo either per query.
    """
    if not _CACHE:
        domain_config = load_domain_config()
        normalized_df, stats, feature_columns = load_and_prepare(FILEPATH, domain_config)

        model = fit_isolation_forest(normalized_df, feature_columns)
        scored_df = score_anomalies(model, normalized_df, feature_columns)

        _CACHE["scored_df"] = scored_df
        _CACHE["domain_config"] = domain_config
        _CACHE["feature_columns"] = feature_columns

    return _CACHE["scored_df"], _CACHE["domain_config"], _CACHE["feature_columns"]


def detect_anomalies(query: str) -> str:
    # 1. Parse the query - same format, same parser, as data_retrieval_tool
    parsed = parse_query(query)

    if "error" in parsed:
        return parsed["error"]

    unit_number = parsed["unit_number"]
    cycles_spec = parsed["cycles_spec"]

    # 2. Load (cached) scored dataframe + config + feature columns
    scored_df, domain_config, feature_columns = _get_cached_scores()

    # 3. Check whether requested unit exists
    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]

    unit_df = scored_df[scored_df[id_column] == unit_number]

    if unit_df.empty:
        valid_units = sorted(scored_df[id_column].unique())
        return (
            f"Unit {unit_number} does not exist. "
            f"Valid unit numbers range from "
            f"{min(valid_units)} to {max(valid_units)}."
        )

    # 4. Get available cycles for this unit
    available_cycles = sorted(unit_df[time_column].unique())

    # 5. Resolve requested cycle specification into an explicit cycle list
    resolved_cycles = resolve_cycle_range(cycles_spec, available_cycles)

    if not resolved_cycles:
        return (
            f"No matching cycles for unit {unit_number} with request "
            f"'{cycles_spec}'. This unit's cycles range from "
            f"{available_cycles[0]} to {available_cycles[-1]}."
        )

    # 6. Subset to exactly the resolved cycles
    df_subset = unit_df[unit_df[time_column].isin(resolved_cycles)]

    # 7. Summarize and return
    return summarize_anomalies(df_subset, domain_config, feature_columns)


@tool
def anomaly_detection_tool(query: str) -> str:
    """Runs Isolation Forest anomaly detection for a specific turbofan
    engine unit and reports whether its recent readings look abnormal.
    Use this when the user asks if something is wrong, unusual, or
    abnormal with a specific engine - typically after data_retrieval_tool,
    or instead of it if the user directly asks about anomalies.

    Input format: 'unit: <id>, cycles: <spec>' where <spec> is 'last N',
    'A-B', a single cycle number, or 'all'. Example: 'unit: 14, cycles:
    last 20'. If cycles is omitted, defaults to the last 20.
    """
    return detect_anomalies(query)


if __name__ == "__main__":
    print("=== detect_anomalies (real dataset) ===")
    for q in [
        "unit: 1, cycles: last 100",   # unit 1 fails at cycle 192 - expect high flag rate
        "unit: 1, cycles: 1-20",      # unit 1's early life - expect low/no flags
        "unit: 500",                  # nonexistent unit
        "unit: 1, cycles: 9999",      # nonexistent cycle for a real unit
    ]:
        print(f"--- query: {q!r} ---")
        print(detect_anomalies(q))
        print()

    print("=== anomaly_detection_tool (the actual LangChain Tool) ===")
    print("name:", anomaly_detection_tool.name)
    print(anomaly_detection_tool.invoke("unit: 1, cycles: last 20"))
