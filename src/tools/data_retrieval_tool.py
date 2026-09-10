import re

from langchain_core.tools import tool

from src.data_loader import load_cleaned_data, load_domain_config

def resolve_cycle_range(cycles_spec: str, available_cycles: list[int]) -> list[int]:
    """
    Resolves a cycle specification into a list of cycle numbers.

    Args:
        cycles_spec (str): The cycle specification string.
        available_cycles (list[int]): A list of available cycle numbers.

    Returns:
        list[int]: A list of resolved cycle numbers.
    """
    if not available_cycles:
        return []

    available_cycles = sorted(available_cycles)

    spec = cycles_spec.strip().lower()

    if spec == "all":
        return available_cycles

    last_match = re.fullmatch(r"last\s+(\d+)", spec)
    if last_match:
        n = int(last_match.group(1))
        return available_cycles[-n:] if n > 0 else []

    range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", spec)

    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))

        if start > end:
            start, end = end, start

        return [
            cycle
            for cycle in available_cycles
            if start <= cycle <= end
        ]

    integer_match = re.fullmatch(r"\d+", spec)

    if integer_match:
        cycle = int(integer_match.group(0))

        return [cycle] if cycle in available_cycles else []

    return available_cycles[-20:]

def summarize_readings(df_subset, domain_config, feature_columns) -> str:
    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]

    df_subset = df_subset.sort_values(time_column)

    unit_number = df_subset[id_column].iloc[0]
    first_cycle = df_subset[time_column].iloc[0]
    last_cycle = df_subset[time_column].iloc[-1]
    reading_count = len(df_subset)

    lines = [
        f"Unit {unit_number}, cycles {first_cycle}-{last_cycle} "
        f"({reading_count} readings):"
    ]

    # operational_settings/sensors are LISTS of dicts in domain.yaml (one
    # entry per column), not dicts keyed by column name - build name -> entry
    # lookups once, rather than indexing the lists directly with a string.
    op_setting_lookup = {
        entry["name"]: entry for entry in domain_config["operational_settings"]
    }
    sensor_lookup = {
        entry["name"]: entry for entry in domain_config["sensors"]
    }

    for column in feature_columns:
        if column in op_setting_lookup:
            config = op_setting_lookup[column]
            identity = config["name"]
            line_prefix = f"{column} ({identity})"

        elif column in sensor_lookup:
            config = sensor_lookup[column]
            symbol = config["symbol"]
            description = config["description"]
            unit = config["unit"]
            line_prefix = (
                f"{column} ({symbol}, {description}, {unit})"
            )

        else:
            raise ValueError(
                f"Column '{column}' not found in domain_config."
            )

        values = df_subset[column]

        mean_value = round(values.mean(), 3)
        min_value = round(values.min(), 3)
        max_value = round(values.max(), 3)
        first_value = round(values.iloc[0], 3)
        last_value = round(values.iloc[-1], 3)

        lines.append(
            f"{line_prefix}: "
            f"mean={mean_value} "
            f"min={min_value} "
            f"max={max_value} "
            f"first={first_value} "
            f"last={last_value}"
        )

    return "\n".join(lines)


_CACHE = {}


def _get_cached_data():
    """Load + clean the dataset once, and derive feature_columns from
    whatever clean() actually kept - not from a separately hand-maintained
    list that could drift out of sync with domain.yaml.
    """
    if not _CACHE:
        domain_config = load_domain_config()
        df = load_cleaned_data(domain_config=domain_config)

        id_column = domain_config["asset"]["id_column"]
        time_column = domain_config["asset"]["time_column"]
        feature_columns = [
            col for col in df.columns if col not in (id_column, time_column)
        ]

        _CACHE["df"] = df
        _CACHE["domain_config"] = domain_config
        _CACHE["feature_columns"] = feature_columns

    return _CACHE["df"], _CACHE["domain_config"], _CACHE["feature_columns"]


def retrieve_data(unit: int, cycles: str = "last 20") -> str:

    df, domain_config, feature_columns = _get_cached_data()

    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]

    unit_df = df[df[id_column] == unit]

    if unit_df.empty:
        valid_units = sorted(df[id_column].unique())

        return (
            f"Unit {unit} does not exist. "
            f"Valid unit numbers range from "
            f"{min(valid_units)} to {max(valid_units)}."
        )

    # 4. Get available cycles for this unit
    available_cycles = sorted(
        unit_df[time_column].unique()
    )

    # 5. Resolve requested cycle specification into an explicit cycle list
    resolved_cycles = resolve_cycle_range(
        cycles,
        available_cycles
    )

    if not resolved_cycles:
        return (
            f"No matching cycles for unit {unit} with request "
            f"'{cycles}'. This unit's cycles range from "
            f"{available_cycles[0]} to {available_cycles[-1]}."
        )

    # 6. Subset the dataframe to exactly the resolved cycles
    df_subset = unit_df[unit_df[time_column].isin(resolved_cycles)]

    # 7. Summarize and return
    return summarize_readings(
        df_subset,
        domain_config,
        feature_columns
    )

@tool(parse_docstring=True)
def data_retrieval_tool(unit: int, cycles: str = "last 20") -> str:
    """Retrieves recent sensor readings and summary statistics for a
    specific turbofan engine unit.

    Args:
        unit: The engine unit number (1-100 for the FD001 dataset).
        cycles: Which cycles to retrieve - 'last N' (e.g. 'last 20'), a
            range 'A-B' (e.g. '10-15'), a single cycle number, or 'all'.
    """
    return retrieve_data(unit, cycles)


if __name__ == "__main__":
    print("=== resolve_cycle_range (against a 1..21 pool) ===")
    pool = list(range(1, 22))
    for spec in ["10-15", "last 5", "all", "75", "not a real spec"]:
        print(repr(spec), "->", resolve_cycle_range(spec, pool))

    print()
    print("=== retrieve_data (real dataset) ===")
    for unit, cycles in [
        (1, "last 5"),      # unit 1 fails at cycle 192
        (3, "10-15"),
        (500, "last 20"),   # nonexistent unit
        (1, "9999"),        # nonexistent cycle for a real unit
    ]:
        print(f"--- unit={unit!r}, cycles={cycles!r} ---")
        print(retrieve_data(unit, cycles))
        print()

    print("=== data_retrieval_tool (the actual LangChain Tool) ===")
    print("name:", data_retrieval_tool.name)
    print("args schema:", data_retrieval_tool.args)
    print(data_retrieval_tool.invoke({"unit": 1, "cycles": "last 1000"}))

    print()
    print("=== what a malformed model tool-call actually looks like now ===")
    try:
        data_retrieval_tool.invoke({"unit": "engine number seven"})
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
