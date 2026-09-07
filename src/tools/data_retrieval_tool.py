import re

def parse_query(query: str) -> dict:
    """
    Parses a query string into a dictionary of key-value pairs.

    Args:
        query (str): The query string to parse.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
    """
    unit_match = re.search(
        r"unit\D{0,10}(\d+)",
        query,
        flags=re.IGNORECASE
    )

    if not unit_match:
        return {
            "error": "Could not find a unit number in the query."
        }

    unit_number = int(unit_match.group(1))

    # Extract cycles specification
    cycles_match = re.search(
        r"cycles?\s*:?\s*(.*)",
        query,
        flags=re.IGNORECASE
    )

    if cycles_match:
        cycles_spec = cycles_match.group(1).strip()
    else:
        cycles_spec = "last 20"

    return {
        "unit_number": unit_number,
        "cycles_spec": cycles_spec
    }

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
    """
    Summarizes the readings in the given DataFrame subset.

    Args:
        df_subset (pd.DataFrame): The subset of the DataFrame to summarize.
        domain_config (dict): Configuration for the domain.
        feature_columns (list[str]): List of feature column names.

    Returns:
        str: A summary string of the readings.
    """
    if df_subset.empty:
        return "No data available for the specified query."

    summary = []
    for feature in feature_columns:
        if feature in df_subset.columns:
            mean_value = df_subset[feature].mean()
            summary.append(f"{feature}: {mean_value:.2f}")

    return "\n".join(summary)

if __name__ == "__main__":
    parsed = parse_query("unit: 3, cycles: 10-15")
    print(parsed)

    avlcycles = resolve_cycle_range(
        parsed["cycles_spec"],
        list(range(1, 22)),
    )
    print(avlcycles)