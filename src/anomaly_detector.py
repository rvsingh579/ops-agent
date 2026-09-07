import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(df, feature_columns, contamination=0.05, random_state=42) -> IsolationForest:
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    model.fit(df[feature_columns])
    return model

def score_anomalies(model, df, feature_columns) -> pd.DataFrame:
    result = df.copy()
    result["anomaly_score"] = -model.decision_function(
        df[feature_columns]
    )
    predictions = model.predict(df[feature_columns])

    result["is_anomaly"] = predictions == -1

    return result

def get_top_contributing_sensors(row, feature_columns, top_n=3) -> list[tuple[str, float]]:
    contributions = [(col, row[col]) for col in feature_columns]

    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:top_n]

def answer_query(df_scored, domain_config, unit_number, time_cycle, feature_columns, top_n=3) -> dict:
    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]

    matched_rows = df_scored[
        (df_scored[id_column] == unit_number)
        & (df_scored[time_column] == time_cycle)
    ]

    if len(matched_rows) == 0:
        raise ValueError(
            f"No row found for {id_column}={unit_number} "
            f"and {time_column}={time_cycle}."
        )

    if len(matched_rows) > 1:
        raise ValueError(
            f"Multiple rows ({len(matched_rows)}) found for "
            f"{id_column}={unit_number} and {time_column}={time_cycle}. "
            "Expected exactly one row."
        )

    row = matched_rows.iloc[0]

    top_contributing_sensors = get_top_contributing_sensors(row, feature_columns, top_n=top_n)

    return {
        "unit_number": row[id_column],
        "time_cycles": row[time_column],
        "anomaly_score": row["anomaly_score"],
        "is_anomaly": row["is_anomaly"],
        "top_contributing_sensors": top_contributing_sensors
    }

if __name__ == "__main__":
    
    from data_loader import load_and_prepare, load_domain_config

    domain_config = load_domain_config()

    filepath = "/Users/raviranjan/Documents/mera_kaam/ops-agent/data/raw/train_FD001.txt"
    df_prepared, stats, feature_columns = load_and_prepare(filepath, domain_config)

    model = fit_isolation_forest(df_prepared, feature_columns)
    df_scored = score_anomalies(model, df_prepared, feature_columns)

    unit_number = 3
    time_cycle = 50
    result = answer_query(df_scored, domain_config, unit_number, time_cycle, feature_columns)

    print(result)