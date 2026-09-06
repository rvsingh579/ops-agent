from pathlib import Path
import pandas as pd
import yaml

def load_domain_config(config_path=None) -> dict:
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent
            / "config"
            / "domain.yaml"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_feature_columns(domain_config: dict) -> dict:
    id_column = domain_config["asset"]["id_column"]
    time_column = domain_config["asset"]["time_column"]
    op_setting_columns = [setting["name"] for setting in domain_config["operational_settings"]]
    sensor_columns = [sensor["name"] for sensor in domain_config["sensors"]]

    feature_columns = {"id_column":id_column,
     "time_column":time_column,
     "op_setting_columns":op_setting_columns,
     "sensor_columns":sensor_columns,
     "all_raw_columns":[id_column, time_column] + op_setting_columns + sensor_columns}
    return feature_columns

def load_raw_cmapss(filepath, domain_config) -> pd.DataFrame:
    feature_columns = get_feature_columns(domain_config)
    rawdf = pd.read_csv(filepath, sep="\s+", header=None)
    rawdf.columns = feature_columns["all_raw_columns"]
    return rawdf

def clean(df, domain_config) -> pd.DataFrame:
    drop_columns = [
        config["name"]
        for config in domain_config["sensors"] + domain_config["operational_settings"]
        if not config.get("informative", True)
    ]
    df = df.drop(columns=drop_columns)
    df_cleaned = df.sort_values(by=[domain_config["asset"]["id_column"], domain_config["asset"]["time_column"]]).reset_index(drop=True)

    null_count = df_cleaned.isnull().sum().sum()
    if null_count != 0:
        raise ValueError(
            f"DataFrame contains {null_count} null values. "
            "This is unexpected for CMAPSS and indicates an upstream data issue."
        )
    duplicate_count = df_cleaned.duplicated(
        subset=[domain_config["asset"]["id_column"], domain_config["asset"]["time_column"]]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"DataFrame contains {duplicate_count} duplicate "
            f"({domain_config['asset']['id_column']}, {domain_config['asset']['time_column']}) pairs. "
            "This is unexpected for CMAPSS and indicates an upstream data issue."
        )
    return df_cleaned

def fit_normalization_stats(df, feature_columns) -> pd.DataFrame:
    stats = pd.DataFrame({
        "mean": df[feature_columns].mean(),
        "std": df[feature_columns].std()
    })

    zero_std_columns = stats.index[stats["std"] == 0].tolist()
    if zero_std_columns:
        raise ValueError(
            f"Zero-std (constant) column(s) found: {zero_std_columns}. "
            "Normalizing these would divide by zero. Mark them "
            "'informative: false' in domain.yaml and exclude them in clean() "
            "before fitting normalization stats."
        )

    return stats

def apply_normalization(df, feature_columns, stats) -> pd.DataFrame:
    df_normalized = df.copy()

    for col in feature_columns:
        df_normalized[col] = (
            df_normalized[col] - stats.loc[col, "mean"]
        ) / stats.loc[col, "std"]

    return df_normalized

def load_and_prepare(filepath, domain_config, stats=None):

    df = load_raw_cmapss(filepath, domain_config)
    df = clean(df, domain_config)

    feature_columns = [
        col for col in df.columns
        if col not in [
            domain_config["asset"]["id_column"],
            domain_config["asset"]["time_column"]
        ]
    ]

    if stats is None:
        stats = fit_normalization_stats(df, feature_columns)

    normalized_df = apply_normalization(df, feature_columns, stats)

    return normalized_df, stats

if __name__ == "__main__":
    domain_config = load_domain_config()
    filepath = "/Users/raviranjan/Documents/mera_kaam/ops-agent/data/raw/train_FD001.txt"
    normalized_df, stats = load_and_prepare(filepath, domain_config)
    print(normalized_df.head())