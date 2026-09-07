import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    apply_normalization,
    clean,
    fit_normalization_stats,
    get_feature_columns,
    load_and_prepare,
    load_domain_config,
    load_raw_cmapss,
)


# --- load_domain_config / get_feature_columns: against the real project config ---


def test_load_domain_config_has_expected_structure():
    config = load_domain_config()
    assert config["asset"]["id_column"] == "unit_number"
    assert config["asset"]["time_column"] == "time_cycles"
    assert len(config["sensors"]) == 21
    assert len(config["operational_settings"]) == 3


def test_get_feature_columns_raw_column_count_is_26():
    config = load_domain_config()
    columns = get_feature_columns(config)
    assert len(columns["all_raw_columns"]) == 26
    assert columns["all_raw_columns"][0] == "unit_number"
    assert columns["all_raw_columns"][1] == "time_cycles"


# --- load_raw_cmapss: regression test for the trailing-double-space gotcha ---


def test_load_raw_cmapss_handles_trailing_double_space(tmp_path, domain_config_stub):
    # Replicates the exact real-file quirk found in Checkpoint 1: each line
    # ends with two spaces before the newline. A naive sep=" " read produces
    # 8 columns instead of 6 (two bogus trailing NaN columns).
    raw_text = "1 1 0.1 100.0 10.0 5.0  \n1 2 0.2 100.0 12.0 5.0  \n"
    filepath = tmp_path / "tiny_raw.txt"
    filepath.write_text(raw_text)

    df = load_raw_cmapss(filepath, domain_config_stub)

    assert df.shape == (2, 6)
    assert list(df.columns) == [
        "unit_number",
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "sensor_1",
        "sensor_2",
    ]


# --- clean ---


def test_clean_drops_noninformative_columns(domain_config_stub, raw_df_stub):
    cleaned = clean(raw_df_stub, domain_config_stub)
    assert "op_setting_2" not in cleaned.columns
    assert "sensor_2" not in cleaned.columns
    assert "op_setting_1" in cleaned.columns
    assert "sensor_1" in cleaned.columns


def test_clean_sorts_by_id_then_time(domain_config_stub, raw_df_stub):
    shuffled = raw_df_stub.sample(frac=1, random_state=0).reset_index(drop=True)
    cleaned = clean(shuffled, domain_config_stub)
    pairs = list(zip(cleaned["unit_number"], cleaned["time_cycles"]))
    assert pairs == sorted(pairs)


def test_clean_raises_on_null_values(domain_config_stub, raw_df_stub):
    broken = raw_df_stub.copy()
    broken.loc[0, "sensor_1"] = np.nan
    with pytest.raises(ValueError, match="null"):
        clean(broken, domain_config_stub)


def test_clean_raises_on_duplicate_id_time_pairs(domain_config_stub, raw_df_stub):
    broken = pd.concat([raw_df_stub, raw_df_stub.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        clean(broken, domain_config_stub)


# --- fit_normalization_stats ---


def test_fit_normalization_stats_values_are_correct():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = fit_normalization_stats(df, ["a"])
    assert np.isclose(stats.loc["a", "mean"], 3.0)
    assert np.isclose(stats.loc["a", "std"], df["a"].std())


def test_fit_normalization_stats_raises_on_zero_std_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [7.0, 7.0, 7.0]})
    with pytest.raises(ValueError, match="Zero-std"):
        fit_normalization_stats(df, ["a", "b"])


# --- apply_normalization ---


def test_apply_normalization_uses_given_stats_not_recomputed():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    # Deliberately "wrong" stats, different from df's own mean/std - this
    # proves apply_normalization uses exactly what it's given rather than
    # silently recomputing from df (the fit/apply separation that avoids
    # train/test leakage).
    fake_stats = pd.DataFrame({"mean": [0.0], "std": [2.0]}, index=["a"])
    result = apply_normalization(df, ["a"], fake_stats)
    assert np.allclose(result["a"], [0.5, 1.0, 1.5])


def test_apply_normalization_result_has_zero_mean_unit_std():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = fit_normalization_stats(df, ["a"])
    result = apply_normalization(df, ["a"], stats)
    assert np.isclose(result["a"].mean(), 0.0, atol=1e-10)
    assert np.isclose(result["a"].std(), 1.0, atol=1e-10)


def test_apply_normalization_leaves_id_and_time_untouched(domain_config_stub, raw_df_stub):
    cleaned = clean(raw_df_stub, domain_config_stub)
    feature_columns = ["op_setting_1", "sensor_1"]
    stats = fit_normalization_stats(cleaned, feature_columns)
    normalized = apply_normalization(cleaned, feature_columns, stats)
    pd.testing.assert_series_equal(normalized["unit_number"], cleaned["unit_number"])
    pd.testing.assert_series_equal(normalized["time_cycles"], cleaned["time_cycles"])


# --- load_and_prepare: integration test against the real dataset ---


def test_load_and_prepare_end_to_end_on_real_fd001(project_root):
    config = load_domain_config()
    filepath = project_root / "data" / "raw" / "train_FD001.txt"
    df, stats, feature_columns = load_and_prepare(filepath, config)

    assert df.shape == (20631, 19)
    assert df.isnull().sum().sum() == 0
    assert len(feature_columns) == 17
    assert not (stats["std"] == 0).any()
