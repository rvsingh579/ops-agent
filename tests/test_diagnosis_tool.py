import pytest

from src.data_loader import load_domain_config
from src.tools.diagnosis_tool import (
    _get_cached_vector_store,
    build_diagnosis_prompt,
    build_fault_documents,
    build_retrieval_query,
    diagnose,
    diagnosis_tool,
    retrieve_fault_context,
)


# --- build_fault_documents: real domain_config, no embedding involved ---


def test_build_fault_documents_returns_all_fault_types():
    domain_config = load_domain_config()
    documents = build_fault_documents(domain_config)

    names = [doc.metadata["name"] for doc in documents]
    assert names == ["hpc_degradation", "fan_degradation", "sensor_instrumentation_anomaly"]


# --- build_retrieval_query: pure string logic, regression test for the bug we found ---


def test_build_retrieval_query_extracts_only_contributing_sensors():
    anomaly_report = (
        "Unit 1, cycles 173-192 (20 readings):\n"
        "anomaly_score: mean=-0.0063 min=-0.0731 max=0.0538\n"
        "8/20 readings flagged anomalous (40.0%)\n"
        "Most recent cycle (192): anomaly_score=0.0248, is_anomaly=True\n"
        "Top contributing sensors at most recent cycle:\n"
        "  sensor_8 (Nf, Physical fan speed): z-score=3.146\n"
        "  sensor_13 (NRf, Corrected fan speed): z-score=3.112\n"
    )

    result = build_retrieval_query(anomaly_report)

    assert result == (
        "Anomalous sensors: sensor_8 (Nf, Physical fan speed): z-score=3.146; "
        "sensor_13 (NRf, Corrected fan speed): z-score=3.112"
    )
    # the boilerplate that caused the original bug must NOT survive distillation
    assert "anomaly_score: mean" not in result


# --- build_diagnosis_prompt: regression test for the dedent fix ---


def test_build_diagnosis_prompt_has_no_stray_indentation():
    prompt = build_diagnosis_prompt("REPORT_TEXT", "CONTEXT_TEXT")

    assert "You are a diagnostic assistant for turbofan engines." in prompt
    assert "\n    " not in prompt


# --- retrieve_fault_context: REAL embeddings (fast, not mocked) - regression
# test for the fan-vs-hpc retrieval fix ---


def test_retrieve_fault_context_discriminates_fan_vs_hpc():
    vector_store = _get_cached_vector_store()

    fan_like_report = (
        "Top contributing sensors at most recent cycle:\n"
        "  sensor_8 (Nf, Physical fan speed): z-score=3.0\n"
        "  sensor_13 (NRf, Corrected fan speed): z-score=3.0\n"
    )

    result = retrieve_fault_context(vector_store, fan_like_report, k=1)

    assert "Fan Degradation" in result


# --- diagnose(): mocked LLM, isolated from the real (slow) chat model.
# detect_anomalies is mocked with a two-argument lambda now - unit/cycles,
# matching its real signature, not a single query string. ---


def test_diagnose_returns_error_passthrough_without_calling_llm(monkeypatch, raising_llm):
    monkeypatch.setattr(
        "src.tools.diagnosis_tool.detect_anomalies",
        lambda unit, cycles="last 20": "Unit 999 does not exist. Valid unit numbers range from 1 to 100.",
    )
    monkeypatch.setattr("src.tools.diagnosis_tool._get_cached_llm", lambda: raising_llm)

    result = diagnose(999)

    assert result == "Unit 999 does not exist. Valid unit numbers range from 1 to 100."


def test_diagnose_builds_prompt_with_anomaly_and_context(monkeypatch, fake_llm):
    fake_report = (
        "Unit 1, cycles 173-192 (20 readings):\n"
        "anomaly_score: mean=-0.0063 min=-0.0731 max=0.0538\n"
        "8/20 readings flagged anomalous (40.0%)\n"
        "Most recent cycle (192): anomaly_score=0.0248, is_anomaly=True\n"
        "Top contributing sensors at most recent cycle:\n"
        "  sensor_8 (Nf, Physical fan speed): z-score=3.146\n"
    )
    monkeypatch.setattr(
        "src.tools.diagnosis_tool.detect_anomalies",
        lambda unit, cycles="last 20": fake_report,
    )
    monkeypatch.setattr("src.tools.diagnosis_tool._get_cached_llm", lambda: fake_llm)

    result = diagnose(1, "last 20")

    assert result == fake_llm.content
    assert fake_report in fake_llm.last_prompt
    # retrieval was NOT mocked here - this confirms the real (fast) embedding
    # step actually ran and found the right document
    assert "Fan Degradation" in fake_llm.last_prompt


def test_diagnosis_tool_invoke_matches_diagnose(monkeypatch, fake_llm):
    monkeypatch.setattr(
        "src.tools.diagnosis_tool.detect_anomalies",
        lambda unit, cycles="last 20": (
            "anomaly_score: mean=0.05\nTop contributing sensors:\n"
            "  sensor_8 (Nf, Physical fan speed): z-score=3.0"
        ),
    )
    monkeypatch.setattr("src.tools.diagnosis_tool._get_cached_llm", lambda: fake_llm)

    result = diagnosis_tool.invoke({"unit": 1, "cycles": "last 20"})

    assert result == fake_llm.content


# --- real end-to-end smoke test - actual chat LLM, skipped by default ---


@pytest.mark.slow
def test_diagnose_real_end_to_end_identifies_fan_degradation():
    result = diagnose(1, "last 20")

    assert "fan degradation" in result.lower()
