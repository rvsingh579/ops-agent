import pytest

from src.tools.recommendation_tool import (
    build_recommendation_prompt,
    recommend,
    recommendation_tool,
)


# --- build_recommendation_prompt: regression test for the dedent-lookalike fix ---


def test_build_recommendation_prompt_has_no_stray_indentation():
    prompt = build_recommendation_prompt("SOME_DIAGNOSIS")

    assert "You are a maintenance planning assistant for turbofan engines." in prompt
    assert "SOME_DIAGNOSIS" in prompt
    assert "\n    " not in prompt


# --- recommend(): mocked LLM, isolated from the real (slow) chat model.
# detect_anomalies/diagnose are mocked with two-argument lambdas now -
# unit/cycles, matching their real signatures, not a single query string. ---


def test_recommend_returns_error_passthrough_without_calling_llm(monkeypatch, raising_llm):
    monkeypatch.setattr(
        "src.tools.recommendation_tool.detect_anomalies",
        lambda unit, cycles="last 20": "Unit 999 does not exist. Valid unit numbers range from 1 to 100.",
    )
    monkeypatch.setattr("src.tools.recommendation_tool._get_cached_llm", lambda: raising_llm)

    result = recommend(999)

    assert result == "Unit 999 does not exist. Valid unit numbers range from 1 to 100."


def test_recommend_builds_prompt_from_diagnosis(monkeypatch, fake_llm):
    monkeypatch.setattr(
        "src.tools.recommendation_tool.detect_anomalies",
        lambda unit, cycles="last 20": "anomaly_score: mean=0.05\nsome real-looking report",
    )
    # "patch where it's used": recommendation_tool.py did
    # `from src.tools.diagnosis_tool import diagnose`, so it has its OWN
    # name binding - patching diagnosis_tool.diagnose would not affect it.
    monkeypatch.setattr(
        "src.tools.recommendation_tool.diagnose",
        lambda unit, cycles="last 20": "Fan Degradation, 80% likelihood.",
    )
    monkeypatch.setattr("src.tools.recommendation_tool._get_cached_llm", lambda: fake_llm)

    result = recommend(1, "last 20")

    assert result == fake_llm.content
    assert "Fan Degradation, 80% likelihood." in fake_llm.last_prompt


def test_recommendation_tool_invoke_matches_recommend(monkeypatch, fake_llm):
    monkeypatch.setattr(
        "src.tools.recommendation_tool.detect_anomalies",
        lambda unit, cycles="last 20": "anomaly_score: mean=0.05\nsome real-looking report",
    )
    monkeypatch.setattr(
        "src.tools.recommendation_tool.diagnose",
        lambda unit, cycles="last 20": "Fan Degradation, 80% likelihood.",
    )
    monkeypatch.setattr("src.tools.recommendation_tool._get_cached_llm", lambda: fake_llm)

    result = recommendation_tool.invoke({"unit": 1, "cycles": "last 20"})

    assert result == fake_llm.content


# --- real end-to-end smoke test - actual chat LLM (chained: diagnose + this
# tool's own call), skipped by default given the ~1 minute cost ---


@pytest.mark.slow
def test_recommend_real_end_to_end_produces_prioritized_actions():
    result = recommend(1, "last 20")

    assert len(result) > 50
    assert not result.startswith("Unit")
    assert not result.startswith("No matching")
