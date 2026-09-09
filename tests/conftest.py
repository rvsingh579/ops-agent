import sys
from pathlib import Path

# Make `from src.data_loader import ...` work regardless of how pytest is
# invoked (bare `pytest`, `python -m pytest`, run from a different cwd, etc.)
# by explicitly putting the repo root on sys.path ourselves, rather than
# relying on pytest's own import-mode guessing.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def domain_config_stub():
    """A minimal, hand-built domain config - NOT the real domain.yaml.

    Unit tests use this instead of the real 21-sensor config so they test
    the logic in data_loader.py / anomaly_detector.py, not the specifics
    of the real CMAPSS dataset (that's what the integration tests, using
    the real file, are for).
    """
    return {
        "asset": {"id_column": "unit_number", "time_column": "time_cycles"},
        "operational_settings": [
            {"name": "op_setting_1"},
            {
                "name": "op_setting_2",
                "informative": False,
                "exclude_reason": "stub: constant by design",
            },
        ],
        "sensors": [
            {"name": "sensor_1"},
            {
                "name": "sensor_2",
                "informative": False,
                "exclude_reason": "stub: constant by design",
            },
        ],
    }


@pytest.fixture
def tool_domain_config_stub():
    """Like domain_config_stub, but with the richer fields the tools/ layer
    needs (sensor symbol/description/unit, fault_types) - kept as a SEPARATE
    fixture rather than bolting these onto domain_config_stub, since that
    one is scoped to data_loader.py's own tests and doesn't need them.
    """
    return {
        "asset": {"id_column": "unit_number", "time_column": "time_cycles"},
        "operational_settings": [
            {"name": "op_setting_1"},
        ],
        "sensors": [
            {
                "name": "sensor_8",
                "symbol": "Nf",
                "description": "Physical fan speed",
                "unit": "rpm",
            },
            {
                "name": "sensor_3",
                "symbol": "T30",
                "description": "HPC outlet temperature",
                "unit": "degR",
            },
        ],
        "fault_types": [
            {
                "name": "fan_degradation",
                "label": "Fan Degradation",
                "description": "Fan degradation stub: Nf, NRf, BPR affected.",
            },
            {
                "name": "hpc_degradation",
                "label": "HPC Degradation",
                "description": "HPC degradation stub: T30, T50, NRc affected.",
            },
        ],
    }


class FakeChatResponse:
    """Mimics the .content attribute on LangChain's AIMessage, without
    needing a real ChatOllama/Ollama server to produce one."""

    def __init__(self, content):
        self.content = content


class FakeChatModel:
    """Drop-in replacement for ChatOllama in tests: same .invoke(prompt)
    interface, returns a fixed response instantly. Stores the prompt it was
    called with, so a test can assert on what was actually sent to the
    "model" without needing the real model to respond well - or at all.
    """

    def __init__(self, content="fake LLM response"):
        self.content = content
        self.last_prompt = None
        self.call_count = 0

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return FakeChatResponse(self.content)


class RaisingChatModel:
    """A fake LLM that raises if invoked at all - use this to PROVE a code
    path never calls the LLM (e.g. an error-passthrough guard), rather than
    just hoping the assertion would have caught a stray call.
    """

    def invoke(self, prompt):
        raise AssertionError(
            "LLM was called when it should have been skipped entirely "
            "(e.g. an error-passthrough guard didn't short-circuit)."
        )


@pytest.fixture
def fake_llm():
    return FakeChatModel()


@pytest.fixture
def raising_llm():
    return RaisingChatModel()


@pytest.fixture
def raw_df_stub():
    """Small synthetic raw dataframe matching domain_config_stub's columns.

    2 units x 4 cycles. op_setting_2 and sensor_2 are constant (simulating
    the real non-informative columns); op_setting_1 and sensor_1 vary.
    """
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 2, 2, 2, 2],
            "time_cycles": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_setting_1": [0.1, 0.2, 0.15, 0.05, -0.1, -0.2, 0.0, 0.1],
            "op_setting_2": [100.0] * 8,
            "sensor_1": [10.0, 12.0, 11.0, 9.0, 20.0, 22.0, 21.0, 19.0],
            "sensor_2": [5.0] * 8,
        }
    )
