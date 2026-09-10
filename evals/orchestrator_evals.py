"""
Orchestrator routing eval harness - deliberately NOT a pytest suite.

Every test in tests/ is deterministic: pure functions, or the LLM mocked
out entirely. These cases are the opposite - a real local model making
real autonomous tool-selection decisions. A different (but still
reasonable) routing choice here isn't a code regression, so forcing this
into pytest's binary pass/fail would misrepresent what's actually being
measured. This reports MISSING tools (a real routing failure - the model
never did something it needed to) separately from EXTRA tools (a real
model made an over-eager but not necessarily wrong choice), rather than
collapsing both into one pass/fail bit.

Run: python3.11 -m evals.orchestrator_evals
"""

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.orchestrator import DEFAULT_MODEL, build_agent, extract_final_answer


@dataclass
class EvalCase:
    query: str
    thread_id: str
    expected_tools: list[str] = field(default_factory=list)
    expected_answer_contains: str | None = None
    notes: str = ""


# Single-turn cases - each gets its OWN thread_id, so none of them
# accidentally inherit context from another (the bug we found and fixed
# earlier: reusing one thread_id across unrelated queries contaminates
# routing with unrelated prior context).
CASES = [
    EvalCase(
        query="What's happening with engine unit 14 right now?",
        thread_id="eval-1",
        expected_tools=["data_retrieval_tool"],
    ),
    EvalCase(
        query="Is anything abnormal with unit 7 in its last 30 cycles?",
        thread_id="eval-2",
        expected_tools=["anomaly_detection_tool"],
    ),
    EvalCase(
        query="Why is unit 1 showing anomalies in its last 20 cycles?",
        thread_id="eval-3",
        expected_tools=["anomaly_detection_tool", "diagnosis_tool"],
        notes="The core multi-hop chaining case - this is the one that failed 3/3 on llama3.2.",
    ),
    EvalCase(
        query="What should be done about unit 1's condition?",
        thread_id="eval-4",
        expected_tools=["recommendation_tool"],
    ),
    EvalCase(
        query="Show me the sensor readings for unit 50 around cycle 100.",
        thread_id="eval-5",
        expected_tools=["data_retrieval_tool"],
    ),
    EvalCase(
        query="What's the probable cause of unit 3's problems in cycles 1-50?",
        thread_id="eval-6",
        expected_tools=["diagnosis_tool"],
        notes="Direct cause question, no prior anomaly_detection_tool call needed.",
    ),
    EvalCase(
        query="Does unit 25 need maintenance based on its last 15 cycles?",
        thread_id="eval-7",
        expected_tools=["recommendation_tool"],
    ),
    EvalCase(
        query="unit 500 status",
        thread_id="eval-8",
        expected_tools=[],
        expected_answer_contains="does not exist",
        notes="Can't predict which tool the model tries first - what matters is graceful degradation in the final answer, not routing.",
    ),
]

# Memory cases share ONE thread_id across two turns, deliberately -
# this is the only thing here that isn't a routing test. It's testing
# whether "that" in turn 2 correctly resolves to "unit 1's last 20
# cycles" from turn 1, using conversation history alone.
MEMORY_CASE = {
    "thread_id": "eval-memory",
    "turns": [
        EvalCase(
            query="Tell me about unit 1's last 20 cycles.",
            thread_id="eval-memory",
            expected_tools=["data_retrieval_tool"],
        ),
        EvalCase(
            query="Is that abnormal?",
            thread_id="eval-memory",
            expected_tools=["anomaly_detection_tool"],
            notes="Tests memory: must resolve 'that' to unit 1 / last 20 cycles without restating it.",
        ),
    ],
}


def get_tools_called(result: dict) -> list[str]:
    return [
        call["name"]
        for msg in result["messages"]
        if getattr(msg, "tool_calls", None)
        for call in msg.tool_calls
    ]


def run_case(agent, case: EvalCase) -> dict:
    config = {"configurable": {"thread_id": case.thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": case.query}]},
        config=config,
    )
    actual_tools = get_tools_called(result)
    final_answer = extract_final_answer(result)

    expected_set = set(case.expected_tools)
    actual_set = set(actual_tools)

    return {
        "case": case,
        "actual_tools": actual_tools,
        "final_answer": final_answer,
        "missing": sorted(expected_set - actual_set),
        "extra": sorted(actual_set - expected_set),
        "answer_ok": (
            case.expected_answer_contains is None
            or case.expected_answer_contains.lower() in final_answer.lower()
        ),
    }


def print_result(label: str, r: dict) -> None:
    case = r["case"]
    ok = not r["missing"] and r["answer_ok"]
    status = "OK" if ok else "REVIEW"
    print(f"[{status}] {label}: {case.query!r}")
    print(f"   expected tools: {case.expected_tools}")
    print(f"   actual tools:   {r['actual_tools']}")
    if r["missing"]:
        print(f"   MISSING (real routing gap): {r['missing']}")
    if r["extra"]:
        print(f"   EXTRA (over-eager, not necessarily wrong): {r['extra']}")
    if case.expected_answer_contains and not r["answer_ok"]:
        print(f"   answer did NOT contain expected: {case.expected_answer_contains!r}")
    if case.notes:
        print(f"   note: {case.notes}")
    print(f"   final answer: {r['final_answer'][:200]}")
    print()


def results_to_dataframe(
    labeled_results: list[tuple[str, dict]],
    model_name: str,
    run_timestamp: datetime.datetime,
) -> pd.DataFrame:
    """model_name and run_timestamp are recorded on EVERY row, not just the
    filename - so multiple eval CSVs can be concatenated later (e.g. to
    compare routing accuracy before/after a model swap like llama3.2 ->
    llama3.1:8b) without losing track of which run produced which row.
    """
    rows = []
    for label, r in labeled_results:
        case = r["case"]
        status = "OK" if (not r["missing"] and r["answer_ok"]) else "REVIEW"
        rows.append({
            "run_timestamp": run_timestamp.isoformat(timespec="seconds"),
            "model": model_name,
            "case": label,
            "query": case.query,
            "expected_tools": ", ".join(case.expected_tools),
            "actual_tools": ", ".join(r["actual_tools"]),
            "missing": ", ".join(r["missing"]),
            "extra": ", ".join(r["extra"]),
            "answer_check_passed": r["answer_ok"],
            "status": status,
            "notes": case.notes,
            "final_answer": r["final_answer"],
        })
    return pd.DataFrame(rows)


def save_results(
    df: pd.DataFrame,
    run_timestamp: datetime.datetime,
    out_dir: str = "evals/results",
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Same run_timestamp used in the data rows AND the filename, rather than
    # generating a second, slightly-later timestamp at save time - keeps
    # the two consistent instead of drifting by however long the run took.
    stamp = run_timestamp.strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"orchestrator_eval_{stamp}.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    model_name = DEFAULT_MODEL
    run_timestamp = datetime.datetime.now()  # one timestamp for this whole run - data rows and filename both use it

    agent = build_agent(model_name=model_name)

    results = [run_case(agent, case) for case in CASES]
    for i, r in enumerate(results, start=1):
        print_result(f"case {i}", r)

    memory_results = [run_case(agent, case) for case in MEMORY_CASE["turns"]]
    for i, r in enumerate(memory_results, start=1):
        print_result(f"memory turn {i}", r)

    labeled = [(f"case {i}", r) for i, r in enumerate(results, start=1)] + [
        (f"memory turn {i}", r) for i, r in enumerate(memory_results, start=1)
    ]

    passed = sum(1 for _, r in labeled if not r["missing"] and r["answer_ok"])
    print(f"=== {passed}/{len(labeled)} cases had no missing tools / met answer expectations ===")
    print("(EXTRA tool calls are reported above but don't count against this - over-chaining is a")
    print(" separate, softer concern from a real routing gap)")

    df = results_to_dataframe(labeled, model_name=model_name, run_timestamp=run_timestamp)
    csv_path = save_results(df, run_timestamp=run_timestamp)
    print(f"\nResults saved to {csv_path} (model={model_name!r})")
