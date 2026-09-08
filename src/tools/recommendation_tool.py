import textwrap

from langchain_core.tools import tool

from src.tools.anomaly_detection_tool import detect_anomalies
from src.tools.diagnosis_tool import diagnose, _get_cached_llm

def build_recommendation_prompt(diagnosis: str) -> str:
    return textwrap.dedent(f"""\
        You are a maintenance planning assistant for turbofan engines.

        Diagnosis: {diagnosis}

        Based on this diagnosis, provide a prioritized list of corrective actions.
        For each, briefly state what to do and why. Order from most to least urgent.
        Be concise and practical.""")

def recommend(query: str) -> str:
    anomaly_report = detect_anomalies(query) 
    if "anomaly_score" not in anomaly_report:
        return anomaly_report
    diagnosis = diagnose(query)
    prompt = build_recommendation_prompt(diagnosis)
    llm = _get_cached_llm()
    response = llm.invoke(prompt)
    return response.content

@tool
def recommendation_tool(query: str) -> str:
    """Generates a prioritized list of corrective maintenance actions for
    a specific engine, based on its diagnosed probable root cause. Use
    this last, after diagnosis_tool has identified why something is
    happening, when the user asks what should be done about it.

    Input format: same as the other tools - 'unit: <id>, cycles: <spec>'.
    """
    return recommend(query)

if __name__ == "__main__":
    import time

    print("=== Step 1: build_recommendation_prompt in isolation (no LLM, no data) ===")
    print("(confirms the prompt text itself is clean before spending time on a real call)")
    fake_prompt = build_recommendation_prompt("Fan Degradation, 80% likelihood.")
    print(fake_prompt)

    print()
    print("=== Step 2: error passthrough - should be instant, zero LLM calls ===")
    start = time.time()
    print(recommend("unit: 500"))
    print(f"(took {time.time() - start:.2f}s)")

    print()
    print("=== Step 3: full recommend() on real data ===")
    print("This chains 2 LLM calls (one inside diagnose(), one here) - expect")
    print("roughly 2x your single-call benchmark, not a quick response.")
    start = time.time()
    result = recommend("unit: 1, cycles: last 20")
    elapsed = time.time() - start
    print(result)
    print(f"(took {elapsed:.1f}s)")

    print()
    print("=== Step 4: the actual LangChain Tool ===")
    print("name:", recommendation_tool.name)