from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.tools.data_retrieval_tool import data_retrieval_tool
from src.tools.anomaly_detection_tool import anomaly_detection_tool
from src.tools.diagnosis_tool import diagnosis_tool
from src.tools.recommendation_tool import recommendation_tool

SYSTEM_PROMPT = """You are an operational assistant for monitoring turbofan
engine health (NASA CMAPSS FD001 dataset, engines numbered 1-100).

Use tools together when a question needs more than one step - do not stop
after the first tool call if it doesn't fully answer what was asked.

- data_retrieval_tool: raw sensor readings/stats. Use for "what's happening"
  questions.
- anomaly_detection_tool: whether readings look abnormal. Use for "is
  something wrong" questions, or as a first step before diagnosing why.
- diagnosis_tool: probable root cause of an anomaly. Use for "why" questions
  - call this AFTER anomaly_detection_tool has confirmed an anomaly, or
  directly if the user explicitly asks for a cause.
- recommendation_tool: prioritized corrective actions. Use for "what should
  be done" questions. It already runs its own diagnosis internally - call
  it directly, don't chain diagnosis_tool yourself first.

If asked "why", don't stop at reporting anomaly scores - go on to call
diagnosis_tool for the actual root cause."""

# Named so evals/orchestrator_evals.py can reference the SAME value when
# recording which model produced a given eval run, rather than a second
# hardcoded "llama3.1:8b" string drifting out of sync with this one.
DEFAULT_MODEL = "llama3.1:8b"

def build_agent(checkpointer=None, model_name: str = DEFAULT_MODEL):
    if checkpointer is None:
        checkpointer = InMemorySaver()
    model = ChatOllama(model=model_name, temperature=0.0, num_predict=512)
    return create_agent(
        model=model,
        tools=[data_retrieval_tool, anomaly_detection_tool, diagnosis_tool, recommendation_tool],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

def extract_final_answer(result: dict) -> str:
    """Extract the user-facing final answer from an agent invoke() result,
    working around a known Ollama/Llama-3.1 quirk where a later,
    unnecessary tool-call attempt can leak as literal text
    (`<|python_tag|>{...}`) instead of being parsed into a real tool call.
    """
    final_message = result["messages"][-1]
    if final_message.content.strip().startswith("<|python_tag|>"):
        for msg in reversed(result["messages"][:-1]):
            if getattr(msg, "content", None) and not getattr(msg, "tool_calls", None):
                return msg.content
    return final_message.content


def ask(agent, message: str, thread_id: str = "default") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return extract_final_answer(result)

if __name__ == "__main__":
    # This is a small interactive demo, not the validation suite - that
    # now lives in evals/orchestrator_evals.py (run it with
    # `python3.11 -m evals.orchestrator_evals`), as ONE place for that
    # logic rather than duplicated inline here too.
    agent = build_agent()
    thread_id = "interactive"

    print("Turbofan Ops Agent - interactive demo. Ctrl+C or 'quit' to exit.")
    print(f"(all messages share thread_id={thread_id!r}, so this is one ongoing conversation)")
    print()

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message or message.lower() in ("quit", "exit"):
            break

        answer = ask(agent, message, thread_id=thread_id)
        print(f"agent> {answer}\n")
