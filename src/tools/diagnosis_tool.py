import textwrap

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from src.data_loader import load_domain_config
from src.tools.anomaly_detection_tool import detect_anomalies

def build_fault_documents(domain_config) -> list[Document]:
    """
    Convert each fault type in domain_config into a LangChain Document.

    The fault description is used as page_content because this is the
    text that will be embedded for semantic retrieval.

    Metadata preserves the fault name and label so the retrieved result
    can be traced back to the original fault type.
    """
    documents = []

    for entry in domain_config["fault_types"]:
        documents.append(
            Document(
                page_content=entry["description"],
                metadata={
                    "name": entry["name"],
                    "label": entry["label"],
                },
            )
        )

    return documents

_VECTOR_STORE_CACHE = None

def _get_cached_vector_store():
    """
    Lazily build and cache the FAISS vector store.

    The index is built only once per process. Subsequent calls return
    the cached vector store without re-embedding the fault documents.
    """
    global _VECTOR_STORE_CACHE

    if _VECTOR_STORE_CACHE is None:
        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:0.6b"
        )
        domain_config = load_domain_config()
        documents = build_fault_documents(domain_config)

        _VECTOR_STORE_CACHE = FAISS.from_documents(
            documents,
            embeddings,
        )

    return _VECTOR_STORE_CACHE

QUERY_INSTRUCTION = (
    "Instruct: Given anomalous turbofan engine sensor readings, retrieve "
    "the fault description that best explains the root cause.\nQuery:"
)

def build_retrieval_query(anomaly_report: str) -> str:
    """
    Distill the full anomaly report down to just the contributing-sensor
    signal before embedding it.

    Empirically verified this matters: embedding the WHOLE structured
    report (with boilerplate like "anomaly_score: mean=...") pulled
    retrieval toward whichever knowledge-base document happened to share
    that generic vocabulary (the "Sensor/Instrumentation Anomaly" entry,
    which literally repeats the word "sensor"), rather than the document
    that's actually the right domain match. A clean natural-language
    distillation, plus the query-instruction prefix Qwen3-Embedding's own
    model card recommends for queries (not documents), fixed it.
    """
    lines = anomaly_report.splitlines()
    contributing_lines = []
    capture = False
    for line in lines:
        if line.strip().lower().startswith("top contributing sensors"):
            capture = True
            continue
        if capture and line.strip():
            contributing_lines.append(line.strip())

    return "Anomalous sensors: " + "; ".join(contributing_lines)

def retrieve_fault_context(vector_store, anomaly_report: str, k=2) -> str:
    """
    Retrieve the k most relevant fault documents for an anomaly report
    and format them as context for the LLM.
    """
    query = QUERY_INSTRUCTION + build_retrieval_query(anomaly_report)

    documents = vector_store.similarity_search(
        query,
        k=k,
    )

    return "\n".join(f"{doc.metadata['label']}: {doc.page_content}" for doc in documents)

def build_diagnosis_prompt(anomaly_report: str, fault_context: str) -> str:
    """
    Build the diagnosis prompt from the anomaly report and
    retrieved fault context.
    """
    return textwrap.dedent(f"""\
        You are a diagnostic assistant for turbofan engines.

        Anomaly report:
        {anomaly_report}

        Reference fault information (most relevant matches from the knowledge base):
        {fault_context}

        Based on the anomaly report and the reference information, identify the
        most probable root cause(s), ranked by likelihood. Be concise.""")

_LLM_CACHE = None

def _get_cached_llm():
    """
    Lazily build and cache the Ollama chat model.

    Constructing a ChatOllama object itself is cheap (it doesn't load model
    weights - that happens inside the Ollama server on first .invoke()), but
    it's still cached here for consistency with every other shared resource
    in this project (vector store, fitted Isolation Forest, cleaned data).
    """
    global _LLM_CACHE

    if _LLM_CACHE is None:
        _LLM_CACHE = ChatOllama(
            model="llama3.2",
            base_url="http://localhost:11434",
            verbose=True,
            temperature=0.0,
            num_predict=512,
        )

    return _LLM_CACHE

def diagnose(query: str) -> str:
    # detect_anomalies() already calls parse_query() internally and returns
    # its error message directly if parsing fails - that error string never
    # contains "anomaly_score", so the guard below catches every failure
    # path (bad query, nonexistent unit, nonexistent cycles) in one place.
    anomaly_report = detect_anomalies(query)
    if "anomaly_score" not in anomaly_report:
        return anomaly_report
    vector_store = _get_cached_vector_store()
    fault_context = retrieve_fault_context(vector_store, anomaly_report)
    prompt = build_diagnosis_prompt(anomaly_report, fault_context)
    llm = _get_cached_llm()
    response = llm.invoke(prompt)
    return response.content

@tool
def diagnosis_tool(query: str) -> str:
    """Identifies probable root cause(s) for a specific engine's anomaly,
    ranked by likelihood, using retrieved fault knowledge. Use this after
    anomaly_detection_tool confirms something looks abnormal, or when the
    user asks WHY something is happening.

    Input format: same as the other tools - 'unit: <id>, cycles: <spec>'.
    """
    return diagnose(query)


if __name__ == "__main__":
    # The two _CACHE variables start as None/empty on purpose - that's the
    # same lazy-loading pattern as every other tool in this project. There
    # is nothing to "set up" by hand: the first call to
    # _get_cached_vector_store() builds the FAISS index (embedding all 3
    # fault documents), and the first call to _get_cached_llm() constructs
    # the chat model wrapper. Both then stay cached for the rest of this
    # process. Watch the cache variables flip from None to populated below.

    print("=== Step 1: build_fault_documents (no embedding yet, just text) ===")
    domain_config = load_domain_config()
    docs = build_fault_documents(domain_config)
    for d in docs:
        print(f"- {d.metadata['name']}: {len(d.page_content)} chars")

    print()
    print("=== Step 2: _get_cached_vector_store (THIS is where embedding happens) ===")
    print("cache before:", _VECTOR_STORE_CACHE)
    vector_store = _get_cached_vector_store()
    print("cache after:", type(vector_store).__name__, "- populated")

    print()
    print("=== Step 3: retrieval ALONE, no LLM involved yet ===")
    print("(cheap + fast - isolates 'did retrieval work' from 'did the LLM behave')")
    fan_like_report = (
        "anomaly_score: mean=0.03. Top contributing sensors: "
        "sensor_8 (Nf, Physical fan speed), sensor_13 (NRf, Corrected fan speed)"
    )
    hpc_like_report = (
        "anomaly_score: mean=0.03. Top contributing sensors: "
        "sensor_3 (T30, HPC outlet temp), sensor_14 (NRc, Corrected core speed)"
    )
    print("--- fan-like report retrieves: ---")
    print(retrieve_fault_context(vector_store, fan_like_report, k=1))
    print("--- HPC-like report retrieves: ---")
    print(retrieve_fault_context(vector_store, hpc_like_report, k=1))

    print()
    print("=== Step 4: error passthrough - should be instant, no LLM call ===")
    print(diagnose("unit: 500"))

    print()
    print("=== Step 5: full diagnose() on real data - this one is slow (LLM call) ===")
    print("unit 1's last 100 cycles are near its actual failure at cycle 192...")
    print(diagnose("unit: 1, cycles: last 100"))

    print()
    print("=== Step 6: the actual LangChain Tool ===")
    print("name:", diagnosis_tool.name)