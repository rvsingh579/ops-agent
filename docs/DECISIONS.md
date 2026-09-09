# Engineering Decisions

A running log of the non-obvious choices made in this project and *why* -
written so the reasoning survives past the moment it was made, not just the
code. Ordered roughly by when each decision came up.

## Phase 1 — Data & anomaly detection

**Dataset subset: FD001, not FD002-004.**
FD001 has one operating condition and one fault mode (HPC degradation).
The other subsets add multiple operating regimes and a second fault mode -
real complexity, but not needed to prove the pipeline works. Simplest
subset first; the domain config is designed so a harder subset is a config
change, not a rewrite.

**Dropped 6 sensors + 1 operational setting, based on measured variance,
not a published list.**
`sensor_1/5/10/16/18/19` and `op_setting_3` are constant (std ≈ 0, or
1e-15/1e-18 floating-point noise) in FD001 - verified directly on the raw
data, not assumed from CMAPSS folklore. `sensor_6` was a genuine borderline
case (std = 1.4e-3) but kept, since that's orders of magnitude above actual
floating-point noise - it's small, real signal. All exclusions are recorded
in `config/domain.yaml` with an `exclude_reason` per column, not silently
dropped.

**Isolation Forest, not a classifier.**
The data gives no per-row "this is a fault" label - only that each training
trajectory *ends* in failure. Knowing *when* an engine fails is not the
same as knowing *which cycles were already anomalous*: failure is a
gradual drift, not a labeled switch. Forcing a classifier here would mean
inventing an arbitrary label (e.g. "last 30 cycles = anomalous") and then
grading the model against a threshold nobody actually knows is right.
Isolation Forest sidesteps this: it learns what normal multivariate sensor
behavior looks like and flags deviation, without a human pre-deciding
where "normal" ends.

**One global model across all 100 engines, not one model per engine.**
All 100 units share the same fault mode, and "normal" (early-life) cycles
vastly outnumber "near-failure" cycles across the whole fleet - exactly
the majority/minority split Isolation Forest needs. A model fit on just
one engine's own cycles would only ever find "the oddest cycle *of this
one engine's short life*," which isn't the same question.

**Fixed `random_state=42` everywhere randomness is involved.**
Isolation Forest is inherently randomized (random feature/split choices
per tree). Without a fixed seed, scores would differ slightly between
runs - bad for a demo where reproducibility matters, and it also makes
caching a fitted model *correct*, not just fast (see below).

**Fit/apply separation for normalization (`fit_normalization_stats` /
`apply_normalization` as two functions, never one).**
Normalization stats (mean/std) must be computed once from training data
and then *applied* - never recomputed - to anything else, including a
future test set. Collapsing this into one step is the single most common
way train/test leakage sneaks into a pipeline unnoticed.

**A `fit_normalization_stats` guard that raises on any zero-std column.**
Not just belt-and-suspenders: this is exactly the bug class that bit us
once already (`op_setting_3` slipped through as un-excluded, causing a
silent divide-by-zero → `NaN` cascade several functions downstream). The
guard turns a silent, hard-to-trace failure into an immediate, specific
error the moment it would occur.

## Phase 2 — Tools

**All four tools share one input contract: `"unit: <id>, cycles: <spec>"`.**
This is a LangChain agent design choice, not an accident: the orchestrator
LLM is the one writing tool inputs, and small local models are worse at
faithfully reproducing long free-text between calls than at repeating one
short, consistent format. One format also means later tools can reuse
earlier tools' parsing (`parse_query`, `resolve_cycle_range`) instead of
re-implementing it.

**Tools re-fetch shared data themselves rather than relying on the LLM to
forward it between calls.**
E.g. `diagnosis_tool` re-calls `detect_anomalies(query)` internally instead
of expecting the orchestrator to paste the previous tool's output back in
as input. Minimizes what has to survive as fragile natural-language text
through the LLM's own context - the data layer is shared directly instead.

**Each tool validates success/failure by checking a deterministic function's
output for a known marker (e.g. `"anomaly_score" in result`), never by
pattern-matching another LLM's free-text output.**
An LLM's successful output has no guaranteed fixed shape to check for -
only the deterministic, tested functions in this codebase do.

**RAG (FAISS + embeddings) for Diagnosis Tool, but *not* for Recommendation
Tool.**
Diagnosis needs domain-private facts the LLM doesn't have (this project's
specific fault taxonomy) - that's what retrieval is for. Recommendation
just needs to turn a known root cause into an action list, which is
general reasoning a capable instruct model already has. Reaching for RAG
everywhere "because this project uses RAG" would have been the wrong
instinct.

**Qwen3-Embedding-0.6B for embeddings; a plain instruct model (`llama3.2`,
no extended "thinking" mode) for chat, both via Ollama.**
Right-sized for a ~3-document knowledge base - embedding-quality
differences mostly matter when disambiguating many similar documents,
which doesn't apply at this scale. Measured directly on this machine
(Apple M1, 16GB): a "thinking"-mode model (`qwen3:latest`) took **2 min
3.96 sec** for a single short answer, spending most of that time in a
hidden reasoning block nobody asked to see; a plain instruct model of
similar size (`llama3.2`) took **27.1 sec** for the same question. The gap
is about model *configuration*, not size - reasoning-mode defaults are a
real, controllable latency cost.

**One shared cached chat model, not a small "router" model + separate
"generator" model.**
A real production system might split cheap routing decisions onto a small
fast model and reserve a larger model for generation. Deliberately not
done here: on 16GB unified memory, keeping two different multi-GB models
warm risks Ollama unloading one to fit the other, causing repeated
cold-start reloads mid-conversation - worse than just being consistently
one model.

**`detect_anomalies`'s `"Top contributing sensors"` heuristic (largest
|z-score| on one row) is an approximation, not true model attribution.**
It's a reasonable, cheap signal, but Isolation Forest doesn't expose *why*
it flagged a row the way SHAP would. Documented as a heuristic everywhere
it's used, rather than oversold as "the model's reasoning."

**Embedding the *whole* structured anomaly report as the retrieval query
was a real bug, not a style choice.**
Verified empirically: doing so pulled retrieval toward whichever
knowledge-base document happened to share generic boilerplate vocabulary
(`"sensor"`, `"anomaly_score"`) rather than the document that was actually
the right domain match. Fixed by distilling the query down to just the
contributing-sensor signal (`build_retrieval_query`) plus the
query-instruction prefix Qwen3-Embedding's own model card recommends for
queries specifically (not documents). Regression-tested in
`test_diagnosis_tool.py`.

**Mock the LLM in tests; never call the real model in the default test
run.**
A test suite that takes minutes and needs a running Ollama server isn't
one that gets run often. `FakeChatModel` (records the prompt it was
called with, returns instantly) covers "did we build the right prompt and
wire the response through correctly." `RaisingChatModel` proves an
error-passthrough guard actually short-circuits - it raises if invoked at
all, rather than hoping an assertion would have noticed a stray call. The
two real end-to-end tests that do call the actual model are marked
`@pytest.mark.slow` and excluded by default (`pytest.ini`); run them
explicitly with `pytest -m slow`.

**"Patch where it's used, not where it's defined."**
`recommendation_tool.py` does `from src.tools.diagnosis_tool import
diagnose` - that creates its *own* name binding. A test that monkeypatches
`diagnosis_tool.diagnose` has no effect on `recommendation_tool.py`'s
already-imported reference; it must patch
`"src.tools.recommendation_tool.diagnose"` instead. Easy to get backwards
the first time.

## Latency, cost & scaling — measured, not assumed

Real numbers from this machine (Apple M1, 16GB RAM, no dedicated GPU),
not textbook estimates:

| Call | Measured time |
|---|---|
| Single chat call, `llama3.2` (no thinking mode) | 27.1 s |
| Single chat call, `qwen3:latest` (thinking mode enabled) | 2 min 3.96 s |
| Full `recommend()` chain (2 sequential LLM calls: `diagnose()` internally + this tool's own) | 64.0 s |

**Why this matters more here than in a typical chat app:** the ReAct
orchestrator (Phase 3) makes *multiple sequential* LLM calls per user
question - one to pick each tool, one to write the final answer, plus one
inside Diagnosis and one inside Recommendation. Calls can't run in
parallel (each depends on the previous step's output). A single question
that touches all four tools could plausibly chain 6-7 LLM calls -
at ~27 s each, that's minutes, before any orchestrator overhead. Cost
scales the same way for the same reason: an agentic system pays for N
calls per question, not one, so it's inherently more expensive per
interaction than a plain chatbot - latency and cost share one root cause
here (sequential calls), not two separate problems.

**What would actually break first if this had many concurrent users, and
what the real fix looks like** (not implemented here - a laptop demo
doesn't need it, but worth naming explicitly rather than pretending it
scales):
- `_CACHE` (module-level, in every tool) lives in one process's memory -
  correct for a single long-running process, but N worker processes would
  each redundantly hold their own copy of the fitted model / cleaned data.
  Fix: an external cache (Redis) shared across workers.
- Ollama serves one request at a time on one machine - no batching. Real
  concurrent-user serving needs a batching-capable server (vLLM, TGI) that
  processes multiple users' requests together on a GPU.
- Conversation memory (Phase 3's `ConversationBufferMemory`), if kept
  in-process, would tie a user to one specific server replica. Fix: an
  external session store (Redis/DB), not in-process state.

## Phase 4 (not yet built) — anticipated decisions

Recorded here in advance so they're not lost by the time we get there:
- Streamlit UI will call the orchestrator, not the four tools directly -
  keeps the agent's tool-selection logic the single source of routing
  truth.
- A demo GIF and screenshot go in the README once the UI exists - not
  faked ahead of time.
