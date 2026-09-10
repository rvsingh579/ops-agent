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

**All four tools share one input contract: separate `unit: int` /
`cycles: str` parameters.** (Revised from the original design - see the
Phase 3 note below on why.) Originally this was a single merged string
(`"unit: <id>, cycles: <spec>"`, parsed by a hand-written `parse_query`
regex) - the right call for a free-text ReAct agent, where the LLM's only
job is producing a sensible string and our own code does the parsing.
Redesigned once Phase 3 revealed `create_agent` uses *native* tool
calling: the model kept trying to decompose the merged string into
separate `unit`/`cycles` keys no matter how the prompt was worded, since
native calling expects a real argument schema, not free text for us to
parse ourselves. `parse_query` was deleted entirely - the framework
extracts typed arguments per each tool's schema now.
`resolve_cycle_range` (interpreting `"last 20"` vs `"10-15"` vs a bare
number) is unchanged either way - that's domain logic, not something the
calling convention affects.

**Tools re-fetch shared data themselves rather than relying on the LLM to
forward it between calls.**
E.g. `diagnosis_tool` re-calls `detect_anomalies(unit, cycles)` internally
instead of expecting the orchestrator to paste the previous tool's output
back in as input. Minimizes what has to survive as fragile natural-language
text through the LLM's own context - the data layer is shared directly
instead.

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

**Qwen3-Embedding-0.6B for embeddings, both via Ollama.**
Right-sized for a ~3-document knowledge base - embedding-quality
differences mostly matter when disambiguating many similar documents,
which doesn't apply at this scale. Measured directly on this machine
(Apple M1, 16GB): a "thinking"-mode model (`qwen3:latest`) took **2 min
3.96 sec** for a single short answer, spending most of that time in a
hidden reasoning block nobody asked to see; a plain instruct model of
similar size (`llama3.2`) took **27.1 sec** for the same question. The gap
is about model *configuration*, not size - reasoning-mode defaults are a
real, controllable latency cost.

**Chat model upgraded from `llama3.2` (3B) to `llama3.1:8b` after Phase 3
testing revealed a real multi-hop tool-chaining failure, not a prompt
problem.** `llama3.2` was the *original* choice purely for convenience -
it was already pulled, small, and fast, so it let Phase 2's tool-building
iterate quickly without a multi-GB download up front, and its single-call
latency (27.1s) looked good next to a reasoning-mode model. `llama3.1:8b`
was actually the model named in the very first project plan, before
anything was built - it just wasn't pulled yet, so testing defaulted to
whatever was convenient rather than what was originally intended. That
gap only became visible once Phase 3 needed genuine multi-step tool
orchestration, not single tool calls. Building the Phase 3 orchestrator
(`create_agent`, native tool
calling), `llama3.2` repeatedly recognized in its own generated text that
a second tool call was needed (e.g. "I would like to run diagnosis_tool to
determine the root cause...") but never actually issued that tool call -
3 for 3 on the exact "why" query the system prompt explicitly instructed
it to chain on. Strengthening the prompt to be much more forceful didn't
fix this correctly - it overcorrected into calling `recommendation_tool`
three redundant times for a question that never asked for
recommendations. Swapping to `llama3.1:8b` - no prompt change at all -
correctly chained `anomaly_detection_tool -> diagnosis_tool` on the first
try, with a clean final answer. Conclusion: reliable multi-step tool
orchestration is a capability that scales with model size here, not
something prompt engineering alone can fully substitute for on a 3B
model. Slower per call (121.3s for the 2-hop chain vs. `llama3.2`'s
~27s/call), but correct on the first try beats fast-but-wrong, especially
since `llama3.2`'s wrong turns (extra/redundant tool calls) were
themselves burning comparable time anyway.

**One shared cached chat model everywhere (`llama3.1:8b`), not a small
"router" model + separate "generator" model, and not different models in
different tool files.**
A real production system might split cheap routing decisions onto a small
fast model and reserve a larger model for generation. Deliberately not
done here: on 16GB unified memory, keeping two different multi-GB models
warm risks Ollama unloading one to fit the other, causing repeated
cold-start reloads mid-conversation - worse than just being consistently
one model. This is also why the model upgrade above has to be applied in
every file that constructs a `ChatOllama` (`diagnosis_tool.py`,
`recommendation_tool.py`, `orchestrator.py`) - not just the orchestrator.

**Known limitation, deliberately not fixed: the model can silently
transcribe a number wrong when constructing a tool call - "unit 500"
became `unit: 50`.** Found via the eval harness (`evals/orchestrator_evals.py`,
case 8, `"unit 500 status"`), then reproduced and pinpointed directly by
inspecting the raw message trace: `tool_calls: [{'name':
'data_retrieval_tool', 'args': {'cycles': 'last 20', 'unit': 50}, ...}]`.
Pydantic validated `50` as a perfectly good integer and passed it
straight through - there was no schema violation, no parsing error,
nothing to catch. `data_retrieval_tool` correctly returned real data for
unit 50 (which genuinely exists), so the final answer was internally
consistent and well-formed - it just silently answered the wrong
question. This is a fundamentally different class of bug from anything
else found in this project: every previous one had a real fix in our own
code or prompt (a wrong parameter name, a leaked format token,
insufficient chaining instructions). This one doesn't - the value itself
was simply wrong, and every validation layer we have correctly assumes
the value it receives is the one the user meant.

One plausible (unverified) mechanism: every tool's docstring states unit
numbers run "1-100 for the FD001 dataset" - it's possible the model's own
learned expectation of the valid range nudged an out-of-range "500"
toward something that fit it, rather than faithfully transcribing what
was actually typed. LLM number-handling being unreliable in exactly this
way is a documented general phenomenon, not something specific to this
project's setup.

Possible solutions considered, with why none were adopted here:
1. **Self-verification / repeat-back before calling the tool** - have the
   model restate its extracted parameters as a separate reasoning step
   before the actual tool call, so a mismatch has a chance to surface.
   *Tradeoff:* extra tokens and latency per call (same shape as the
   Corrective RAG tradeoff), and does not actually guarantee correctness
   - the same transcription error could just as easily occur in the
   restated version, since the underlying weakness is reading the digits
   correctly in the first place, not the number of times it's asked to.
2. **Cross-check the tool's chosen number against digits present in the
   raw user message** - flag a mismatch (a number in the tool call that
   never appeared in what the user actually typed) for confirmation.
   *Tradeoff:* reintroduces a hand-written parsing/matching layer over
   the user's raw text - close to the `parse_query` regex approach this
   project deliberately moved away from for tool arguments - and doesn't
   generalize to legitimate cases with no literal digit in the current
   message at all (e.g. a memory-based follow-up like "is that
   abnormal?", which correctly relies on conversation history rather than
   a number in the current message).
3. **State the unit number explicitly at the start of the final answer**
   (e.g. "Checking unit 50...") so a human reader has a fair chance to
   notice a mismatch against their own question. *Tradeoff:* cheap (a
   prompt instruction, no extra LLM calls), but doesn't prevent the error
   - only makes it easier for a careful human to catch, and depends on
   the model reliably following the instruction, which prior testing here
   (the chaining-prompt experiments) showed isn't guaranteed.
4. **A larger/more capable model** - the same lever that fixed the
   multi-hop chaining failure earlier likely reduces this too, since
   larger models are generally more faithful at copying exact tokens from
   context. *Tradeoff:* the same one already documented for that
   decision (latency, resource cost) - plus this failure mode isn't fully
   eliminated by model size alone even at the frontier, only made less
   frequent.
5. **Document and accept, as done here** - zero implementation cost,
   appropriate for this project's actual scope (a portfolio demo, not a
   system steering real maintenance decisions), but a real correctness
   gap in a setting where it would matter: if this pattern controlled
   real fleet maintenance actions, silently inspecting/recommending
   action on the wrong physical engine is a materially different kind of
   mistake than "the demo answered slightly imprecisely" - worth being
   honest that the stakes, not just the mechanism, are part of why this
   would need solving properly before any real deployment.

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

**Known limitation, deliberately not fixed: `retrieve_fault_context`
always returns its top-k, with no confidence check on whether they're
actually a good match.** Verified directly: fed it a deliberately
nonsense report (a single near-constant operational setting, no real
fault signal), and it still confidently returned a ranked top-3 with
real-looking distance scores (best: 0.9093) - in the *same numeric range*
as the genuinely good matches found for real anomalies (0.77-0.80 for
fan/HPC degradation earlier). A similarity-score threshold does not
solve this: there is no cutoff that reliably separates "real match" from
"no real match" here, since a nonsense query and a real one land in the
same numeric neighborhood. Consequence: our knowledge base only covers 3
fault types (FD001's actual HPC degradation, plus fan degradation and
generic sensor noise for retrieval to discriminate against). Any anomaly
that is genuinely none of these three would still get silently matched to
whichever is least-dissimilar, handed to the LLM as if it were solid
reference material, with no signal the match was actually weak.

The real fix is a different mechanism, not a better number: **Corrective
RAG** - grade each retrieved candidate with the LLM itself before trusting
it, since relevance is a judgment call the LLM is better suited for than
a raw vector distance.

```
current (implemented):
  anomaly report -> embed -> FAISS top-k -> stuff into prompt -> diagnosis
                                             (top-k trusted blindly)

corrective RAG (designed, not implemented):
  anomaly report -> embed -> FAISS top-k -> grade each candidate (LLM)
                                                    |
                                     any graded relevant?
                                      /                  \
                                   yes                    no
                                    |                      |
                           use only relevant ones   "no confident match -
                           -> diagnosis (grounded)   escalate for review"
```

**Why this stayed a design, not code: the tradeoff was decided explicitly,
not skipped by accident.** Grading adds one more LLM call to the
diagnosis chain - given the latency already measured for this hardware,
that roughly *doubles* this step's cost for every single diagnosis
request, not just the rare bad-match case. Given the actual scope here
(FD001, one real fault mode, a 3-entry knowledge base built to
demonstrate the pattern rather than cover a production fault taxonomy),
that cost wasn't justified. Instead, top-k retrieval quality was verified
manually across the realistic queries this project actually exercises
(fan-like and HPC-like anomaly reports - see the retrieval-fix entry
above) and found acceptable *for this scenario*. This is the right
tradeoff to name explicitly in review, not the right tradeoff to leave
unstated: correctness safety net vs. doubled latency, decided in favor of
latency here because the failure mode (a genuinely novel, unrepresented
fault type) is rare at this project's actual scope - it would flip the
other way for a knowledge base large enough that "is this really a good
match" stops being something a human can spot-check by hand.

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
