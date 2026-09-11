import uuid

import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.orchestrator import ask, build_agent
from src.tools.anomaly_detection_tool import _get_cached_scores
from src.tools.data_retrieval_tool import _get_cached_data, resolve_cycle_range

st.set_page_config(page_title="Turbofan Ops Agent", page_icon="✈️", layout="wide")


# --- Shared, expensive resources: built once, cached across every rerun
# AND every visitor - see docs/DECISIONS.md for why this matters (rebuilding
# the agent per rerun would silently wipe conversation memory each time). ---


@st.cache_resource
def get_agent():
    return build_agent()


@st.cache_resource
def get_domain_info():
    df, domain_config, feature_columns = _get_cached_data()
    return domain_config, feature_columns


agent = get_agent()
domain_config, feature_columns = get_domain_info()
id_column = domain_config["asset"]["id_column"]
time_column = domain_config["asset"]["time_column"]
sensor_lookup = {s["name"]: s for s in domain_config["sensors"]}


def sensor_label(column: str) -> str:
    if column in sensor_lookup:
        return f"{column} ({sensor_lookup[column]['symbol']})"
    return column


# --- Per-session state: each browser tab gets its own conversation thread,
# not a fixed shared one - required once more than one person can use this
# at the same time (unlike the earlier CLI demo, which only ever had one
# user, so a fixed thread_id was fine there but would NOT be fine here). ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# --- Sidebar: drives the chart panel only, NOT a constraint on the chat ---

with st.sidebar:
    st.header("Sensor chart controls")
    selected_unit = st.number_input(
        "Engine unit", min_value=1, max_value=100, value=1, step=1
    )
    n_cycles = st.slider("Show last N cycles", min_value=5, max_value=200, value=20)

    preferred_defaults = ["sensor_8", "sensor_13", "sensor_3"]
    default_sensors = [c for c in preferred_defaults if c in feature_columns] or feature_columns[:3]
    selected_sensors = st.multiselect(
        "Sensors to chart",
        options=[c for c in feature_columns if c in sensor_lookup],
        default=default_sensors,
        format_func=sensor_label,
    )


# --- Header ---

st.title("Turbofan Ops Agent")
st.caption(
    "NASA CMAPSS FD001 - ask about any engine unit (1-100). "
    "Runs on a local, free model - answers can take 30s-2min, not instant."
)

with st.expander("How this works"):
    st.markdown(
        "Your question goes to an **orchestrator agent** that decides which "
        "of four tools to call - possibly more than one, in sequence, if "
        "the question needs it (e.g. \"why\" questions chain anomaly "
        "detection into diagnosis)."
    )
    st.code(
        """\
You ask a question
        |
        v
Orchestrator Agent  (decides which tool(s) to call, remembers this
        |             conversation via a per-session thread)
        |
        +--> Data Retrieval Tool        - raw sensor readings/stats
        +--> Anomaly Detection Tool     - Isolation Forest scoring
        +--> Diagnosis Tool             - RAG + LLM root-cause analysis
        +--> Recommendation Tool        - LLM corrective actions
                |
                v
        Final answer, in this chat
""",
        language=None,
    )
    st.markdown(
        "Full reasoning behind every design choice - including real bugs "
        "found and fixed along the way - is in `docs/DECISIONS.md`."
    )


# --- Chat (main panel) ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about an engine unit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (local model calls can take 30s-2min)"):
            try:
                answer = ask(agent, prompt, thread_id=st.session_state.thread_id)
            except Exception as exc:
                answer = f"Something went wrong reaching the local model: {exc}"
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


# --- Sensor chart (bottom panel) - reuses the SAME tested functions the
# tools already call, not new data logic. ---

st.subheader(f"Sensor readings - unit {selected_unit}")

df, _, _ = _get_cached_data()
unit_df = df[df[id_column] == selected_unit].sort_values(time_column)

if unit_df.empty:
    st.warning(f"Unit {selected_unit} does not exist in this dataset.")
elif not selected_sensors:
    st.info("Pick at least one sensor in the sidebar to chart.")
else:
    available_cycles = sorted(unit_df[time_column].unique())
    resolved_cycles = resolve_cycle_range(f"last {n_cycles}", available_cycles)
    chart_df = unit_df[unit_df[time_column].isin(resolved_cycles)]

    scored_df, _, _ = _get_cached_scores()
    chart_df = chart_df.merge(
        scored_df[[id_column, time_column, "is_anomaly"]],
        on=[id_column, time_column],
        how="left",
    )

    fig = make_subplots(
        rows=len(selected_sensors),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[sensor_label(s) for s in selected_sensors],
        vertical_spacing=0.06,
    )

    for i, sensor in enumerate(selected_sensors, start=1):
        fig.add_trace(
            go.Scatter(
                x=chart_df[time_column],
                y=chart_df[sensor],
                mode="lines",
                line=dict(width=2, color="#2563EB"),
                name=sensor_label(sensor),
                showlegend=False,
                hovertemplate="cycle %{x}<br>%{y:.3f}<extra></extra>",
            ),
            row=i,
            col=1,
        )

        flagged = chart_df[chart_df["is_anomaly"].fillna(False)]
        if not flagged.empty:
            fig.add_trace(
                go.Scatter(
                    x=flagged[time_column],
                    y=flagged[sensor],
                    mode="markers",
                    marker=dict(
                        size=9,
                        symbol="circle-open",
                        line=dict(width=2, color="#DC2626"),
                    ),
                    name="Flagged anomalous",
                    showlegend=(i == 1),
                    hovertemplate="cycle %{x}<br>%{y:.3f} (flagged anomalous)<extra></extra>",
                ),
                row=i,
                col=1,
            )

    fig.update_xaxes(title_text="cycle", row=len(selected_sensors), col=1)
    fig.update_layout(
        height=220 * len(selected_sensors),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
