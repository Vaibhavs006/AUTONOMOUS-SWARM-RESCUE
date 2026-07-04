import sys
import logging
import random
import time
import warnings
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. BRUTE-FORCE SYS.STDERR STREAM INTERCEPTION (PERMANENT SILENCE)
# ==============================================================================
class ContextWarningStreamFilter:
    def __init__(self, target_stream):
        self.target_stream = target_stream

    def write(self, message):
        # Catch and discard low-level thread printouts containing the target warning
        if "missing ScriptRunContext" in message:
            return
        self.target_stream.write(message)

    def flush(self):
        self.target_stream.flush()

# Redirect stderr through our gatekeeper filter
sys.stderr = ContextWarningStreamFilter(sys.stderr)

# Standard sub-logger and warnings module fallback configuration
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ==============================================================================
# UI AND CONFIGURATION SETUP
# ==============================================================================
st.set_page_config(layout="wide", page_title="Swarm Telemetry Control")

st.markdown(
    """
    <style>
        .stApp {
            background-color: #090d16;
            color: #ffffff;
        }
        section[data-testid="stSidebar"] {
            background-color: #0d1527 !important;
            border-right: 1px solid #1e293b;
        }
        h1, h2, h3, h4, p, label {
            font-family: 'Courier New', Courier, monospace !important;
        }
        div[data-testid="stMetricValue"] {
            color: #22c55e !important;
            font-family: 'Courier New', Courier, monospace !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# 2. INITIALIZE MATRIX AND SWARM STATE MATRIX (35x35 Grid Enclosure)
GRID_SIZE = 35
BASE_X, BASE_Y = 17, 17

if "grid" not in st.session_state:
    np.random.seed(42)
    raw_grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.75, 0.25])
    raw_grid[BASE_X-2:BASE_X+3, BASE_Y-2:BASE_Y+3] = 0
    st.session_state.grid = raw_grid

if "explored" not in st.session_state:
    st.session_state.explored = np.zeros((GRID_SIZE, GRID_SIZE))
    st.session_state.explored[BASE_X-2:BASE_X+3, BASE_Y-2:BASE_Y+3] = 1

if "history_coverage" not in st.session_state:
    st.session_state.history_coverage = [2.0]

if "simulation_step" not in st.session_state:
    st.session_state.simulation_step = 0

if "agents" not in st.session_state:
    agents = []
    for i in range(8):
        agents.append(
            {
                "id": i,
                "pos": [BASE_X, BASE_Y],
                "state": "EXPLORE",
                "backtrack_stack": [(BASE_X, BASE_Y)],
                "isolation_timer": 0,
            }
        )
    st.session_state.agents = agents

# 3. SIDEBAR DECK CONFIGURATION CONTROLS
st.sidebar.markdown(
    "<h2 style='color:#22c55e; font-size:1.5rem;'>SWARM CONFIGURATION</h2>",
    unsafe_allow_html=True,
)
swarm_size = st.sidebar.slider("Active Node Swarm Size", 2, 8, 8)
comm_range = st.sidebar.slider("Mesh Comm Range (Tiles)", 3, 15, 12)
isolation_limit = st.sidebar.slider("Isolation Limit (Frames)", 5, 50, 50)
wall_density = st.sidebar.slider("Cave Wall Density (%)", 10, 60, 25)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h2 style='color:#22c55e; font-size:1.5rem;'>MISSION ACTIONS</h2>",
    unsafe_allow_html=True,
)

auto_run = st.sidebar.checkbox("Auto-Run Simulation Loop", value=False)
step_button = st.sidebar.button("Step Single Frame Execution")
reset_button = st.sidebar.button("Reset Mission Grid")

if reset_button:
    st.session_state.grid = np.random.choice(
        [0, 1], size=(GRID_SIZE, GRID_SIZE), p=[1 - wall_density / 100, wall_density / 100]
    )
    st.session_state.grid[BASE_X-2:BASE_X+3, BASE_Y-2:BASE_Y+3] = 0
    st.session_state.explored = np.zeros((GRID_SIZE, GRID_SIZE))
    st.session_state.explored[BASE_X-2:BASE_X+3, BASE_Y-2:BASE_Y+3] = 1
    st.session_state.history_coverage = [2.0]
    st.session_state.simulation_step = 0

    agents = []
    for i in range(swarm_size):
        agents.append(
            {
                "id": i,
                "pos": [BASE_X, BASE_Y],
                "state": "EXPLORE",
                "backtrack_stack": [(BASE_X, BASE_Y)],
                "isolation_timer": 0,
            }
        )
    st.session_state.agents = agents
    st.rerun()


# 4. SIMULATION EXECUTION LOGIC
def run_simulation_step():
    st.session_state.simulation_step += 1
    grid = st.session_state.grid
    explored = st.session_state.explored
    agents = st.session_state.agents

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for a in agents[:swarm_size]:
        curr_x, curr_y = a["pos"]
        connected_to_mesh = (
            abs(curr_x - BASE_X) <= comm_range and abs(curr_y - BASE_Y) <= comm_range
        )

        if not connected_to_mesh:
            a["isolation_timer"] += 1
        else:
            a["isolation_timer"] = max(0, a["isolation_timer"] - 2)

        if a["isolation_timer"] >= isolation_limit:
            a["state"] = "STATE_RECONNECT"

        if a["state"] == "STATE_RECONNECT":
            if len(a["backtrack_stack"]) > 1:
                a["backtrack_stack"].pop()
                prev_pos = a["backtrack_stack"][-1]
                a["pos"] = list(prev_pos)
            else:
                a["pos"] = [BASE_X, BASE_Y]
                a["state"] = "EXPLORE"
                a["isolation_timer"] = 0
        else:
            valid_moves = []
            frontier_moves = []

            for dx, dy in directions:
                nx, ny = curr_x + dx, curr_y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    if grid[nx, ny] == 0:
                        valid_moves.append((nx, ny))
                        if explored[nx, ny] == 0:
                            frontier_moves.append((nx, ny))

            if frontier_moves:
                next_pos = random.choice(frontier_moves)
            elif valid_moves:
                next_pos = random.choice(valid_moves)
            else:
                next_pos = (curr_x, curr_y)

            a["pos"] = list(next_pos)
            a["backtrack_stack"].append(next_pos)
            explored[next_pos[0], next_pos[1]] = 1

    total_walkable = np.sum(grid == 0)
    total_explored = np.sum((explored == 1) & (grid == 0))
    coverage_pct = (total_explored / total_walkable) * 100.0
    st.session_state.history_coverage.append(coverage_pct)


if step_button or auto_run:
    run_simulation_step()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h2 style='color:#22c55e; font-size:1.1rem;'>LIVE METRICS</h2>",
    unsafe_allow_html=True,
)
st.sidebar.metric("Simulation Frame", f"{st.session_state.simulation_step}")
st.sidebar.metric(
    "Global Map Coverage", f"{st.session_state.history_coverage[-1]:.1f} %"
)

# 5. RENDER TWO-COLUMN WORKSPACE MAIN INTERFACE
st.markdown(
    "<h1 style='color:#22c55e; text-align:center;'>AUTONOMOUS SWARM RESCUE MISSION DECK</h1>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.2, 1.0])

with col1:
    st.markdown(
        "<h3 style='color:#22c55e;'>TACTICAL MISSION CONTROL MAP</h3>",
        unsafe_allow_html=True,
    )

    display_matrix = np.copy(st.session_state.grid)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if display_matrix[x, y] == 0:
                if st.session_state.explored[x, y] == 1:
                    display_matrix[x, y] = 2

    fig_map = go.Figure()
    
    fig_map.add_trace(
        go.Heatmap(
            z=display_matrix,
            colorscale=[
                [0.0, "#05070c"],
                [0.5, "#1e1e24"],
                [1.0, "#0f2b5c"],
            ],
            showscale=False,
            hoverinfo="none",
        )
    )

    agents_to_plot = st.session_state.agents[:swarm_size]
    agent_x = [a["pos"][1] for a in agents_to_plot]
    agent_y = [a["pos"][0] for a in agents_to_plot]
    agent_labels = [f"N{a['id']}" for a in agents_to_plot]

    for a in agents_to_plot:
        ax, ay = a["pos"]
        if abs(ax - BASE_X) <= comm_range and abs(ay - BASE_Y) <= comm_range:
            fig_map.add_trace(
                go.Scatter(
                    x=[BASE_Y, ay],
                    y=[BASE_X, ax],
                    mode="lines",
                    line=dict(color="#22c55e", width=1.5, dash="dash"),
                    showlegend=False,
                    hoverinfo="none"
                )
            )

    fig_map.add_trace(
        go.Scatter(
            x=[BASE_Y],
            y=[BASE_X],
            mode="markers",
            marker=dict(size=24, color="rgba(0, 255, 255, 0.2)", symbol="circle", line=dict(width=1.5, color="#00ffff")),
            showlegend=False,
            hoverinfo="none",
        )
    )

    fig_map.add_trace(
        go.Scatter(
            x=[BASE_Y],
            y=[BASE_X],
            mode="markers",
            marker=dict(size=14, color="#00ffff", symbol="hexagram", line=dict(width=1, color="#ffffff")),
            name="Base Station",
            hovertext="Central Ground Control Base Terminal",
            hoverinfo="text",
        )
    )

    fig_map.add_trace(
        go.Scatter(
            x=agent_x,
            y=agent_y,
            mode="markers+text",
            marker=dict(size=19, color="#22c55e", line=dict(width=1.5, color="#ffffff")),
            text=agent_labels,
            textposition="middle center",
            textfont=dict(color="#000000", size=9, weight="bold"),
            hovertext=[
                f"Node {a['id']}<br>State: {a['state']}<br>Isolation Clock: {a['isolation_timer']}"
                for a in agents_to_plot
            ],
            hoverinfo="text",
            name="Exploring Node"
        )
    )

    fig_map.update_layout(
        template="plotly_dark",
        width=600,
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor="#111827", zeroline=False, showticklabels=False, range=[-0.5, GRID_SIZE-0.5]),
        yaxis=dict(showgrid=True, gridcolor="#111827", zeroline=False, showticklabels=False, autorange="reversed", range=[GRID_SIZE-0.5, -0.5]),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        showlegend=False
    )

    st.plotly_chart(fig_map, config={"displayModeBar": False}, use_container_width=True)

with col2:
    st.markdown(
        "<h3 style='color:#22c55e;'>SYSTEMIC REAL-TIME ANALYTICS</h3>",
        unsafe_allow_html=True,
    )

    fig_coverage = go.Figure()
    fig_coverage.add_trace(
        go.Scatter(
            y=st.session_state.history_coverage,
            mode="lines",
            line=dict(color="#22c55e", width=3),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.1)",
        )
    )
    fig_coverage.update_layout(
        title="SYSTEMIC EXPLORATION COVERAGE AREA (%) OVER TIME",
        title_font=dict(color="#22c55e", family="Courier New"),
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        xaxis=dict(showgrid=False, title="Simulation Frames"),
        yaxis=dict(showgrid=False, title="Percentage (%)", range=[0, 101]),
    )
    st.plotly_chart(fig_coverage, config={"displayModeBar": False}, use_container_width=True)

    node_names = [f"Node {a['id']}" for a in agents_to_plot]
    stack_sizes = [len(a["backtrack_stack"]) for a in agents_to_plot]
    isolation_clocks = [a["isolation_timer"] for a in agents_to_plot]

    fig_telemetry = go.Figure()
    fig_telemetry.add_trace(
        go.Bar(
            x=node_names,
            y=stack_sizes,
            name="Backtrack Stack Size",
            marker_color="#06b6d4",
        )
    )
    fig_telemetry.add_trace(
        go.Bar(
            x=node_names,
            y=isolation_clocks,
            name="Isolation Timer",
            marker_color="#ef4444",
        )
    )

    fig_telemetry.update_layout(
        title="AGENT INDIVIDUAL STATUS TELEMETRY",
        title_font=dict(color="#22c55e", family="Courier New"),
        barmode="group",
        template="plotly_dark",
        height=280,
        margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="Metrics Clock Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_telemetry, config={"displayModeBar": False}, use_container_width=True)

# 6. FORCE REFRESH INTERACTION LOOP IF AUTO-RUN IS ACTIVATED
if auto_run:
    time.sleep(0.05)
    st.rerun()