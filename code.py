import time
import random
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Tuple, Dict, Any
from datetime import datetime
from collections import deque

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION LAYER (Strict Schema + JSON Export/Load)
# ==============================================================================
@dataclass(frozen=True)
class SystemConfig:
    """Immutable system configuration. Frozen dataclass prevents runtime mutations."""
    GRID_SIZE: int = 35
    BASE_COORD: Tuple[int, int] = (17, 17)
    SAFETY_ZONE: int = 2
    MAX_AGENTS: int = 8
    MAX_LOG_CAPACITY: int = 500
    CONVERGENCE_WINDOW: int = 30  # Ticks to check for plateau
    CONVERGENCE_THRESHOLD: float = 0.5  # % change threshold to detect convergence
    
    # Color Palette (Production-Grade Naming)
    BG_MAIN: str = "#090d16"
    BG_PANEL: str = "#0d1527"
    COLOR_PRIMARY: str = "#22c55e"    # Neon Green
    COLOR_SECONDARY: str = "#00ffff"  # Cyan
    COLOR_WARNING: str = "#ef4444"    # Red
    COLOR_INFO: str = "#06b6d4"       # Blue
    COLOR_GRID: str = "#111827"
    COLOR_TEXT: str = "#ffffff"
    COLOR_LOG_BG: str = "#050811"
    COLOR_LOG_TEXT: str = "#10b981"
    
    @classmethod
    def from_json(cls, json_path: str) -> "SystemConfig":
        """Load configuration from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except FileNotFoundError:
            return cls()  # Return defaults if file not found
    
    def to_json(self, json_path: str) -> None:
        """Export configuration to JSON file (immutable snapshot)."""
        config_dict = asdict(self)
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @property
    def colors(self) -> Dict[str, str]:
        """Expose color palette as dictionary for backward compatibility."""
        return {
            "bg_main": self.BG_MAIN,
            "bg_panel": self.BG_PANEL,
            "primary": self.COLOR_PRIMARY,
            "secondary": self.COLOR_SECONDARY,
            "warning": self.COLOR_WARNING,
            "info": self.COLOR_INFO,
            "grid": self.COLOR_GRID,
            "text": self.COLOR_TEXT,
            "log_bg": self.COLOR_LOG_BG,
            "log_text": self.COLOR_LOG_TEXT,
        }

# Initialize singleton config
CONFIG = SystemConfig()

# ==============================================================================
# 2. EVENT-DRIVEN LOGGING PIPELINE (Structured Telemetry)
# ==============================================================================
@dataclass
class SystemEvent:
    """Typed event payload for structured logging."""
    timestamp: float
    frame: int
    event_type: str  # "CONNECTION_LOST", "RECONNECT_SUCCESS", "MOVE", "EXPLORE", "ERROR"
    node_id: int
    position: Tuple[int, int]
    message: str
    severity: str = "INFO"  # "INFO", "WARNING", "CRITICAL"
    
    def to_log_line(self) -> str:
        """Format as telemetry log line."""
        return f"[Frame {self.frame}] Node {self.node_id} @ ({self.position[0]},{self.position[1]}): {self.message}"

class EventLogger:
    """Central event queue with ringbuffer for memory efficiency."""
    
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.events: deque = deque(maxlen=capacity)
    
    def log(self, frame: int, node_id: int, pos: Tuple[int, int], 
            event_type: str, message: str, severity: str = "INFO") -> None:
        """Append structured event to queue."""
        event = SystemEvent(
            timestamp=time.time(),
            frame=frame,
            event_type=event_type,
            node_id=node_id,
            position=pos,
            message=message,
            severity=severity
        )
        self.events.append(event)
    
    def get_logs(self, limit: int = None) -> List[str]:
        """Return formatted log lines (most recent first)."""
        logs = list(self.events)
        if limit:
            logs = logs[-limit:]
        return [event.to_log_line() for event in reversed(logs)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute telemetry statistics from event stream."""
        if not self.events:
            return {}
        
        event_types = {}
        node_events = {}
        severity_counts = {}
        
        for event in self.events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            node_events[event.node_id] = node_events.get(event.node_id, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        return {
            "total_events": len(self.events),
            "event_types": event_types,
            "node_activity": node_events,
            "severity_distribution": severity_counts,
        }
    
    def clear(self) -> None:
        """Reset event queue."""
        self.events.clear()

# ==============================================================================
# 3. CORE DATA MODELS
# ==============================================================================
class NodeState(str, Enum):
    """State machine for swarm agents."""
    EXPLORE = "EXPLORE"
    RECONNECT = "RECONNECT"

@dataclass
class SwarmAgent:
    """Strictly typed swarm node."""
    id: int
    pos: Tuple[int, int]
    state: NodeState = NodeState.EXPLORE
    backtrack_stack: List[Tuple[int, int]] = field(default_factory=list)
    isolation_timer: int = 0
    tiles_explored: int = 0  # Track unique tiles this agent discovered
    total_moves: int = 0  # Track total moves (for strategy comparison)

# ==============================================================================
# 4. MISSION CONTROL LOGIC
# ==============================================================================
class SwarmController:
    """Algorithmic execution engine with event logging integration."""
    
    @staticmethod
    def boot_system(wall_density: int, swarm_size: int, logger: EventLogger) -> None:
        """Initialize environment and deploy agents."""
        np.random.seed(42)
        base_x, base_y = CONFIG.BASE_COORD
        
        # Matrix generation
        prob_wall = wall_density / 100.0
        grid = np.random.choice(
            [0, 1], 
            size=(CONFIG.GRID_SIZE, CONFIG.GRID_SIZE), 
            p=[1 - prob_wall, prob_wall]
        )
        
        # Carve safety zone
        r = CONFIG.SAFETY_ZONE
        grid[base_x - r : base_x + r + 1, base_y - r : base_y + r + 1] = 0
        
        # Exploration mask
        explored = np.zeros((CONFIG.GRID_SIZE, CONFIG.GRID_SIZE))
        explored[base_x - r : base_x + r + 1, base_y - r : base_y + r + 1] = 1
        
        # Trajectory heatmap
        trajectory_heatmap = np.zeros((CONFIG.GRID_SIZE, CONFIG.GRID_SIZE))
        trajectory_heatmap[base_x, base_y] = swarm_size
        
        # Agent initialization
        agents = [
            SwarmAgent(id=i, pos=CONFIG.BASE_COORD, backtrack_stack=[CONFIG.BASE_COORD])
            for i in range(swarm_size)
        ]
        
        logger.log(
            frame=0, 
            node_id=-1, 
            pos=(base_x, base_y),
            event_type="SYSTEM_BOOT",
            message=f"Deployed swarm of {swarm_size} nodes | Wall density: {wall_density}%",
            severity="INFO"
        )
        
        st.session_state.update({
            "is_booted": True,
            "grid": grid,
            "explored": explored,
            "agents": agents,
            "coverage_history": [2.0],
            "clock": 0,
            "total_steps_taken": 0,
            "state_history": {"EXPLORE": [swarm_size], "RECONNECT": [0]},
            "trajectory_heatmap": trajectory_heatmap,
            "event_logger": logger,
            "has_converged": False,
            "convergence_tick": None,
            "agent_explore_stats": {i: {"tiles": 0, "moves": 0} for i in range(swarm_size)},
        })

    @staticmethod
    def step_forward(swarm_size: int, comm_range: int, isolation_limit: int) -> None:
        """Execute single algorithmic frame with structured event logging."""
        state = st.session_state
        logger = state["event_logger"]
        state.clock += 1
        base_x, base_y = CONFIG.BASE_COORD
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        current_explore_count = 0
        current_reconnect_count = 0

        for agent in state.agents[:swarm_size]:
            curr_x, curr_y = agent.pos
            
            # Telemetry link check
            is_linked = (abs(curr_x - base_x) <= comm_range and 
                         abs(curr_y - base_y) <= comm_range)

            if not is_linked:
                agent.isolation_timer += 1
                if agent.isolation_timer == 1:
                    logger.log(
                        frame=state.clock,
                        node_id=agent.id,
                        pos=agent.pos,
                        event_type="CONNECTION_LOST",
                        message=f"Lost comms link; routing unavailable",
                        severity="WARNING"
                    )
            else:
                if agent.isolation_timer > 0:
                    logger.log(
                        frame=state.clock,
                        node_id=agent.id,
                        pos=agent.pos,
                        event_type="CONNECTION_RESTORED",
                        message=f"Comms link re-established",
                        severity="INFO"
                    )
                agent.isolation_timer = max(0, agent.isolation_timer - 2)

            # Failsafe triggered
            if agent.isolation_timer >= isolation_limit:
                agent.state = NodeState.RECONNECT
                logger.log(
                    frame=state.clock,
                    node_id=agent.id,
                    pos=agent.pos,
                    event_type="FAILSAFE_TRIGGERED",
                    message=f"Isolation timeout exceeded; entering RECONNECT state",
                    severity="CRITICAL"
                )

            if agent.state == NodeState.RECONNECT:
                current_reconnect_count += 1
                if len(agent.backtrack_stack) > 1:
                    agent.backtrack_stack.pop()
                    agent.pos = agent.backtrack_stack[-1]
                    logger.log(
                        frame=state.clock,
                        node_id=agent.id,
                        pos=agent.pos,
                        event_type="BACKTRACK",
                        message=f"Retracing path to base; {len(agent.backtrack_stack)} hops remaining",
                        severity="INFO"
                    )
                else:
                    agent.pos = CONFIG.BASE_COORD
                    agent.state = NodeState.EXPLORE
                    agent.isolation_timer = 0
                    logger.log(
                        frame=state.clock,
                        node_id=agent.id,
                        pos=agent.pos,
                        event_type="RECONNECT_SUCCESS",
                        message=f"Successfully returned to base; resuming EXPLORE",
                        severity="INFO"
                    )
            else:
                current_explore_count += 1
                valid_moves, frontier_moves = [], []

                for dx, dy in directions:
                    nx, ny = curr_x + dx, curr_y + dy
                    if 0 <= nx < CONFIG.GRID_SIZE and 0 <= ny < CONFIG.GRID_SIZE:
                        if state.grid[nx, ny] == 0:
                            valid_moves.append((nx, ny))
                            if state.explored[nx, ny] == 0:
                                frontier_moves.append((nx, ny))

                if frontier_moves:
                    next_pos = random.choice(frontier_moves)
                    event_type = "EXPLORE"
                elif valid_moves:
                    next_pos = random.choice(valid_moves)
                    event_type = "WANDER"
                else:
                    next_pos = (curr_x, curr_y)
                    event_type = "STUCK"

                agent.pos = next_pos
                agent.backtrack_stack.append(next_pos)
                
                # Track if this is a new tile discovery
                if state.explored[next_pos[0], next_pos[1]] == 0:
                    state.agent_explore_stats[agent.id]["tiles"] += 1
                
                state.explored[next_pos[0], next_pos[1]] = 1
                
                if event_type == "EXPLORE":
                    logger.log(
                        frame=state.clock,
                        node_id=agent.id,
                        pos=next_pos,
                        event_type="EXPLORE",
                        message=f"Discovered frontier tile",
                        severity="INFO"
                    )
            
            # Track total moves per agent (optimization: only increment here)
            agent.total_moves += 1
            state.agent_explore_stats[agent.id]["moves"] += 1

            # Accumulate metrics
            state.total_steps_taken += 1
            state.trajectory_heatmap[agent.pos[0], agent.pos[1]] += 1

        state.state_history["EXPLORE"].append(current_explore_count)
        state.state_history["RECONNECT"].append(current_reconnect_count)

        # Coverage calculation (optimize: cache walkable count)
        walkable = np.sum(state.grid == 0)
        explored_count = np.sum((state.explored == 1) & (state.grid == 0))
        current_coverage = (explored_count / walkable) * 100.0
        state.coverage_history.append(current_coverage)
        
        # CONVERGENCE DETECTION: Check if exploration has plateaued
        if state.clock >= CONFIG.CONVERGENCE_WINDOW:
            recent_coverage = state.coverage_history[-CONFIG.CONVERGENCE_WINDOW:]
            coverage_change = recent_coverage[-1] - recent_coverage[0]
            
            if coverage_change < CONFIG.CONVERGENCE_THRESHOLD and not state.has_converged:
                state.has_converged = True
                state.convergence_tick = state.clock
                logger.log(
                    frame=state.clock,
                    node_id=-1,
                    pos=(base_x, base_y),
                    event_type="CONVERGENCE_DETECTED",
                    message=f"Exploration plateau detected: {coverage_change:.2f}% change in {CONFIG.CONVERGENCE_WINDOW} ticks",
                    severity="INFO"
                )

# ==============================================================================
# 5. RENDERING ENGINE
# ==============================================================================
class UIEngine:
    """Plotly rendering with clean UI/UX."""

    @staticmethod
    def render_map(swarm_size: int, comm_range: int) -> go.Figure:
        state = st.session_state
        base_x, base_y = CONFIG.BASE_COORD
        
        display = np.copy(state.grid)
        display[(display == 0) & (state.explored == 1)] = 2

        fig = go.Figure()
        
        # Base topography
        fig.add_trace(go.Heatmap(
            z=display,
            colorscale=[[0.0, "#05070c"], [0.5, "#1e1e24"], [1.0, "#0f2b5c"]],
            showscale=False, hoverinfo="none"
        ))

        agents = state.agents[:swarm_size]
        
        # Communication tethers and node status coloring
        node_colors = []
        for a in agents:
            ax, ay = a.pos
            is_linked = abs(ax - base_x) <= comm_range and abs(ay - base_y) <= comm_range
            # Green if connected, Red if isolated
            node_colors.append(CONFIG.COLOR_PRIMARY if is_linked else CONFIG.COLOR_WARNING)
            
            # Draw tether only if connected
            if is_linked:
                fig.add_trace(go.Scatter(
                    x=[base_y, ay], y=[base_x, ax], mode="lines",
                    line=dict(color=CONFIG.COLOR_PRIMARY, width=1.0, dash="dot"),
                    showlegend=False, hoverinfo="none"
                ))

        # Master hub
        fig.add_trace(go.Scatter(
            x=[base_y], y=[base_x], mode="markers",
            marker=dict(size=24, color="rgba(0, 255, 255, 0.15)", symbol="circle", 
                        line=dict(width=1.5, color=CONFIG.COLOR_SECONDARY)),
            showlegend=False, hoverinfo="none"
        ))

        # Agent array with dynamic connection status colors
        fig.add_trace(go.Scatter(
            x=[a.pos[1] for a in agents], y=[a.pos[0] for a in agents],
            mode="markers+text",
            marker=dict(size=16, color=node_colors, line=dict(width=1, color=CONFIG.COLOR_TEXT)),
            text=[f"N{a.id}" for a in agents], textposition="middle center", 
            textfont=dict(color="#000000", size=8, weight="bold"),
            hovertext=[f"Node {a.id} | State: {a.state} | Isolation: {a.isolation_timer}ms | Connected: {'Yes' if node_colors[i] == CONFIG.COLOR_PRIMARY else 'No (ISOLATED)'}" 
                      for i, a in enumerate(agents)], 
            hoverinfo="text", showlegend=False
        ))

        fig.update_layout(
            template="plotly_dark", height=650, margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=True, gridcolor=CONFIG.COLOR_GRID, zeroline=False, showticklabels=False, 
                      range=[-0.5, CONFIG.GRID_SIZE - 0.5]),
            yaxis=dict(showgrid=True, gridcolor=CONFIG.COLOR_GRID, zeroline=False, showticklabels=False, 
                      autorange="reversed", range=[CONFIG.GRID_SIZE - 0.5, -0.5]),
            paper_bgcolor=CONFIG.BG_MAIN, plot_bgcolor=CONFIG.BG_MAIN,
            uirevision="constant_map_lock"
        )
        return fig

    @staticmethod
    def render_analytics(swarm_size: int) -> go.Figure:
        state = st.session_state

        # Coverage trend
        fig_cov = go.Figure(go.Scatter(
            y=state.coverage_history, mode="lines",
            line=dict(color=CONFIG.COLOR_PRIMARY, width=2),
            fill="tozeroy", fillcolor="rgba(34, 197, 94, 0.1)"
        ))
        fig_cov.update_layout(
            title="Mesh Network Coverage (%)", title_font=dict(color=CONFIG.COLOR_PRIMARY, family="Courier New", size=14),
            template="plotly_dark", height=280, margin=dict(l=30, r=20, t=40, b=20),
            paper_bgcolor=CONFIG.BG_MAIN, plot_bgcolor=CONFIG.BG_MAIN, yaxis=dict(range=[0, 101])
        )
        
        return fig_cov

    @staticmethod
    def render_network_topology(swarm_size: int, comm_range: int) -> go.Figure:
        """Render network connection graph and node states."""
        state = st.session_state
        base_x, base_y = CONFIG.BASE_COORD
        agents = state.agents[:swarm_size]
        
        # Count connected vs isolated nodes
        connected_nodes = []
        isolated_nodes = []
        
        for a in agents:
            ax, ay = a.pos
            is_linked = abs(ax - base_x) <= comm_range and abs(ay - base_y) <= comm_range
            if is_linked:
                connected_nodes.append(a.id)
            else:
                isolated_nodes.append(a.id)
        
        # Create pie chart showing network partition
        labels = ["Connected\n(EXPLORE/RECONNECT)", "Isolated\n(OFFLINE)"]
        values = [len(connected_nodes), len(isolated_nodes)]
        colors_pie = [CONFIG.COLOR_PRIMARY, CONFIG.COLOR_WARNING]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors_pie),
            textinfo="label+value",
            hovertext=[f"Nodes: {connected_nodes}" if len(connected_nodes) > 0 else "None",
                      f"Nodes: {isolated_nodes}" if len(isolated_nodes) > 0 else "None"],
            hoverinfo="label+value"
        )])
        
        fig.update_layout(
            title="Network Connectivity Partition",
            title_font=dict(color=CONFIG.COLOR_PRIMARY, family="Courier New", size=13),
            template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=CONFIG.BG_MAIN, plot_bgcolor=CONFIG.BG_MAIN,
            showlegend=True
        )
        
        return fig

    @staticmethod
    def render_efficiency_breakdown() -> go.Figure:
        """Render path efficiency: useful steps vs wasted backtracks."""
        state = st.session_state
        
        total_steps = state.total_steps_taken
        unique_tiles = float(np.sum((state.explored == 1) & (state.grid == 0)))
        wasted_steps = total_steps - unique_tiles
        
        # Efficiency breakdown
        labels = ["Productive Exploration", "Wasted / Backtracking"]
        values = [unique_tiles, max(0, wasted_steps)]
        colors_breakdown = [CONFIG.COLOR_PRIMARY, CONFIG.COLOR_WARNING]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors_breakdown),
            textinfo="label+percent",
            hovertext=[f"{int(unique_tiles)} tiles discovered", 
                      f"{int(wasted_steps)} redundant steps"],
            hoverinfo="label+value"
        )])
        
        fig.update_layout(
            title="Path Efficiency Breakdown",
            title_font=dict(color=CONFIG.COLOR_PRIMARY, family="Courier New", size=13),
            template="plotly_dark", height=280, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=CONFIG.BG_MAIN, plot_bgcolor=CONFIG.BG_MAIN,
            showlegend=True
        )
        
        return fig

    @staticmethod
    def render_agent_strategy(swarm_size: int) -> go.Figure:
        """Compare agent exploration efficiency: tiles discovered vs moves taken."""
        state = st.session_state
        agents = state.agents[:swarm_size]
        
        # Calculate efficiency per agent
        agent_ids = [f"N{a.id}" for a in agents]
        tiles_discovered = [state.agent_explore_stats[a.id]["tiles"] for a in agents]
        total_moves = [state.agent_explore_stats[a.id]["moves"] for a in agents]
        
        # Efficiency = tiles / moves (avoid division by zero)
        efficiency = [
            (tiles_discovered[i] / max(1, total_moves[i])) * 100 
            for i in range(len(agents))
        ]
        
        fig = go.Figure()
        
        # Bar 1: Tiles discovered
        fig.add_trace(go.Bar(
            x=agent_ids, y=tiles_discovered, name="Tiles Discovered",
            marker_color=CONFIG.COLOR_PRIMARY, opacity=0.8
        ))
        
        # Bar 2: Total moves (scaled for comparison)
        fig.add_trace(go.Bar(
            x=agent_ids, y=total_moves, name="Total Moves",
            marker_color=CONFIG.COLOR_INFO, opacity=0.6
        ))
        
        fig.update_layout(
            title="Agent Strategy Comparison: Exploration Contribution",
            title_font=dict(color=CONFIG.COLOR_PRIMARY, family="Courier New", size=13),
            template="plotly_dark", height=300, margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=CONFIG.BG_MAIN, plot_bgcolor=CONFIG.BG_MAIN,
            barmode="group",
            yaxis=dict(title="Count"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig

    @staticmethod
    def render_summary_deck() -> Tuple[go.Figure, go.Figure]:
        """Generates retrospective timelines and heatmap traffic overlays."""
        state = st.session_state
        ticks = list(range(len(state.state_history["EXPLORE"])))

        # Network density timeline
        fig_downtime = go.Figure()
        fig_downtime.add_trace(go.Scatter(
            x=ticks, y=state.state_history["EXPLORE"], name="Exploring Nodes",
            mode="lines", line=dict(width=0.5, color=CONFIG.COLOR_PRIMARY),
            stackgroup="one", groupnorm="percent", fillcolor="rgba(34, 197, 94, 0.25)"
        ))
        fig_downtime.add_trace(go.Scatter(
            x=ticks, y=state.state_history["RECONNECT"], name="Reconnecting Nodes",
            mode="lines", line=dict(width=0.5, color=CONFIG.COLOR_WARNING),
            stackgroup="one", fillcolor="rgba(239, 68, 68, 0.4)"
        ))
        fig_downtime.update_layout(
            title="Network Node Density Profile (EXPLORE vs RECONNECT Ratio)",
            title_font=dict(color=CONFIG.COLOR_TEXT, family="Courier New", size=13),
            template="plotly_dark", height=300, margin=dict(l=40, r=20, t=40, b=30),
            paper_bgcolor=CONFIG.BG_PANEL, plot_bgcolor=CONFIG.BG_PANEL,
            xaxis=dict(title="System Ticks", showgrid=False),
            yaxis=dict(title="Percentage Allocation", ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )

        # Path trajectory heatmap
        fig_traffic = go.Figure(go.Heatmap(
            z=state.trajectory_heatmap,
            colorscale="Hot",
            showscale=True,
            colorbar=dict(title="Visits", title_font=dict(size=10), tickfont=dict(size=8))
        ))
        fig_traffic.update_layout(
            title="Spatial Trajectory Density Overlay",
            title_font=dict(color=CONFIG.COLOR_TEXT, family="Courier New", size=13),
            template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=CONFIG.BG_PANEL, plot_bgcolor=CONFIG.BG_PANEL,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False, autorange="reversed")
        )

        return fig_downtime, fig_traffic

    @staticmethod
    def render_event_log(logger: EventLogger, max_lines: int = 20) -> str:
        """Render structured event log as terminal-style box."""
        logs = logger.get_logs(limit=max_lines)
        if not logs:
            return "```\n[No events recorded]\n```"
        
        log_text = "\n".join(logs)
        return f"```\n{log_text}\n```"

# ==============================================================================
# 6. APPLICATION LAUNCH
# ==============================================================================
def apply_css():
    st.markdown(f"""
        <style>
            .stApp {{ background-color: {CONFIG.BG_MAIN}; color: {CONFIG.COLOR_TEXT}; }}
            section[data-testid="stSidebar"] {{ background-color: {CONFIG.BG_PANEL} !important; border-right: 1px solid #1e293b; }}
            h1, h2, h3, p, label, .stMetric label {{ font-family: 'Courier New', Courier, monospace !important; }}
            div[data-testid="stMetricValue"] {{ color: {CONFIG.COLOR_PRIMARY} !important; font-weight: bold; }}
            .block-container {{ padding-top: 1.5rem !important; }}
            div[data-testid="stExpander"] {{ background-color: {CONFIG.BG_PANEL} !important; border: 1px solid #1e293b; }}
            code {{ background-color: {CONFIG.COLOR_LOG_BG}; color: {CONFIG.COLOR_LOG_TEXT}; padding: 12px; border-radius: 4px; }}
        </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide", page_title="Swarm Ops", page_icon="🛰️")
    apply_css()

    # Initialize event logger once per session
    if "event_logger" not in st.session_state:
        st.session_state.event_logger = EventLogger(capacity=CONFIG.MAX_LOG_CAPACITY)

    with st.sidebar:
        st.markdown(f"<h2 style='color:{CONFIG.COLOR_PRIMARY}; text-align:center;'>CONTROL CENTER</h2>", unsafe_allow_html=True)
        
        with st.expander("🛠️ HARDWARE PARAMS", expanded=True):
            swarm_size = st.slider("Active Node Array", 2, CONFIG.MAX_AGENTS, CONFIG.MAX_AGENTS)
            comm_range = st.slider("Mesh Link Radius", 3, 15, 12)
            
        with st.expander("🌍 ENVIRONMENT PARAMS", expanded=False):
            wall_density = st.slider("Cave Density (%)", 10, 60, 25)
            isolation_limit = st.slider("Signal Failsafe Delay", 5, 50, 50)
        
        # Initialize if unbooted
        if "is_booted" not in st.session_state:
            SwarmController.boot_system(wall_density, swarm_size, st.session_state.event_logger)

        st.markdown("---")
        auto_run = st.toggle("🚀 ENGAGE AUTO-PILOT", value=False)
        
        col1, col2 = st.columns(2)
        if col1.button("STEP FWD", use_container_width=True) or auto_run:
            SwarmController.step_forward(swarm_size, comm_range, isolation_limit)
        if col2.button("WIPE SYS", type="primary", use_container_width=True):
            st.session_state.clear()
            st.session_state.event_logger = EventLogger(capacity=CONFIG.MAX_LOG_CAPACITY)
            SwarmController.boot_system(wall_density, swarm_size, st.session_state.event_logger)
            st.rerun()

        st.markdown("---")
        
        # REAL-TIME PERFORMANCE METRICS DASHBOARD
        st.markdown(f"<h3 style='color:{CONFIG.COLOR_PRIMARY}; font-size:14px;'>LIVE TELEMETRY</h3>", unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        m1.metric("SYS CLOCK", f"{st.session_state.clock} ticks")
        m2.metric("MAPPED", f"{st.session_state.coverage_history[-1]:.1f} %")
        
        # Calculate real-time performance metrics
        total_steps = st.session_state.total_steps_taken
        unique_tiles = float(np.sum((st.session_state.explored == 1) & (st.session_state.grid == 0)))
        
        # Exploration rate (new tiles per tick)
        if st.session_state.clock > 1:
            avg_exploration_rate = unique_tiles / st.session_state.clock
        else:
            avg_exploration_rate = 0
        
        # Path efficiency (unique tiles vs total steps)
        if total_steps > 0:
            path_efficiency = (unique_tiles / total_steps) * 100
        else:
            path_efficiency = 0
        
        # Network health (exploring vs reconnecting)
        current_explore_count = st.session_state.state_history["EXPLORE"][-1] if st.session_state.state_history["EXPLORE"] else 0
        current_reconnect_count = st.session_state.state_history["RECONNECT"][-1] if st.session_state.state_history["RECONNECT"] else 0
        online_agents = current_explore_count + current_reconnect_count
        network_health = (current_explore_count / online_agents * 100) if online_agents > 0 else 0
        
        m3, m4, m5 = st.columns(3)
        m3.metric("EXPLORATION RATE", f"{avg_exploration_rate:.2f} tiles/tick")
        m4.metric("PATH EFFICIENCY", f"{path_efficiency:.1f}%", 
                  help="Unique tiles discovered / total steps taken")
        m5.metric("NETWORK HEALTH", f"{network_health:.0f}%",
                  help="% of agents in EXPLORE state vs RECONNECT")
        
        # CONVERGENCE STATUS
        if st.session_state.has_converged:
            st.markdown(f"<div style='background-color:#1e293b; padding:10px; border-radius:4px; border-left:4px solid {CONFIG.COLOR_INFO};'><span style='color:{CONFIG.COLOR_INFO}; font-weight:bold;'>✓ CONVERGENCE DETECTED</span><br/><span style='color:#94a3b8; font-size:12px;'>Exploration plateau at tick {st.session_state.convergence_tick}</span></div>", unsafe_allow_html=True)
        else:
            ticks_remaining = CONFIG.CONVERGENCE_WINDOW - (st.session_state.clock % CONFIG.CONVERGENCE_WINDOW)
            st.markdown(f"<div style='background-color:#1e293b; padding:10px; border-radius:4px; border-left:4px solid {CONFIG.COLOR_WARNING};'><span style='color:{CONFIG.COLOR_WARNING}; font-weight:bold;'>○ EXPLORING</span><br/><span style='color:#94a3b8; font-size:12px;'>Convergence check in {ticks_remaining} ticks</span></div>", unsafe_allow_html=True)

    # Main workspace
    st.markdown(f"<h2 style='color:{CONFIG.COLOR_PRIMARY};'>AUTONOMOUS SWARM RESCUE TERMINAL</h2>", unsafe_allow_html=True)
    map_col, chart_col = st.columns([1.3, 1.0], gap="large")

    with map_col:
        st.plotly_chart(UIEngine.render_map(swarm_size, comm_range), config={"displayModeBar": False}, use_container_width=True)
        
        # LEGEND
        st.markdown("**Map Legend**")
        leg_col1, leg_col2, leg_col3 = st.columns(3)
        leg_col1.markdown(f"🟩 <span style='color:{CONFIG.COLOR_PRIMARY}'>Explored</span>", unsafe_allow_html=True)
        leg_col2.markdown(f"⬛ <span style='color:#0f2b5c'>Walls</span>", unsafe_allow_html=True)
        leg_col3.markdown(f"🔵 <span style='color:{CONFIG.COLOR_SECONDARY}'>Base Hub</span>", unsafe_allow_html=True)
        
        leg_col1b, leg_col2b, leg_col3b = st.columns(3)
        leg_col1b.markdown(f"🟢 <span style='color:{CONFIG.COLOR_PRIMARY}'>Connected</span>", unsafe_allow_html=True)
        leg_col2b.markdown(f"🔴 <span style='color:{CONFIG.COLOR_WARNING}'>Isolated</span>", unsafe_allow_html=True)
        leg_col3b.markdown(f"⤪ <span style='color:{CONFIG.COLOR_PRIMARY}'>Comm Link</span>", unsafe_allow_html=True)

    with chart_col:
        fig_cov = UIEngine.render_analytics(swarm_size)
        st.plotly_chart(fig_cov, config={"displayModeBar": False}, use_container_width=True)
        
        # NEW: Network topology + efficiency breakdown
        net_col, eff_col = st.columns(2)
        with net_col:
            fig_net = UIEngine.render_network_topology(swarm_size, comm_range)
            st.plotly_chart(fig_net, config={"displayModeBar": False}, use_container_width=True)
        with eff_col:
            fig_eff = UIEngine.render_efficiency_breakdown()
            st.plotly_chart(fig_eff, config={"displayModeBar": False}, use_container_width=True)
    
    # NEW: Agent strategy comparison
    st.markdown("---")
    st.markdown(f"<h3 style='color:{CONFIG.COLOR_PRIMARY};'>AGENT PERFORMANCE ANALYSIS</h3>", unsafe_allow_html=True)
    fig_strategy = UIEngine.render_agent_strategy(swarm_size)
    st.plotly_chart(fig_strategy, config={"displayModeBar": False}, use_container_width=True)

    # NEW: STRUCTURED EVENT-DRIVEN LOGGING PIPELINE
    st.markdown("---")
    
    # SESSION METADATA & CONFIG EXPORT
    with st.expander("⚙️ SESSION CONFIGURATION & EXPORT", expanded=False):
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        col_meta1.metric("SESSION PARAMS", f"Swarm: {swarm_size} | Range: {comm_range}")
        col_meta2.metric("ENVIRONMENT", f"Density: {wall_density}% | Timeout: {isolation_limit}t")
        col_meta3.metric("RUNTIME", f"{st.session_state.clock} ticks | {st.session_state.total_steps_taken} total steps")
        
        # Config export button
        st.markdown("**Export Session Data**")
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "swarm_size": swarm_size,
            "comm_range": comm_range,
            "wall_density": wall_density,
            "isolation_limit": isolation_limit,
            "final_coverage": st.session_state.coverage_history[-1],
            "total_ticks": st.session_state.clock,
            "total_steps": st.session_state.total_steps_taken,
            "unique_tiles_discovered": int(unique_tiles),
            "path_efficiency": path_efficiency,
            "network_health": network_health,
        }
        
        export_json = json.dumps(session_data, indent=2)
        st.download_button(
            label="📥 Download Session JSON",
            data=export_json,
            file_name=f"swarm_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with st.expander("📡 SYSTEM EVENT TELEMETRY LOG", expanded=False):
        logger = st.session_state.event_logger
        
        # Display log statistics
        stats = logger.get_statistics()
        if stats:
            log_col1, log_col2, log_col3 = st.columns(3)
            log_col1.metric("TOTAL EVENTS", stats.get("total_events", 0))
            
            event_types_str = ", ".join([f"{k}: {v}" for k, v in list(stats.get("event_types", {}).items())[:3]])
            log_col2.metric("TOP EVENTS", event_types_str[:25])
            
            critical_count = stats.get("severity_distribution", {}).get("CRITICAL", 0)
            log_col3.metric("CRITICAL ALERTS", critical_count)
        
        # Auto-scrolling terminal box
        st.markdown("**Event Stream (Latest First):**")
        st.markdown(UIEngine.render_event_log(logger, max_lines=25), unsafe_allow_html=True)

    # RETROSPECTIVE PERFORMANCE SUMMARY
    st.markdown("---")
    with st.expander("📊 RETROSPECTIVE PERFORMANCE MISSION SUMMARY DECK", expanded=True):
        unique_tiles = float(np.sum((st.session_state.explored == 1) & (st.session_state.grid == 0)))
        total_steps = st.session_state.total_steps_taken
        
        efficiency_ratio = (unique_tiles / total_steps * 100.0) if total_steps > 0 else 0.0
        
        # Metrics with convergence status
        c1, c2, c3 = st.columns(3)
        c1.metric(label="TOTAL ACCUMULATED STEPS", value=f"{total_steps} actions")
        c2.metric(label="UNIQUE DISCOVERED TILES", value=f"{int(unique_tiles)} nodes")
        
        if st.session_state.has_converged:
            c3.metric(label="MISSION STATUS", value="✓ CONVERGED", 
                      help=f"Exploration converged at tick {st.session_state.convergence_tick}")
        else:
            c3.metric(label="EXPLORATION EFFICIENCY", value=f"{efficiency_ratio:.2f}%", 
                      help="Unique Tiles / Total Steps. Higher = lower algorithmic drift.")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visuals
        deck_col1, deck_col2 = st.columns(2)
        fig_downtime, fig_traffic = UIEngine.render_summary_deck()
        
        with deck_col1:
            st.plotly_chart(fig_downtime, config={"displayModeBar": False}, use_container_width=True)
        with deck_col2:
            st.plotly_chart(fig_traffic, config={"displayModeBar": False}, use_container_width=True)
        
        # HEATMAP ANALYSIS: Identify bottlenecks
        st.markdown("**Spatial Hotspot Analysis**")
        heatmap = st.session_state.trajectory_heatmap
        max_traffic = np.max(heatmap) if np.max(heatmap) > 0 else 1
        top_coords = np.argsort(heatmap.flatten())[-5:][::-1]
        
        hotspot_details = []
        for idx in top_coords:
            x, y = np.unravel_index(idx, heatmap.shape)
            traffic = heatmap[x, y]
            pct = (traffic / max_traffic) * 100
            hotspot_details.append(f"({x}, {y}): {int(traffic)} visits ({pct:.0f}% of peak)")
        
        st.markdown("**Top 5 Traffic Hotspots:**")
        for detail in hotspot_details:
            st.write(f"• {detail}")

    if auto_run:
        time.sleep(0.04)
        st.rerun()

if __name__ == "__main__":
    main()
    