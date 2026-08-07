import streamlit as st
import requests
import time
import json
import io
from PIL import Image
from datetime import datetime
import hashlib
import graphviz
import tempfile
import os
# pip install watchdog

# ======== Configuration ========
st.set_page_config(
    page_title="Agent State Visualizer",
    page_icon="🤖",
    layout="wide",
)

# ======== State Management ========
if 'api_data' not in st.session_state:
    st.session_state.api_data = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'polling_active' not in st.session_state:
    st.session_state.polling_active = False
if 'poll_interval' not in st.session_state:
    st.session_state.poll_interval = 1.0
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = {"status": "disconnected", "message": "Not connected"}
if 'fsm_hash' not in st.session_state:
    st.session_state.fsm_hash = None
if 'fsm_image' not in st.session_state:
    st.session_state.fsm_image = None

# ======== API Functions ========
def fetch_agent_data(api_host, api_port, timeout=5):
    """Fetch main agent data from /agent endpoint"""
    try:
        agent_url = f"http://{api_host}:{api_port}/agent"
        response = requests.get(agent_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_fsm_data(api_host, api_port, timeout=5):
    """Fetch FSM data from the /agent/fsm endpoint"""
    try:
        fsm_url = f"http://{api_host}:{api_port}/agent/fsm"
        response = requests.get(fsm_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def calculate_fsm_hash(fsm_data):
    """Calculate a hash of the FSM data to detect changes (excluding rendering metadata)"""
    if not fsm_data:
        return None

    # Only hash the structural data, not rendering details
    hash_data = {
        'current_state': fsm_data.get("fsm_info", {}).get("current_state"),
        'states': fsm_data.get("graph", {}).get("states", []),
        'transitions': fsm_data.get("graph", {}).get("transitions", [])
    }

    serialized = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

# ======== UI Layout ========
st.title("Agent State Visualizer")
st.markdown("Visualize and monitor the state of the Agent FSM in real-time.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API endpoint configuration
    st.subheader("API Connection")
    col1, col2 = st.columns(2)
    with col1:
        api_host = st.text_input("Host", value="localhost")
    with col2:
        api_port = st.text_input("Port", value="40000")

    st.text(f"Agent API: http://{api_host}:{api_port}/agent")
    st.text(f"FSM API: http://{api_host}:{api_port}/agent/fsm")

    # Polling configuration
    st.subheader("Polling Settings")
    new_poll_interval = st.slider("Poll Interval (seconds)",
                             min_value=0.1,
                             max_value=5.0,
                             value=st.session_state.poll_interval,
                             step=0.1,
                             key="poll_interval_slider")
    st.session_state.poll_interval = new_poll_interval

    # Connection status display
    st.subheader("Connection Status")
    status_container = st.empty()

    status = st.session_state.connection_status.get("status", "unknown")
    message = st.session_state.connection_status.get("message", "")

    if status == "connected":
        status_container.success(message)
    elif status == "connecting":
        status_container.info(message)
    elif status == "disconnected":
        status_container.warning(message)
    elif status == "error":
        status_container.error(message)

    if st.session_state.last_update_time:
        st.text(f"Last update: {st.session_state.last_update_time.strftime('%H:%M:%S')}")

    # Connection control buttons
    if st.session_state.polling_active:
        if st.button("Disconnect", key="stop_polling_button"):
            st.session_state.polling_active = False
            st.session_state.api_data = None
            st.session_state.last_update_time = None
            st.session_state.connection_status = {
                "status": "disconnected",
                "message": "Disconnected"
            }
            st.rerun()
    else:
        # Test connection button
        if st.button("Test API", key="test_api_button"):
            data = fetch_agent_data(api_host, api_port)
            if "error" not in data:
                st.session_state.connection_status = {
                    "status": "connected",
                    "message": "API available"
                }
                st.session_state.api_data = data
                st.session_state.last_update_time = datetime.now()

                # Test FSM endpoint separately
                fsm_data = fetch_fsm_data(api_host, api_port)
                if "error" not in fsm_data:
                    new_hash = calculate_fsm_hash(fsm_data)
                    if new_hash != st.session_state.fsm_hash:
                        st.session_state.fsm_hash = new_hash
                        st.session_state.fsm_image = visualize_fsm(fsm_data, data)
            else:
                st.session_state.connection_status = {
                    "status": "error",
                    "message": f"API unavailable: {data.get('error')}"
                }
            st.rerun()

        # Connect button
        if st.button("Connect", key="connect_button"):
            st.session_state.polling_active = True
            st.session_state.connection_status = {
                "status": "connecting",
                "message": "Starting connection..."
            }
            st.rerun()

# Main content area
st.header("Finite State Machine")
graph_container = st.empty()

st.header("Agent Information")
agent_info_container = st.container()

# ======== Visualization Functions ========
def visualize_fsm(fsm_data, agent_data=None):
    """Create a visualization of the FSM graph using Graphviz"""
    if not fsm_data:
        return None

    try:
        # Get FSM structure
        fsm_info = fsm_data.get("fsm_info", {})
        current_state = fsm_info.get("current_state")
        graph_data = fsm_data.get("graph", {})
        states = graph_data.get("states", [])
        transitions = graph_data.get("transitions", [])

        # Get previous state from agent data
        previous_state = None

        if agent_data:
            # Get the last action from recent activity to find previous state
            recent_actions = agent_data.get("recent_activity", {}).get("last_5_actions", [])
            if recent_actions:
                last_action = recent_actions[-1]
                # Parse the transition string "prev_state → curr_state"
                transition_str = last_action.get("transition", "")
                if " → " in transition_str:
                    previous_state = transition_str.split(" → ")[0]

        # Create a directed graph with optimized DPI
        dot = graphviz.Digraph('FSM', format='png')
        dot.attr(rankdir='LR', size='14,10', dpi='96')  # Sweet spot for web display
        dot.attr('node', shape='circle', style='filled', fontsize='12')

        # Add states as nodes
        for state in states:
            if state == current_state:
                dot.node(state, style='filled', color='lightgreen', penwidth='3')
            elif state == previous_state:
                dot.node(state, style='filled', color='darkgreen', fontcolor='white', penwidth='2')
            else:
                dot.node(state, style='filled', color='lightblue2')

        # Add transitions as edges
        for transition in transitions:
            from_state = transition.get("from")
            to_state = transition.get("to")
            action = transition.get("action")
            transition_type = transition.get("type", "success")

            # Style edges based on type
            if transition_type == "fallback":
                dot.edge(from_state, to_state, label=action,
                        style="dashed", color="red", fontcolor="red", fontsize='10')
            else:
                dot.edge(from_state, to_state, label=action,
                        color="blue", fontcolor="black", fontsize='10')

        # Add legend with reverted size
        dot.attr(label=r'\n\n\nLEGEND:\l\l• Light Green Node: Current state\l• Dark Green Node: Previous state\l• Blue Edge: Available transition\l• Red Dashed Edge: Failure fallback\l\l',
                 fontsize='12', labelloc='b', labeljust='l')

        # Render to PNG in memory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "fsm")
            rendered_path = dot.render(output_path, cleanup=True)

            # Load the image
            with open(rendered_path, 'rb') as f:
                image_data = f.read()

            return Image.open(io.BytesIO(image_data))

    except Exception as e:
        st.error(f"Error visualizing FSM: {str(e)}")
        return None

def display_agent_info(data):
    """Display agent information using new API structure"""
    if not data:
        agent_info_container.info("No agent information available.")
        return

    try:
        config = data.get("config", {})
        status = data.get("status", {})
        performance = data.get("performance", {})
        recent_activity = data.get("recent_activity", {})
        resources = data.get("resources", {})

        tabs = agent_info_container.tabs(["Status & Performance", "Recent Activity", "Resources", "Raw Data"])

        # Tab 1: Status & Performance
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("### Agent Status")
                st.markdown(f"**Name:** {config.get('name', 'Unknown')}")
                st.markdown(f"**Current State:** {status.get('current_state', 'Unknown')}")
                st.markdown(f"**Current Action:** {status.get('current_action', 'None')}")
                st.markdown(f"**Iteration:** {status.get('iteration', 0)}")

                # Available actions
                actions = status.get('available_actions', [])
                if actions:
                    st.markdown("**Available Actions:**")
                    action_html = '<div style="display: flex; flex-wrap: wrap; gap: 5px;">'
                    for action in actions:
                        color = "#e0f7ea" if "failed" not in action.lower() else "#ffebee"
                        text_color = "#1b5e20" if "failed" not in action.lower() else "#c62828"
                        action_html += f'<div style="background-color: {color}; color: {text_color}; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin-bottom: 5px;">{action}</div>'
                    action_html += '</div>'
                    st.markdown(action_html, unsafe_allow_html=True)

            with col2:
                st.markdown("### Performance")
                st.markdown(f"**Actions Executed:** {performance.get('actions_executed', 0)}")
                st.markdown(f"**Success Rate:** {performance.get('success_rate', 0)}%")
                st.markdown(f"**Errors:** {performance.get('errors_count', 0)}")

                # Repository progress
                repo_progress = performance.get('repository_progress', {})
                st.markdown("### Repository Progress")
                completed = repo_progress.get('completed', 0)
                total = repo_progress.get('total', 0)
                percentage = repo_progress.get('percentage', 0)
                st.markdown(f"**Completed:** {completed}/{total}")
                st.markdown(f"**Progress:** {percentage}%")

                # Progress bar
                if total > 0:
                    st.progress(completed / total)

        # Tab 2: Recent Activity
        with tabs[1]:
            recent_actions = recent_activity.get('last_5_actions', [])
            recent_errors = recent_activity.get('recent_errors', [])

            if recent_actions:
                st.markdown("### Recent Actions")
                for action in recent_actions:
                    result_color = "🟢" if action.get('result') == 'success' else "🔴"
                    st.markdown(f"{result_color} **Iteration {action.get('iteration')}:** {action.get('action')} - {action.get('transition')}")

            if recent_errors:
                st.markdown("### Recent Errors")
                for error in recent_errors:
                    st.error(f"Iteration {error.get('iteration')} in {error.get('state')}: {error.get('exception')}")

        # Tab 3: Resources
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### OpenAI Usage")
                cost = resources.get('openai_cost', 0)
                st.markdown(f"**Cost:** ${cost:.4f}")
                if resources.get('cost_limit'):
                    remaining = resources.get('budget_remaining', 0)
                    st.markdown(f"**Budget Remaining:** ${remaining:.4f}")
                st.markdown(f"**API Calls:** {resources.get('api_calls', 0)}")
                st.markdown(f"**Active Files:** {resources.get('active_files', 0)}")

            with col2:
                st.markdown("### Configuration")
                st.markdown(f"**FSM Type:** {config.get('fsm_type', 'Unknown')}")
                st.markdown(f"**Repository:** {config.get('repository', 'Unknown')}")
                st.markdown(f"**Strategy:** {config.get('strategy', 'Unknown')}")

        # Tab 4: Raw data
        with tabs[3]:
            st.json(data)

    except Exception as e:
        agent_info_container.error(f"Error displaying agent info: {str(e)}")

# ======== Main Polling Logic ========
if st.session_state.polling_active:
    # Fetch main agent data
    data = fetch_agent_data(api_host, api_port)

    if "error" not in data:
        st.session_state.api_data = data
        st.session_state.last_update_time = datetime.now()
        st.session_state.connection_status = {
            "status": "connected",
            "message": f"Connected at {st.session_state.last_update_time.strftime('%H:%M:%S')}"
        }

        # Fetch FSM data separately
        fsm_data = fetch_fsm_data(api_host, api_port)

        if "error" not in fsm_data:
            # Check if FSM data has changed
            new_hash = calculate_fsm_hash(fsm_data)

            if new_hash != st.session_state.fsm_hash:
                st.session_state.fsm_hash = new_hash
                st.session_state.fsm_image = visualize_fsm(fsm_data, data)

            if st.session_state.fsm_image:
                graph_container.image(st.session_state.fsm_image, use_container_width=True)
            else:
                graph_container.warning("Cannot visualize FSM. Incomplete or invalid data.")
        else:
            graph_container.error(f"FSM Error: {fsm_data.get('error')}")

        # Display agent information
        display_agent_info(data)
    else:
        st.session_state.connection_status = {
            "status": "error",
            "message": f"Connection error: {data.get('error')}"
        }
        graph_container.error("Error connecting to API")
        agent_info_container.error("Unable to retrieve agent information")

    time.sleep(st.session_state.poll_interval)
    st.rerun()

# Static display when not polling
if not st.session_state.polling_active:
    if st.session_state.api_data:
        # Display cached FSM image or fetch new FSM data
        if st.session_state.fsm_image:
            graph_container.image(st.session_state.fsm_image, use_container_width=True)
        else:
            # Fetch FSM data since we don't have the image cached
            fsm_data = fetch_fsm_data(api_host, api_port)
            if "error" not in fsm_data:
                st.session_state.fsm_image = visualize_fsm(fsm_data, st.session_state.api_data)
                if st.session_state.fsm_image:
                    graph_container.image(st.session_state.fsm_image, use_container_width=True)
                else:
                    graph_container.warning("Cannot visualize FSM. Incomplete or invalid data.")
            else:
                graph_container.warning("FSM data not available. Click 'Test API' to refresh.")

        # Display agent information
        display_agent_info(st.session_state.api_data)
    else:
        graph_container.info("Waiting for data from API...")
        agent_info_container.info("No agent information available")

# Footer
st.markdown("---")
st.markdown("Agent State Visualizer | Made with Streamlit")