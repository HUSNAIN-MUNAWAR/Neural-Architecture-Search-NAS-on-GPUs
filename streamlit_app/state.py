"""Session state helpers for the Streamlit application."""

from __future__ import annotations

import subprocess
from typing import Any, Optional

import streamlit as st


DEFAULTS = {
    "search_process": None,
    "search_output_dir": "search_checkpoints",
    "last_config_path": "streamlit_runtime/search_config.yaml",
    "last_log_path": "search_checkpoints/streamlit_subprocess.log",
    "selected_preset": "CPU smoke mode",
}


def init_state() -> None:
    """Initialize known Streamlit session keys."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_state(key: str, value: Any) -> None:
    """Set one session-state value."""
    st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """Get one session-state value."""
    return st.session_state.get(key, default)


def get_process_status(process: Optional[subprocess.Popen]) -> str:
    """Return a human-readable subprocess status."""
    if process is None:
        return "No active search process"
    exit_code = process.poll()
    if exit_code is None:
        return f"Running (PID {process.pid})"
    return f"Exited with code {exit_code}"
