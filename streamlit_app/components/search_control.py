"""Streamlit controls for launching and managing DARTS search jobs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict

import streamlit as st
import torch
import yaml

from streamlit_app.state import get_process_status, set_state


PRESETS: Dict[str, Dict[str, object]] = {
    "CPU smoke mode": {
        "epochs": 1,
        "batch_size": 4,
        "val_batch_size": 4,
        "init_channels": 4,
        "layers": 2,
        "num_workers": 0,
        "amp": False,
        "multi_gpu": False,
        "save_dir": "search_checkpoints/smoke_cpu",
        "log_interval": 1,
        "use_synthetic_data": True,
        "synthetic_train_size": 8,
        "synthetic_val_size": 8,
        "synthetic_test_size": 8,
    },
    "GPU realistic mode": {
        "epochs": 5,
        "batch_size": 64,
        "val_batch_size": 128,
        "init_channels": 16,
        "layers": 8,
        "num_workers": 4,
        "amp": True,
        "multi_gpu": True,
        "save_dir": "search_checkpoints/demo_gpu",
        "log_interval": 25,
        "use_synthetic_data": False,
        "synthetic_train_size": 64,
        "synthetic_val_size": 64,
        "synthetic_test_size": 64,
    },
}


def _write_runtime_config(config: Dict[str, object], path: str) -> str:
    """Write a search configuration from UI controls."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return str(output)


def _safe_tail(path: Path, limit: int = 12000) -> str:
    """Read the tail of a log file safely."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _terminate_process(process: subprocess.Popen) -> None:
    """Terminate a child process in a cross-platform friendly way."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.kill(process.pid, signal.SIGTERM)


def render_search_controls() -> None:
    """Render the search-launching form and subprocess output."""
    st.subheader("Launch DARTS Search")
    st.warning(
        "NAS can be expensive. CPU smoke mode uses a tiny synthetic dataset so users can verify the full pipeline "
        "offline. Use GPU realistic mode with CIFAR-10 for meaningful experiments."
    )

    preset_name = st.selectbox("Preset", list(PRESETS), index=0 if not torch.cuda.is_available() else 1)
    preset = PRESETS[preset_name]

    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            data_root = st.text_input("Dataset root", value="./data")
            epochs = st.number_input("Search epochs", min_value=1, max_value=200, value=int(preset["epochs"]), step=1)
            batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=int(preset["batch_size"]), step=1)
            val_batch_size = st.number_input("Validation batch size", min_value=1, max_value=1024, value=int(preset["val_batch_size"]), step=1)
        with col2:
            init_channels = st.number_input("Initial channels", min_value=2, max_value=64, value=int(preset["init_channels"]), step=2)
            layers = st.number_input("Layers", min_value=2, max_value=30, value=int(preset["layers"]), step=1)
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
            num_workers = st.number_input("Data workers", min_value=0, max_value=32, value=int(preset["num_workers"]), step=1)
        with col3:
            learning_rate = st.number_input("Network LR", min_value=0.0001, max_value=1.0, value=0.025, format="%.5f")
            arch_learning_rate = st.number_input("Architecture LR", min_value=0.00001, max_value=0.1, value=0.0003, format="%.5f")
            output_dir = st.text_input("Output directory", value=str(preset["save_dir"]))
            log_interval = st.number_input("Log interval", min_value=1, max_value=500, value=int(preset["log_interval"]), step=1)
            use_synthetic_data = st.checkbox(
                "Use offline synthetic smoke data",
                value=bool(preset.get("use_synthetic_data", False)),
                help="Use only for CI/local plumbing checks. It is not a benchmark and does not train on CIFAR-10.",
            )
        unrolled = st.checkbox("Enable second-order unrolled approximation", value=False)
        amp = st.checkbox("Use mixed precision AMP", value=bool(preset["amp"]), disabled=not torch.cuda.is_available())
        cutout = st.checkbox("Use cutout augmentation", value=False)
        start = st.form_submit_button("Start Search", type="primary")

    process = st.session_state.get("search_process")
    if start:
        if process is not None and process.poll() is None:
            st.error("A search process is already running. Stop it before starting another run.")
        else:
            config = {
                "data_root": data_root,
                "batch_size": int(batch_size),
                "val_batch_size": int(val_batch_size),
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "arch_learning_rate": float(arch_learning_rate),
                "arch_weight_decay": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0003,
                "unrolled": bool(unrolled),
                "grad_clip": 5.0,
                "num_workers": int(num_workers),
                "save_dir": output_dir,
                "seed": int(seed),
                "init_channels": int(init_channels),
                "layers": int(layers),
                "num_classes": 10,
                "amp": bool(amp),
                "multi_gpu": bool(preset.get("multi_gpu", False)),
                "distributed": False,
                "cutout": bool(cutout),
                "cutout_length": 16,
                "validation_fraction": 0.5,
                "log_interval": int(log_interval),
                "use_synthetic_data": bool(use_synthetic_data),
                "synthetic_train_size": int(preset.get("synthetic_train_size", 64)),
                "synthetic_val_size": int(preset.get("synthetic_val_size", 64)),
                "synthetic_test_size": int(preset.get("synthetic_test_size", 64)),
                "download_data": True,
            }
            config_path = _write_runtime_config(config, "streamlit_runtime/search_config.yaml")
            command = [sys.executable, "scripts/run_search.py", "--config", config_path]
            log_file = Path(output_dir) / "streamlit_subprocess.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handle = log_file.open("w", encoding="utf-8")
            proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2])
            set_state("search_process", proc)
            set_state("search_output_dir", output_dir)
            set_state("last_config_path", config_path)
            set_state("last_log_path", str(log_file))
            st.success(f"Search started with PID {proc.pid}. Logs: {log_file}")

    process = st.session_state.get("search_process")
    st.write("**Process status:**", get_process_status(process))
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh logs"):
            st.rerun()
    with col_b:
        if st.button("Stop process", disabled=process is None or process.poll() is not None):
            _terminate_process(process)
            st.warning("Stop signal sent to search process.")
            st.rerun()

    output_dir = Path(st.session_state.get("search_output_dir", "search_checkpoints"))
    log_path = output_dir / "streamlit_subprocess.log"
    fallback_log = output_dir / "search.log"
    shown_log = log_path if log_path.exists() else fallback_log
    if shown_log.exists():
        st.text_area("Live process output", _safe_tail(shown_log), height=340)
        st.download_button("Download log", data=shown_log.read_bytes(), file_name=shown_log.name)
    else:
        st.info("No logs found yet for this run directory.")
