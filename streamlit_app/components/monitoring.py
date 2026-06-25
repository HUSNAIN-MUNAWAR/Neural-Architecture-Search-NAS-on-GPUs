"""Streamlit monitoring dashboard for search metrics and TensorBoard events."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load metrics from JSON lines."""
    records: list[dict[str, float]] = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(records)


def _load_tensorboard_scalars(log_dir: Path) -> pd.DataFrame:
    """Load scalar events from TensorBoard files when tensorboard is installed."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return pd.DataFrame()
    if not log_dir.exists():
        return pd.DataFrame()
    event_files = [p for p in log_dir.rglob("events.out.tfevents.*") if p.is_file()]
    if not event_files:
        return pd.DataFrame()
    rows = []
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file.parent))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    rows.append({"step": event.step, "tag": tag, "value": event.value, "source": event_file.name})
        except Exception:
            continue
    return pd.DataFrame(rows)


def _plot_metric(dataframe: pd.DataFrame, metric: str) -> None:
    """Plot one metric if available."""
    if metric in dataframe.columns and "epoch" in dataframe.columns:
        fig = px.line(dataframe, x="epoch", y=metric, markers=True, title=metric.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)


def render_monitoring(output_dir: str) -> None:
    """Render metrics curves and raw metrics table."""
    st.subheader("Monitor Runs")
    run_dir = Path(output_dir)
    st.caption(f"Reading from `{run_dir}`")
    if st.button("Refresh metrics"):
        st.rerun()

    metrics_path = run_dir / "metrics.jsonl"
    dataframe = _load_jsonl(metrics_path)
    if dataframe.empty:
        st.info("No JSONL metrics found yet. Start a search or check the output directory.")
    else:
        st.success(f"Loaded {len(dataframe)} metric rows from {metrics_path}.")
        st.dataframe(dataframe.tail(30), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            _plot_metric(dataframe, "train_loss")
            _plot_metric(dataframe, "arch_loss")
        with c2:
            _plot_metric(dataframe, "train_acc")
            _plot_metric(dataframe, "learning_rate")
        st.download_button("Download metrics.jsonl", data=metrics_path.read_bytes(), file_name="metrics.jsonl")

    tb_df = _load_tensorboard_scalars(run_dir / "tensorboard")
    if tb_df.empty:
        st.caption("No TensorBoard scalar events found yet.")
    else:
        st.caption("TensorBoard scalar events")
        st.dataframe(tb_df.tail(100), use_container_width=True)
        selected_tags = st.multiselect(
            "TensorBoard tags", sorted(tb_df["tag"].unique()), default=list(sorted(tb_df["tag"].unique()))[:3]
        )
        if selected_tags:
            subset = tb_df[tb_df["tag"].isin(selected_tags)]
            fig = px.line(subset, x="step", y="value", color="tag", title="TensorBoard Scalars")
            st.plotly_chart(fig, use_container_width=True)

    log_candidates = [run_dir / "streamlit_subprocess.log", run_dir / "search.log"]
    for log_path in log_candidates:
        if log_path.exists():
            with st.expander(f"Log: {log_path.name}", expanded=log_path.name == "streamlit_subprocess.log"):
                text = log_path.read_text(encoding="utf-8", errors="replace")
                st.text_area(log_path.name, text[-12000:], height=260)
                st.download_button(f"Download {log_path.name}", data=log_path.read_bytes(), file_name=log_path.name)
