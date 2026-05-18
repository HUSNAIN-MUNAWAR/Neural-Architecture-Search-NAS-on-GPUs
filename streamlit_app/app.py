"""Professional Streamlit dashboard for DARTS NAS search and deployment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch

from streamlit_app.components.architecture_view import render_architecture_view
from streamlit_app.components.monitoring import render_monitoring
from streamlit_app.components.search_control import render_search_controls
from streamlit_app.state import get_process_status, init_state
from utils.gpu_utils import gpu_memory_stats


def _metric_card(label: str, value: str, help_text: str = "") -> None:
    """Render a compact dashboard metric."""
    st.metric(label=label, value=value, help=help_text or None)


def render_overview() -> None:
    """Render the dashboard overview page."""
    st.subheader("GPU-Aware DARTS NAS Pipeline")
    st.info(
        "Real neural architecture search is compute-heavy. Use CPU smoke mode for quick verification and GPU realistic "
        "mode for meaningful search experiments."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Data", "CIFAR-10", "Default image classification benchmark")
    with c2:
        _metric_card("Search", "DARTS", "Differentiable architecture search")
    with c3:
        _metric_card("Runtime", "AMP/GPU-aware", "Uses CUDA features when available")
    with c4:
        _metric_card("Exports", "ONNX / TorchScript", "Deployment-friendly artifacts")

    st.markdown(
        """
        ### Control Flow

        1. **Load data** with CIFAR-10 transforms and validation split.
        2. **Search architecture** using MixedOps and alpha parameters.
        3. **Export genotype** as JSON.
        4. **Train derived model** from scratch with evaluation config.
        5. **Export model** to ONNX or TorchScript.
        6. **Deploy dashboard** locally, with Docker, or as an always-on VPS service.
        """
    )
    diagram = PROJECT_ROOT / "docs" / "architecture.png"
    if diagram.exists():
        st.image(str(diagram), caption="System architecture", use_container_width=True)
    else:
        st.warning("Architecture diagram not found. Run `python scripts/generate_diagram.py`.")


def render_export_deployment() -> None:
    """Render artifact download and deployment guidance."""
    st.subheader("Export / Deployment")
    output_dir = Path(st.session_state.get("search_output_dir", "search_checkpoints"))
    st.caption(f"Current run directory: `{output_dir}`")

    downloadable = [
        output_dir / "genotype.json",
        output_dir / "metrics.jsonl",
        output_dir / "search.log",
        output_dir / "streamlit_subprocess.log",
    ]
    exported = list(output_dir.rglob("*.onnx")) + list(output_dir.rglob("*.ts")) + list(output_dir.rglob("*.torchscript"))
    downloadable.extend(exported)

    available = [path for path in downloadable if path.exists() and path.is_file()]
    if not available:
        st.info("No downloadable artifacts found yet. Launch a search or run evaluation/export first.")
    for path in available:
        st.download_button(
            label=f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/octet-stream",
        )

    st.markdown(
        """
        ### Always-on deployment

        Streamlit Community Cloud is fine for demos but may sleep after inactivity. For a serious always-on portfolio demo,
        use Docker Compose or systemd on a VPS.

        ```bash
        docker compose up -d --build
        ```

        GPU Docker deployment:

        ```bash
        docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
        ```

        See `docs/DEPLOYMENT.md` and `deployment/vps/README.md` for Nginx and systemd examples.
        """
    )


def render_docs() -> None:
    """Render useful documentation links and snippets."""
    st.subheader("Documentation")
    docs = [
        ("Deployment", "docs/DEPLOYMENT.md"),
        ("Reproducibility", "docs/REPRODUCIBILITY.md"),
        ("Benchmarks", "docs/BENCHMARKS.md"),
        ("Project Report", "docs/PROJECT_REPORT.md"),
        ("Architecture", "docs/ARCHITECTURE.md"),
        ("Demo Guide", "docs/DEMO_GUIDE.md"),
        ("Portfolio Pitch", "docs/PORTFOLIO_PITCH.md"),
        ("Upwork Proposal Snippets", "docs/UPWORK_PROPOSAL_SNIPPETS.md"),
    ]
    for title, rel_path in docs:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            with st.expander(title):
                st.markdown(path.read_text(encoding="utf-8")[:8000])
        else:
            st.warning(f"Missing documentation file: {rel_path}")


def main() -> None:
    """Run the Streamlit NAS dashboard."""
    st.set_page_config(page_title="NAS GPU Platform", layout="wide", page_icon="🧠")
    init_state()

    with st.sidebar:
        st.title("NAS GPU Platform")
        st.caption("DARTS • PyTorch • Streamlit • Docker")
        process = st.session_state.get("search_process")
        st.write("**Runtime status**")
        st.info(get_process_status(process))
        output_dir = st.text_input(
            "Current run directory",
            value=st.session_state.get("search_output_dir", "search_checkpoints"),
        )
        st.session_state["search_output_dir"] = output_dir
        st.write("**Device info**")
        if torch.cuda.is_available():
            st.success(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            st.json(gpu_memory_stats())
        else:
            st.warning("CUDA not detected. CPU smoke mode is recommended.")

    st.title("Neural Architecture Search on GPUs")
    st.caption("Production-oriented AutoML portfolio project using DARTS, CUDA-aware PyTorch, TensorBoard, and Streamlit.")

    tabs = st.tabs([
        "Overview",
        "Launch Search",
        "Monitor Runs",
        "Architecture Viewer",
        "Export / Deployment",
        "Documentation",
    ])
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_search_controls()
    with tabs[2]:
        render_monitoring(st.session_state.get("search_output_dir", "search_checkpoints"))
    with tabs[3]:
        render_architecture_view(st.session_state.get("search_output_dir", "search_checkpoints"))
    with tabs[4]:
        render_export_deployment()
    with tabs[5]:
        render_docs()


if __name__ == "__main__":
    main()
