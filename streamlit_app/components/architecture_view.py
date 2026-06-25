"""Streamlit architecture visualization tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from search.utils import genotype_to_string, plot_cell
from utils.file_utils import load_checkpoint, load_json


def _validate_genotype(genotype: dict[str, list]) -> None:
    """Validate the minimal DARTS genotype structure expected by the visualizer."""
    required = {"normal", "normal_concat", "reduce", "reduce_concat"}
    missing = required - set(genotype)
    if missing:
        raise ValueError(f"Missing genotype keys: {sorted(missing)}")
    for key in ["normal", "reduce"]:
        if not isinstance(genotype[key], list) or len(genotype[key]) == 0:
            raise ValueError(f"Genotype key `{key}` must be a non-empty list.")


def _load_genotype_from_upload() -> dict[str, list] | None:
    """Load genotype JSON from a Streamlit file uploader."""
    uploaded = st.file_uploader("Upload genotype JSON", type=["json"])
    if uploaded is None:
        return None
    content = uploaded.read().decode("utf-8")
    genotype = json.loads(content)
    if not isinstance(genotype, dict):
        raise ValueError("Uploaded genotype must be a JSON object.")
    _validate_genotype(genotype)
    return genotype


def _load_genotype_from_path(path: str) -> dict[str, list] | None:
    """Load genotype from a JSON or checkpoint path."""
    if not path:
        return None
    input_path = Path(path)
    if not input_path.exists():
        st.error(f"Path does not exist: {input_path}")
        return None
    try:
        if input_path.suffix.lower() == ".json":
            genotype = load_json(str(input_path))
        else:
            checkpoint = load_checkpoint(str(input_path), map_location="cpu")
            genotype = checkpoint.get("genotype")
        if not isinstance(genotype, dict):
            st.error("File does not contain a genotype dictionary.")
            return None
        _validate_genotype(genotype)
        return genotype
    except Exception as exc:
        st.error(f"Could not load genotype: {exc}")
        return None


def render_architecture_view(default_output_dir: str) -> None:
    """Render uploaded or checkpoint genotype diagrams."""
    st.subheader("Architecture Viewer")
    path = st.text_input("Load genotype/checkpoint path", value=str(Path(default_output_dir) / "genotype.json"))
    genotype = None
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load from path"):
            genotype = _load_genotype_from_path(path)
    with col2:
        st.caption("Supported: genotype `.json`, `best_search.pt`, or `last_search.pt` checkpoint.")
    try:
        uploaded_genotype = _load_genotype_from_upload()
        if uploaded_genotype is not None:
            genotype = uploaded_genotype
    except Exception as exc:
        st.error(f"Invalid uploaded genotype: {exc}")
        return
    if genotype is None and Path(path).exists():
        genotype = _load_genotype_from_path(path)
    if genotype is None:
        st.info("Upload a genotype JSON or load one from a search checkpoint.")
        return

    st.code(genotype_to_string(genotype), language="json")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            normal_png = plot_cell(genotype, "normal", str(Path(tmpdir) / "normal.png"))
            reduce_png = plot_cell(genotype, "reduce", str(Path(tmpdir) / "reduce.png"))
            c1, c2 = st.columns(2)
            with c1:
                st.image(normal_png, caption="Normal Cell", use_container_width=True)
            with c2:
                st.image(reduce_png, caption="Reduction Cell", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render architecture graph: {exc}")
    st.download_button(
        "Download genotype JSON",
        data=genotype_to_string(genotype),
        file_name="genotype.json",
        mime="application/json",
    )
