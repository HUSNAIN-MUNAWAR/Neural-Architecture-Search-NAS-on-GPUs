"""Generate architecture diagram PNG and project PDF report."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"


def _draw_diagram_matplotlib(output_path: Path) -> None:
    """Generate a professional architecture diagram using matplotlib fallback."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    boxes = [
        (0.8, 5.5, "CIFAR-10\nData Loader"),
        (3.2, 5.5, "Search Space\nMixedOps + Cells"),
        (5.8, 5.5, "DARTS Trainer\nBilevel Loop"),
        (8.4, 5.5, "GPU Cluster\nAMP + DP/DDP"),
        (10.9, 5.5, "TensorBoard\nMetrics"),
        (5.8, 2.8, "Derived\nGenotype"),
        (8.4, 2.8, "Evaluation\nTrain from Scratch"),
        (10.9, 2.8, "Export\nONNX + TorchScript"),
        (3.2, 2.8, "Streamlit App\nLaunch + Monitor"),
    ]
    for x, y, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            1.9,
            0.9,
            boxstyle="round,pad=0.08",
            linewidth=1.6,
            edgecolor="#263238",
            facecolor="#E3F2FD",
        )
        ax.add_patch(patch)
        ax.text(x + 0.95, y + 0.45, label, ha="center", va="center", fontsize=10, fontweight="bold")
    arrows = [
        ((2.7, 5.95), (3.2, 5.95)),
        ((5.1, 5.95), (5.8, 5.95)),
        ((7.7, 5.95), (8.4, 5.95)),
        ((10.3, 5.95), (10.9, 5.95)),
        ((6.75, 5.5), (6.75, 3.7)),
        ((7.7, 3.25), (8.4, 3.25)),
        ((10.3, 3.25), (10.9, 3.25)),
        ((5.1, 3.25), (5.8, 3.25)),
        ((4.1, 3.7), (6.7, 5.5)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=1.5, color="#37474F")
        ax.add_patch(arrow)
    ax.text(7, 7.3, "NAS GPU Project - DARTS System Architecture", ha="center", fontsize=16, fontweight="bold")
    ax.text(
        7,
        0.7,
        "Config-driven pipelines, CUDA acceleration, TensorBoard logging, Streamlit control plane",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_architecture_png(output_path: str | Path | None = None) -> Path:
    """Generate docs/architecture.png using diagrams when available, with a robust fallback."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else DOCS_DIR / "architecture.png"
    try:
        from diagrams import Cluster, Diagram
        from diagrams.aws.compute import GPU
        from diagrams.generic.compute import Rack
        from diagrams.generic.storage import Storage
        from diagrams.onprem.analytics import Spark
        from diagrams.onprem.client import Users
        from diagrams.onprem.monitoring import Grafana

        with Diagram(
            "NAS GPU Project - DARTS Architecture",
            filename=str(output.with_suffix("")),
            outformat="png",
            show=False,
            direction="LR",
        ):
            data = Storage("CIFAR-10 Data Loader")
            search = Rack("Search Space")
            trainer = Spark("DARTS Trainer")
            gpu = GPU("GPU Cluster AMP / DP / DDP")
            eval_node = Spark("Evaluation Pipeline")
            app = Users("Streamlit App")
            board = Grafana("TensorBoard")
            with Cluster("Artifacts"):
                genotype = Storage("Genotype JSON")
                exports = Storage("ONNX / TorchScript")
            data >> search >> trainer >> gpu >> genotype >> eval_node >> exports
            trainer >> board
            app >> trainer
            app >> board
            app >> genotype
    except Exception:
        _draw_diagram_matplotlib(output)
    return output


def _pdf_story() -> list[tuple[str, str]]:
    """Return report sections."""
    return [
        (
            "1. Introduction to NAS and DARTS",
            "Neural Architecture Search automates the design of high-performing neural networks. This project implements Differentiable Architecture Search, where candidate operations are relaxed into a continuous weighted mixture. The first-order optimization path updates network weights on training batches and architecture weights on validation batches, making the search practical for CIFAR-10 on modern GPUs.",
        ),
        (
            "2. System Architecture Description",
            "The architecture diagram in docs/architecture.png shows the flow from CIFAR-10 dataloading into the DARTS search space, the bilevel optimization trainer, GPU execution layer, TensorBoard monitoring, Streamlit control plane, evaluation pipeline, and model export artifacts.",
        ),
        (
            "3. Code Structure and Module Details",
            "The data package owns augmentation and dataloaders. The models package implements operations, cells, differentiable search networks, and discrete evaluation networks. The search package owns architecture optimization, genotype decoding, and the trainer. The evaluation package performs full training, metrics, latency measurement, and exports. Utilities centralize configuration, logging, GPU selection, and file IO.",
        ),
        (
            "4. Search Space and Operations",
            "Each cell is a directed acyclic graph with four intermediate nodes. Each edge chooses among sep_conv_3x3, sep_conv_5x5, dil_conv_3x3, dil_conv_5x5, max_pool_3x3, avg_pool_3x3, skip_connect, and zero. Learned alpha tensors are converted to a discrete genotype by selecting the top two strongest non-zero operations for each intermediate node.",
        ),
        (
            "5. GPU Acceleration Techniques Used",
            "The project enables CuDNN benchmarking, least-free-memory GPU selection, automatic mixed precision, pinned-memory dataloading, non-blocking host-to-device copies, gradient clipping, DataParallel, and an environment-driven DistributedDataParallel helper. These choices make the code suitable for one GPU workstations and multi-GPU servers.",
        ),
        (
            "6. Streamlit Application Features",
            "The Streamlit application provides tabs for launching a search, monitoring JSONL and TensorBoard metrics with Plotly curves, inspecting GPU memory, loading genotypes from JSON or checkpoints, rendering normal and reduction cell diagrams, and downloading architecture definitions.",
        ),
        (
            "7. Instructions to Run the Project",
            "Install dependencies with pip install -r requirements.txt. Run architecture search with python scripts/run_search.py --config configs/default_search.yaml. Train the derived architecture with python scripts/run_evaluation.py --config configs/default_eval.yaml --genotype search_checkpoints/genotype.json. Launch the UI with streamlit run streamlit_app/app.py. Docker users can run docker compose up --build.",
        ),
    ]


def generate_pdf_report(output_path: str | Path | None = None) -> Path:
    """Generate a formatted PDF project report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else DOCS_DIR / "project_report.pdf"
    diagram_path = DOCS_DIR / "architecture.png"
    if not diagram_path.exists():
        generate_architecture_png(diagram_path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    story.append(Paragraph("Neural Architecture Search on GPUs - Project Report", styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            "DARTS first-order architecture search for CIFAR-10 using PyTorch, CUDA, AMP, TensorBoard, and Streamlit.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.35 * inch))
    if diagram_path.exists():
        story.append(Image(str(diagram_path), width=6.6 * inch, height=3.8 * inch))
    story.append(PageBreak())
    for heading, body in _pdf_story():
        story.append(Paragraph(heading, styles["Heading1"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("Example Commands", styles["Heading1"]))
    code = "python scripts/run_search.py --config configs/default_search.yaml<br/>python scripts/run_evaluation.py --config configs/default_eval.yaml --genotype search_checkpoints/genotype.json<br/>streamlit run streamlit_app/app.py"
    story.append(Paragraph(code, styles["Code"]))
    doc.build(story)
    return output


def main() -> None:
    """Generate all documentation artifacts."""
    generate_architecture_png()
    generate_pdf_report()
    print(f"Generated documentation in {DOCS_DIR}")


if __name__ == "__main__":
    main()
