"""Search utilities for genotype derivation and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from models.operations import PRIMITIVES

Genotype = Dict[str, List]


def _parse(weights: torch.Tensor, k: int = 2) -> List[List[object]]:
    """Parse edge weights into a DARTS gene list."""
    gene: List[List[object]] = []
    n = 2
    start = 0
    primitive_names = list(PRIMITIVES)
    zero_index = primitive_names.index("zero")
    for _ in range(4):
        end = start + n
        edge_scores = []
        for edge_idx in range(start, end):
            edge_weight = weights[edge_idx]
            valid_scores = [
                (float(edge_weight[op_idx].item()), op_idx)
                for op_idx in range(len(primitive_names))
                if op_idx != zero_index
            ]
            best_score, best_op_idx = max(valid_scores, key=lambda item: item[0])
            edge_scores.append((best_score, best_op_idx, edge_idx - start))
        selected = sorted(edge_scores, key=lambda item: item[0], reverse=True)[:k]
        for _, op_idx, input_idx in selected:
            gene.append([primitive_names[op_idx], int(input_idx)])
        start = end
        n += 1
    return gene


def derive_genotype(alphas_normal: torch.Tensor, alphas_reduce: torch.Tensor, k: int = 2) -> Genotype:
    """Derive a discrete genotype from continuous architecture parameters."""
    normal_weights = F.softmax(alphas_normal.detach().cpu(), dim=-1)
    reduce_weights = F.softmax(alphas_reduce.detach().cpu(), dim=-1)
    return {
        "normal": _parse(normal_weights, k=k),
        "normal_concat": [2, 3, 4, 5],
        "reduce": _parse(reduce_weights, k=k),
        "reduce_concat": [2, 3, 4, 5],
    }


def genotype_to_string(genotype: Genotype) -> str:
    """Format a genotype as stable JSON."""
    return json.dumps(genotype, indent=2)


def plot_cell(genotype: Genotype, cell_type: str, output_path: str) -> str:
    """Render a normal or reduction cell diagram to PNG."""
    if cell_type not in {"normal", "reduce"}:
        raise ValueError("cell_type must be 'normal' or 'reduce'")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    gene = genotype[cell_type]
    try:
        from graphviz import Digraph

        graph = Digraph(format="png", graph_attr={"rankdir": "LR", "splines": "spline"})
        graph.node("0", "c_{k-2}", fillcolor="lightgray", style="filled")
        graph.node("1", "c_{k-1}", fillcolor="lightgray", style="filled")
        for node_idx in range(2, 6):
            graph.node(str(node_idx), str(node_idx), fillcolor="lightblue", style="filled")
        graph.node("out", "concat", fillcolor="lightgreen", style="filled")
        for step in range(4):
            for pair_idx in range(2):
                op_name, source_idx = gene[2 * step + pair_idx]
                graph.edge(str(source_idx), str(step + 2), label=str(op_name))
        for node_idx in genotype.get(f"{cell_type}_concat", [2, 3, 4, 5]):
            graph.edge(str(node_idx), "out")
        rendered = graph.render(str(output.with_suffix("")), cleanup=True)
        return rendered
    except Exception:
        import matplotlib.pyplot as plt
        import networkx as nx

        graph_nx = nx.DiGraph()
        labels = {0: "c{k-2}", 1: "c{k-1}", 2: "2", 3: "3", 4: "4", 5: "5", 6: "concat"}
        for node in labels:
            graph_nx.add_node(node)
        edge_labels = {}
        for step in range(4):
            for pair_idx in range(2):
                op_name, source_idx = gene[2 * step + pair_idx]
                dst = step + 2
                graph_nx.add_edge(int(source_idx), dst)
                edge_labels[(int(source_idx), dst)] = str(op_name)
        for node_idx in genotype.get(f"{cell_type}_concat", [2, 3, 4, 5]):
            graph_nx.add_edge(int(node_idx), 6)
            edge_labels[(int(node_idx), 6)] = "concat"
        pos = {0: (0, 1), 1: (0, -1), 2: (1, 0.8), 3: (2, 0.2), 4: (3, -0.2), 5: (4, -0.8), 6: (5, 0)}
        plt.figure(figsize=(10, 4))
        nx.draw(graph_nx, pos, labels=labels, with_labels=True, node_size=1800, arrows=True)
        nx.draw_networkx_edge_labels(graph_nx, pos, edge_labels=edge_labels, font_size=8)
        plt.title(f"{cell_type.title()} Cell")
        plt.tight_layout()
        plt.savefig(output, dpi=160)
        plt.close()
        return str(output)
