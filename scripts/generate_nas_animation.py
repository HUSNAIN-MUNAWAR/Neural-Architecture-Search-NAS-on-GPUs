"""Generate a DARTS search-loop animation and reproducibility manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "nas_animation_fixture.synthetic.json"
REFERENCE_GENOTYPE = ROOT / "examples" / "smoke_genotype.synthetic.json"
MANIFEST = ROOT / "docs" / "animation" / "nas_search_sequence.json"
REEL = ROOT / "frontend" / "public" / "animation" / "nas-search-reel.html"


def genotype_summary(genotype: dict[str, list]) -> dict[str, object]:
    def counts(key: str) -> dict[str, int]:
        return dict(Counter(str(item[0]) for item in genotype[key]))

    return {
        "normal_edges": len(genotype.get("normal", [])),
        "reduction_edges": len(genotype.get("reduce", [])),
        "normal_operation_counts": counts("normal"),
        "reduction_operation_counts": counts("reduce"),
        "normal_concat": genotype.get("normal_concat", []),
        "reduce_concat": genotype.get("reduce_concat", []),
    }


def run_actual_model(fixture: dict[str, object]) -> dict[str, object] | None:
    try:
        import torch

        sys.path.insert(0, str(ROOT))
        from data.dataset import get_synthetic_dataloaders
        from models.architecture import DartsNetworkCIFAR, NetworkCIFAR
        from search.utils import derive_genotype

        torch.set_num_threads(1)
        torch.manual_seed(int(fixture["seed"]))
        loaders = get_synthetic_dataloaders(
            batch_size=int(fixture["batch_size"]),
            val_batch_size=int(fixture["batch_size"]),
            num_workers=0,
            num_classes=int(fixture["num_classes"]),
            train_size=int(fixture["synthetic_train_size"]),
            val_size=int(fixture["synthetic_val_size"]),
            test_size=int(fixture["synthetic_test_size"]),
            seed=int(fixture["seed"]),
        )
        images, labels = next(iter(loaders.train))
        criterion = torch.nn.CrossEntropyLoss()
        search_model = DartsNetworkCIFAR(
            init_channels=int(fixture["init_channels"]),
            num_classes=int(fixture["num_classes"]),
            layers=int(fixture["layers"]),
            criterion=criterion,
            steps=int(fixture["steps"]),
            multiplier=int(fixture["multiplier"]),
        )
        search_model.eval()
        with torch.no_grad():
            logits = search_model(images)
            loss = criterion(logits, labels)
        genotype = derive_genotype(search_model.alphas_normal, search_model.alphas_reduce)
        derived = NetworkCIFAR(
            genotype=genotype,
            init_channels=int(fixture["init_channels"]),
            num_classes=int(fixture["num_classes"]),
            layers=int(fixture["layers"]),
            auxiliary=False,
            steps=int(fixture["steps"]),
            multiplier=int(fixture["multiplier"]),
        )
        derived.eval()
        with torch.no_grad():
            derived_logits = derived(images)
        return {
            "execution_mode": "actual_forward_pass",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "batch_shape": list(images.shape),
            "label_histogram": Counter(labels.tolist()),
            "search_logits_shape": list(logits.shape),
            "search_loss": round(float(loss.item()), 6),
            "derived_logits_shape": list(derived_logits.shape),
            "parameter_count": sum(parameter.numel() for parameter in search_model.parameters()),
            "genotype": genotype,
        }
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        return {"execution_mode": "deterministic_genotype_reference", "fallback_reason": str(exc)}


def build_manifest() -> dict[str, object]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reference_genotype = json.loads(REFERENCE_GENOTYPE.read_text(encoding="utf-8"))
    actual = run_actual_model(fixture)
    genotype = actual.get("genotype", reference_genotype) if actual else reference_genotype
    model_execution = actual or {"execution_mode": "deterministic_genotype_reference"}
    fixture_bytes = FIXTURE.read_bytes()
    summary = genotype_summary(genotype)
    return {
        "schema_version": "1.0",
        "fixture": fixture["fixture_id"],
        "source": {
            "path": FIXTURE.relative_to(ROOT).as_posix(),
            "bytes": len(fixture_bytes),
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "synthetic": True,
            "config_reference": fixture["config_reference"],
            "seed": fixture["seed"],
            "image_shape": fixture["image_shape"],
            "dataset_sizes": {"train": fixture["synthetic_train_size"], "val": fixture["synthetic_val_size"], "test": fixture["synthetic_test_size"]},
        },
        "model": {
            "name": "DARTS-CIFAR search network",
            "search_implementation": "models/architecture.py::DartsNetworkCIFAR",
            "derived_implementation": "models/architecture.py::NetworkCIFAR",
            "genotype_derivation": "search/utils.py::derive_genotype",
            "candidate_operations": ["sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3", "dil_conv_5x5", "max_pool_3x3", "avg_pool_3x3", "skip_connect", "zero"],
            "execution": model_execution,
            "claim_boundary": "Synthetic data verifies model plumbing and genotype export; it is not a classification benchmark.",
        },
        "architecture": summary,
        "events": [
            {"id": "seed", "short": "Seed", "title": "Synthetic data is fixed", "detail": "The offline CIFAR-shaped loader creates reproducible train, validation, and test tensors from one seed.", "metric": f"seed {fixture['seed']} · {fixture['image_shape']} images", "status": "reproducible"},
            {"id": "search-space", "short": "Search", "title": "Mixed operations are instantiated", "detail": "Every DARTS edge carries a weighted mixture of the repository’s candidate operations.", "metric": f"{len(summary['normal_operation_counts']) + len(summary['reduction_operation_counts'])} active op families · {fixture['steps']} steps", "status": "continuous"},
            {"id": "forward", "short": "Forward", "title": "Search network produces logits", "detail": "A real synthetic batch flows through the search network when PyTorch is available; otherwise the committed genotype reference remains explicit.", "metric": f"mode {model_execution['execution_mode']} · output {model_execution.get('search_logits_shape', [fixture['batch_size'], fixture['num_classes']])}", "status": "computed"},
            {"id": "derive", "short": "Derive", "title": "Architecture weights become a genotype", "detail": "The repository’s derive_genotype utility selects non-zero operations and records normal/reduction edges.", "metric": f"{summary['normal_edges']} normal + {summary['reduction_edges']} reduction edges", "status": "discrete"},
            {"id": "compile", "short": "Compile", "title": "Discrete network is rebuilt", "detail": "NetworkCIFAR compiles the exported genotype into a deployment-shaped architecture for a second forward path.", "metric": f"concat nodes {summary['normal_concat']} · {summary['reduce_concat']}", "status": "derived"},
            {"id": "artifact", "short": "Export", "title": "Experiment evidence is exportable", "detail": "The genotype, synthetic configuration, tensor shapes, and execution mode are stored as a reviewable manifest.", "metric": "JSON manifest · HTML reel · no benchmark claim", "status": "recorded"},
        ],
    }


def reel_html(manifest: dict[str, object]) -> str:
    sequence = json.dumps(manifest, indent=2, sort_keys=True).replace("</", "<\\/")
    return dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>NAS on GPUs — Search reference reel</title>
          <style>
            :root {{ color-scheme: dark; --ink: #eef4fa; --muted: #9bb0c2; --line: #294759; --accent: #79d4ff; }}
            * {{ box-sizing: border-box; }}
            body {{ margin: 0; min-width: 320px; background: radial-gradient(circle at 15% 0%, #0e4862 0, #071620 53%, #050a10 100%); color: var(--ink); font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif; }}
            .reel {{ max-width: 1180px; margin: 0 auto; padding: 42px 28px 34px; }}
            .eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
            h1 {{ max-width: 820px; margin: 8px 0 10px; font-size: clamp(28px, 5vw, 52px); line-height: 1.02; letter-spacing: -.04em; }}
            .lede {{ max-width: 790px; margin: 0; color: var(--muted); font-size: 17px; }}
            .model-line {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 22px; color: var(--muted); font-size: 12px; }}
            .model-line span::before {{ content: "•"; color: var(--accent); margin-right: 8px; }}
            .stage {{ margin-top: 36px; overflow: hidden; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(145deg, rgba(15,49,67,.94), rgba(6,17,25,.97)); box-shadow: 0 22px 60px rgba(0,0,0,.28); }}
            svg {{ display: block; width: 100%; height: auto; min-height: 300px; }}
            .route {{ fill: none; stroke: #2d566c; stroke-width: 4; stroke-linecap: round; stroke-dasharray: 7 12; }}
            .station circle {{ fill: #081720; stroke: #46738a; stroke-width: 3; transition: fill .25s ease, stroke .25s ease, transform .25s ease; transform-origin: center; }}
            .station text {{ fill: #a0b8c6; font-size: 14px; text-anchor: middle; }}
            .station .index {{ fill: #d2e8f0; font-size: 13px; font-weight: 700; }}
            .station.active circle {{ fill: #125269; stroke: var(--accent); transform: scale(1.1); }}
            .station.active text {{ fill: var(--ink); }}
            .station.done circle {{ fill: #173d50; stroke: #61a0b8; }}
            .token {{ fill: var(--accent); stroke: #d9f7ff; stroke-width: 3; filter: drop-shadow(0 0 12px rgba(121,212,255,.65)); }}
            .token-label {{ fill: #06202a; font-size: 10px; font-weight: 800; text-anchor: middle; }}
            .panel {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 18px; padding: 22px 24px 24px; border-top: 1px solid var(--line); }}
            .panel h2 {{ margin: 0 0 3px; font-size: 24px; letter-spacing: -.02em; }}
            .panel p {{ margin: 0; color: var(--muted); }}
            .metric {{ color: var(--accent); font-size: 13px; font-weight: 700; text-align: right; }}
            .controls {{ display: flex; gap: 8px; margin-top: 16px; }}
            button {{ border: 1px solid #3c667b; border-radius: 999px; padding: 9px 15px; background: #102d3c; color: var(--ink); font: inherit; cursor: pointer; }}
            button:hover, button:focus-visible {{ border-color: var(--accent); outline: 2px solid rgba(121,212,255,.35); outline-offset: 2px; }}
            .timeline {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 7px; margin-top: 18px; }}
            .timeline button {{ min-height: 60px; border-radius: 12px; padding: 8px; color: var(--muted); text-align: left; font-size: 12px; }}
            .timeline button[aria-current="step"] {{ border-color: var(--accent); background: #16485d; color: var(--ink); }}
            .timeline strong {{ display: block; color: inherit; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
            .disclaimer {{ margin-top: 20px; color: var(--muted); font-size: 12px; }}
            @media (max-width: 760px) {{ .reel {{ padding: 28px 16px; }} .panel {{ grid-template-columns: 1fr; }} .metric {{ text-align: left; }} .timeline {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} svg {{ min-height: 420px; }} }}
            @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }} }}
          </style>
        </head>
        <body>
          <main class="reel" aria-labelledby="title">
            <div class="eyebrow">Neural Architecture Search · production reference reel</div>
            <h1 id="title">From synthetic batch to derived genotype</h1>
            <p class="lede">A DARTS-style search network turns reproducible CIFAR-shaped data into inspectable continuous weights, a discrete genotype, and a derived model path.</p>
            <div class="model-line" aria-label="Model and data boundary"><span id="model-name"></span><span id="source-name"></span><span>synthetic data only</span></div>
            <section class="stage" aria-label="Animated NAS sequence">
              <svg viewBox="0 0 1210 390" role="img" aria-labelledby="diagram-title diagram-description">
                <title id="diagram-title">Animated DARTS search stages</title>
                <desc id="diagram-description">A token travels through synthetic data seeding, mixed operations, forward computation, genotype derivation, discrete compilation, and evidence export.</desc>
                <path class="route" d="M 80 190 C 220 82, 360 82, 510 190 S 800 298, 1130 100" />
                <g id="stations"></g>
                <g aria-hidden="true"><circle id="token" class="token" cx="80" cy="190" r="22"></circle><text id="token-label" class="token-label" x="80" y="194">NAS</text></g>
              </svg>
              <div class="panel" aria-live="polite"><div><h2 id="event-title"></h2><p id="event-detail"></p></div><div id="event-metric" class="metric"></div></div>
            </section>
            <div class="controls"><button id="toggle" type="button">Pause</button><button id="replay" type="button">Replay</button></div>
            <div id="timeline" class="timeline" role="list" aria-label="Animation stages"></div>
            <p class="disclaimer">Synthetic smoke fixture · DARTS model plumbing · not a measured accuracy benchmark or GPU performance claim.</p>
          </main>
          <script>
            const sequence = {sequence};
            const events = sequence.events;
            const points = [[80,190],[270,96],[460,96],[650,190],[840,284],[1020,284],[1130,100]];
            const stations = document.getElementById('stations'), timeline = document.getElementById('timeline');
            const token = document.getElementById('token'), tokenLabel = document.getElementById('token-label');
            const title = document.getElementById('event-title'), detail = document.getElementById('event-detail'), metric = document.getElementById('event-metric');
            const toggle = document.getElementById('toggle');
            let current = 0, started = performance.now(), playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
            const duration = 1850;
            points.forEach((point, index) => {{ const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'); group.classList.add('station'); group.innerHTML = `<circle cx="${{point[0]}}" cy="${{point[1]}}" r="30"></circle><text class="index" x="${{point[0]}}" y="${{point[1] + 5}}">${{index + 1}}</text><text x="${{point[0]}}" y="${{point[1] + 57}}">${{events[index].short}}</text>`; stations.appendChild(group); }});
            events.forEach((event, index) => {{ const item = document.createElement('button'); item.type = 'button'; item.setAttribute('role', 'listitem'); item.innerHTML = `<strong>${{String(index + 1).padStart(2, '0')}} · ${{event.status}}</strong>${{event.title}}`; item.addEventListener('click', () => {{ current = index; started = performance.now(); playing = false; toggle.textContent = 'Play'; render(0); }}); timeline.appendChild(item); }});
            document.getElementById('model-name').textContent = `model ${{sequence.model.name}}`;
            document.getElementById('source-name').textContent = `mode ${{sequence.model.execution.execution_mode}}`;
            function render(progress) {{ const next = (current + 1) % points.length, from = points[current], to = points[next]; const x = from[0] + (to[0] - from[0]) * progress, y = from[1] + (to[1] - from[1]) * progress; token.setAttribute('cx', x); token.setAttribute('cy', y); tokenLabel.setAttribute('x', x); tokenLabel.setAttribute('y', y + 4); title.textContent = events[current].title; detail.textContent = events[current].detail; metric.textContent = events[current].metric; document.querySelectorAll('.station').forEach((node, index) => {{ node.classList.toggle('active', index === current); node.classList.toggle('done', index < current); }}); timeline.querySelectorAll('button').forEach((node, index) => node.setAttribute('aria-current', index === current ? 'step' : 'false')); }}
            function animate(now) {{ if (playing) {{ const progress = Math.min(1, (now - started) / duration); render(progress); if (progress >= 1) {{ current = (current + 1) % events.length; started = now; }} }} else {{ render(0); }} requestAnimationFrame(animate); }}
            toggle.addEventListener('click', () => {{ playing = !playing; started = performance.now(); toggle.textContent = playing ? 'Pause' : 'Play'; }}); document.getElementById('replay').addEventListener('click', () => {{ current = 0; started = performance.now(); playing = true; toggle.textContent = 'Pause'; }}); render(0); requestAnimationFrame(animate);
          </script>
        </body>
        </html>
        """
    )


def main() -> None:
    manifest = build_manifest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REEL.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    REEL.write_text(reel_html(manifest), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"wrote {REEL.relative_to(ROOT)}")
    print(f"execution mode: {manifest['model']['execution']['execution_mode']}")
    print(f"normal edges: {manifest['architecture']['normal_edges']}")
    print(f"reduction edges: {manifest['architecture']['reduction_edges']}")


if __name__ == "__main__":
    main()
