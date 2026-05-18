# Demo Guide

Use this guide to record real screenshots and a demo video. Do not add fake screenshots.

## 1. Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## 2. Run CPU Smoke Mode

In the dashboard:

1. Open **Launch Search**.
2. Select **CPU smoke mode**.
3. Keep `epochs=1`, `init_channels=4`, `layers=2`.
4. Click **Start Search**.
5. Open **Monitor Runs** and show logs/metrics.

CLI equivalent:

```bash
python scripts/run_search.py --config configs/smoke_cpu.yaml
```

## 3. Screenshots To Capture

Add real screenshots to `docs/screenshots/`:

- `overview.png` - dashboard overview and architecture diagram.
- `launch_search.png` - search controls with CPU/GPU presets.
- `monitor_metrics.png` - metrics curves after a run.
- `architecture_viewer.png` - normal/reduction cell diagram.
- `deployment_tab.png` - export/deployment page.

## 4. Recommended Demo Video Flow

1. Open README and explain project value.
2. Start Streamlit dashboard.
3. Launch CPU smoke search.
4. Monitor live logs and metrics.
5. Open architecture viewer and load genotype.
6. Show export/download controls.
7. Show Docker deployment command.
8. Mention limitations honestly: real NAS needs GPU time.

## 5. Optional GPU Demo

```bash
python scripts/run_search.py --config configs/demo_gpu.yaml
```

Record GPU details with:

```bash
nvidia-smi
python scripts/smoke_test.py
```
