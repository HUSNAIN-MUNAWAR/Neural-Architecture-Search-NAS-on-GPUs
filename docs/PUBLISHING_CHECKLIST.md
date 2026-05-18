# Publishing Checklist

Use this checklist before posting the repository on LinkedIn, Upwork, or a job application.

## Must Be Done Before Public Posting

- [ ] Replace every placeholder live URL in README/demo docs with either a real link or `Not deployed yet`.
- [ ] Run the commands below and paste the terminal output into a GitHub issue, release note, or README verification section:

```bash
pip install -r requirements.txt
pytest
python scripts/smoke_test.py
python scripts/verify_project.py
python scripts/run_search.py --config configs/smoke_cpu.yaml
```

- [ ] Push to GitHub and confirm the CI badge is green.
- [ ] Capture real Streamlit screenshots from the local app and place them in `docs/screenshots/`.
- [ ] Record a 60-90 second demo video showing the dashboard, smoke search, metrics, genotype viewer, and Docker command.
- [ ] Do **not** publish accuracy numbers until a real GPU CIFAR-10 or client-dataset run is completed.

## Strongly Recommended Before Upwork Proposals

- [ ] Deploy the dashboard on a small VPS with Docker Compose or systemd.
- [ ] Add the live app URL to the README.
- [ ] Add one real GPU run row to `docs/BENCHMARKS.md` with hardware, config, metrics, and commit hash.
- [ ] Add a short LinkedIn post explaining the engineering decisions, not just the model.
- [ ] Add a custom dataset adapter branch or note if applying to industrial defect inspection jobs.

## Avoid These Claims

Do not say:

- "state-of-the-art" unless benchmarked against published baselines.
- "production distributed training" unless DDP has been tested on target multi-GPU hardware.
- "always free hosted app" because free Streamlit hosting may sleep.
- "benchmark accuracy" from the synthetic smoke run.

Better wording:

- "production-oriented ML engineering project"
- "GPU-aware DARTS implementation"
- "deployment-ready local dashboard"
- "MLOps-ready structure"
- "designed for extension to industrial datasets"
