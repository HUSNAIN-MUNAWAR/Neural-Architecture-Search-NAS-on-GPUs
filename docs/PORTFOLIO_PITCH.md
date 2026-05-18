# Portfolio Pitch

## 30-Second Recruiter Pitch

I built a GPU-aware Neural Architecture Search platform using PyTorch, DARTS, Streamlit, Docker, and MLOps-style practices. It is not just a notebook: it includes modular training code, YAML configs, TensorBoard logging, model export to ONNX/TorchScript, a dashboard, tests, CI, and VPS deployment documentation. It shows that I can build end-to-end ML systems that are reproducible, inspectable, and client-ready.

## Upwork Proposal Pitch

I can build AutoML and computer vision training systems that go beyond basic model scripts. This NAS platform demonstrates how I structure GPU training pipelines, dashboards, experiment tracking, model export, and deployment. The same pattern can be adapted to your image classification, defect inspection, medical imaging, or product recognition dataset.

## Technical Summary

- DARTS-style differentiable architecture search.
- CIFAR-10 data pipeline with augmentation.
- MixedOps and genotype export.
- Full evaluation training path.
- TensorBoard and JSONL metrics.
- Streamlit dashboard for local job control.
- Docker and VPS always-on deployment assets.

## Business / Client Value

Clients often need fast model iteration but do not know which architecture works best. A NAS pipeline can explore candidate CNN blocks, produce reproducible artifacts, and help compare model trade-offs before deployment.

## Adaptation Examples

### Industrial defect inspection

Search for a compact CNN architecture optimized for defect/non-defect classification on factory images.

### Medical image classification research

Use the search pipeline for controlled experiments on validated medical datasets, with clear benchmark reporting.

### Retail product classification

Adapt the dataloader to product images and export the trained model for app/backend inference.

### Edge model optimization

Add latency and parameter constraints to search smaller architectures for embedded devices.

### AutoML research pipeline

Extend the search space and experiment tracker for reproducible academic or applied AutoML studies.
