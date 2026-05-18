# Security Policy

## Supported Scope

This project is a research and portfolio-grade ML engineering repository. Security support focuses on dependency hygiene, safe configuration practices, and responsible reporting for issues that could affect users running the Streamlit dashboard or Docker images.

## Reporting a Vulnerability

Please open a private security advisory on GitHub or contact the repository owner directly. Include:

- A clear description of the issue.
- Steps to reproduce.
- Affected files or commands.
- Potential impact.
- Suggested remediation if available.

Do not publicly disclose a vulnerability before maintainers have had reasonable time to review and patch it.

## Dependency Safety

- Keep PyTorch, Streamlit, TensorBoard, and Docker base images patched.
- Run `pip list --outdated` or a dependency scanner before production deployment.
- Never commit API keys, private datasets, credentials, SSH keys, or cloud tokens.
- Use `.streamlit/secrets.toml` locally only; commit only `.streamlit/secrets.toml.example`.

## Deployment Safety

- Do not expose the Streamlit app publicly without authentication if it can launch long-running jobs.
- Use a reverse proxy, TLS, firewall rules, and basic access control for public demos.
- Mount datasets/checkpoints as volumes rather than baking them into Docker images.
- Treat uploaded genotype/checkpoint files as untrusted input.
