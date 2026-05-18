# Always-on VPS Deployment Guide

This guide deploys the Streamlit dashboard on an Ubuntu VPS so it does not sleep after inactivity.

## 1. Prepare Ubuntu

```bash
sudo apt update
sudo apt install -y git curl docker.io docker-compose-plugin nginx
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

Log out and log back in so Docker group permissions apply.

## 2. Clone and start with Docker Compose

```bash
git clone https://github.com/your-org/Neural-Architecture-Search-NAS-on-GPUs.git
cd Neural-Architecture-Search-NAS-on-GPUs
docker compose up -d --build
```

Check health:

```bash
docker compose ps
curl http://localhost:8501/_stcore/health
```

## 3. GPU VPS option

Install NVIDIA drivers and NVIDIA Container Toolkit, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Verify GPU access:

```bash
docker compose exec nas-streamlit python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 4. systemd option without Docker

Copy the service file:

```bash
sudo cp deployment/vps/systemd/nas-streamlit.service /etc/systemd/system/nas-streamlit.service
sudo systemctl daemon-reload
sudo systemctl enable --now nas-streamlit
sudo systemctl status nas-streamlit
```

Edit paths inside the service file before using it on your server.

## 5. Nginx reverse proxy

```bash
sudo cp deployment/vps/nginx/nas-streamlit.conf /etc/nginx/sites-available/nas-streamlit
sudo ln -s /etc/nginx/sites-available/nas-streamlit /etc/nginx/sites-enabled/nas-streamlit
sudo nginx -t
sudo systemctl reload nginx
```

Add TLS with Certbot for public deployments.

## 6. Security notes

- Do not expose a job-launching dashboard publicly without access control.
- Put the app behind Nginx, HTTPS, and firewall rules.
- Store data/checkpoints on mounted volumes.
- Do not commit secrets or private datasets.
