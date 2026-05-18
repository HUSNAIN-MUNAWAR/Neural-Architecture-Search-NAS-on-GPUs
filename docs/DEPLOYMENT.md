# Deployment Guide

This project supports local, Docker, GPU Docker, and VPS always-on deployment.

## Local Streamlit

```bash
streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port 8501
```

Use this for development and quick demos on your own machine.

## Docker Compose

```bash
docker compose up -d --build
```

This starts a CPU-friendly Streamlit container with mounted volumes:

- `./data`
- `./search_checkpoints`
- `./eval_checkpoints`
- `./streamlit_runtime`

Check status:

```bash
docker compose ps
docker compose logs -f nas-streamlit
```

## GPU Docker Compose

Install NVIDIA Container Toolkit first, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Verify GPU visibility:

```bash
docker compose exec nas-streamlit python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## VPS Always-on Deployment

For an always-on demo, use a small Ubuntu VPS and run:

```bash
git clone https://github.com/your-org/Neural-Architecture-Search-NAS-on-GPUs.git
cd Neural-Architecture-Search-NAS-on-GPUs
docker compose up -d --build
```

Docker restart policy is set to `unless-stopped`, so the app restarts after crashes or reboots.

## systemd Deployment

A non-Docker systemd service is included:

```bash
sudo cp deployment/vps/systemd/nas-streamlit.service /etc/systemd/system/nas-streamlit.service
sudo systemctl daemon-reload
sudo systemctl enable --now nas-streamlit
sudo systemctl status nas-streamlit
```

Edit `User`, `WorkingDirectory`, and `ExecStart` before enabling the service.

## Nginx Reverse Proxy

Use `deployment/vps/nginx/nas-streamlit.conf` as a starting point:

```bash
sudo cp deployment/vps/nginx/nas-streamlit.conf /etc/nginx/sites-available/nas-streamlit
sudo ln -s /etc/nginx/sites-available/nas-streamlit /etc/nginx/sites-enabled/nas-streamlit
sudo nginx -t
sudo systemctl reload nginx
```

Add TLS with Certbot for public demos.

## Streamlit Community Cloud Limitation

Free Streamlit Community Cloud is useful for simple demos, but apps may sleep after inactivity and are not ideal for heavy NAS jobs. For always-on portfolio or client demos, prefer:

- VPS + Docker Compose
- VPS + systemd
- Paid always-on hosts
- Cloud VM with Nginx and TLS

Keep-alive pings are a weak demo workaround and should not be presented as production deployment.
