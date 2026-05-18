#!/usr/bin/env bash
set -euo pipefail
URL="${1:-http://localhost:8501/_stcore/health}"
curl -fsS "$URL" >/dev/null
echo "NAS Streamlit healthcheck OK: $URL"
