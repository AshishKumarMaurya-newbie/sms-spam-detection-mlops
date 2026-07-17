#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py

docker build -t text-spam-detection .
docker run -d -p 8000:8000 --name text-spam-detection-container text-spam-detection

echo "Application is running on http://localhost:8000"
