"""
SentimentLens Multi-Model Production Gateway Entrypoint.
Root server adapter for Render, Docker, and Cloud Deployments.
Serves Model V1, Model V2, and Model V3 through unified REST API.
"""

import os
import sys

# Ensure root directory is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend_app.app import app, p1, p2, p3

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"[SentimentLens Gateway] Starting production server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
