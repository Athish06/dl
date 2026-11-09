"""
dl Multi-Model Production Gateway Entrypoint.
Root server adapter for Render, Docker, and Cloud Deployments.
Serves Model V1, Model V2, and Model V3 through unified REST API.
"""

import os
import sys

# Ensure both current directory and parent directory are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in [CURRENT_DIR, PARENT_DIR]:
    if path and path not in sys.path:
        sys.path.insert(0, path)

try:
    from backend_app.app import app, p1, p2, p3
except (ModuleNotFoundError, ImportError):
    from app import app, p1, p2, p3

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"[dl Gateway] Starting production server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
