# Containerizing the Marketing Intelligence API

This project includes a Dockerfile and docker-compose configuration to build and run the Flask API (`src/api/main.py`).

Quick start (from project root):

1. Build the image:

```bash
docker compose build
```

2. Run the service:

```bash
docker compose up
```

3. The API will be available on http://localhost:8000

Notes and tips
- The image installs heavy ML libraries (torch, faiss, sentence-transformers) from `requirements.txt`. The build can be large and may take a while depending on your network.
- If you run into native build errors for `faiss` or `torch` on certain host platforms, prefer running Docker on a Linux host or use pre-built compatible wheels.
- The container sets several environment variables (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, etc.) to reduce runtime threading conflicts that can cause crashes.
- For development iteration you can keep the source mounted via the `volumes` entry in `docker-compose.yml` so changes on the host reflect inside the container.

Troubleshooting
- If the container fails during `pip install` due to wheels or platform issues, try building on a Linux machine or remove problematic packages for a slim image.
- To debug which import causes segmentation faults, try editing `src/api/main.py` to lazy-load `faiss` and `sentence_transformers` only when endpoints are called.

If you want, I can:
- Add a multi-stage build to reduce final image size.
- Implement lazy-loading of heavy ML libs inside `src/api/main.py` so the container can start even when some optional artifacts are missing.
