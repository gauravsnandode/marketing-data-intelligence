FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
       git \
       curl \
       ca-certificates \
       libsndfile1 \
       libopenblas-dev \
       libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency spec
COPY requirements.txt /app/requirements.txt

# Prepare a constraints file to pin NumPy and related binary packages to 1.x to avoid ABI
# incompatibilities (prevents accidental upgrade to numpy 2.x during dependency resolution)
RUN printf "numpy==1.24.3\nscipy==1.13.1\nscikit-learn==1.2.2\npandas==1.5.3\n" > /app/constraints.txt

# Upgrade pip and install core numeric packages first (ensure binary wheels are used)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -c /app/constraints.txt \
        "numpy==1.24.3" "scipy==1.13.1" "scikit-learn==1.2.2" "pandas==1.5.3"

# Install CPU-only PyTorch (explicit CPU wheel) to avoid pulling CUDA/NVIDIA wheels
RUN pip install --no-cache-dir -c /app/constraints.txt "torch==2.2.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

# Install faiss-cpu (manylinux wheel) so faiss is available at runtime (CPU-only)
RUN pip install --no-cache-dir -c /app/constraints.txt faiss-cpu

# Install remaining deps but strip heavy GPU/faiss packages from requirements
RUN sed -E '/^(torch|torchvision|torchaudio|faiss|faiss-cpu|numpy|scipy|scikit-learn|pandas)/I d' /app/requirements.txt > /app/requirements_no_torch.txt && \
    # Use a constraints file to pin numpy to a 1.x release so binary extensions compiled
    # against NumPy 1.x remain compatible and pip won't upgrade numpy to 2.x.
    printf "numpy==1.24.3\n" > /app/constraints.txt && \
    pip install --no-cache-dir -r /app/requirements_no_torch.txt -c /app/constraints.txt

# Ensure Flask is installed (in case it's missing from requirements.txt)
RUN python -c "import pkgutil, sys; \
    exit(0) if pkgutil.find_loader('flask') else sys.exit(1)" || pip install --no-cache-dir flask

# Copy the project
COPY . /app

EXPOSE 8000

CMD ["python", "src/api/main.py"]
