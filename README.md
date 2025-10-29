# Marketing Data Intelligence — Project Report

## Executive summary

This project builds an end-to-end Marketing Data Intelligence system that:

- Predicts product discount percentage using a supervised model trained on e-commerce data.
- Provides an LLM-powered assistant that answers product/customer queries using a Retrieval-Augmented Generation (RAG) layer over product data.
- Exposes both capabilities via containerized APIs (`/predict_discount`, `/answer_question`) implemented in `src/api/main.py`.
- Is packaged with Docker (`Dockerfile`, `docker-compose.yml`) and includes testing harnesses such as `test_api.py`.

Key outcomes so far:

- A predictive model (XGBoost / LightGBM-style training) was trained — logs are in `training_process.txt`. Example predictions returned by the running API are in `results_for_test_api.txt`.
- RAG retrieval works when RAG artifacts exist; FAISS-based index created in `src/api/model/product_index.faiss`. The API lazy-loads FAISS and sentence-transformers to avoid startup failures in constrained images.
- Docker build was tuned to avoid GPU/CUDA wheels and NumPy ABI mismatches (constraints pinning to numpy 1.24.3).

---

## System architecture (high level)

- Data layer
  - Raw dataset: `data/amazon.csv` (used in training scripts).
  - Vector store: `src/api/model/product_index.faiss`
  - Product metadata: pickled product_data (`product_data.pkl`) in same directory.
- Model layer
  - Predictive model artifact: `discount_predictor.pkl`
  - Model feature metadata: `model_columns.pkl`
- RAG + LLM layer
  - SentenceTransformer embeddings (e.g., `all-MiniLM-L6-v2`) for retrieval.
  - Retrieval via FAISS index; optional LLM (open-source) performs generation when integrated.
- API layer
  - Flask app in `src/api/main.py` exposing:
    - `POST /predict_discount` → returns predicted discount percent
    - `POST /answer_question` → returns retrieval results (and optionally generated answers)
    - Test client: `test_api.py`
- Deployment
  - `Dockerfile` and `docker-compose.yml` to build/run the service.
  - Constraints and dependency ordering added to ensure binary compatibility for numeric libs.

Diagram (textual)

- Client → Docker container (Flask API)
- Flask API → (1) discount model (loaded at startup) → prediction
- Flask API → (2) RAG resources (lazy-loaded FAISS index + embedding model) → retrieval → (optional) LLM generation

---

## Dataset

- Source: Amazon-style product dataset (similar to Kaggle Amazon Sales Dataset). The project used `data/amazon.csv`.
- Key features used (from training logs): `['main_category', 'sub_category_1', ..., 'actual_price', 'rating', 'rating_count']`
- Preprocessing steps (implemented in `src/scripts/train_model.py`):
  - Cleaning and normalization of price fields, missing value imputation for `rating` and `rating_count`.
  - Categorical encoding for hierarchical category fields.
  - Feature engineering for pricing and popularity signals.

Files of interest:
- `data/amazon.csv` — raw dataset
- `src/scripts/train_model.py` — training script (produces `discount_predictor.pkl`, `model_columns.pkl`)
- `src/scripts/create_vector_store.py` — builds FAISS index and `product_data.pkl`

---

## Predictive model (discount prediction)

- Model type: XGBoost / XGBoost-like (training logs show XGBoost-style iteration output). The training script uses RandomizedSearchCV to tune hyperparameters.
- Target: Discount percentage (regression).
- Training evidence: `training_process.txt` contains validation MAE across iterations (example MAE improvements down to ~13–14).
- Example API predictions (from `results_for_test_api.txt`):
  - Input 1 → predicted_discount_percentage: 25.44
  - Input 2 → predicted_discount_percentage: 14.61

Evaluation (recommended and available)

- Logged during training: validation MAE (shown in `training_process.txt`).
- Recommended metrics to compute and report:
  - MAE, RMSE, R² on holdout/test dataset.
  - Example commands:
    - Run evaluation script in repository (or add one) that loads `discount_predictor.pkl`, evaluates on `test.csv`, and prints MAE/RMSE/R².
- Explainability:
  - Use SHAP or feature importance from model to show which features drive discount decisions.
  - Example: `shap.Explainer` on `discount_predictor` and produce top features.

API contract (predict_discount)

- Endpoint: POST `/predict_discount`
- Input: JSON object matching `model_columns` (fields used during training). Example:
  - {"category": 10, "actual_price": 2999.0, "rating": 4.5, "rating_count": 12500.0}
- Output: JSON
  - Success: {"predicted_discount_percentage": float}
  - Errors: 400 for malformed input, 503 if model unavailable
- Implementation: `src/api/main.py` — loads `discount_predictor.pkl` and `model_columns.pkl`.

Edge cases & error modes
- Missing fields → endpoint returns 400 and lists required fields.
- Model artifact missing → endpoint returns 503.

---

## RAG + LLM Assistant

Design
- Index: FAISS index (`product_index.faiss`) + `product_data.pkl` (list of product dicts).
- Embeddings: SentenceTransformer (`all-MiniLM-L6-v2`) for compact embeddings.
- RAG flow:
  1. Encode incoming query into embedding.
  2. Search FAISS for top-k nearest products.
  3. Retrieve product text and optionally pass to an LLM (open-source) to generate a grounded answer.
- Implementation details:
  - RAG resources are now lazy-loaded on demand (`_load_rag_resources()` in `src/api/main.py`) to avoid import-time failures in minimal containers.
  - `answer_question` returns retrieved context (product dicts and similarity score). Optional generation step can be plugged in later.

Fine-tuning / domain adaptation
- Approaches:
  - Fine-tune an open-source LLM using LoRA/PEFT on domain-specific QA pairs or product descriptions to get tone/domain-aware generations.
  - Optionally fine-tune SentenceTransformer on in-domain text for better retrieval embeddings.
- Tools:
  - `transformers`, `peft`, `accelerate` for model fine-tuning; `sentence-transformers` for embedding adaptation.
- Data for fine-tuning:
  - Use product Q&A pairs (if available), curated marketing texts, and paraphrase augmentation.

RAG grounding & factuality
- Evaluate grounding accuracy (does the LLM output base facts found in retrieved documents?) with:
  - Precision@k for retrieval (how often ground-truth docs included)
  - Human evaluation for factuality, or automated checks comparing entity values to retrieved documents.

---

## Implementation details & notable changes (what was implemented)

- `src/api/main.py`
  - Environment variable controls and thread limits (OMP/OpenBLAS/etc.) are set early to avoid segfaults.
  - Lazy loading of FAISS and SentenceTransformer (`_load_rag_resources`) to avoid image startup crashes.
  - Numpy/pandas/floats converted to native types before JSON serialization.
- `Dockerfile` and `docker-compose.yml`
  - Containerized service exposing port 8000.
  - Dockerfile installs core numeric packages first, pins NumPy to 1.24.3 via a constraints file, installs CPU-only PyTorch and `faiss-cpu` to avoid pulling GPU wheels and to maintain ABI compatibility.
  - `.dockerignore` added to reduce build context.
- Test harness
  - `test_api.py` exercises the two endpoints; results logged to `results_for_test_api.txt`.

---

## Deployment & containerization

How to run locally (non-docker)
- Activate venv and run:
```bash
source venv/bin/activate
python src/api/main.py
```
How to build & run with Docker:
```bash
# From project root
docker compose build --no-cache
docker compose up
```
Notes / gotchas encountered
- Pip initially pulled Nvidia/CUDA wheels for torch-related packages. Fixed by:
  - installing CPU-only torch wheel (`torch==2.2.0+cpu` with the PyTorch CPU index),
  - installing `faiss-cpu`,
  - pinning NumPy to `1.24.3` via constraint file to avoid ABI mismatch with compiled C-extensions.
- Lazy-loading FAISS / sentence_transformers was added so the app can start even if some heavy packages fail in some environments.

---

## Testing, validation & CI suggestions

Unit & integration tests
- Unit tests: add pytest tests for:
  - Input validation in `/predict_discount`.
  - Serialization and edge-handling in JSON outputs.
  - `_load_rag_resources()` behaviors (mock missing artifacts).
- Integration tests:
  - `test_api.py` already provided. Run it while service is running to validate endpoints.
Load testing
- Use `locust` or `k6` to validate the API under load:
  - Script example: simulate concurrent POSTs to `/predict_discount` and `/answer_question`.
- Example: `locustfile.py` with endpoints.

Safety & rules validation
- Add a safety layer to the LLM generation (if added) to block sensitive or harmful outputs.
- Add rate-limiting on API (e.g., via nginx or in-app token bucket) for abuse protection.

Observability & monitoring
- Add Prometheus metrics (requests, latency, errors) and Grafana dashboards.
- Add logs to stdout in structured JSON to integrate with ELK/Cloud logging.
- Add Sentry (optional) for exception tracking.

Continuous retraining & drift detection (nice to have)
- Pipeline:
  - Automate nightly or periodic retraining via CI/CD or Airflow.
  - Monitor input data distributions and model predictive distributions for drift (KL divergence, pop stats).
  - When drift exceeds thresholds, trigger retraining pipeline and CI validations.
- Tools:
  - MLflow/Alembic for model versioning; Evidently/WhyLogs for drift detection.

Explainability
- Use SHAP for local and global interpretation of the discount model.
- Expose an admin endpoint that returns top-5 SHAP feature contributions for a given prediction to explain why a discount was suggested.

Security
- Do not include secret keys in images.
- If using hosted LLMs or model registries, use secure credentials and rotation.
- Limit model artifact permissions and sanitize inputs for injection attacks.

---

## Evaluation plan (how to score & report)

Regression metrics (predictive model)
- MAE, RMSE, R² on a holdout test set. Use `sklearn.metrics`.
- Example:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)
```
RAG grounding & factuality
- Retrieval: Precision@k (how often ground-truth or desired product is in top-k).
- Grounded generation: measure percentage of factual statements that can be traced to retrieved docs (manual or automated checks).
Latency & throughput
- Endpoint latency and max QPS under given hardware; measure with `wrk`/`locust`.

Reported results (from current run)
- Training: validation MAE decreased from ~18 down to ~13–14 across hyperparameter search (see `training_process.txt`).
- API: `test_api.py` shows successful integration:
  - `/predict_discount` responses: 25.44, 14.61 (see `results_for_test_api.txt`).
  - `/answer_question` returned contextual products with similarity scores in the responses.

---

## Artifacts delivered (in repository)
- `src/api/main.py` — Flask API with `/predict_discount` and `/answer_question`, lazy RAG loader.
- `src/api/model/` — FAISS index `product_index.faiss` (vector store).
- Model artifacts expected in `src/api/model/`:
  - `discount_predictor.pkl`
  - `model_columns.pkl`
  - `product_data.pkl`
- `src/scripts/train_model.py`, `src/scripts/create_vector_store.py`
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- `test_api.py` — endpoint test harness
- Logs: `training_process.txt`, `results_for_test_api.txt`

---

## Useful commands & quick checklist

Run API locally:
```bash
source venv/bin/activate
python src/api/main.py
```

Run tests (API must be running):
```bash
python test_api.py
```

Build and run with Docker:
```bash
docker compose build --no-cache
docker compose up
```

Check installed packages in container:
```bash
docker compose run --rm api pip list
docker compose run --rm api python -c "import numpy as np; print(np.__version__)"
```

Compute model evaluation (example script you can add):
```python
# eval_model.py - example
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model = joblib.load('src/api/model/discount_predictor.pkl')
cols = joblib.load('src/api/model/model_columns.pkl')
test = pd.read_csv('data/test.csv')  # add test split
X = test[cols]
y = test['target_discount']
pred = model.predict(X)
print('MAE', mean_absolute_error(y, pred))
print('RMSE', mean_squared_error(y, pred, squared=False))
print('R2', r2_score(y, pred))
```

---