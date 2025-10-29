import os

# --- Set thread/environment variables before importing heavy native libs ---
# Limit threads for OpenMP/BLAS/MKL to avoid runtime conflicts that can cause segfaults
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Disable tokenizers parallelism early
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
# Suppress LibreSSL/OpenSSL compatibility warning emitted during urllib3 import
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

from urllib3.exceptions import NotOpenSSLWarning
# Suppress the specific urllib3 NotOpenSSLWarning category
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Mitigate segmentation fault by importing torch and setting thread count first.
# This can help avoid conflicts between libraries that use OpenMP.
import torch
torch.set_num_threads(1)

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
# NOTE: heavy ML libs (faiss, sentence_transformers) will be lazy-loaded when needed

# Define the path to the model artifacts
MODEL_DIR = os.path.dirname(__file__)
ARTIFACTS_PATH = os.path.join(MODEL_DIR, "model")

# --- 1. Load Models and Artifacts at Startup (lightweight) ---
# Load only the discount prediction model at startup. RAG resources are loaded lazily.
try:
    # Load discount prediction model
    discount_model = joblib.load(os.path.join(ARTIFACTS_PATH, "discount_predictor.pkl"))
    model_columns = joblib.load(os.path.join(ARTIFACTS_PATH, "model_columns.pkl"))
    print("Discount prediction model loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading discount model artifacts: {e}")
    print("Please ensure you have run 'train_model.py' successfully.")
    discount_model = None
    model_columns = None

# RAG resources will be lazily loaded on first request
rag_index = None
product_data = None
embedding_model = None

def _load_rag_resources():
    """Lazy-load FAISS index, product data, and embedding model.

    This function is safe to call multiple times; it will load resources only once.
    """
    global rag_index, product_data, embedding_model
    if rag_index is not None and product_data is not None and embedding_model is not None:
        return

    try:
        # Import heavy libs only when needed
        import faiss
        from sentence_transformers import SentenceTransformer

        # Load RAG artifacts
        rag_index = faiss.read_index(os.path.join(ARTIFACTS_PATH, "product_index.faiss"))
        product_data = joblib.load(os.path.join(ARTIFACTS_PATH, "product_data.pkl"))
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("RAG artifacts and embedding model loaded successfully (lazy load).")

    except FileNotFoundError as e:
        print(f"Error loading RAG artifacts: {e}")
        rag_index = None
        product_data = None
        embedding_model = None
    except Exception as e:
        # Log any other exception but don't crash the app at startup
        print(f"Failed to lazy-load RAG resources: {e}")
        rag_index = None
        product_data = None
        embedding_model = None

# --- 2. Create Flask App ---
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['TITLE'] = "Marketing Intelligence API"
app.config['DESCRIPTION'] = "API for predicting product discounts and answering product questions."
app.config['VERSION'] = "1.0.0"

# --- 3. Define API Endpoints ---
@app.route("/")
def index():
    return jsonify(
    title="Marketing Intelligence API",
    description="API for predicting product discounts and answering product questions.",
    version="1.0.0"
)

@app.route("/predict_discount", methods=["POST"])
def predict_discount():
    """
    Predicts the discount percentage for a product based on its features.
    """
    if not discount_model or not model_columns:
        return jsonify({"error": "Discount model is not available. Please check server logs."}), 503

    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

    try:
        # Convert input data to a pandas DataFrame, ensuring column order matches the model's training data
        input_df = pd.DataFrame([input_data], columns=model_columns)
    except (KeyError, ValueError):
        return jsonify({"error": f"Invalid input format. Required fields: {model_columns}"}), 400

    prediction = discount_model.predict(input_df)
    print("prediction: {}".format(prediction))

    # Convert numpy/scalar types to native Python types for JSON serialization
    try:
        pred_value = float(prediction[0])
    except Exception:
        # Fallback: use item() if available
        pred_value = float(getattr(prediction[0], "item", lambda: prediction[0])())

    return jsonify({"predicted_discount_percentage": round(pred_value, 2)})

@app.route("/answer_question", methods=["POST"])
def answer_question():
    """
    Retrieves relevant product information to answer a user's query using a RAG system.
    """
    # Ensure RAG resources are available; load them lazily if needed
    _load_rag_resources()
    if not rag_index or not embedding_model or not product_data:
        return jsonify({"error": "RAG system is not available. Please check server logs."}), 503

    input_data = request.get_json()
    if not input_data or 'query' not in input_data:
        return jsonify({"error": "Invalid input: 'query' field is required."}), 400

    query = input_data['query']
    top_k = input_data.get('top_k', 3) # Default to 3 if not provided

    # 1. Encode the user's query
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')

    # 2. Search the FAISS index for the top_k most similar products
    distances, indices = rag_index.search(query_embedding, top_k)

    # 3. Retrieve the corresponding product data
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        # The 'product_data' is a list of dicts, so we can access by index
        # Ensure similarity score is a native float for JSON serialization
        sim_score = float(1 - distances[0][i])
        results.append({
            "retrieved_product": product_data[idx],
            "similarity_score": sim_score
        })

    # The 'results' list contains the context that would be passed to an LLM.
    # For now, we return this context directly.
    return jsonify({"query": query, "retrieved_context": results})

# --- 4. Run the Flask App ---
if __name__ == "__main__":
    # Use host='0.0.0.0' to make it accessible from outside the container
    app.run(host='0.0.0.0', port=8000, debug=True)