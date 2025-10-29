
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from pathlib import Path

# Resolve repository root (two levels up from scripts directory)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Paths (relative to repo root)
DATA_PATH = REPO_ROOT / 'data' / 'amazon.csv'
OUTPUT_DIR = REPO_ROOT / 'src' / 'api' / 'model'
VECTOR_STORE_PATH = OUTPUT_DIR / 'vector_store.faiss'
TEXTS_PKL_PATH = OUTPUT_DIR / 'texts.pkl'

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Prepare the text for embedding
df['text'] = df['product_name'] + " " + df['about_product'].fillna('')
texts = df['text'].tolist()

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index and the texts
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

faiss.write_index(index, str(VECTOR_STORE_PATH))
with open(TEXTS_PKL_PATH, 'wb') as f:
    pickle.dump(texts, f)

print(f"Vector store built and saved successfully to {OUTPUT_DIR}")
