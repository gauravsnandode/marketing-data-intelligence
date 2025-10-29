import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
import os

print("Starting the process to create the vector store...")

# --- 1. Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/amazon.csv')
MODEL_NAME = 'all-MiniLM-L6-v2'  # A good starting model for sentence embeddings

# Define output paths within the api/model directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../api/model/')
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, 'product_index.faiss')
PRODUCT_DATA_PATH = os.path.join(OUTPUT_DIR, 'product_data.pkl')

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Load and Prepare Data ---
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Successfully loaded data from {DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

# Combine relevant text fields to create a comprehensive description for embedding
df['combined_text'] = df['product_name'].fillna('') + " | " + df['about_product'].fillna('')
documents = df['combined_text'].tolist()

# Store the original data that corresponds to each document for later retrieval
# --- Normalize price field to a numeric value (strip currency symbols and commas) ---
def _parse_price(x):
    try:
        s = str(x)
        s = s.replace('₹', '').replace(',', '').strip()
        return float(s) if s not in ['', 'nan', 'None'] else None
    except Exception:
        return None

df['actual_price_numeric'] = df['actual_price'].apply(_parse_price)

# --- Split category into main and sub categories for convenient filtering ---
category_levels = df['category'].fillna('').str.split('|', expand=True)
df['main_category'] = category_levels[0].fillna('').astype(str)
for i in range(1, 7):
    col = f'sub_category_{i}'
    if i < category_levels.shape[1]:
        df[col] = category_levels[i].fillna('').astype(str)
    else:
        df[col] = ''

product_data = df[['product_id', 'product_name', 'category', 'main_category', 'sub_category_1', 'sub_category_2', 'sub_category_3', 'actual_price', 'actual_price_numeric', 'about_product']].to_dict(orient='records')

# --- 3. Generate Embeddings ---
print(f"Loading sentence transformer model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for product data... (This may take a while)")
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32') # FAISS requires float32

# --- 4. Build and Save FAISS Index ---
d = embeddings.shape[1]  # Get the dimension of the embeddings
index = faiss.IndexFlatL2(d)  # Using a simple L2 distance index
index.add(embeddings)

print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
faiss.write_index(index, d subcategorie)

print(f"Saving product data mapping to {PRODUCT_DATA_PATH}")
joblib.dump(product_data, PRODUCT_DATA_PATH)

print("\nVector store creation complete!")