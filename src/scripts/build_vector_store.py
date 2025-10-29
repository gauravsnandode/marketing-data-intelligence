
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

# Load the dataset
df = pd.read_csv('/Users/coditas/Desktop/Study/Assignments/HG/data/amazon.csv')

# Prepare the text for embedding
df['text'] = df['product_name'] + " " + df['about_product'].fillna('')
texts = df['text'].tolist()

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index and the texts
if not os.path.exists('/Users/coditas/Desktop/Study/Assignments/HG/src/api/model'):
    os.makedirs('/Users/coditas/Desktop/Study/Assignments/HG/src/api/model')

faiss.write_index(index, '/Users/coditas/Desktop/Study/Assignments/HG/src/api/model/vector_store.faiss')
with open('/Users/coditas/Desktop/Study/Assignments/HG/src/api/model/texts.pkl', 'wb') as f:
    pickle.dump(texts, f)

print("Vector store built and saved successfully.")
