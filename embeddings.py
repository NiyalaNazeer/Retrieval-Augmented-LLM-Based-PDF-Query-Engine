import os
import hashlib
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create a cache directory for embeddings
CACHE_DIR = "embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

def hash_text(text):
    """
    Generates an MD5 hash from the text content.
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def embed_text_chunks(chunks):
    """
    Embeds and caches embeddings using a hash of the combined text.

    Args:
        chunks (List[str]): List of text chunks

    Returns:
        List[List[float]]: List of embedding vectors as plain lists
    """
    # Combine all chunks into one string
    text_combined = "\n".join(chunks)
    hash_key = hash_text(text_combined)
    cache_file = os.path.join(CACHE_DIR, f"{hash_key}.json")

    # If cached, load and return
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Else compute embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)
    embeddings_list = [embedding.tolist() for embedding in embeddings]

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(embeddings_list, f)

    return embeddings_list
