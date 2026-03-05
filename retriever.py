import faiss
import numpy as np


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def retrieve_top_k(query_embedding, index, chunks, k=3):
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results
