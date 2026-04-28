import numpy as np
import pandas as pd


def normalize_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def cosine_top_k(query_embedding, gallery_embeddings, gallery_metadata, top_k=5):
    gallery_embeddings = normalize_embeddings(gallery_embeddings)
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    query_embedding = normalize_embeddings(query_embedding)[0]

    similarities = gallery_embeddings @ query_embedding
    top_k = min(int(top_k), len(similarities))
    indices = np.argsort(-similarities)[:top_k]

    rows = []
    for rank, idx in enumerate(indices, start=1):
        item = dict(gallery_metadata[idx])
        item.update({"rank": rank, "similarity": float(similarities[idx])})
        rows.append(item)
    return pd.DataFrame(rows)
