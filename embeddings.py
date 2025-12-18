import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_embedding_store(documents):
    texts = [d["text"] for d in documents]
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def semantic_search(query, documents, embeddings, top_k=5):
    query_emb = model.encode([query], normalize_embeddings=True)
    scores = np.dot(embeddings, query_emb.T).squeeze()
    top_idx = scores.argsort()[::-1][:top_k]
    return top_idx
