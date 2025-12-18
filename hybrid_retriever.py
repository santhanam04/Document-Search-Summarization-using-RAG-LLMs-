def hybrid_retrieve(query, documents, embeddings, tfidf_engine, semantic_fn, top_k=5):
    sem_idx = semantic_fn(query, documents, embeddings, top_k*2)
    tfidf_idx = tfidf_engine.search(query, top_k*2)

    combined_idx = list(set(sem_idx.tolist() + tfidf_idx.tolist()))
    combined_idx = combined_idx[:top_k]

    return [documents[i] for i in combined_idx]
