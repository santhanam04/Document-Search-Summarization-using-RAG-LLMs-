import streamlit as st
from ingestion import load_documents
from embeddings import build_embedding_store, semantic_search
from tfidf_search import TFIDFSearch
from hybrid_retriever import hybrid_retrieve
from summarizer import summarize
from evaluate import rouge_scores   # ✅ FIXED IMPORT

st.set_page_config(page_title="RAG Search", layout="centered")
st.title("📄 Document Search & Summarization (RAG)")

# Load documents
docs = load_documents("data/raw")
if not docs:
    st.error("No documents found in data/raw")
    st.stop()

# Build stores
embeddings = build_embedding_store(docs)
tfidf_engine = TFIDFSearch(docs)

# User input
query = st.text_input("Enter query")
length = st.selectbox("Summary length", ["short", "medium", "detailed"])

# Search
if st.button("Search"):
    if not query.strip():
        st.warning("Enter a query")
    else:
        results = hybrid_retrieve(
            query,
            docs,
            embeddings,
            tfidf_engine,
            semantic_search
        )

        summary = summarize([r["text"] for r in results], length)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Top Documents")
        for r in results:
            st.write(r["id"])

        # ✅ Evaluation (INSIDE SEARCH BLOCK)
        st.subheader("Evaluation (ROUGE Scores)")
        reference = " ".join([r["text"] for r in results])[:1000]
        scores = rouge_scores(reference, summary)
        st.json(scores)
