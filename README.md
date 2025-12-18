📄 Document Search & Summarization using RAG (LLMs)
🚀 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that enables efficient document search and summarization using Large Language Models (LLMs).

The system combines traditional information retrieval techniques, semantic embeddings, and LLM-based summarization, all exposed through a Streamlit web interface.

This project was built as part of a GenAI / LLM interview assignment and follows industry-standard design principles.

✨ Key Features

📥 Document Ingestion – Supports PDF and TXT files

🧹 Text Cleaning & Preprocessing

🔎 Hybrid Retrieval

TF-IDF (keyword-based)

Sentence-Transformer embeddings (semantic search)

🧠 LLM-Based Summarization

Adjustable summary length (short / medium / detailed)

📊 Evaluation

ROUGE-1, ROUGE-2, ROUGE-L metrics

🖥️ Interactive UI

Built using Streamlit

🚀 Scalable Architecture

FAISS-ready vector indexing

🧱 System Architecture
User Query
   ↓
Hybrid Retrieval (TF-IDF + Embeddings)
   ↓
Top-N Relevant Documents
   ↓
LLM Summarization
   ↓
ROUGE Evaluation

🛠️ Technology Stack

Python

Sentence Transformers – all-MiniLM-L6-v2

Hugging Face Transformers – facebook/bart-large-cnn

Scikit-learn – TF-IDF

FAISS – Scalable vector search (optional)

Streamlit – Web interface

ROUGE Score – Summary evaluation

📁 Project Structure
rag-document-search/
│
├── app.py                  # Streamlit application
├── ingestion.py            # Document loading & cleaning
├── embeddings.py           # Embedding generation
├── tfidf_search.py         # Keyword-based search
├── hybrid_retriever.py     # Hybrid retrieval logic
├── summarizer.py           # LLM summarization
├── evaluation.py           # ROUGE evaluation
├── requirements.txt
├── README.md
│
└── data/
    └── raw/                # Input PDF/TXT documents

⚙️ Installation & Usage
1️⃣ Clone Repository
git clone https://github.com/<your-username>/rag-document-search.git
cd rag-document-search

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Documents

Place PDF or TXT files inside:

data/raw/

4️⃣ Run the Application
streamlit run app.py

🧪 Evaluation Methodology

Search Relevance: Verified via correct document retrieval

Summary Quality: Evaluated using:

ROUGE-1 (unigram overlap)

ROUGE-2 (bigram overlap)

ROUGE-L (longest common subsequence)

📌 Short summaries naturally have lower recall and higher precision.
