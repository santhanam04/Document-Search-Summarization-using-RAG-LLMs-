from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFSearch:
    def __init__(self, documents):
        self.texts = [d["text"] for d in documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().squeeze()
        return scores.argsort()[::-1][:top_k]
