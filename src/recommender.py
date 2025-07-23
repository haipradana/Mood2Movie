"""simple cosine similarity recommender"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import EMBEDDING_FILE, CSV_FILE
from .embedder import embed_texts, _get_model

class MovieRecommender:
    def __init__(self):
        self.movies: pd.DataFrame = pd.read_csv(CSV_FILE)
        self.embeddings: np.ndarray = np.load(EMBEDDING_FILE)

    def recommend(self, query: str, top_k: int = 6):
        q_embd = embed_texts([query])[0].reshape(1, -1)
        sims = cosine_similarity(q_embd, self.embeddings)[0]
        top_i = np.argsort(-sims)[:top_k]
        result = self.movies.iloc[top_i].copy()
        result["similarity"] = sims[top_i]
        return result.reset_index(drop=True)