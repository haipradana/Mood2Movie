"""simple cosine similarity recommender"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .config import EMBEDDING_FILE, CSV_FILE
from .embedder import embed_texts, _get_model

GENRE_PRIORITIES = {
    "calm": [10751, 18, 16, 10749],  # Family, Drama, Animation, Romance
    "soothing": [10751, 18, 16, 10749],
    "relaxing": [10751, 18, 16, 10749],
    "adrenaline": [28, 53],  # Action, Thriller
    "thrilling": [28, 53],
    "angry": [28, 53],
    "sad": [18],                         # Drama
    "emotional": [18, 10749],            # Drama + Romance
    "heartwarming": [10751, 10749, 16],  # Family, Romance, Animation
    "funny": [35],                       # Comedy
    "romantic": [10749],                 # Romance
    "family": [10751],                   # Family
    "mystery": [9648],                   # Mystery
    "suspense": [53, 9648],              # Thriller, Mystery
    "violent": [28, 80],                 # Action, Crime
    "scary": [27, 9648],                 # Horror, Mystery
    "fantasy": [14, 16],                 # Fantasy, Animation
    "war": [10752],                      # War
    "sport": [99, 28],                   # Documentary or Action (for race themes)
    "epic": [12, 14],                    # Adventure, Fantasy
    "historical": [36],                  # History
    "space": [878, 12],                  # Sci-fi + Adventure
    "sci-fi": [878],
    "crime": [80],
    "documentary": [99],
    "anime": [16],
    "animated": [16],
    "action": [28],
    "drama": [18],
    "romance": [10749],
    "adventure": [12],
}

def compute_genre_score(genre_ids, tags):
    score = 0
    for tag in tags:
        for gid in GENRE_PRIORITIES.get(tag.lower(), []):
            if gid in genre_ids:
                score += 1
    return score / max(1, len(genre_ids))

class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv(CSV_FILE)
        self.embs = np.load(EMBEDDING_FILE)

    def recommend(self, query: str, tags: list[str] = [], top_k: int = 6):
        q_emb = embed_texts([query])[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self.embs)[0]
        genre_scores = self.movies["genre_ids"].apply(lambda g: compute_genre_score(eval(g), tags))
        final_score = 0.8 * sims + 0.2 * genre_scores
        top_i = np.argsort(-final_score)[:top_k]
        results = self.movies.iloc[top_i].copy()
        results["similarity"] = sims[top_i]
        results["genre_boost"] = genre_scores.iloc[top_i].values
        results["score"] = final_score[top_i]
        return results.reset_index(drop=True)