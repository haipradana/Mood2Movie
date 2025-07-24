"""simple cosine similarity recommender"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from .config import EMBEDDING_FILE, CSV_FILE
from .embedder import embed_texts, _get_model

GENRE_PRIORITIES = {
    "calm": [10751, 18, 16, 10749],  # Family, Drama, Animation, Romance
    "soothing": [10751, 18, 16, 10749],
    "relaxing": [10751, 18, 16, 10749],
    "adrenaline": [28, 53],  # Action, Thriller
    "thrilling": [28, 53],
    "angry": [28, 53],
    "sad": [18, 10749, 10751],  # Drama, Romance, Family
    "emotional": [18, 10749, 10751, 16],
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

INCOMPATIBLE_GENRES = {
    "sad": [53, 27, 80],        # Avoid Thriller, Horror, Crime
    "emotional": [53, 27, 80],  
    "thrilling": [10751, 16],   # Avoid Family, Animation
    "suspense": [10751, 16],    
}

# fuzzywuzzy untuk menghitung kesamaan judul
def compute_title_similarity(query: str, title: str) -> float:
    ratio = fuzz.ratio(query.lower(), title.lower())
    partial_ratio = fuzz.partial_ratio(query.lower(), title.lower())
    return max(ratio, partial_ratio)/100.0

def compute_genre_score(genre_ids: str, tags: list[str]) -> float:
    if not tags:
        return 0.5
    
    score = 0
    genre_ids_list = eval(genre_ids)
    
    # Cek incompatible genres
    for tag in tags:
        incompatible = INCOMPATIBLE_GENRES.get(tag.lower(), [])
        if any(g in genre_ids_list for g in incompatible):
            return 0.1
        
    for tag in tags:
        tag_genres = GENRE_PRIORITIES.get(tag.lower(), [])
        matches = sum(1 for g in genre_ids_list if g in tag_genres)
        if matches > 0:
            score += (matches / len(tag_genres))
    return min(1.0, score / max(1, len(tags)))

class MovieRecommender:
    def __init__(self):
        self.movies = pd.read_csv(CSV_FILE)
        self.embs = np.load(EMBEDDING_FILE)
        self.title_embs = embed_texts(self.movies["title"].tolist())

    def recommend(self, query: str, tags: list[str] = [], top_k: int = 6):
        q_emb = embed_texts([query])[0].reshape(1, -1)


        overview_sims = cosine_similarity(q_emb, self.embs)[0]
        title_sims = cosine_similarity(q_emb, self.title_embs)[0]
        title_fuzzy = self.movies["title"].apply(
            lambda t: compute_title_similarity(query, t)
        )

        genre_scores = self.movies["genre_ids"].apply(
            lambda g: compute_genre_score(eval(g), tags)
            )
        
        # adaptive scoring based on title similarity
        if title_fuzzy.max() > 0.8:
            final_score = (
                0.5 * title_fuzzy + 0.2 * overview_sims + 0.3 * genre_scores
            )
        else:
            final_score = (
                0.2 * overview_sims + 0.45 * genre_scores + 0.35 * title_sims
            )

        top_i = np.argsort(-final_score)[:top_k]
        results = self.movies.iloc[top_i].copy()
        
        results["overview_sim"] = overview_sims[top_i]
        results["title_sim"] = title_sims[top_i]
        results["genre_score"] = genre_scores[top_i]
        results["final_score"] = final_score[top_i]

        return results.reset_index(drop=True)