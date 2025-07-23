"""load embedder for movie overview"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .config import EMBEDDING_MODEL, CSV_FILE, EMBEDDING_FILE

_model: SentenceTransformer | None = None #global singleton

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list[str], batch_size: int = 32):
    model = _get_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

def build_embeddings(force: bool = False):
    if EMBEDDING_FILE.exists() and not force:
        print(f"Embeddings already exist in {EMBEDDING_FILE}")
        return
    df = pd.read_csv(CSV_FILE)
    embs = embed_texts(df["overview"]).fillna("").tolist()
    np.save(EMBEDDING_FILE, embs)
    print(f"Embeddings saved to {EMBEDDING_FILE}")