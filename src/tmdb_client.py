"fetch and cache movie metadata from TMDB"
import time
import requests
import pandas as pd
from .config import TMDB_API_KEY, CSV_FILE, DATA_PATH

_BASE = "https://api.themoviedb.org/3"
_HEADERS = {"Accept": "application/json"}

def _get(endpoint: str, params: dict | None = None):
    if TMDB_API_KEY == "BELOM_ADA":
        raise ValueError("TMDB_API_KEY is not set")
    params = params or {}
    params.update({"api_key": TMDB_API_KEY, "language": "en-US",})
    r = requests.get(f"{_BASE}/{endpoint}", headers=_HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def fetch_popular(pages: int = 5) -> pd.DataFrame:
    """grab *pages* of popular movies"""
    movies = []
    for page in range(1, pages + 1):
        # res = _get("movie/popular", {"page": page})
        # movies.extend(res["results"])
        movies.extend(_get("movie/popular", {"page": page})["results"])
        movies.extend(_get("movie/top_rated", {"page": page})["results"])
        time.sleep(0.5)
    df = pd.json_normalize(movies)
    keep = ["id", "title", "overview", "genre_ids", "vote_average", "poster_path"]
    df = df[df["overview"].str.len() > 50].drop_duplicates("id")
    return df[keep]

def cache_movies(pages: int = 5):
    df = fetch_popular(pages)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"cached {len(df)} movies -> {CSV_FILE.relative_to(DATA_PATH.parent)}")