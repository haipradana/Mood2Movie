import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# load .env in root dir
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "BELOM_ADA")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# uncomment this if you want to use local .env file
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_FILE = DATA_PATH / "movie_embeddings.npy"
TITLE_EMBEDDING_FILE = DATA_PATH / "title_embeddings.npy"
CSV_FILE = DATA_PATH / "movies_cache.csv"