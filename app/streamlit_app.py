"""frontend streamlit"""
import streamlit as st
from pathlib import Path
import pandas as pd
from src.recommender import MovieRecommender
from src.config import CSV_FILE, BASE_POSTER_URL, EMBEDDING_FILE

st.set_page_config(page_title="Mood2Movie Film Recommender", page_icon=":movie_camera:", layout="wide")
st.title("Mood2Movie - Find the perfect movie for your mood")

#Guard clauses
if not Path(CSV_FILE).exists() or not Path(EMBEDDING_FILE).exists():
    st.error("Data files not found. Please run the script to generate them.")
    st.stop()

recommender = MovieRecommender()

query = st.text_input(
    "Enter your mood or a movie title to get recommendations",
    placeholder="action film that makes me laugh",
)

if st.button("Find Movies") and query:
    with st.spinner("Finding movies for you..."):
        recs = recommender.recommend(query, top_k=6)
    
    cols = st.columns(3)
    for i, row in recs.iterrows():
        with cols[i % 3]:
            if pd.notna(row.poster_url):
                st.image(f"{BASE_POSTER_URL}{row.poster_path}"),
            st.subheader(row.title)
            st.text(f"⭐ {row.vote_average}/10 | Similarity: {row.similarity:.2f}")
            st.caption(row.overview[:150] + "...")
