import sys
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="Mood2Movie Film Recommender", page_icon=":movie_camera:", layout="wide")

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.recommender import MovieRecommender
from src.config import CSV_FILE, BASE_POSTER_URL, EMBEDDING_FILE
from src.gemini_enhancer import enhance_prompt

st.title("Mood2Movie - Find the perfect movie for your mood")

#Guard clauses
if not Path(CSV_FILE).exists() or not Path(EMBEDDING_FILE).exists():
    st.error("Data files not found. Please run the script to generate them.")
    st.stop()

recommender = MovieRecommender()

query_raw = st.text_input("What's your mood today? Or type what you want to watch", placeholder="I'm in a thrilling mood, maybe something of suspense and action")
if st.button("Recommend") and query_raw:
    with st.spinner("Understanding your mood ..."):
        enhanced_query, tags = enhance_prompt(query_raw)
        recs = recommender.recommend(enhanced_query, tags=tags, top_k=6)
    
    cols = st.columns(3)
    for i, row in recs.iterrows():
        with cols[i % 3]:
            if pd.notna(row.poster_path):
                st.image(f"{BASE_POSTER_URL}{row.poster_path}")
            st.subheader(row.title)
            st.text(f"⭐ {row.vote_average:.1f}/10 | score: {row.final_score:.2f}")
            overview_text = row.overview if pd.notna(row.overview) else ""
            st.caption(overview_text[:150] + "...")
