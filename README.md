# Mood2Movie : Film Recommender from Your Mood

A smart-simple movie recommender that understands your mood and suggest the perfect movies to watch. Built with semantic search, sentiment analysis, and mood-based filtering.

**Try it now: [mood2movie.streamlit.app](https://mood2movie.streamlit.app/)**

[Lihat Demo Video](https://github.com/haipradana/Mood2Movie/blob/main/demo.mp4)
<a href="https://youtu.be/_Q6pRmcKGks" target="_blank">
  <img src="https://github.com/haipradana/Mood2Movie/blob/main/screenshot.png?raw=true" width="25%">
</a>
## Apa Fiturnya?

- **Mood-Based Recommendations**: Rekomendasi film berdasarkan input mood atau ketertarikan
- **Natural Language Understanding**: Jelaskan aja apa yang kamu sedang rasakan atau yang ingin kamu lihat
- **Content Based Filtering**: Semantic search untuk optimasi rekomendasi berdasrkan pencocokan overview film dengan input menggunakan Sentence Transformer
- **AI-Enhanced Query**: query yang disempurnakan oleh LLM for "better understanding"

## Try It Out Bro!

### Opsi 1: Use the Deployed App (Recommended)

visit [mood2movie.streamlit.app](https://mood2movie.streamlit.app/) to use the app directly!

### Opsi 2: Run Locally

If you want to run the app locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Mood2Movie.git
cd Mood2Movie
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up API keys:

   - In `src/config.py`, uncomment the local environment section:

   ```python
   # COMMENT THIS
   # import streamlit as st
   # AND THIS
   # GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

   # THEN

   # Uncomment these lines
   # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
   # TMDB_API_KEY = os.getenv("TMDB_API_KEY")
   ```

   - Create `.env` file in root directory:

   ```env
   TMDB_API_KEY=your_tmdb_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. Build the movie database and embeddings:
   You can simply run all the notebook in notebook/test_pipeline.ipynb or

```bash
python -c "from src.tmdb_client import cache_movies; cache_movies()"
python -c "from src.embedder import build_embeddings; build_embeddings()"
```

5. Run the Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## API Keys Required

1. **TMDB API Key**:

   - Required for movie data collection
   - Get from [TMDB](https://www.themoviedb.org/)
   - For local setup: Add to `.env`
   - For deployment: Already configured in live app

2. **Google Gemini API Key**:
   - Required for query enhancement
   - Get from [Google AI Studio](https://makersuite.google.com/)
   - For local setup: Add to `.env`
   - For deployment: Already configured in live app

Thank you
