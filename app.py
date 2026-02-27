
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    return movies

def main():
    st.title("🎬 Movie Suggestion Engine")
    st.markdown("Enter a movie you liked, and I'll find 5 similar ones based on genres.")

    df = load_data()
    
    # Precompute Similarity (Cached for speed)
    tfidf = TfidfVectorizer(stop_words='english')
    df['genres_space'] = df['genres'].str.replace('|', ' ', regex=False)
    genre_matrix = tfidf.fit_transform(df['genres_space'])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    # UI Components
    movie_list = df['title'].values
    selected_movie = st.selectbox("Type or select a movie:", movie_list)

    if st.button('Recommend'):
        idx = df[df['title'] == selected_movie].index[0]
        distances = cosine_sim[idx]
        
        # Sort by similarity and get top 5
        rec_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.subheader(f"Top 5 picks for fans of '{selected_movie}':")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                movie_name = df.iloc[rec_indices[i][0]]['title']
                genre_name = df.iloc[rec_indices[i][0]]['genres']
                st.info(f"**{movie_name}**")
                st.caption(f"_{genre_name}_")

if __name__ == "__main__":
    main()