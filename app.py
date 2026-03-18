import streamlit as st
import pandas as pd
import numpy as np

from src.semantic_search import search_movies_by_description, recommend_movies_by_title

st.set_page_config(page_title="FilmFinder AI", page_icon="🎬", layout="wide")


@st.cache_data
def load_movies():
    return pd.read_csv("data/clean_movies.csv")


@st.cache_resource
def load_embeddings():
    return np.load("data/movie_embeddings.npy")


movies_df = load_movies()
movie_embeddings = load_embeddings()

st.title("🎬 FilmFinder AI")
st.subheader("An Intelligent Movie Discovery and Recommendation System")
st.write("Discover movies from natural language descriptions or get recommendations from a movie title.")

tab1, tab2 = st.tabs(["🔍 Search by Description", "🎯 Recommend by Title"])

with tab1:
    st.markdown("### Describe a movie you remember")

    description_query = st.text_area(
        "Enter a movie description",
        placeholder="Example: a prison escape movie with a very intelligent main character",
        height=120
    )

    top_k_search = st.slider("Number of results", 5, 20, 5, key="search_slider")

    if st.button("Search Movies"):
        if description_query.strip():
            results = search_movies_by_description(
                description_query,
                movies_df,
                movie_embeddings,
                top_k=top_k_search
            )

            st.success(f"Found {len(results)} matching movies.")

            for _, row in results.iterrows():
                st.markdown(f"## 🎬 {row['title']}")
                st.write(f"**Genres:** {row['genres']}")
                st.write(f"**Release Date:** {row['release_date']}")
                st.write(f"**Vote Average:** {row['vote_average']}")
                st.write(f"**Vote Count:** {int(row['vote_count']) if pd.notna(row['vote_count']) else 0}")
                st.write(f"**Popularity:** {row['popularity']:.2f}")
                st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
                st.write(f"**Final Score:** {row['final_score']:.4f}")
                st.write(f"**Overview:** {row['overview']}")
                st.markdown("---")
        else:
            st.warning("Please enter a movie description.")

with tab2:
    st.markdown("### Recommend movies from a title")

    movie_title = st.text_input(
        "Enter a movie title",
        placeholder="Example: Interstellar"
    )

    top_k_recommend = st.slider("Number of recommendations", 5, 20, 5, key="recommend_slider")

    if st.button("Recommend Movies"):
        if movie_title.strip():
            results = recommend_movies_by_title(
                movie_title,
                movies_df,
                movie_embeddings,
                top_k=top_k_recommend
            )

            if results is None:
                st.error("Movie title not found in dataset.")
            else:
                st.success(f"Showing recommendations similar to '{movie_title}'")

                for _, row in results.iterrows():
                    st.markdown(f"## 🎬 {row['title']}")
                    st.write(f"**Genres:** {row['genres']}")
                    st.write(f"**Release Date:** {row['release_date']}")
                    st.write(f"**Vote Average:** {row['vote_average']}")
                    st.write(f"**Vote Count:** {int(row['vote_count']) if pd.notna(row['vote_count']) else 0}")
                    st.write(f"**Popularity:** {row['popularity']:.2f}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.4f}")
                    st.write(f"**Final Score:** {row['final_score']:.4f}")
                    st.write(f"**Overview:** {row['overview']}")
                    st.markdown("---")
        else:
            st.warning("Please enter a movie title.")