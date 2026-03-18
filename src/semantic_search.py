import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def prepare_movies_dataframe(movies_df):
    """
    Keep only rows that have semantic_text, same as embedding generation step.
    Reset index so dataframe rows align correctly with embeddings.
    """
    return movies_df.dropna(subset=["semantic_text"]).copy().reset_index(drop=True)


def compute_final_scores(results_df):
    """
    Build a stronger ranking score using:
    - semantic similarity
    - vote_average + vote_count
    - popularity
    """
    results = results_df.copy()

    results["vote_average"] = pd.to_numeric(results["vote_average"], errors="coerce").fillna(0)
    results["vote_count"] = pd.to_numeric(results["vote_count"], errors="coerce").fillna(0)
    results["popularity"] = pd.to_numeric(results["popularity"], errors="coerce").fillna(0)

    results["vote_score"] = results["vote_average"] * np.log1p(results["vote_count"])

    max_vote_score = results["vote_score"].max()
    if pd.notna(max_vote_score) and max_vote_score > 0:
        results["vote_score"] = results["vote_score"] / max_vote_score
    else:
        results["vote_score"] = 0.0

    max_popularity = results["popularity"].max()
    if pd.notna(max_popularity) and max_popularity > 0:
        results["popularity_score"] = results["popularity"] / max_popularity
    else:
        results["popularity_score"] = 0.0

    results["final_score"] = (
        results["similarity_score"] * 0.6 +
        results["vote_score"] * 0.25 +
        results["popularity_score"] * 0.15
    )

    return results


def search_movies_by_description(query, movies_df, movie_embeddings, top_k=5):
    movies_df = prepare_movies_dataframe(movies_df)

    query_embedding = model.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, movie_embeddings)[0]

    candidate_count = min(max(top_k * 5, 25), len(movies_df))
    top_indices = np.argsort(scores)[::-1][:candidate_count]

    results = movies_df.iloc[top_indices][
        [
            "title",
            "genres",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
        ]
    ].copy()

    results["similarity_score"] = scores[top_indices]
    results = compute_final_scores(results)
    results = results.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

    return results


def recommend_movies_by_title(title, movies_df, movie_embeddings, top_k=5):
    movies_df = prepare_movies_dataframe(movies_df)

    matched_movies = movies_df[movies_df["title"].str.lower() == title.lower()]

    if matched_movies.empty:
        return None

    movie_index = matched_movies.index[0]
    target_embedding = movie_embeddings[movie_index].reshape(1, -1)
    scores = cosine_similarity(target_embedding, movie_embeddings)[0]

    results = movies_df[
        [
            "title",
            "genres",
            "overview",
            "release_date",
            "vote_average",
            "vote_count",
            "popularity",
        ]
    ].copy()

    results["similarity_score"] = scores
    results = results.drop(index=movie_index).reset_index(drop=True)

    scores_without_self = np.delete(scores, movie_index)
    results["similarity_score"] = scores_without_self

    results = compute_final_scores(results)
    results = results.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

    return results


if __name__ == "__main__":
    df = pd.read_csv("data/clean_movies.csv")
    embeddings = np.load("data/movie_embeddings.npy")

    query = "a prison escape movie with a very intelligent main character"
    results = search_movies_by_description(query, df, embeddings, top_k=5)

    print(results)




