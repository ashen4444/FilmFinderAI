import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../data/clean_movies.csv")
embeddings = np.load("../data/movie_embeddings.npy")

# keep only rows that have semantic_text, same as embedding script
df = df.dropna(subset=["semantic_text"]).copy().reset_index(drop=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def search_movies_by_description(query, top_k=5):
    query_embedding = model.encode([query], normalize_embeddings=True)

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = df.iloc[top_indices][["title", "genres", "overview", "release_date", "vote_average"]].copy()
    results["similarity_score"] = scores[top_indices]

    results["final_score"] = (
            results["similarity_score"] * 0.8 +
            (results["vote_average"].fillna(0) / 10) * 0.2
    )
    results = results.sort_values("final_score", ascending=False)

    return results

# test query
query = "a prison escape movie with a very intelligent main character"
results = search_movies_by_description(query)

print(results)

#for _, row in results.iterrows():
 #   print("\nTitle:", row["title"])
  #  print("Genres:", row["genres"])
   # print("Release Date:", row["release_date"])
 #   print("Vote Average:", row["vote_average"])
   # print("Score:", row["similarity_score"])
   # print("Overview:", row["overview"])




