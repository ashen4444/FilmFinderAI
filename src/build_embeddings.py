import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/clean_movies.csv")

# remove rows where semantic_text is missing
df = df.dropna(subset=["semantic_text"]).copy()

# force everything to string
texts = df["semantic_text"].astype(str).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings = True
)

np.save("../data/movie_embeddings.npy", embeddings)

print("Embeddings saved to data/movie_embeddings.npy")
print("Embedding shape:", embeddings.shape)