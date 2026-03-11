import pandas as pd
import ast

df = pd.read_csv("../data/movies_metadata.csv", low_memory=False)

def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return " ".join([g["name"] for g in genres])
    except:
        return ""

df["genres"] = df["genres"].apply(extract_genres)

print(df[["title", "genres"]].head(10))

df["semantic_text"] = (
    "Title: " + df["title"] +
    ". Genres: " + df["genres"] +
    ". Overview: " + df["overview"]
)

pd.set_option('display.max_colwidth', None)
print(df[["semantic_text"]].head(3))

df = df[
    [
        "title",
        "genres",
        "overview",
        "semantic_text",
        "release_date",
        "vote_average",
        "vote_count",
        "popularity"
    ]
]
df.to_csv("../data/clean_movies.csv", index=False)
print("Saved: ../data/clean_movies.csv")