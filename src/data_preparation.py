import pandas as pd

# File paths
movies_path = "../data/ml-latest-small/movies.csv"
tags_path = "../data/ml-latest-small/tags.csv"

# Load datasets
movies = pd.read_csv(movies_path)
tags = pd.read_csv(tags_path)

# ---------------------------
# Explore datasets
# ---------------------------
print("Movies Shape:", movies.shape)
print("Tags Shape:", tags.shape)

print("\nMovies Columns:", movies.columns.tolist())
print("Tags Columns:", tags.columns.tolist())

print("\nMissing Values in Movies:")
print(movies.isnull().sum())

print("\nMissing Values in Tags:")
print(tags.isnull().sum())

print("\nMovies Preview:")
print(movies.head())

print("\nTags Preview:")
print(tags.head())

# ---------------------------
# Clean movies data
# ---------------------------
movies["genres"] = movies["genres"].fillna("")
movies["genres"] = movies["genres"].replace("(no genres listed)", "")
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# ---------------------------
# Clean tags data
# ---------------------------
tags = tags[["movieId", "tag"]]
tags = tags.dropna(subset=["tag"])
tags["tag"] = tags["tag"].str.lower().str.strip()
tags = tags[tags["tag"] != ""]

# ---------------------------
# Group tags by movie
# ---------------------------
grouped_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
grouped_tags.rename(columns={"tag": "all_tags"}, inplace=True)

# ---------------------------
# Merge movies and tags
# ---------------------------
movies_with_tags = pd.merge(movies, grouped_tags, on="movieId", how="left")
movies_with_tags["all_tags"] = movies_with_tags["all_tags"].fillna("")

# ---------------------------
# Create combined feature column
# ---------------------------
movies_with_tags["features"] = (
    movies_with_tags["genres"] + " " + movies_with_tags["all_tags"]
)

# Text cleanup
movies_with_tags["features"] = movies_with_tags["features"].str.lower()
movies_with_tags["features"] = movies_with_tags["features"].str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
movies_with_tags["features"] = movies_with_tags["features"].str.replace(r"\s+", " ", regex=True).str.strip()

# ---------------------------
# Final dataframe
# ---------------------------
final_movies_df = movies_with_tags[["movieId", "title", "features"]].copy()
final_movies_df = final_movies_df[final_movies_df["features"] != ""]

print("\nFinal Dataset Preview:")
print(final_movies_df.head(10))

print("\nFinal Dataset Shape:", final_movies_df.shape)

# Save cleaned data
final_movies_df.to_csv("../data/cleaned_movies.csv", index=False)
print("\nCleaned dataset saved as data/cleaned_movies.csv")