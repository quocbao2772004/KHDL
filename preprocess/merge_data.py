import pandas as pd

# --- paths ---
tmdb_path   = "/home/anonymous/code/KHDL/btl/web/backend/data/tmdb_cleaned.csv"
ratings_path = "/home/anonymous/code/KHDL/btl/MovieLens/ratings.csv"
links_path   = "/home/anonymous/code/KHDL/btl/MovieLens/links.csv"

# --- load ---
tmdb = pd.read_csv(tmdb_path)
ratings = pd.read_csv(ratings_path)
links = pd.read_csv(links_path)

# --- clean types ---
tmdb["tmdb_id"] = pd.to_numeric(tmdb["tmdb_id"], errors="coerce").astype("Int64")

links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
links = links.dropna(subset=["tmdbId"])  # chỉ giữ dòng có tmdbId

# --- map ratings -> tmdbId ---
rat = ratings.merge(
    links[["movieId", "tmdbId"]],
    on="movieId",
    how="inner"
)

rat = rat.dropna(subset=["tmdbId"])
rat["tmdbId"] = rat["tmdbId"].astype("Int64")

# --- join with your TMDB dataset ---
rat_tmdb = rat.merge(
    tmdb,
    left_on="tmdbId",
    right_on="tmdb_id",
    how="inner"
)

# --- results ---
print("ratings:", len(ratings))
print("after movieId->tmdbId mapping:", len(rat))
print("after join with your tmdb_cleaned:", len(rat_tmdb))

print("unique users:", rat_tmdb["userId"].nunique())
print("unique tmdb movies:", rat_tmdb["tmdb_id"].nunique())

# gợi ý: chỉ giữ cột cần cho CF + metadata
cols_keep = ["userId", "tmdb_id", "rating", "timestamp", "title", "genres", "overview"]
cols_keep = [c for c in cols_keep if c in rat_tmdb.columns]
rat_tmdb = rat_tmdb[cols_keep]

rat_tmdb.to_csv("/home/anonymous/code/KHDL/btl/web/backend/data/ratings_mapped_to_tmdb.csv",
                index=False, encoding="utf-8-sig")

print("Saved: ratings_mapped_to_tmdb.csv")
