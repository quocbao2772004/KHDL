import pandas as pd

rat = pd.read_csv("/home/anonymous/code/KHDL/btl/web/backend/data/ratings_mapped_to_tmdb.csv")

rat_cf = rat[["userId","tmdb_id","rating","timestamp"]].copy()

# giảm memory
rat_cf["userId"] = rat_cf["userId"].astype("int32")
rat_cf["tmdb_id"] = rat_cf["tmdb_id"].astype("int32")
rat_cf["rating"] = rat_cf["rating"].astype("float32")
rat_cf["timestamp"] = rat_cf["timestamp"].astype("int64")

rat_cf.to_parquet("/home/anonymous/code/KHDL/btl/web/backend/data/ratings_cf.parquet", index=False)

print("Saved ratings_cf.parquet:", len(rat_cf))
