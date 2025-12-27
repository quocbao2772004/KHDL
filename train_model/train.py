import os
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from surprise import Dataset, Reader, SVD, accuracy


# =========================
# CONFIG
# =========================
RATINGS_PARQUET = "/home/anonymous/code/KHDL/btl/web/backend/data/ratings_cf.parquet"
OUT_DIR = "/home/anonymous/code/KHDL/btl/web/backend/models"
MODEL_PATH = os.path.join(OUT_DIR, "svd_cf_best.joblib")
META_PATH  = os.path.join(OUT_DIR, "svd_cf_meta.joblib")

# Giữ đủ phim nhưng giảm số dòng để khỏi kill
# Mỗi phim lấy tối đa CAP_PER_MOVIE ratings (vẫn không mất phim!)
CAP_PER_MOVIE = 4000   # 2000/3000/4000/5000 tuỳ RAM. 16GB thường 3000-5000 OK

# Split ratio
TEST_SIZE  = 0.20
VALID_SIZE = 0.10

# Metrics
K = 10
THRESHOLD = 4.0
RATING_SCALE = (0.5, 5.0)

# Train params (đủ ngon + không quá nặng)
# (mày đã chạy ra best ~ 100 factors, 20 epochs -> giữ luôn)
SVD_PARAMS = dict(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

# Nếu mày muốn grid nhỏ để chọn params (nặng hơn), bật True
USE_GRID = False
GRID = [
    dict(n_factors=50,  n_epochs=10, lr_all=0.005, reg_all=0.02, random_state=42),
    dict(n_factors=80,  n_epochs=15, lr_all=0.005, reg_all=0.02, random_state=42),
    dict(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
]


# =========================
# UTILS
# =========================
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Precision@K & Recall@K cho recommender.
    Relevant: true rating >= threshold
    Recommended: (trong top K) predicted >= threshold
    """
    user_est_true = defaultdict(list)
    for p in predictions:
        user_est_true[p.uid].append((p.est, p.r_ui))

    precisions, recalls = [], []
    for uid, est_true in user_est_true.items():
        est_true.sort(key=lambda x: x[0], reverse=True)
        top_k = est_true[:k]

        n_rel = sum(true_r >= threshold for (_, true_r) in est_true)
        n_rec_k = sum(est >= threshold for (est, _) in top_k)
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for (est, true_r) in top_k)

        if n_rec_k > 0:
            precisions.append(n_rel_and_rec_k / n_rec_k)
        if n_rel > 0:
            recalls.append(n_rel_and_rec_k / n_rel)

    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    return p, r


def cap_ratings_keep_all_movies(df, cap_per_movie=4000, seed=42):
    """
    Giữ đủ phim (tmdb_id) nhưng giới hạn số rating mỗi phim.
    Không mất phim, giảm số dòng => nhẹ RAM + train nhanh.
    """
    # groupby sample theo phim (tmdb_id)
    # NOTE: apply sẽ tốn chút thời gian nhưng RAM ổn
    return (df.groupby("tmdb_id", group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), cap_per_movie), random_state=seed))
              .reset_index(drop=True))


def split_indices(n, test_size=0.2, valid_size=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(test_size * n)
    n_valid = int(valid_size * n)

    test_idx = idx[:n_test]
    valid_idx = idx[n_test:n_test + n_valid]
    train_idx = idx[n_test + n_valid:]
    return train_idx, valid_idx, test_idx


def build_trainset_and_evalsets(uids, iids, rats, train_idx, valid_idx, test_idx, rating_scale=(0.5, 5.0)):
    """
    RAM-friendly:
      - build trainset 1 lần từ train_df nhỏ (3 cột)
      - valid/test là list tuples (uid, iid, r) -> algo.test() dùng trực tiếp
    """
    reader = Reader(rating_scale=rating_scale)

    train_df = pd.DataFrame({
        "userId": uids[train_idx],
        "itemId": iids[train_idx],
        "rating": rats[train_idx],
    })

    train_data = Dataset.load_from_df(train_df, reader)
    trainset = train_data.build_full_trainset()

    validset = list(zip(uids[valid_idx], iids[valid_idx], rats[valid_idx]))
    testset  = list(zip(uids[test_idx],  iids[test_idx],  rats[test_idx]))

    return trainset, validset, testset


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading parquet:", RATINGS_PARQUET)
    df = pd.read_parquet(RATINGS_PARQUET, columns=["userId", "tmdb_id", "rating"])
    df = df.dropna()

    # dtype nhẹ
    df["userId"] = df["userId"].astype("int32")
    df["tmdb_id"] = df["tmdb_id"].astype("int32")
    df["rating"] = df["rating"].astype("float32")

    print("Original rows:", len(df))
    print("Users:", df["userId"].nunique(), "Movies:", df["tmdb_id"].nunique())

    # cap per movie để không kill RAM nhưng vẫn giữ đủ phim
    if CAP_PER_MOVIE is not None:
        print(f"Capping ratings per movie to <= {CAP_PER_MOVIE} (keep all movies)...")
        df = cap_ratings_keep_all_movies(df, cap_per_movie=CAP_PER_MOVIE, seed=42)

    print("Rows after cap:", len(df))
    print("Users:", df["userId"].nunique(), "Movies:", df["tmdb_id"].nunique())

    # Convert to numpy arrays (string ids cho Surprise)
    uids = df["userId"].astype(str).to_numpy()
    iids = df["tmdb_id"].astype(str).to_numpy()
    rats = df["rating"].to_numpy()

    # Split indices
    n = len(df)
    train_idx, valid_idx, test_idx = split_indices(n, TEST_SIZE, VALID_SIZE, seed=42)
    print(f"Split sizes: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

    # Build trainset (1 lần) + valid/test lists
    trainset, validset, testset = build_trainset_and_evalsets(
        uids, iids, rats, train_idx, valid_idx, test_idx, rating_scale=RATING_SCALE
    )

    best_algo = None
    best_params = None
    best_valid_rmse = 1e9

    if USE_GRID:
        print("\n=== GRID SEARCH (VALID RMSE) ===")
        for params in GRID:
            algo = SVD(**params)
            algo.fit(trainset)

            preds_valid = algo.test(validset)
            rmse_v = accuracy.rmse(preds_valid, verbose=False)
            mae_v  = accuracy.mae(preds_valid, verbose=False)
            print("Params:", params, "| Valid RMSE:", f"{rmse_v:.4f}", "| MAE:", f"{mae_v:.4f}")

            if rmse_v < best_valid_rmse:
                best_valid_rmse = rmse_v
                best_algo = algo
                best_params = params
    else:
        print("\n=== TRAIN ONE MODEL ===")
        best_params = SVD_PARAMS
        best_algo = SVD(**best_params)
        best_algo.fit(trainset)

        preds_valid = best_algo.test(validset)
        best_valid_rmse = accuracy.rmse(preds_valid, verbose=False)
        mae_v = accuracy.mae(preds_valid, verbose=False)
        print("Valid RMSE:", f"{best_valid_rmse:.4f}", "| MAE:", f"{mae_v:.4f}")

    print("\n=== BEST PARAMS ===")
    print(best_params)
    print("Best Valid RMSE:", best_valid_rmse)

    # Final test
    preds_test = best_algo.test(testset)
    rmse = accuracy.rmse(preds_test, verbose=False)
    mae  = accuracy.mae(preds_test, verbose=False)
    p10, r10 = precision_recall_at_k(preds_test, k=K, threshold=THRESHOLD)

    print("\n===== TEST METRICS =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"Precision@{K}: {p10:.4f}")
    print(f"Recall@{K}   : {r10:.4f}")

    # Save
    joblib.dump(best_algo, MODEL_PATH)
    joblib.dump(
        {
            "cap_per_movie": CAP_PER_MOVIE,
            "test_size": TEST_SIZE,
            "valid_size": VALID_SIZE,
            "k": K,
            "threshold": THRESHOLD,
            "rating_scale": RATING_SCALE,
            "best_params": best_params,
            "valid_rmse": best_valid_rmse,
            "test_rmse": rmse,
            "test_mae": mae,
            "test_precision_at_k": p10,
            "test_recall_at_k": r10,
        },
        META_PATH
    )
    print("\nSaved model:", MODEL_PATH)
    print("Saved meta :", META_PATH)


if __name__ == "__main__":
    # giảm thread để đỡ ngốn RAM/CPU spike (hợp Lightning AI)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    main()
