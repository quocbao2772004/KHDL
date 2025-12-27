# -*- coding: utf-8 -*-
"""
Train LightGCN model cho dữ liệu ratings_cf.parquet
Dựa trên code mẫu từ code_lightgcn.py
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import joblib

# =========================
# CONFIG
# =========================
RATINGS_PARQUET = "/home/anonymous/code/KHDL/btl/web/backend/data/ratings_cf.parquet"
OUT_DIR = "/home/anonymous/code/KHDL/btl/web/backend/models"
MODEL_PATH = os.path.join(OUT_DIR, "lightgcn_model.pth")
META_PATH = os.path.join(OUT_DIR, "lightgcn_meta.joblib")
MAPPING_PATH = os.path.join(OUT_DIR, "lightgcn_mapping.joblib")

# Giảm dữ liệu nếu cần (tương tự train.py)
CAP_PER_MOVIE = None  # None để dùng toàn bộ, hoặc số để giới hạn ratings mỗi phim

# Split ratio
TEST_SIZE = 0.20
VALID_SIZE = 0.10

# Model params
EMBEDDING_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.002  # Tăng learning rate một chút để converge nhanh hơn
NUM_EPOCHS = 100  # Giảm max epochs vì có early stopping
BATCH_SIZE = 16384  # Tăng batch size để train nhanh hơn (giảm số lần forward pass)
WEIGHT_DECAY = 1e-6

# Early stopping
PATIENCE = 10

# Tối ưu hóa
USE_MIXED_PRECISION = True  # Dùng mixed precision trên GPU để tăng tốc
VAL_FREQ = 5  # Chỉ validate mỗi N epochs để tiết kiệm thời gian

# Metrics
K_VALUES = [5, 10, 20]
THRESHOLD = 4.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =========================
# UTILS
# =========================
def cap_ratings_keep_all_movies(df, cap_per_movie=4000, seed=42):
    """
    Giữ đủ phim (tmdb_id) nhưng giới hạn số rating mỗi phim.
    Không mất phim, giảm số dòng => nhẹ RAM + train nhanh.
    """
    return (df.groupby("tmdb_id", group_keys=False)
              .apply(lambda x: x.sample(n=min(len(x), cap_per_movie), random_state=seed))
              .reset_index(drop=True))


def create_edge_index_weights(data, num_users):
    """
    Tạo edge_index và edge_weight cho graph.
    data phải có columns: user_id, movie_id, rating (đã reindex)
    """
    users = data['user_id'].tolist()
    items = [i + num_users for i in data['movie_id'].tolist()]
    ratings = data['rating'].tolist()

    # Tạo các cạnh hai chiều (u->i và i->u)
    row = users + items
    col = items + users

    edge_index = torch.tensor([row, col], dtype=torch.long).to(device)
    edge_weight = torch.tensor(ratings + ratings, dtype=torch.float32).to(device)

    return edge_index, edge_weight


# =========================
# MODEL
# =========================
class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # Khởi tạo embeddings
        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, edge_index, edge_weight):
        """
        edge_index: [2, 2*E]  (mỗi cạnh được duplicate 2 chiều)
        edge_weight: [2*E]    (trọng số rating cho mỗi cạnh)
        """
        x = self.embedding.weight  # [num_nodes, embedding_dim]

        # row và col
        row, col = edge_index
        # Tính degree cho mỗi node
        deg = torch.zeros(x.size(0), dtype=torch.float, device=device)
        deg.index_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # Tạo norm = A_{i,j} / sqrt(d_i * d_j)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Message passing qua num_layers lớp
        all_embeddings = [x]
        for _ in range(self.num_layers):
            x = torch.sparse.mm(
                torch.sparse_coo_tensor(
                    edge_index, norm, (x.size(0), x.size(0))
                ),
                x
            )
            all_embeddings.append(x)

        # LightGCN: lấy trung bình embedding của tất cả các tầng
        all_embeddings = torch.stack(all_embeddings, dim=1)  # [N, num_layers+1, dim]
        final_embeddings = torch.mean(all_embeddings, dim=1)  # [N, dim]

        return final_embeddings

    def predict(self, user_ids, item_ids, embeddings):
        user_embeddings = embeddings[user_ids]
        item_embeddings = embeddings[self.num_users + item_ids]
        # Dot product
        return (user_embeddings * item_embeddings).sum(dim=1)


# =========================
# METRICS
# =========================
def calculate_metrics(model, data, embeddings, k_values=[5, 10, 20], threshold=4.0):
    model.eval()
    with torch.no_grad():
        user_ids = torch.tensor(data['user_id'].values, dtype=torch.long).to(device)
        item_ids = torch.tensor(data['movie_id'].values, dtype=torch.long).to(device)
        true_ratings = data['rating'].values

        predicted_ratings = model.predict(user_ids, item_ids, embeddings).cpu().numpy()
        
        # Tính MSE, RMSE, MAE
        mse = np.mean((predicted_ratings - true_ratings) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted_ratings - true_ratings))

        # Tạo dictionary để lưu predictions và true relevant items
        user_predictions = {}
        user_true_items = {}

        for user, item, rating, pred in zip(user_ids.cpu().numpy(), item_ids.cpu().numpy(), true_ratings, predicted_ratings):
            if user not in user_predictions:
                user_predictions[user] = []
                user_true_items[user] = set()

            user_predictions[user].append((item, pred))

            if rating >= threshold:  # Ratings >= threshold là relevant
                user_true_items[user].add(item)

        # Tính Recall@K và Precision@K
        metrics = {'mse': mse, 'rmse': rmse, 'mae': mae}

        for k in k_values:
            recall_sum = 0
            precision_sum = 0
            num_users = 0

            for user in user_predictions:
                if len(user_true_items[user]) == 0:
                    continue  # Không có items relevant cho user này

                # Sắp xếp items theo pred_rating giảm dần
                sorted_items = sorted(user_predictions[user], key=lambda x: x[1], reverse=True)
                top_k_items = set([x[0] for x in sorted_items[:k]])
                num_relevant = len(top_k_items & user_true_items[user])

                recall = num_relevant / len(user_true_items[user])
                precision = num_relevant / k

                recall_sum += recall
                precision_sum += precision
                num_users += 1

            if num_users > 0:
                metrics[f'recall@{k}'] = recall_sum / num_users
                metrics[f'precision@{k}'] = precision_sum / num_users
            else:
                metrics[f'recall@{k}'] = 0.0
                metrics[f'precision@{k}'] = 0.0

        return metrics


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

    # Cap per movie để không kill RAM nhưng vẫn giữ đủ phim
    if CAP_PER_MOVIE is not None:
        print(f"Capping ratings per movie to <= {CAP_PER_MOVIE} (keep all movies)...")
        df = cap_ratings_keep_all_movies(df, cap_per_movie=CAP_PER_MOVIE, seed=42)

    print("Rows after cap:", len(df))
    print("Users:", df["userId"].nunique(), "Movies:", df["tmdb_id"].nunique())

    # Reindex user_ids và movie_ids để có continuous indices (0, 1, 2, ...)
    original_user_ids = df['userId'].unique()
    original_movie_ids = df['tmdb_id'].unique()
    
    user_id_map = {old: new for new, old in enumerate(sorted(original_user_ids))}
    movie_id_map = {old: new for new, old in enumerate(sorted(original_movie_ids))}
    
    # Lưu reverse mapping để convert lại sau
    reverse_user_map = {new: old for old, new in user_id_map.items()}
    reverse_movie_map = {new: old for old, new in movie_id_map.items()}

    # Apply mapping
    ratings_df = df.copy()
    ratings_df['user_id'] = ratings_df['userId'].map(user_id_map)
    ratings_df['movie_id'] = ratings_df['tmdb_id'].map(movie_id_map)
    ratings_df = ratings_df[['user_id', 'movie_id', 'rating']].dropna()

    num_users = len(user_id_map)
    num_items = len(movie_id_map)
    print(f"Reindexed: {num_users} users, {num_items} items")

    # Chia dữ liệu thành train (80%), validation (10%), và test (10%)
    train_data, temp_data = train_test_split(ratings_df, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    # Tạo edge_index và edge_weight từ train data
    print("Creating graph edges...")
    edge_index, edge_weight = create_edge_index_weights(train_data, num_users)
    print(f"Graph: {edge_index.shape[1] // 2} edges (bidirectional)")

    # Khởi tạo mô hình
    model = LightGCN(num_users, num_items, EMBEDDING_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    
    # Mixed precision training (nếu có GPU)
    scaler = None
    if USE_MIXED_PRECISION and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training")
    
    # Pre-convert train data to tensors để tăng tốc
    print("Pre-converting train data to tensors...")
    train_user_ids = torch.tensor(train_data['user_id'].values, dtype=torch.long).to(device)
    train_item_ids = torch.tensor(train_data['movie_id'].values, dtype=torch.long).to(device)
    train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32).to(device)
    
    # Pre-convert validation data
    val_user_ids = torch.tensor(val_data['user_id'].values, dtype=torch.long).to(device)
    val_item_ids = torch.tensor(val_data['movie_id'].values, dtype=torch.long).to(device)
    val_ratings = torch.tensor(val_data['rating'].values, dtype=torch.float32).to(device)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    # Training loop
    train_losses = []
    val_losses = []

    print(f"\n=== TRAINING (max {NUM_EPOCHS} epochs) ===")
    print(f"Batch size: {BATCH_SIZE}, Validation frequency: every {VAL_FREQ} epochs")
    
    num_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE
    # Tính embeddings update frequency (tính lại sau mỗi N batches để cân bằng tốc độ/chính xác)
    EMBEDDING_UPDATE_FREQ = max(1, num_batches // 10)  # Update embeddings ~10 lần mỗi epoch
    print(f"Embedding update frequency: every {EMBEDDING_UPDATE_FREQ} batches (~{num_batches // EMBEDDING_UPDATE_FREQ} times per epoch)")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        # Tạo batch indices
        indices = torch.randperm(len(train_data), device=device)
        
        # Tính embeddings ban đầu cho epoch
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            embeddings = model(edge_index, edge_weight)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(train_data))
            batch_indices = indices[start_idx:end_idx]

            user_ids_batch = train_user_ids[batch_indices]
            item_ids_batch = train_item_ids[batch_indices]
            ratings_batch = train_ratings[batch_indices]

            # Forward pass với mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    predicted_ratings = model.predict(user_ids_batch, item_ids_batch, embeddings)
                    loss = loss_fn(predicted_ratings, ratings_batch)
                
                # Backward pass với mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                predicted_ratings = model.predict(user_ids_batch, item_ids_batch, embeddings)
                loss = loss_fn(predicted_ratings, ratings_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(batch_indices)
            
            # Tính lại embeddings sau mỗi EMBEDDING_UPDATE_FREQ batches (tối ưu tốc độ)
            if (batch_idx + 1) % EMBEDDING_UPDATE_FREQ == 0:
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    embeddings = model(edge_index, edge_weight)

        avg_train_loss = total_loss / len(train_data)
        train_losses.append(avg_train_loss)

        # Tính toán loss trên tập validation (chỉ mỗi VAL_FREQ epochs)
        if (epoch + 1) % VAL_FREQ == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    val_embeddings = model(edge_index, edge_weight)
                    val_predicted_ratings = model.predict(val_user_ids, val_item_ids, val_embeddings)
                    val_loss = loss_fn(val_predicted_ratings, val_ratings).item()
            val_losses.append(val_loss)
        else:
            # Nếu không validate, dùng giá trị cũ hoặc train loss
            val_loss = val_losses[-1] if val_losses else avg_train_loss
            val_losses.append(val_loss)

        # Kiểm tra Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f} (validated every {VAL_FREQ} epochs)')

        if patience_counter >= PATIENCE:
            print(f'\nEarly Stopping triggered at epoch {epoch+1}.')
            break

    print(f'\nBest Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}')

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Tính embeddings trên toàn bộ dữ liệu train
    model.eval()
    with torch.no_grad():
        full_embeddings = model(edge_index, edge_weight)

    # Đánh giá trên tập train, validation và test
    print("\n=== FINAL EVALUATION (MSE, RMSE, MAE) ===")
    print("Calculating metrics on all sets...")
    
    train_metrics = calculate_metrics(model, train_data, full_embeddings, k_values=K_VALUES, threshold=THRESHOLD)
    val_metrics = calculate_metrics(model, val_data, full_embeddings, k_values=K_VALUES, threshold=THRESHOLD)
    test_metrics = calculate_metrics(model, test_data, full_embeddings, k_values=K_VALUES, threshold=THRESHOLD)

    print("\n" + "="*60)
    print("TRAIN SET METRICS:")
    print("="*60)
    print(f"  MSE  (Mean Squared Error):     {train_metrics['mse']:.6f}")
    print(f"  RMSE (Root Mean Squared Error): {train_metrics['rmse']:.6f}")
    print(f"  MAE  (Mean Absolute Error):     {train_metrics['mae']:.6f}")
    for k in K_VALUES:
        if f'precision@{k}' in train_metrics:
            print(f"  Precision@{k}: {train_metrics[f'precision@{k}']:.4f}")
        if f'recall@{k}' in train_metrics:
            print(f"  Recall@{k}:   {train_metrics[f'recall@{k}']:.4f}")

    print("\n" + "="*60)
    print("VALIDATION SET METRICS:")
    print("="*60)
    print(f"  MSE  (Mean Squared Error):     {val_metrics['mse']:.6f}")
    print(f"  RMSE (Root Mean Squared Error): {val_metrics['rmse']:.6f}")
    print(f"  MAE  (Mean Absolute Error):     {val_metrics['mae']:.6f}")
    for k in K_VALUES:
        if f'precision@{k}' in val_metrics:
            print(f"  Precision@{k}: {val_metrics[f'precision@{k}']:.4f}")
        if f'recall@{k}' in val_metrics:
            print(f"  Recall@{k}:   {val_metrics[f'recall@{k}']:.4f}")

    print("\n" + "="*60)
    print("TEST SET METRICS:")
    print("="*60)
    print(f"  MSE  (Mean Squared Error):     {test_metrics['mse']:.6f}")
    print(f"  RMSE (Root Mean Squared Error): {test_metrics['rmse']:.6f}")
    print(f"  MAE  (Mean Absolute Error):     {test_metrics['mae']:.6f}")
    for k in K_VALUES:
        if f'precision@{k}' in test_metrics:
            print(f"  Precision@{k}: {test_metrics[f'precision@{k}']:.4f}")
        if f'recall@{k}' in test_metrics:
            print(f"  Recall@{k}:   {test_metrics[f'recall@{k}']:.4f}")
    print("="*60)

    # Vẽ đồ thị loss
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
        plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Train và Validation Loss theo Epoch')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(OUT_DIR, "lightgcn_loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"\nSaved loss plot: {loss_plot_path}")
    except Exception as e:
        print(f"Warning: Could not save loss plot: {e}")

    # Save model và metadata
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

    # Save mappings và metadata
    joblib.dump(
        {
            "user_id_map": user_id_map,
            "movie_id_map": movie_id_map,
            "reverse_user_map": reverse_user_map,
            "reverse_movie_map": reverse_movie_map,
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "cap_per_movie": CAP_PER_MOVIE,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        META_PATH
    )
    print(f"Saved metadata: {META_PATH}")

    # Save mappings riêng để dễ load
    joblib.dump(
        {
            "user_id_map": user_id_map,
            "movie_id_map": movie_id_map,
            "reverse_user_map": reverse_user_map,
            "reverse_movie_map": reverse_movie_map,
        },
        MAPPING_PATH
    )
    print(f"Saved mappings: {MAPPING_PATH}")


if __name__ == "__main__":
    # Giảm thread để đỡ ngốn RAM/CPU
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    main()

