"""
LightGCN-based recommendation system.
Load trained LightGCN model and provide recommendations.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgcn_model.pth")
META_PATH = os.path.join(MODEL_DIR, "lightgcn_meta.joblib")
MAPPING_PATH = os.path.join(MODEL_DIR, "lightgcn_mapping.joblib")
RATINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ratings_cf.parquet")
TMDB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tmdb_cleaned.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# MODEL DEFINITION
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
# LOAD MODEL & DATA
# =========================
_model = None
_meta = None
_mappings = None
_ratings = None
_tmdb = None
_edge_index = None
_edge_weight = None
_embeddings = None

def _load_model():
    """Load LightGCN model and metadata."""
    global _model, _meta, _mappings, device
    
    if _model is not None:
        return _model, _meta, _mappings
    
    try:
        # Force check device again
        if not torch.cuda.is_available():
            device = torch.device('cpu')
            
        print(f"Loading LightGCN model on {device}...")
        _meta = joblib.load(META_PATH)
        _mappings = joblib.load(MAPPING_PATH)
        
        # Create model structure
        model = LightGCN(
            num_users=_meta['num_users'],
            num_items=_meta['num_items'],
            embedding_dim=_meta['embedding_dim'],
            num_layers=_meta['num_layers']
        )
        
        try:
            # Try loading on default device (CUDA if available)
            model = model.to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Error loading on {device}: {e}")
            print("Falling back to CPU...")
            device = torch.device('cpu')
            model = model.to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        _model = model
        _model.eval()
        
        print(f"Loaded LightGCN model on {device}: {_meta['num_users']} users, {_meta['num_items']} items")
        return _model, _meta, _mappings
    except Exception as e:
        print(f"Error loading LightGCN model: {e}")
        # Return empty/safe values instead of crashing
        return None, None, None


def _load_data():
    """Load ratings and tmdb data."""
    global _ratings, _tmdb
    
    if _ratings is not None and _tmdb is not None:
        return _ratings, _tmdb
    
    try:
        print("Loading ratings and tmdb data...")
        _ratings = pd.read_parquet(RATINGS_PATH)
        _ratings["userId"] = _ratings["userId"].astype("int32")
        _ratings["tmdb_id"] = _ratings["tmdb_id"].astype("int32")
        _ratings["rating"] = _ratings["rating"].astype("float32")
        
        _tmdb = pd.read_csv(TMDB_PATH)
        return _ratings, _tmdb
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def _create_graph_and_embeddings():
    """Create graph from ratings and compute embeddings."""
    global _edge_index, _edge_weight, _embeddings
    
    if _embeddings is not None:
        return _edge_index, _edge_weight, _embeddings
    
    model, meta, mappings = _load_model()
    ratings, _ = _load_data()
    
    # Reindex ratings to model's internal indices
    ratings_reindexed = ratings.copy()
    ratings_reindexed['user_id'] = ratings_reindexed['userId'].map(mappings['user_id_map'])
    ratings_reindexed['movie_id'] = ratings_reindexed['tmdb_id'].map(mappings['movie_id_map'])
    ratings_reindexed = ratings_reindexed.dropna(subset=['user_id', 'movie_id'])
    
    # Create edge_index and edge_weight
    num_users = meta['num_users']
    users = ratings_reindexed['user_id'].tolist()
    items = [int(i + num_users) for i in ratings_reindexed['movie_id'].tolist()]
    rating_vals = ratings_reindexed['rating'].tolist()
    
    row = users + items
    col = items + users
    
    _edge_index = torch.tensor([row, col], dtype=torch.long).to(device)
    _edge_weight = torch.tensor(rating_vals + rating_vals, dtype=torch.float32).to(device)
    
    # Compute embeddings once and cache
    print("Computing embeddings (this may take a moment)...")
    with torch.no_grad():
        _embeddings = model(_edge_index, _edge_weight)
    
    print("Embeddings computed and cached.")
    return _edge_index, _edge_weight, _embeddings


# =========================
# RECOMMENDATION FUNCTIONS
# =========================
def recommend_cf(user_id, top_k=10):
    """
    Recommend movies for a user using LightGCN.
    
    Args:
        user_id: Original user ID (not reindexed)
        top_k: Number of recommendations
    
    Returns:
        DataFrame with columns: tmdb_id, title, poster_path, score
    """
    model, meta, mappings = _load_model()
    _, tmdb = _load_data()
    _, _, embeddings = _create_graph_and_embeddings()
    
    # Convert user_id to internal index
    if user_id not in mappings['user_id_map']:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    user_idx = mappings['user_id_map'][user_id]
    
    # Get all movie indices
    all_movie_indices = list(range(meta['num_items']))
    
    # Get seen movies for this user
    ratings, _ = _load_data()
    seen_tmdb_ids = set(ratings[ratings['userId'] == user_id]['tmdb_id'].astype(int))
    
    # Predict scores for all movies
    user_ids_tensor = torch.tensor([user_idx] * len(all_movie_indices), dtype=torch.long).to(device)
    movie_ids_tensor = torch.tensor(all_movie_indices, dtype=torch.long).to(device)
    
    with torch.no_grad():
        scores = model.predict(user_ids_tensor, movie_ids_tensor, embeddings).cpu().numpy()
    
    # Create results DataFrame
    results = []
    for movie_idx, score in zip(all_movie_indices, scores):
        # Convert back to original tmdb_id
        if movie_idx in mappings['reverse_movie_map']:
            tmdb_id = mappings['reverse_movie_map'][movie_idx]
            if tmdb_id not in seen_tmdb_ids:
                results.append({
                    'tmdb_id': int(tmdb_id),
                    'score': float(score)
                })
    
    if not results:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False).head(top_k)
    
    # Merge with tmdb to get title and poster_path
    results_df = results_df.merge(
        tmdb[['tmdb_id', 'title', 'poster_path']],
        on='tmdb_id',
        how='left'
    )
    
    # Normalize scores to [0.6, 0.9]
    if len(results_df) > 0:
        scores_raw = results_df['score'].values
        score_max = scores_raw.max()
        score_min = scores_raw.min()
        
        if score_max > score_min:
            scores_normalized = (scores_raw - score_min) / (score_max - score_min)
            results_df['score'] = 0.6 + 0.3 * scores_normalized
        elif score_max > 0:
            results_df['score'] = np.linspace(0.9, 0.7, len(results_df))
        else:
            results_df['score'] = 0.7
    
    return results_df[['tmdb_id', 'title', 'poster_path', 'score']].copy()


def recommend_by_movie(movie_id, top_k=10, user_rating=None):
    """
    Recommend movies based on a movie using LightGCN.
    
    Strategy:
    - Find users who rated this movie (optionally with similar rating)
    - Recommend movies that these users rated highly
    
    Args:
        movie_id: Original tmdb_id (not reindexed)
        top_k: Number of recommendations
        user_rating: Optional user rating for this movie (to filter similar users)
    
    Returns:
        DataFrame with columns: tmdb_id, title, poster_path, score
    """
    model, meta, mappings = _load_model()
    ratings, tmdb = _load_data()
    _, _, embeddings = _create_graph_and_embeddings()
    
    movie_id = int(movie_id)
    
    # Convert movie_id to internal index
    if movie_id not in mappings['movie_id_map']:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    movie_idx = mappings['movie_id_map'][movie_id]
    
    # Find users who rated this movie
    movie_ratings = ratings[ratings['tmdb_id'] == movie_id].copy()
    
    if movie_ratings.empty:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Filter by user_rating if provided
    if user_rating is not None:
        # Find users with similar rating (±0.5)
        rating_low = max(0.5, user_rating - 0.5)
        rating_high = min(5.0, user_rating + 0.5)
        movie_ratings = movie_ratings[
            (movie_ratings['rating'] >= rating_low) & 
            (movie_ratings['rating'] <= rating_high)
        ]
        
        if movie_ratings.empty:
            # Fallback: expand range
            rating_low = max(0.5, user_rating - 1.0)
            rating_high = min(5.0, user_rating + 1.0)
            movie_ratings = ratings[
                (ratings['tmdb_id'] == movie_id) &
                (ratings['rating'] >= rating_low) & 
                (ratings['rating'] <= rating_high)
            ]
    
    if movie_ratings.empty:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Get user indices (internal)
    user_ids_original = movie_ratings['userId'].unique()
    user_indices = [mappings['user_id_map'][uid] for uid in user_ids_original if uid in mappings['user_id_map']]
    
    if not user_indices:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Find movies rated by these users (excluding the current movie)
    other_movies = ratings[
        (ratings['userId'].isin(user_ids_original)) &
        (ratings['tmdb_id'] != movie_id)
    ]
    
    if other_movies.empty:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Get unique movie IDs and their average ratings from these users
    movie_scores = other_movies.groupby('tmdb_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_scores.columns = ['tmdb_id', 'avg_rating', 'rating_count']
    
    # Filter to movies that exist in our model
    movie_scores = movie_scores[movie_scores['tmdb_id'].isin(mappings['movie_id_map'].keys())]
    
    if movie_scores.empty:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Calculate score: avg_rating * log(rating_count + 1)
    movie_scores['score'] = movie_scores['avg_rating'] * np.log1p(movie_scores['rating_count'])
    
    # Sort and get top_k
    movie_scores = movie_scores.sort_values('score', ascending=False).head(top_k)
    
    # Merge with tmdb
    result = movie_scores.merge(
        tmdb[['tmdb_id', 'title', 'poster_path']],
        on='tmdb_id',
        how='left'
    )
    
    # Normalize score to [0.6, 0.9]
    if len(result) > 0:
        scores_raw = result['score'].values
        score_max = scores_raw.max()
        score_min = scores_raw.min()
        
        if score_max > score_min:
            scores_normalized = (scores_raw - score_min) / (score_max - score_min)
            result['score'] = 0.6 + 0.3 * scores_normalized
        elif score_max > 0:
            result['score'] = np.linspace(0.8, 0.7, len(result))
        else:
            result['score'] = 0.7
    
    return result[['tmdb_id', 'title', 'poster_path', 'score']].copy()


def add_rating(user_id, movie_id, rating):
    """
    Add or update a rating (for future use with online learning).
    Currently just updates the in-memory ratings DataFrame.
    """
    global _ratings
    
    ratings, _ = _load_data()
    
    user_id_int = int(user_id)
    movie_id_int = int(movie_id)
    
    mask = (ratings['userId'] == user_id_int) & (ratings['tmdb_id'] == movie_id_int)
    if mask.any():
        ratings.loc[mask, 'rating'] = float(rating)
    else:
        new_row = pd.DataFrame({
            'userId': [user_id_int], 
            'tmdb_id': [movie_id_int], 
            'rating': [float(rating)], 
            'timestamp': [0]
        })
        ratings = pd.concat([ratings, new_row], ignore_index=True)
    
    # Invalidate embeddings cache (would need to recompute)
    global _embeddings
    _embeddings = None

