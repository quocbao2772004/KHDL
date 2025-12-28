import pandas as pd
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# Add backend path to sys.path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "backend")
sys.path.append(backend_path)

# Mock bcrypt if needed (though it should be installed)
try:
    import bcrypt
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['bcrypt'] = MagicMock()

from recommendation import rcm_lightgcn

def evaluate_lightgcn(num_users_to_eval=50, top_k=10):
    print(f"Evaluating LightGCN with {num_users_to_eval} users, top_k={top_k}...")
    
    # Load ratings
    ratings, _ = rcm_lightgcn._load_data()
    
    # Split into train/test (simple random split for evaluation)
    # In a real scenario, we'd use time-based split or leave-one-out
    test_ratio = 0.2
    test_indices = np.random.choice(ratings.index, size=int(len(ratings) * test_ratio), replace=False)
    test_df = ratings.loc[test_indices]
    
    # We evaluate on users who have at least some ratings in test set
    eval_users = test_df['userId'].unique()
    if len(eval_users) > num_users_to_eval:
        eval_users = np.random.choice(eval_users, num_users_to_eval, replace=False)
    
    mse_list = []
    precisions = []
    recalls = []
    
    # Load model and embeddings
    model, meta, mappings = rcm_lightgcn._load_model()
    _, _, embeddings = rcm_lightgcn._create_graph_and_embeddings()
    
    if model is None:
        print("Error: Model not loaded.")
        return

    num_users = meta['num_users']
    
    for user_id in tqdm(eval_users):
        if user_id not in mappings['user_id_map']:
            continue
            
        user_idx = mappings['user_id_map'][user_id]
        
        # Ground truth for this user in test set
        user_test = test_df[test_df['userId'] == user_id]
        if user_test.empty:
            continue
            
        # 1. MSE / RMSE
        # Predict ratings for movies in test set
        test_movie_ids = user_test['tmdb_id'].tolist()
        actual_ratings = user_test['rating'].tolist()
        
        predicted_scores = []
        valid_actuals = []
        
        for i, tid in enumerate(test_movie_ids):
            if tid in mappings['movie_id_map']:
                movie_idx = mappings['movie_id_map'][tid]
                # LightGCN prediction (dot product)
                u_emb = embeddings[user_idx]
                i_emb = embeddings[num_users + movie_idx]
                score = torch.dot(u_emb, i_emb).item()
                # Scale score to [0.5, 5.0] range for MSE (heuristic scaling)
                # In LightGCN, scores are arbitrary, but we can normalize them
                predicted_scores.append(score)
                valid_actuals.append(actual_ratings[i])
        
        if predicted_scores:
            # Normalize predicted scores to [0.5, 5.0] for MSE comparison
            p_min, p_max = min(predicted_scores), max(predicted_scores)
            if p_max > p_min:
                norm_preds = [0.5 + 4.5 * (s - p_min) / (p_max - p_min) for s in predicted_scores]
            else:
                norm_preds = [2.5] * len(predicted_scores)
            
            mse = mean_squared_error(valid_actuals, norm_preds)
            mse_list.append(mse)
            
        # 2. Precision / Recall @ K
        # Get top-K recommendations for this user
        try:
            recs = rcm_lightgcn.recommend_cf(user_id, top_k=top_k)
            if recs.empty:
                continue
                
            rec_ids = set(recs['tmdb_id'].tolist())
            # Liked movies in test set (rating >= 4.0)
            liked_test_ids = set(user_test[user_test['rating'] >= 4.0]['tmdb_id'].tolist())
            
            if not liked_test_ids:
                continue
                
            hits = len(rec_ids.intersection(liked_test_ids))
            precisions.append(hits / top_k)
            recalls.append(hits / len(liked_test_ids))
            
        except Exception as e:
            # print(f"Error recommending for user {user_id}: {e}")
            continue
            
    avg_mse = np.mean(mse_list) if mse_list else 0
    avg_rmse = np.sqrt(avg_mse)
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    print("\n--- LightGCN Evaluation Results ---")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average Precision@{top_k}: {avg_precision:.4f}")
    print(f"Average Recall@{top_k}: {avg_recall:.4f}")
    print("-----------------------------------\n")

if __name__ == "__main__":
    evaluate_lightgcn(num_users_to_eval=30, top_k=10)
