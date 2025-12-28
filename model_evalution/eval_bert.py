import pandas as pd
import numpy as np
import os
import sys
import ast
from tqdm import tqdm

# Add backend path to sys.path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "backend")
sys.path.append(backend_path)

from recommendation import rcm_bert

def evaluate_bert(num_samples=100, top_k=10):
    print(f"Evaluating BERT with {num_samples} samples, top_k={top_k}...")
    
    # Load data for genre ground truth
    DATA_PATH = os.path.join(backend_path, "data", "tmdb_cleaned.csv")
    df = pd.read_csv(DATA_PATH)
    
    # Parse genres
    def parse_genres(g):
        try:
            if isinstance(g, str):
                return set(ast.literal_eval(g))
            return set(g) if isinstance(g, list) else set()
        except:
            return set()

    df['genre_set'] = df['genres'].apply(parse_genres)
    
    # Sample movies that have genres
    sample_df = df[df['genre_set'].map(len) > 0].sample(min(num_samples, len(df)))
    
    precisions = []
    recalls = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        title = row['title']
        target_genres = row['genre_set']
        
        try:
            recs = rcm_bert.recommend_by_title(title, top_k=top_k)
            if recs.empty:
                continue
                
            # Get genres of recommended movies
            rec_ids = recs['tmdb_id'].tolist()
            rec_genres = df[df['tmdb_id'].isin(rec_ids)]['genre_set'].tolist()
            
            # Precision: how many recs share at least one genre with target
            hits = sum(1 for g in rec_genres if not g.isdisjoint(target_genres))
            precisions.append(hits / len(recs))
            
            # Recall: hits / total movies in dataset sharing at least one genre (simplified)
            # For BERT, recall is hard to define without user data, so we use genre-based relevance
            total_relevant = len(df[df['genre_set'].apply(lambda g: not g.isdisjoint(target_genres))]) - 1 # exclude itself
            if total_relevant > 0:
                recalls.append(hits / min(total_relevant, top_k)) # Normalized recall
            else:
                recalls.append(0)
                
        except Exception as e:
            # print(f"Error recommending for {title}: {e}")
            continue
            
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    print("\n--- BERT Evaluation Results ---")
    print(f"Average Precision@{top_k}: {avg_precision:.4f}")
    print(f"Average Recall@{top_k} (Genre-based): {avg_recall:.4f}")
    print("-------------------------------\n")

if __name__ == "__main__":
    evaluate_bert(num_samples=50, top_k=10)
