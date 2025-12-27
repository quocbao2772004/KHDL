import sys
import os
import pandas as pd
import numpy as np

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from recomendation.rcm_bert import recommend_by_title
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

title = "Avatar: Fire and Ash"
print(f"Testing recommendation for: {title}")

try:
    recommendations = recommend_by_title(title, top_k=5)
    print("Recommendation successful. Result head:")
    print(recommendations.head())
    
    # Load full data to simulate API environment
    df = pd.read_csv("data/tmdb_cleaned.csv")
    
    # Merge
    merged = recommendations.merge(df[['tmdb_id', 'overview', 'release_date', 'genres']], on='tmdb_id', how='left')
    
    print("\nTesting serialization with merged data...")
    results = []
    for _, row in merged.iterrows():
        item = {
            "title": row['title'],
            "overview": row['overview'],
            "release_date": row['release_date'],
            "genres": row['genres'],
            "poster_path": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if pd.notna(row['poster_path']) else None,
            "score": float(row['score'])
        }
        print(f"Item score type: {type(item['score'])}")
        results.append(item)
        
    import json
    # Simulate JSON serialization which FastAPI does
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    # Standard json dump to see if it fails without custom encoder (FastAPI usually handles some, but not all)
    try:
        json.dumps(results)
        print("Standard JSON serialization successful.")
    except TypeError as e:
        print(f"Standard JSON serialization FAILED: {e}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
