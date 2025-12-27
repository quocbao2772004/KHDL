# -*- coding: utf-8 -*-
"""
Create BERT embeddings từ dữ liệu đã preprocess
Sử dụng file tmdb_cleaned_for_bert.csv đã được xử lý bởi preprocess_for_bert.py
"""

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Paths
PREPROCESSED_CSV = "/home/anonymous/code/KHDL/btl/web/backend/data/tmdb_cleaned_for_bert.csv"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output paths (trong thư mục recommendation)
OUTPUT_EMB = os.path.join(CURRENT_DIR, "movie_embeddings.npy")
OUTPUT_INDEX = os.path.join(CURRENT_DIR, "movie_index.csv")

def main():
    print("=" * 60)
    print("Creating BERT embeddings from preprocessed data")
    print("=" * 60)
    
    # Kiểm tra file đã preprocess
    if not os.path.exists(PREPROCESSED_CSV):
        print(f"ERROR: File {PREPROCESSED_CSV} not found!")
        print("Please run preprocess_for_bert.py first to create the preprocessed file.")
        return
    
    # Load dữ liệu đã preprocess
    print(f"\n1. Loading preprocessed data from {PREPROCESSED_CSV}...")
    df = pd.read_csv(PREPROCESSED_CSV)
    print(f"   Loaded {len(df)} movies")
    
    # Kiểm tra cột bert_text
    if "bert_text" not in df.columns:
        print("ERROR: Column 'bert_text' not found in preprocessed file!")
        print("Please make sure preprocess_for_bert.py ran successfully.")
        return
    
    # Lấy bert_text
    print("\n2. Preparing texts for embedding...")
    texts = df["bert_text"].fillna("").astype(str).tolist()
    print(f"   Total texts: {len(texts)}")
    print(f"   Empty texts: {sum(1 for t in texts if not t.strip())}")
    
    # Sample để kiểm tra
    print("\n3. Sample bert_text (first 3):")
    for i in range(min(3, len(texts))):
        print(f"   [{i}] {texts[i][:100]}...")
    
    # Load model và encode
    print("\n4. Loading BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("\n5. Encoding texts to embeddings (this may take a while)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Quan trọng: giúp cosine similarity nhanh & ổn hơn
    )
    
    print(f"\n6. Embeddings shape: {embeddings.shape}")
    
    # Lưu embeddings
    print(f"\n7. Saving embeddings to {OUTPUT_EMB}...")
    np.save(OUTPUT_EMB, embeddings)
    print("   ✓ Saved embeddings")
    
    # Lưu index (cần title + tmdb_id + poster_path)
    print(f"\n8. Saving movie index to {OUTPUT_INDEX}...")
    keep_cols = ["tmdb_id", "title"]
    if "poster_path" in df.columns:
        keep_cols.append("poster_path")
    
    df[keep_cols].to_csv(OUTPUT_INDEX, index=False, encoding="utf-8-sig")
    print("   ✓ Saved index")
    
    print("\n" + "=" * 60)
    print("SUCCESS! Embeddings and index created.")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Embeddings: {OUTPUT_EMB}")
    print(f"  - Index: {OUTPUT_INDEX}")
    print(f"\nYou can now use these files with rcm_bert.py")
    
    # Test một vài trường hợp
    print("\n" + "=" * 60)
    print("Testing recommendations...")
    print("=" * 60)
    
    # Load lại để test
    idx_df = pd.read_csv(OUTPUT_INDEX)
    emb = np.load(OUTPUT_EMB)
    
    # Load file đã preprocess để xem bert_text
    df_preprocessed = pd.read_csv(PREPROCESSED_CSV)
    
    # Map title -> index (lowercase)
    title_to_idx = pd.Series(idx_df.index, index=idx_df["title"].astype(str).str.lower()).drop_duplicates()
    
    def test_recommend_by_title(title, top_k=5):
        key = str(title).strip().lower()
        
        # Kiểm tra có bao nhiêu phim có title này
        matching_movies = idx_df[idx_df["title"].str.lower() == key]
        print(f"   Found {len(matching_movies)} movie(s) with title '{title}':")
        for idx, row in matching_movies.iterrows():
            print(f"      - Index {idx}: {row['title']} (tmdb_id: {row.get('tmdb_id', 'N/A')})")
        
        if key not in title_to_idx:
            print(f"   ❌ '{title}' not found in title_to_idx mapping")
            # Tìm gần đúng
            cand = idx_df[idx_df["title"].str.lower().str.contains(key, na=False)]["title"].head(5).tolist()
            if cand:
                print(f"   Similar titles: {cand}")
            return None
        
        q = int(title_to_idx[key])
        print(f"   Using index {q} for recommendations")
        
        # Kiểm tra phim tại index q
        movie_at_q = idx_df.iloc[q]
        tmdb_id_q = movie_at_q.get('tmdb_id', None)
        print(f"   Movie at index {q}: {movie_at_q['title']} (tmdb_id: {tmdb_id_q})")
        
        # Kiểm tra bert_text của phim này
        if tmdb_id_q and 'tmdb_id' in df_preprocessed.columns:
            movie_data = df_preprocessed[df_preprocessed['tmdb_id'] == tmdb_id_q]
            if not movie_data.empty:
                bert_text = movie_data.iloc[0].get('bert_text', '')
                print(f"   bert_text preview: {bert_text[:200]}...")
        
        sims = emb @ emb[q]
        top_idx = np.argsort(sims)[::-1][:top_k+1]
        top_idx = [i for i in top_idx if i != q][:top_k]
        
        rec = idx_df.iloc[top_idx].copy()
        rec["score"] = sims[top_idx]
        print(f"   ✓ Recommendations (similarity scores):")
        for _, row in rec.iterrows():
            print(f"      - {row['title']} (score: {row['score']:.4f}, tmdb_id: {row.get('tmdb_id', 'N/A')})")
        
        # Kiểm tra xem có phim "Anaconda" khác trong recommendations không
        if "anaconda" in key:
            anaconda_in_recs = rec[rec['title'].str.lower().str.contains('anaconda', na=False)]
            if not anaconda_in_recs.empty:
                print(f"   ⚠️  Found other Anaconda movies in recommendations:")
                for _, row in anaconda_in_recs.iterrows():
                    print(f"      - {row['title']} (score: {row['score']:.4f})")
            else:
                print(f"   ⚠️  No other Anaconda movies in top {top_k} recommendations")
        
        return rec
    
    # Test cases
    test_titles = ["Anaconda", "Avatar"]
    for title in test_titles:
        print(f"\nTesting: '{title}'")
        test_recommend_by_title(title, top_k=3)

if __name__ == "__main__":
    main()

