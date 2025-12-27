import pandas as pd
import numpy as np
import ast
import re

import os
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
current_dir = os.path.dirname(os.path.abspath(__file__))
idx_path = os.path.join(current_dir, "movie_index.csv")
emb_path = os.path.join(current_dir, "movie_embeddings.npy")

# Load full data for director/actor matching
DATA_PATH = os.path.join(os.path.dirname(current_dir), "data", "tmdb_cleaned.csv")

idx_df = pd.read_csv(idx_path)
emb = np.load(emb_path)

# Load full df for director/actor matching
try:
    df_full = pd.read_csv(DATA_PATH)
    # Parse cast_top5 if it's a string
    if 'cast_top5' in df_full.columns:
        df_full['cast_top5'] = df_full['cast_top5'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') 
            else (x if isinstance(x, list) else [])
        )
except:
    df_full = None

# map title -> index (lowercase)
title_to_idx = pd.Series(idx_df.index, index=idx_df["title"].astype(str).str.lower()).drop_duplicates()

def normalize_name(name):
    """Normalize name: remove _, parentheses, lowercase"""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    name = re.sub(r'\([^)]*\)', '', name)
    name = name.replace('_', ' ')
    name = ' '.join(name.split()).lower().strip()
    return name

def recommend_by_title(title, top_k=10):
    key = str(title).strip().lower()
    if key not in title_to_idx:
        # gợi ý gần đúng
        cand = idx_df[idx_df["title"].str.lower().str.contains(key, na=False)]["title"].head(10).tolist()
        raise ValueError(f"Không tìm thấy title chính xác. Gợi ý gần giống: {cand}")

    q = int(title_to_idx[key])

    # vì embeddings đã normalize -> cosine = dot product
    sims = emb @ emb[q]   # (N,)
    # lấy top_k + 1 để bỏ chính nó
    top_idx = np.argsort(sims)[::-1][:top_k+1]

    # bỏ chính nó
    top_idx = [i for i in top_idx if i != q][:top_k]

    rec = idx_df.iloc[top_idx].copy()
    # Clamp scores về [0, 1] để đảm bảo không vượt quá 1
    rec["score"] = np.clip(sims[top_idx], 0, 1)
    # Đảm bảo có poster_path
    if 'poster_path' not in rec.columns and 'poster_path' in idx_df.columns:
        rec['poster_path'] = idx_df.iloc[top_idx]['poster_path'].values
    return rec.reset_index(drop=True)

def recommend_by_actor(actor_name, top_k=10):
    """
    Recommend movies by actor name.
    Boost score for movies that have this actor in cast_top5.
    """
    query = actor_name.strip().lower()
    query_norm = normalize_name(actor_name)

    # Tạo embedding cho tên diễn viên
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Cosine similarity với toàn bộ phim
    sims = emb @ q_emb

    # Boost score cho phim có actor này trong cast_top5
    if df_full is not None and 'cast_top5' in df_full.columns:
        boost_mask = pd.Series([False] * len(idx_df))
        for idx, row in idx_df.iterrows():
            tmdb_id = row.get('tmdb_id')
            if pd.notna(tmdb_id):
                movie_data = df_full[df_full['tmdb_id'] == tmdb_id]
                if not movie_data.empty:
                    cast_list = movie_data.iloc[0].get('cast_top5', [])
                    if isinstance(cast_list, list):
                        # Check cả format gốc và normalized
                        cast_str = [str(c).lower() for c in cast_list]
                        cast_norm = [normalize_name(str(c)) for c in cast_list]
                        if query in cast_str or query_norm in cast_norm:
                            boost_mask.iloc[idx] = True
        
        # Boost 0.3 cho exact match
        sims = np.where(boost_mask, sims + 0.3, sims)

    # Clamp scores về [0, 1]
    sims = np.clip(sims, 0, 1)

    top_idx = np.argsort(sims)[::-1][:top_k]

    rec = idx_df.iloc[top_idx].copy()
    rec["score"] = sims[top_idx]
    return rec.reset_index(drop=True)

def recommend_by_director(name, top_k=10):
    """
    Recommend movies by director name.
    Boost score for movies that have this director.
    """
    query = name.strip().lower()
    query_norm = normalize_name(name)

    # Tạo embedding cho tên director
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Cosine similarity với toàn bộ phim
    sims = emb @ q_emb

    # Boost score cho phim có director này
    if df_full is not None and 'director' in df_full.columns:
        boost_mask = pd.Series([False] * len(idx_df))
        for idx, row in idx_df.iterrows():
            tmdb_id = row.get('tmdb_id')
            if pd.notna(tmdb_id):
                movie_data = df_full[df_full['tmdb_id'] == tmdb_id]
                if not movie_data.empty:
                    director = str(movie_data.iloc[0].get('director', '')).lower()
                    director_norm = normalize_name(director)
                    if query in director or query_norm == director_norm:
                        boost_mask.iloc[idx] = True
        
        # Boost 0.4 cho exact match (director quan trọng hơn actor)
        sims = np.where(boost_mask, sims + 0.4, sims)

    # Clamp scores về [0, 1]
    sims = np.clip(sims, 0, 1)

    top_idx = np.argsort(sims)[::-1][:top_k]

    rec = idx_df.iloc[top_idx].copy()
    rec["score"] = sims[top_idx]
    return rec.reset_index(drop=True)

def recommend_by_description(description, top_k=10):
    """
    Recommend movies by description/overview text.
    """
    query = str(description).strip()
    if not query:
        raise ValueError("Description cannot be empty")

    # Tạo embedding cho description
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Cosine similarity với toàn bộ phim
    sims = emb @ q_emb

    # Clamp scores về [0, 1]
    sims = np.clip(sims, 0, 1)

    top_idx = np.argsort(sims)[::-1][:top_k]

    rec = idx_df.iloc[top_idx].copy()
    rec["score"] = sims[top_idx]
    return rec.reset_index(drop=True)

from rapidfuzz import process, fuzz
import numpy as np

# nếu mày chưa có thì tạo list titles
titles = idx_df["title"].astype(str).tolist()
title_lower = idx_df["title"].astype(str).str.lower()
def recommend_by_query_text(query_text, top_k=10):
    q = str(query_text).strip()
    if not q:
        raise ValueError("Query rỗng")

    # Encode query -> vector
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Cosine similarity (vì emb đã normalize)
    sims = emb @ q_emb  # (N,)
    
    # Boost score cho các phim có title chứa query (partial match)
    q_low = q.lower()
    boost_mask = title_lower.str.contains(q_low, na=False, case=False)
    sims = np.where(boost_mask, sims + 0.2, sims)  # boost 0.2 cho partial match
    
    # Clamp scores về [0, 1] để đảm bảo không vượt quá 1
    sims = np.clip(sims, 0, 1)

    top_idx = np.argsort(sims)[::-1][:top_k]

    rec = idx_df.iloc[top_idx].copy()
    # Clamp scores về [0, 1] để đảm bảo không vượt quá 1
    rec["score"] = np.clip(sims[top_idx], 0, 1)
    # Đảm bảo có poster_path
    if 'poster_path' not in rec.columns and 'poster_path' in idx_df.columns:
        rec['poster_path'] = idx_df.iloc[top_idx]['poster_path'].values
    return rec.reset_index(drop=True)

def recommend(query, top_k=10):
    q = str(query).strip()
    q_low = q.lower()
    q_norm = normalize_name(q)

    # 1) Nếu match title chính xác
    if q_low in title_to_idx:
        return recommend_by_title(q, top_k=top_k)

    # 2) Check nếu là director name (trước khi check partial title match)
    if df_full is not None and 'director' in df_full.columns:
        # Kiểm tra xem có phim nào có director này không
        director_matches = df_full[
            (df_full['director'].astype(str).str.lower().str.contains(q_low, na=False, case=False)) |
            (df_full['director'].astype(str).apply(normalize_name).str.contains(q_norm, na=False, case=False))
        ]
        if not director_matches.empty:
            # Nếu có nhiều phim với director này, dùng recommend_by_director
            return recommend_by_director(q, top_k=top_k)

    # 3) Check nếu là actor name
    if df_full is not None and 'cast_top5' in df_full.columns:
        # Kiểm tra xem có phim nào có actor này không
        def has_actor(row):
            cast_list = row.get('cast_top5', [])
            if isinstance(cast_list, list):
                cast_str = [str(c).lower() for c in cast_list]
                cast_norm = [normalize_name(str(c)) for c in cast_list]
                return q_low in cast_str or q_norm in cast_norm
            return False
        
        actor_matches = df_full[df_full.apply(has_actor, axis=1)]
        if not actor_matches.empty:
            # Nếu có nhiều phim với actor này, dùng recommend_by_actor
            return recommend_by_actor(q, top_k=top_k)

    # 4) Tìm các phim có title chứa query (partial match) - ưu tiên exact match và phần đầu tiên
    partial_matches = idx_df[title_lower.str.contains(q_low, na=False, case=False)].copy()
    if not partial_matches.empty:
        # Normalize title để so sánh: loại bỏ khoảng trắng thừa, ký tự đặc biệt
        def normalize_title(title):
            import re
            # Chuyển về lowercase, loại bỏ khoảng trắng thừa
            normalized = re.sub(r'\s+', ' ', str(title).lower().strip())
            return normalized
        
        q_normalized = normalize_title(q)
        partial_matches['title_normalized'] = partial_matches['title'].apply(normalize_title)
        
        # Check exact match sau khi normalize
        exact_match_mask = partial_matches['title_normalized'] == q_normalized
        partial_matches['is_exact'] = exact_match_mask
        
        # Sắp xếp: ưu tiên exact match, sau đó là phim bắt đầu bằng query, sau đó là chứa query
        partial_matches['starts_with'] = partial_matches['title'].astype(str).str.lower().str.startswith(q_low)
        
        # Nếu có nhiều phim, ưu tiên phần đầu tiên (không có số hoặc có số 1)
        def get_sequel_number(title):
            # Tìm pattern như "Movie 2", "Movie: Part 2", "Movie II"
            import re
            patterns = [
                r'\b(\d+)\b',  # số đơn giản
                r'[:\s]+part\s+(\d+)',  # part 2
                r'[:\s]+(\d+)\s*$',  # số ở cuối
            ]
            for pattern in patterns:
                match = re.search(pattern, str(title).lower())
                if match:
                    return int(match.group(1))
            return 0  # không có số = phần đầu tiên
        
        partial_matches['seq_num'] = partial_matches['title'].apply(get_sequel_number)
        # Sắp xếp: exact match > starts_with > sequel number
        partial_matches = partial_matches.sort_values(['is_exact', 'starts_with', 'seq_num'], ascending=[False, False, True])
        
        # Tính score dựa trên vị trí và độ tương đồng
        n_matches = len(partial_matches)
        # Exact match có score cao nhất
        partial_matches['score'] = np.where(
            partial_matches['is_exact'],
            0.95,  # Score cao cho exact match
            0.8 - (np.arange(n_matches) * 0.05)  # Giảm dần cho các phim khác
        )
        partial_matches['score'] = np.clip(partial_matches['score'], 0.3, 0.95)
        
        # Lấy top_k phim từ partial matches (hoặc tất cả nếu ít hơn top_k)
        result = partial_matches.head(top_k)[['tmdb_id', 'title', 'poster_path', 'score']].copy()
        return result.reset_index(drop=True)

    # 5) Nếu match gần đúng title (fuzzy) -> coi như user gõ sai tên phim
    best = process.extractOne(q, titles, scorer=fuzz.WRatio)
    if best and best[1] >= 85:   # giảm ngưỡng xuống 85 để linh hoạt hơn
        return recommend_by_title(best[0], top_k=top_k)

    # 6) Còn lại: free-text / keyword đều xử lý chung bằng embedding query
    return recommend_by_query_text(q, top_k=top_k)
import re

def parse_log_detail(action_type, detail):
    # detail ví dụ: "Viewed: Avatar: Fire and Ash"
    # hoặc: "Searched for: fire "
    if not isinstance(detail, str):
        return None
    
    if action_type == "view_details":
        m = re.match(r"Viewed:\s*(.*)$", detail)
        return ("movie", m.group(1).strip()) if m else None
    
    if action_type == "search":
        m = re.match(r"Searched for:\s*(.*)$", detail)
        return ("query", m.group(1).strip()) if m else None
    
    return None
import numpy as np

def build_user_profile_from_logs(log_rows, top_recent=30):
    """
    log_rows: list tuple dạng (id, user_id, action_type, detail, timestamp)
    """
    vectors = []
    weights = []
    seen_titles = set()

    # lấy log mới nhất trước
    recent = log_rows[:top_recent]

    for _id, user_id, action, detail, ts in recent:
        parsed = parse_log_detail(action, detail)
        if not parsed:
            continue
        
        kind, value = parsed
        
        if kind == "movie":
            title = value.strip()
            t_low = title.lower()
            seen_titles.add(t_low)

            # nếu title có trong index -> dùng embedding phim
            if t_low in title_to_idx:
                idx = int(title_to_idx[t_low])
                vectors.append(emb[idx])
                weights.append(2.0)   # view mạnh hơn
            else:
                # nếu title không có (crawl thiếu) -> encode text
                q_emb = model.encode([title], convert_to_numpy=True, normalize_embeddings=True)[0]
                vectors.append(q_emb)
                weights.append(1.5)

        elif kind == "query":
            q = value.strip()
            if not q:
                continue
            q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
            vectors.append(q_emb)
            weights.append(1.0)

    if not vectors:
        return None, seen_titles

    V = np.vstack(vectors)
    w = np.array(weights).reshape(-1, 1)

    profile = (V * w).sum(axis=0)
    profile = profile / (np.linalg.norm(profile) + 1e-12)  # normalize

    return profile, seen_titles
def recommend_for_user_from_logs(log_rows, top_k=30):
    print(f"DEBUG: recommend_for_user_from_logs called with top_k={top_k}")
    print(f"DEBUG: idx_df shape: {idx_df.shape}")
    profile, seen_titles = build_user_profile_from_logs(log_rows, top_recent=30)
    if profile is None:
        # fallback: hot/popular (nếu có cột popularity) hoặc random
        # Ở đây trả top theo score embedding gần 0: chọn vài phim đầu
        return idx_df.head(top_k).assign(score=np.nan)

    sims = emb @ profile
    top_idx = np.argsort(sims)[::-1]

    rec_idx = []
    for i in top_idx:
        title_low = str(idx_df.iloc[i]["title"]).lower()
        if title_low in seen_titles:
            continue
        rec_idx.append(i)
        if len(rec_idx) >= top_k:
            break

    rec = idx_df.iloc[rec_idx].copy()
    # Clamp scores về [0, 1] để đảm bảo không vượt quá 1
    rec["score"] = np.clip(sims[rec_idx], 0, 1)
    return rec.reset_index(drop=True)
# logs = [
#  (20, 1, 'view_details', "Viewed: Now You See Me: Now You Don't", '2025-12-27 08:53:58'),
#  (19, 1, 'view_details', "Viewed: Now You See Me: Now You Don't", '2025-12-27 08:53:55'),
#  (18, 1, 'search', 'Searched for: fire ', '2025-12-27 08:53:36'),
#  (17, 1, 'view_details', 'Viewed: Muzzle: City of Wolves', '2025-12-27 08:52:25'),
#  (16, 1, 'view_details', "Viewed: Five Nights at Freddy's 2", '2025-12-27 08:52:23'),
# ]

# rec = recommend_for_user_from_logs(logs, top_k=12)
# print(rec)

# print(recommend("space war alien", top_k=10))
# print(recommend_by_actor("Matthew Mcconaughey", top_k=10))

# print(recommend_by_title("Avatar", top_k=10))
# print(recommend_by_director("Rich Lee", top_k=10))
# print(recommend("Rich Lee", top_k=10))
# print(recommend("Anaconda", top_k=10))