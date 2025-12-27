import joblib
import pandas as pd

model = joblib.load("/home/anonymous/code/KHDL/btl/web/backend/models/models_svd_cf_best.joblib")

ratings = pd.read_parquet("/home/anonymous/code/KHDL/btl/web/backend/data/ratings_cf.parquet")
tmdb = pd.read_csv("/home/anonymous/code/KHDL/btl/web/backend/data/tmdb_cleaned.csv")


def add_rating(user_id, movie_id, rating):
    global ratings
    # Check if user already rated this movie
    # userId trong ratings là int32, nên cần convert user_id về int
    user_id_int = int(user_id)
    movie_id_int = int(movie_id)
    mask = (ratings['userId'] == user_id_int) & (ratings['tmdb_id'] == movie_id_int)
    if mask.any():
        ratings.loc[mask, 'rating'] = float(rating)
    else:
        new_row = pd.DataFrame({'userId': [user_id_int], 'tmdb_id': [movie_id_int], 'rating': [float(rating)], 'timestamp': [0]})
        ratings = pd.concat([ratings, new_row], ignore_index=True)

def recommend_cf(user_id, top_k=10):
    user_id_int = int(user_id)

    # Filter seen movies
    seen = set(ratings[ratings.userId == user_id_int]["tmdb_id"].astype(int))

    all_movies = tmdb["tmdb_id"].dropna().astype(int).unique()

    scores = []
    for m in all_movies:
        if m in seen:
            continue
        try:
            # Model cần string cho user và movie
            est = model.predict(str(user_id_int), str(m)).est
            scores.append((m, est))
        except:
            continue

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]

    rec = tmdb[tmdb.tmdb_id.isin([m for m,_ in top])][
        ["tmdb_id","title","poster_path","genres"]
    ].copy()

    rec["score"] = rec["tmdb_id"].map(dict(top))
    return rec.sort_values("score", ascending=False)

def recommend_by_movie(movie_id, top_k=10, user_rating=None):
    """
    Recommend movies based on collaborative filtering.
    
    Logic cải thiện để phân biệt các mức rating chi tiết:
    - Nếu user_rating < 2.5: Tìm users có rating tương tự (thấp), recommend phim họ rate cao (>= 4.0)
    - Nếu 2.5 <= user_rating < 3.5: Tìm users có rating tương tự (trung bình), recommend phim họ rate tốt (>= 3.5)
    - Nếu 3.5 <= user_rating < 4.5: Tìm users có rating tương tự (khá), recommend phim họ rate khá trở lên (>= 4.0)
    - Nếu user_rating >= 4.5: Tìm users có rating tương tự (rất cao), recommend phim họ rate rất cao (>= 4.5)
    - Nếu user_rating = None: Dùng logic mặc định (tìm users rate cao >= 4.0)
    """
    movie_id = int(movie_id)
    import numpy as np
    
    # Xác định strategy dựa trên user_rating chi tiết
    if user_rating is not None:
        # Tìm users có rating gần với user_rating (trong khoảng ±0.5 hoặc ±1.0)
        # Và xác định threshold cho phim được recommend
        if user_rating < 2.5:
            # User đánh giá rất thấp: tìm users cũng đánh giá thấp (< 3.0)
            rating_low = max(0.5, user_rating - 1.0)
            rating_high = min(3.0, user_rating + 1.0)
            similar_raters = ratings[
                (ratings['tmdb_id'] == movie_id) & 
                (ratings['rating'] >= rating_low) & 
                (ratings['rating'] < rating_high)
            ]
            # Recommend phim mà những users này rate cao (>= 4.0)
            recommend_threshold = 4.0
            recommend_fallback = 3.5
        elif user_rating < 3.5:
            # User đánh giá trung bình: tìm users có rating tương tự (2.5-3.5)
            rating_low = max(2.0, user_rating - 0.5)
            rating_high = min(4.0, user_rating + 0.5)
            similar_raters = ratings[
                (ratings['tmdb_id'] == movie_id) & 
                (ratings['rating'] >= rating_low) & 
                (ratings['rating'] < rating_high)
            ]
            # Recommend phim mà những users này rate tốt (>= 3.5)
            recommend_threshold = 3.5
            recommend_fallback = 3.0
        elif user_rating < 4.5:
            # User đánh giá khá: tìm users có rating tương tự (3.5-4.5)
            rating_low = max(3.0, user_rating - 0.5)
            rating_high = min(5.0, user_rating + 0.5)
            similar_raters = ratings[
                (ratings['tmdb_id'] == movie_id) & 
                (ratings['rating'] >= rating_low) & 
                (ratings['rating'] < rating_high)
            ]
            # Recommend phim mà những users này rate khá trở lên (>= 4.0)
            recommend_threshold = 4.0
            recommend_fallback = 3.5
        else:
            # User đánh giá rất cao (>= 4.5): tìm users có rating tương tự (4.0-5.0)
            rating_low = max(3.5, user_rating - 0.5)
            rating_high = 5.0
            similar_raters = ratings[
                (ratings['tmdb_id'] == movie_id) & 
                (ratings['rating'] >= rating_low) & 
                (ratings['rating'] <= rating_high)
            ]
            # Recommend phim mà những users này rate rất cao (>= 4.5)
            recommend_threshold = 4.5
            recommend_fallback = 4.0
        
        # Nếu không tìm thấy users có rating tương tự, mở rộng range
        if similar_raters.empty:
            if user_rating < 3.0:
                similar_raters = ratings[(ratings['tmdb_id'] == movie_id) & (ratings['rating'] < 3.5)]
            elif user_rating < 4.0:
                similar_raters = ratings[(ratings['tmdb_id'] == movie_id) & (ratings['rating'] >= 2.5) & (ratings['rating'] < 4.5)]
            else:
                similar_raters = ratings[(ratings['tmdb_id'] == movie_id) & (ratings['rating'] >= 3.5)]
        
        if similar_raters.empty:
            return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
        
        # Lấy danh sách user IDs có rating tương tự
        similar_users = set(similar_raters['userId'].values)
        
        # Tìm những phim khác mà những user này rate >= recommend_threshold
        other_movies = ratings[
            (ratings['userId'].isin(similar_users)) & 
            (ratings['tmdb_id'] != movie_id) &
            (ratings['rating'] >= recommend_threshold)
        ]
        
        if other_movies.empty:
            # Fallback: giảm threshold xuống recommend_fallback
            other_movies = ratings[
                (ratings['userId'].isin(similar_users)) & 
                (ratings['tmdb_id'] != movie_id) &
                (ratings['rating'] >= recommend_fallback)
            ]
    else:
        # Strategy mặc định: User chưa đánh giá -> tìm những user đánh giá cao, recommend phim họ thích
        # Tìm những người dùng đã rate phim này >= 4.0 (high rating)
        high_raters = ratings[(ratings['tmdb_id'] == movie_id) & (ratings['rating'] >= 4.0)]
        
        if high_raters.empty:
            # Nếu không có ai rate cao, lấy tất cả người đã rate (>= 3.0)
            high_raters = ratings[(ratings['tmdb_id'] == movie_id) & (ratings['rating'] >= 3.0)]
        
        if high_raters.empty:
            # Nếu vẫn không có, trả về empty
            return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
        
        # Lấy danh sách user IDs đã rate phim này cao
        similar_users = set(high_raters['userId'].values)
        
        # Tìm những phim khác mà những user này cũng rate cao (>= 4.0)
        other_movies = ratings[
            (ratings['userId'].isin(similar_users)) & 
            (ratings['tmdb_id'] != movie_id) &
            (ratings['rating'] >= 4.0)
        ]
        
        if other_movies.empty:
            # Nếu không có phim nào rate >= 4.0, giảm xuống >= 3.5
            other_movies = ratings[
                (ratings['userId'].isin(similar_users)) & 
                (ratings['tmdb_id'] != movie_id) &
                (ratings['rating'] >= 3.5)
            ]
    
    if other_movies.empty:
        return pd.DataFrame(columns=['tmdb_id', 'title', 'poster_path', 'score'])
    
    # Tính điểm cho mỗi phim: trung bình rating từ những user tương tự
    movie_scores = other_movies.groupby('tmdb_id').agg({
        'rating': ['mean', 'count']  # mean rating và số lượng ratings
    }).reset_index()
    
    movie_scores.columns = ['tmdb_id', 'avg_rating', 'rating_count']
    
    # Tính score kết hợp: avg_rating * log(rating_count + 1) để ưu tiên phim có nhiều ratings
    movie_scores['score'] = movie_scores['avg_rating'] * np.log1p(movie_scores['rating_count'])
    
    # Sắp xếp và lấy top_k
    movie_scores = movie_scores.sort_values('score', ascending=False).head(top_k)
    
    # Merge với tmdb để lấy title và poster_path
    result = movie_scores.merge(
        tmdb[['tmdb_id', 'title', 'poster_path']],
        on='tmdb_id',
        how='left'
    )
    
    # Normalize score về [0.6, 0.9] để tránh phim đầu tiên luôn = 100%
    # Sử dụng min-max nhưng scale về range nhỏ hơn và đảm bảo phân bố đều
    if len(result) > 0:
        scores_raw = result['score'].values
        score_max = scores_raw.max()
        score_min = scores_raw.min()
        
        if score_max > score_min:
            # Dùng min-max normalization nhưng scale về [0.6, 0.9] thay vì [0, 1]
            # Điều này đảm bảo không có phim nào = 100% và scores được phân bố đều
            scores_normalized = (scores_raw - score_min) / (score_max - score_min)
            # Scale về [0.6, 0.9]
            result['score'] = 0.6 + 0.3 * scores_normalized
        elif score_max > 0:
            # Nếu tất cả score bằng nhau, set về 0.7-0.8 range để có variation nhẹ
            result['score'] = np.linspace(0.8, 0.7, len(result))
        else:
            result['score'] = 0.7
    
    return result[['tmdb_id', 'title', 'poster_path', 'score']].copy()