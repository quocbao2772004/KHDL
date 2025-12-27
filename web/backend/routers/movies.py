from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Movie, PaginatedMovies
from recommendation import rcm_bert, rcm_lightgcn
import database as db

router = APIRouter()

# Request models
class RateRequest(BaseModel):
    user_id: int
    movie_id: int  # Frontend gửi movie_id (tương đương tmdb_id)
    rating: float

# =========================
# DATA LOADING & CACHING
# =========================
_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_current_dir, "data")
TMDB_PATH = os.path.join(DATA_DIR, "tmdb_cleaned.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings_cf.parquet")

# Load tmdb data
_df = None
_ratings_df = None
_average_ratings_cache = None

def _load_data():
    global _df, _ratings_df
    if _df is None:
        print("Loading tmdb data...")
        _df = pd.read_csv(TMDB_PATH)
        print(f"Loaded {len(_df)} movies")
    if _ratings_df is None:
        print("Loading ratings data...")
        _ratings_df = pd.read_parquet(RATINGS_PATH)
        _ratings_df["userId"] = _ratings_df["userId"].astype("int32")
        _ratings_df["tmdb_id"] = _ratings_df["tmdb_id"].astype("int32")
        _ratings_df["rating"] = _ratings_df["rating"].astype("float32")
        print(f"Loaded {len(_ratings_df)} ratings")
    return _df, _ratings_df

def get_average_ratings():
    """Calculate and cache average ratings for all movies."""
    global _average_ratings_cache
    if _average_ratings_cache is None:
        _, ratings = _load_data()
        print("Calculating average ratings...")
        _average_ratings_cache = ratings.groupby("tmdb_id")["rating"].mean().to_dict()
        print(f"Cached average ratings for {len(_average_ratings_cache)} movies")
    return _average_ratings_cache

def get_movie_average_rating(tmdb_id: int) -> Optional[float]:
    """Get average rating for a specific movie."""
    avg_ratings = get_average_ratings()
    return avg_ratings.get(int(tmdb_id))

def get_user_rating(user_id: int, tmdb_id: int) -> Optional[float]:
    """Get user's rating for a specific movie."""
    _, ratings = _load_data()
    user_rating = ratings[
        (ratings["userId"] == int(user_id)) & 
        (ratings["tmdb_id"] == int(tmdb_id))
    ]
    if not user_rating.empty:
        return float(user_rating.iloc[0]["rating"])
    return None

def _df_to_movie(row: pd.Series, score: Optional[float] = None) -> Movie:
    """Convert DataFrame row to Movie model."""
    tmdb_id = int(row.get("tmdb_id", 0)) if pd.notna(row.get("tmdb_id")) else None
    poster_path = row.get("poster_path", "")
    if pd.notna(poster_path) and isinstance(poster_path, str) and poster_path:
        if not poster_path.startswith("http"):
            poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
    else:
        poster_path = None
    
    return Movie(
        title=str(row.get("title", "")),
        overview=str(row.get("overview", "")) if pd.notna(row.get("overview")) else None,
        release_date=str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else None,
        genres=str(row.get("genres", "")) if pd.notna(row.get("genres")) else None,
        cast=str(row.get("cast_top5", "")) if pd.notna(row.get("cast_top5")) else None,
        director=str(row.get("director", "")) if pd.notna(row.get("director")) else None,
        poster_path=poster_path,
        score=score if score is not None else None,
        average_rating=get_movie_average_rating(tmdb_id) if tmdb_id else None,
        tmdb_id=tmdb_id
    )

# =========================
# ENDPOINTS
# =========================

@router.get("/movies", response_model=PaginatedMovies)
def get_movies(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None
):
    """Get paginated list of movies."""
    df, _ = _load_data()
    
    if search:
        search_lower = search.lower()
        mask = df["title"].astype(str).str.lower().str.contains(search_lower, na=False)
        df = df[mask]
    
    total = len(df)
    start = (page - 1) * limit
    end = start + limit
    
    movies = []
    for idx in range(start, min(end, total)):
        row = df.iloc[idx]
        movies.append(_df_to_movie(row))
    
    return PaginatedMovies(
        total=total,
        page=page,
        limit=limit,
        movies=movies
    )

@router.get("/movie/{identifier}", response_model=Movie)
def get_movie_details(identifier: str):
    """Get details of a specific movie by tmdb_id (int) or title (string)."""
    df, _ = _load_data()
    
    # Try to parse as integer (tmdb_id)
    try:
        tmdb_id = int(identifier)
        movie = df[df["tmdb_id"] == tmdb_id]
    except ValueError:
        # Not an integer, treat as title
        # Decode URL encoding
        from urllib.parse import unquote
        title = unquote(identifier)
        # Try exact match first (case-insensitive)
        movie = df[df["title"].astype(str).str.lower() == title.lower()]
        # If not found, try contains match
        if movie.empty:
            movie = df[df["title"].astype(str).str.lower().str.contains(title.lower(), na=False, regex=False)]
    
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return _df_to_movie(movie.iloc[0])

@router.get("/recommend/movie/{tmdb_id}", response_model=List[Movie])
def get_recommendations(
    tmdb_id: int,
    top_k: int = Query(10, ge=1, le=50),
    user_id: Optional[int] = Query(None),
    user_rating: Optional[float] = Query(None, ge=0.5, le=5.0)
):
    """
    Get recommendations for a movie.
    - If user has rated the movie: use LightGCN
    - If user hasn't rated: use BERT
    """
    df, _ = _load_data()
    
    # Check if movie exists
    movie = df[df["tmdb_id"] == tmdb_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Determine which recommendation system to use
    use_cf = False
    actual_user_rating = user_rating
    
    if user_id is not None:
        # Get actual user rating if not provided
        if actual_user_rating is None:
            actual_user_rating = get_user_rating(user_id, tmdb_id)
        
        # If user has rated, use CF
        if actual_user_rating is not None:
            use_cf = True
    
    recommendations = None
    
    if use_cf:
        # Use LightGCN for collaborative filtering
        try:
            recommendations = rcm_lightgcn.recommend_by_movie(
                movie_id=tmdb_id,
                top_k=top_k,
                user_rating=actual_user_rating
            )
        except Exception as e:
            print(f"LightGCN failed: {e}, falling back to BERT")
            use_cf = False
    
    if not use_cf or recommendations is None or recommendations.empty:
        # Use BERT-based recommendations
        try:
            movie_title = movie.iloc[0]["title"]
            recommendations = rcm_bert.recommend_by_title(movie_title, top_k=top_k)
        except Exception as e:
            print(f"BERT recommendation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
    
    # Merge with df to get full movie info
    if recommendations is not None and not recommendations.empty:
        # Create poster_map from recommendations before merging
        poster_map = {}
        score_map = {}
        for idx, row in recommendations.iterrows():
            rec_tmdb_id = int(row.get("tmdb_id", 0))
            if rec_tmdb_id:
                poster_map[rec_tmdb_id] = row.get("poster_path", "")
                score_map[rec_tmdb_id] = row.get("score", None)
        
        # Merge with df
        recommendations = recommendations.merge(
            df,
            on="tmdb_id",
            how="left",
            suffixes=("", "_df")
        )
        
        # Prioritize poster_path from recommendations, fallback to df
        if "poster_path_df" in recommendations.columns:
            recommendations["poster_path"] = recommendations.apply(
                lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
                axis=1
            )
        
        # Ensure score is preserved
        if "score" not in recommendations.columns or recommendations["score"].isna().all():
            recommendations["score"] = recommendations["tmdb_id"].map(score_map)
    
    # Convert to Movie objects
    movies = []
    if recommendations is not None and not recommendations.empty:
        for _, row in recommendations.iterrows():
            score = row.get("score")
            movies.append(_df_to_movie(row, score=float(score) if pd.notna(score) else None))
    
    return movies

@router.get("/recommend/director/{director_name}", response_model=List[Movie])
def recommend_by_director(director_name: str, top_k: int = Query(10, ge=1, le=50)):
    """Get recommendations by director name."""
    recommendations = rcm_bert.recommend_by_director(director_name, top_k=top_k)
    df, _ = _load_data()
    
    # Merge with df
    recommendations = recommendations.merge(df, on="tmdb_id", how="left", suffixes=("", "_df"))
    
    # Preserve poster_path and score from recommendations
    poster_map = {int(r["tmdb_id"]): r.get("poster_path", "") for _, r in recommendations.iterrows()}
    score_map = {int(r["tmdb_id"]): r.get("score", None) for _, r in recommendations.iterrows()}
    
    if "poster_path_df" in recommendations.columns:
        recommendations["poster_path"] = recommendations.apply(
            lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
            axis=1
        )
    
    movies = []
    for _, row in recommendations.iterrows():
        score = score_map.get(int(row["tmdb_id"]))
        movies.append(_df_to_movie(row, score=float(score) if score is not None else None))
    
    return movies

@router.get("/recommend/actor/{actor_name}", response_model=List[Movie])
def recommend_by_actor(actor_name: str, top_k: int = Query(10, ge=1, le=50)):
    """Get recommendations by actor name."""
    recommendations = rcm_bert.recommend_by_actor(actor_name, top_k=top_k)
    df, _ = _load_data()
    
    # Merge with df
    recommendations = recommendations.merge(df, on="tmdb_id", how="left", suffixes=("", "_df"))
    
    # Preserve poster_path and score from recommendations
    poster_map = {int(r["tmdb_id"]): r.get("poster_path", "") for _, r in recommendations.iterrows()}
    score_map = {int(r["tmdb_id"]): r.get("score", None) for _, r in recommendations.iterrows()}
    
    if "poster_path_df" in recommendations.columns:
        recommendations["poster_path"] = recommendations.apply(
            lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
            axis=1
        )
    
    movies = []
    for _, row in recommendations.iterrows():
        score = score_map.get(int(row["tmdb_id"]))
        movies.append(_df_to_movie(row, score=float(score) if score is not None else None))
    
    return movies

@router.get("/recommend/description", response_model=List[Movie])
def recommend_by_description(
    description: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=50)
):
    """Get recommendations by description text."""
    recommendations = rcm_bert.recommend_by_description(description, top_k=top_k)
    df, _ = _load_data()
    
    # Merge with df
    recommendations = recommendations.merge(df, on="tmdb_id", how="left", suffixes=("", "_df"))
    
    # Preserve poster_path and score from recommendations
    poster_map = {int(r["tmdb_id"]): r.get("poster_path", "") for _, r in recommendations.iterrows()}
    score_map = {int(r["tmdb_id"]): r.get("score", None) for _, r in recommendations.iterrows()}
    
    if "poster_path_df" in recommendations.columns:
        recommendations["poster_path"] = recommendations.apply(
            lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
            axis=1
        )
    
    movies = []
    for _, row in recommendations.iterrows():
        score = score_map.get(int(row["tmdb_id"]))
        movies.append(_df_to_movie(row, score=float(score) if score is not None else None))
    
    return movies

@router.get("/search", response_model=List[Movie])
def search_movies(
    q: str = Query(..., min_length=1),
    top_k: int = Query(20, ge=1, le=50)
):
    """Search movies using BERT."""
    df, _ = _load_data()
    
    # Get recommendations from BERT
    recommendations = rcm_bert.recommend(q, top_k=top_k)
    
    if recommendations is None or recommendations.empty:
        return []
    
    # Create poster_map and score_map from recommendations before merging
    poster_map = {}
    score_map = {}
    for idx, row in recommendations.iterrows():
        rec_tmdb_id = int(row.get("tmdb_id", 0))
        if rec_tmdb_id:
            poster_map[rec_tmdb_id] = row.get("poster_path", "")
            score_map[rec_tmdb_id] = row.get("score", None)
    
    # Merge with df to get full movie info
    recommendations = recommendations.merge(
        df,
        on="tmdb_id",
        how="left",
        suffixes=("", "_df")
    )
    
    # Prioritize poster_path from recommendations, fallback to df
    if "poster_path_df" in recommendations.columns:
        recommendations["poster_path"] = recommendations.apply(
            lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
            axis=1
        )
    
    # Ensure score is preserved
    if "score" not in recommendations.columns or recommendations["score"].isna().all():
        recommendations["score"] = recommendations["tmdb_id"].map(score_map)
    
    # Convert to Movie objects
    movies = []
    for _, row in recommendations.iterrows():
        score = row.get("score")
        movies.append(_df_to_movie(row, score=float(score) if pd.notna(score) else None))
    
    return movies

@router.get("/recommend/user/{user_id}", response_model=List[Movie])
def get_user_recommendations(
    user_id: int,
    top_k: int = Query(30, ge=1, le=100)
):
    """Get personalized recommendations for a user based on their activity logs."""
    df, _ = _load_data()
    
    # Get user logs
    logs = db.get_user_logs(user_id, limit=100)
    
    if not logs:
        # No logs, return empty or popular movies
        return []
    
    # Get recommendations from BERT based on logs
    recommendations = rcm_bert.recommend_for_user_from_logs(logs, top_k=top_k)
    
    if recommendations is None or recommendations.empty:
        return []
    
    # Merge with df
    recommendations = recommendations.merge(df, on="tmdb_id", how="left", suffixes=("", "_df"))
    
    # Preserve poster_path and score
    poster_map = {int(r["tmdb_id"]): r.get("poster_path", "") for _, r in recommendations.iterrows()}
    score_map = {int(r["tmdb_id"]): r.get("score", None) for _, r in recommendations.iterrows()}
    
    if "poster_path_df" in recommendations.columns:
        recommendations["poster_path"] = recommendations.apply(
            lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
            axis=1
        )
    
    movies = []
    for _, row in recommendations.iterrows():
        score = score_map.get(int(row["tmdb_id"]))
        movies.append(_df_to_movie(row, score=float(score) if score is not None else None))
    
    return movies

@router.get("/recommend/{identifier}", response_model=List[Movie])
def get_recommendations_by_identifier(
    identifier: str,
    top_k: int = Query(10, ge=1, le=50),
    user_id: Optional[int] = Query(None),
    user_rating: Optional[float] = Query(None, ge=0.5, le=5.0)
):
    """
    Get recommendations for a movie by title or tmdb_id.
    This endpoint must be placed after all specific routes like /recommend/director/, /recommend/actor/, etc.
    - If user has rated the movie: use LightGCN
    - If user hasn't rated: use BERT
    """
    df, _ = _load_data()
    
    # Try to parse as integer (tmdb_id)
    try:
        tmdb_id = int(identifier)
        movie = df[df["tmdb_id"] == tmdb_id]
    except ValueError:
        # Not an integer, treat as title
        from urllib.parse import unquote
        title = unquote(identifier)
        # Try exact match first (case-insensitive)
        movie = df[df["title"].astype(str).str.lower() == title.lower()]
        # If not found, try contains match
        if movie.empty:
            movie = df[df["title"].astype(str).str.lower().str.contains(title.lower(), na=False, regex=False)]
        if not movie.empty:
            tmdb_id = int(movie.iloc[0]["tmdb_id"])
        else:
            raise HTTPException(status_code=404, detail="Movie not found")
    
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    # Call the main recommendation function
    return get_recommendations(tmdb_id, top_k, user_id, user_rating)

@router.post("/rate", response_model=dict)
def rate_movie(request: RateRequest):
    """Rate a movie and return LightGCN recommendations."""
    if request.rating < 0.5 or request.rating > 5.0:
        raise HTTPException(status_code=400, detail="Rating must be between 0.5 and 5.0")
    
    # Update in-memory ratings
    _, ratings = _load_data()
    
    user_id_int = int(request.user_id)
    tmdb_id_int = int(request.movie_id)  # movie_id từ frontend = tmdb_id
    rating_float = float(request.rating)
    
    mask = (ratings["userId"] == user_id_int) & (ratings["tmdb_id"] == tmdb_id_int)
    
    if mask.any():
        ratings.loc[mask, "rating"] = rating_float
    else:
        new_row = pd.DataFrame({
            "userId": [user_id_int],
            "tmdb_id": [tmdb_id_int],
            "rating": [rating_float],
            "timestamp": [0]
        })
        global _ratings_df
        _ratings_df = pd.concat([ratings, new_row], ignore_index=True)
    
    # Invalidate average ratings cache
    global _average_ratings_cache
    _average_ratings_cache = None
    
    # Update in recommendation modules
    try:
        rcm_lightgcn.add_rating(user_id_int, tmdb_id_int, rating_float)
    except Exception as e:
        print(f"Error updating LightGCN rating: {e}")
    
    # Get LightGCN recommendations ngay sau khi rate
    recommendations = []
    try:
        cf_recs = rcm_lightgcn.recommend_by_movie(
            movie_id=tmdb_id_int,
            top_k=10,
            user_rating=rating_float
        )
        
        if cf_recs is not None and not cf_recs.empty:
            df, _ = _load_data()
            # Merge với df để lấy đầy đủ thông tin
            cf_recs = cf_recs.merge(df, on="tmdb_id", how="left", suffixes=("", "_df"))
            
            # Preserve poster_path và score
            poster_map = {int(r["tmdb_id"]): r.get("poster_path", "") for _, r in cf_recs.iterrows()}
            score_map = {int(r["tmdb_id"]): r.get("score", None) for _, r in cf_recs.iterrows()}
            
            if "poster_path_df" in cf_recs.columns:
                cf_recs["poster_path"] = cf_recs.apply(
                    lambda r: poster_map.get(int(r["tmdb_id"]), "") or r.get("poster_path_df", ""),
                    axis=1
                )
            
            # Convert to Movie objects
            for _, row in cf_recs.iterrows():
                score = score_map.get(int(row["tmdb_id"]))
                recommendations.append(_df_to_movie(row, score=float(score) if score is not None else None))
    except Exception as e:
        print(f"Error getting LightGCN recommendations: {e}")
    
    return {
        "message": "Rating updated successfully",
        "recommendations": [m.dict() for m in recommendations]
    }

@router.get("/ratings/{user_id}", response_model=List[dict])
def get_user_ratings(user_id: int):
    """Get all ratings for a specific user."""
    _, ratings = _load_data()
    
    user_ratings = ratings[ratings["userId"] == user_id]
    
    if user_ratings.empty:
        return []
        
    return user_ratings[["tmdb_id", "rating"]].to_dict(orient="records")

