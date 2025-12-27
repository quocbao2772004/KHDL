from pydantic import BaseModel
from typing import Optional

class Movie(BaseModel):
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    genres: Optional[str] = None
    cast: Optional[str] = None
    director: Optional[str] = None
    poster_path: Optional[str] = None
    score: Optional[float] = None
    average_rating: Optional[float] = None
    tmdb_id: Optional[int] = None

class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class LogEntry(BaseModel):
    user_id: int
    action: str
    details: Optional[str] = None

from typing import List
class PaginatedMovies(BaseModel):
    total: int
    page: int
    limit: int
    movies: List[Movie]
