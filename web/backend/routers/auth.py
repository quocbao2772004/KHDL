from fastapi import APIRouter, HTTPException
from models import UserRegister, UserLogin, LogEntry
import database as db

router = APIRouter()

@router.post("/register")
def register(user: UserRegister):
    if db.create_user(user.username, user.password):
        return {"message": "User created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Username already exists")

@router.post("/login")
def login(user: UserLogin):
    user_id = db.verify_user(user.username, user.password)
    if user_id:
        return {"user_id": user_id, "username": user.username}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.post("/log")
def log_activity(entry: LogEntry):
    db.log_action(entry.user_id, entry.action, entry.details)
    return {"message": "Logged"}
