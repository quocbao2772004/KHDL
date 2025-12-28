import sqlite3
import bcrypt
import datetime
import os

DB_NAME = "user_data.db"

def init_db():
    """Initialize the database with users and logs tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # Create logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    # Create ratings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            tmdb_id INTEGER,
            rating REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, tmdb_id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(username, password):
    """Create a new user. Returns True if successful, False if username exists."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
        
    return success

def verify_user(username, password):
    """Verify user credentials. Returns user_id if valid, None otherwise."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if user:
        user_id, stored_hash = user
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return user_id
            
    return None

def log_action(user_id, action, details=None):
    """Log a user action."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('INSERT INTO logs (user_id, action, details) VALUES (?, ?, ?)', (user_id, action, details))
    conn.commit()
    conn.close()

def get_logs(limit=50):
    """Get recent logs."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        SELECT logs.timestamp, users.username, logs.action, logs.details 
        FROM logs 
        JOIN users ON logs.user_id = users.id 
        ORDER BY logs.timestamp DESC LIMIT ?
    ''', (limit,))
    
    logs = c.fetchall()
    conn.close()
    return logs

def get_user_logs(user_id, limit=50):
    """Get recent logs for a specific user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        SELECT id, user_id, action, details, timestamp
        FROM logs 
        WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT ?
    ''', (user_id, limit))
    
    logs = c.fetchall()
    conn.close()
    return logs

def save_rating(user_id, tmdb_id, rating):
    """Save or update a user rating."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''
        INSERT OR REPLACE INTO ratings (user_id, tmdb_id, rating, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (user_id, tmdb_id, rating))
    
    conn.commit()
    conn.close()

def get_all_user_ratings(user_id):
    """Get all ratings for a specific user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('SELECT tmdb_id, rating FROM ratings WHERE user_id = ?', (user_id,))
    ratings = c.fetchall()
    conn.close()
    return [{"tmdb_id": r[0], "rating": r[1]} for r in ratings]

def get_all_ratings():
    """Get all ratings from all users (for initial loading/sync)."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('SELECT user_id, tmdb_id, rating FROM ratings')
    ratings = c.fetchall()
    conn.close()
    return ratings

# Initialize DB on import
if not os.path.exists(DB_NAME):
    init_db()
else:
    # Ensure tables exist even if file exists (e.g. empty file)
    init_db()
