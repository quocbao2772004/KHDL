import sqlite3
import os

DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")

def check_logs():
    if not os.path.exists(DB_NAME):
        print("Database not found.")
        return

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    try:
        c.execute("SELECT * FROM logs ORDER BY id ASC")
        rows = c.fetchall()
        if not rows:
            print("No logs found.")
        else:
            print("Recent Logs:")
            for row in rows:
                print(row)
    except Exception as e:
        print(f"Error reading logs: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_logs()
