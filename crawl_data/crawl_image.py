import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = "btl/data/tmdb_crawled_fast.csv"
IMG_DIR = "btl/data/posters"
BASE_IMG_URL = "https://image.tmdb.org/t/p/w500"

os.makedirs(IMG_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

def download_poster(row):
    pid = row["tmdb_id"]
    path = row.get("poster_path", None)

    if not isinstance(path, str) or not path.strip():
        return None

    url = BASE_IMG_URL + path
    out_path = os.path.join(IMG_DIR, f"{pid}.jpg")

    if os.path.exists(out_path):
        return out_path

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return out_path
    except:
        pass
    return None

# Dùng nhiều luồng để tải nhanh
max_workers = 16  # 16–32 OK vì chỉ tải ảnh
saved = 0

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(download_poster, row) for _, row in df.iterrows()]
    for i, fut in enumerate(as_completed(futures), 1):
        res = fut.result()
        if res:
            saved += 1
        if i % 200 == 0:
            print(f"Downloaded {saved}/{i}")

print("Total posters saved:", saved)
