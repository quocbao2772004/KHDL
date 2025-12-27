import requests, time, random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import API_KEY

BASE = "https://api.themoviedb.org/3"

session = requests.Session()

def get_json(url, params=None, max_retries=6):
    if params is None:
        params = {}
    params["api_key"] = API_KEY

    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=25)

        # TMDB rate limit
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if retry_after else (2 ** attempt) + random.random()
            time.sleep(wait)
            continue

        if r.status_code >= 500:
            time.sleep((2 ** attempt) + random.random())
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError(f"Retries exceeded: {url}")

def get_movie_details(movie_id):
    det = get_json(f"{BASE}/movie/{movie_id}", params={"language": "en-US"})
    cre = get_json(f"{BASE}/movie/{movie_id}/credits", params={"language": "en-US"})
    key = get_json(f"{BASE}/movie/{movie_id}/keywords")

    cast_names = [c.get("name","") for c in cre.get("cast", [])[:5] if c.get("name")]
    director = ""
    for person in cre.get("crew", []):
        if person.get("job") == "Director":
            director = person.get("name","")
            break

    keywords = [k.get("name","") for k in key.get("keywords", [])[:10] if k.get("name")]

    return {
        "tmdb_id": det.get("id"),
        "title": det.get("title"),
        "overview": det.get("overview"),
        "release_date": det.get("release_date"),
        "genres": [g["name"] for g in det.get("genres", [])],
        "runtime": det.get("runtime"),
        "vote_average": det.get("vote_average"),
        "vote_count": det.get("vote_count"),
        "popularity": det.get("popularity"),
        "cast_top5": cast_names,
        "director": director,
        "keywords": keywords,
        "poster_path": det.get("poster_path"),
    }

def fetch_ids(list_type="popular", pages=50):
    # 50 pages * 20 = ~1000 phim
    ids = []
    for page in range(1, pages + 1):
        data = get_json(f"{BASE}/movie/{list_type}", params={"page": page, "language": "en-US"})
        ids += [m["id"] for m in data.get("results", []) if "id" in m]
    return ids

def collect_ids_fast():
    # Gộp nhiều list để đủ >= 2000 phim nhanh, ít trùng
    ids = []
    ids += fetch_ids("popular", pages=60)      # ~1200
    ids += fetch_ids("top_rated", pages=40)    # ~800
    ids += fetch_ids("now_playing", pages=20)  # ~400
    ids += fetch_ids("upcoming", pages=20)     # ~400
    # unique giữ thứ tự
    ids = list(dict.fromkeys(ids))
    return ids

def crawl_fast(max_workers=10, target=2200):
    ids = collect_ids_fast()
    ids = ids[:max(target, 2000)]  # lấy dư chút
    print("Total IDs prepared:", len(ids))

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(get_movie_details, mid): mid for mid in ids}
        done = 0
        for fut in as_completed(futs):
            mid = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                # lỗi thì bỏ, vì lấy dư rồi
                pass
            done += 1
            if done % 200 == 0:
                print(f"Done {done}/{len(ids)} | collected={len(rows)}")
            if len(rows) >= target:
                break

    df = pd.DataFrame(rows).drop_duplicates(subset=["tmdb_id"])
    return df

df = crawl_fast(max_workers=10, target=2200)
print("Final rows:", len(df))
df.to_csv("btl/data/tmdb_crawled_fast.csv", index=False, encoding="utf-8-sig")
print("Saved: tmdb_crawled_fast.csv")
