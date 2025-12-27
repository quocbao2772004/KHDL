# -*- coding: utf-8 -*-
"""
Preprocess data for BERT embeddings
Chuẩn hóa dữ liệu để search chính xác hơn:
- Director: chuyển rick_roessler -> rick roessler (và giữ cả 2 format)
- Actor: loại bỏ ngoặc, chuyển _ thành space
- Title: normalize để search chính xác
"""

import pandas as pd
import ast
import re
import os

# Paths
INPUT_CSV = "/home/anonymous/code/KHDL/btl/web/backend/data/tmdb_cleaned.csv"
OUTPUT_CSV = "/home/anonymous/code/KHDL/btl/web/backend/data/tmdb_cleaned_for_bert.csv"

def normalize_name(name):
    """
    Chuẩn hóa tên người: chuyển _ thành space, loại bỏ ngoặc, normalize
    Ví dụ: rick_roessler -> rick roessler
           denis_rovira_van_boekholt -> denis rovira van boekholt
           "John Doe (voice)" -> john doe
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    # Loại bỏ ngoặc và nội dung trong ngoặc
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Chuyển _ thành space
    name = name.replace('_', ' ')
    
    # Loại bỏ khoảng trắng thừa và lowercase
    name = ' '.join(name.split()).lower().strip()
    
    return name

def normalize_title(title):
    """
    Chuẩn hóa title: loại bỏ ký tự đặc biệt, normalize
    """
    if pd.isna(title) or not isinstance(title, str):
        return ""
    
    # Giữ nguyên title nhưng normalize khoảng trắng
    title = ' '.join(title.split()).strip()
    
    return title

def to_list_normalized(x):
    """
    Parse list và normalize từng phần tử
    """
    if pd.isna(x):
        return []
    
    if isinstance(x, list):
        return [normalize_name(str(i)) for i in x if pd.notna(i) and str(i).strip()]
    
    if isinstance(x, str):
        s = x.strip()
        # Nếu là string dạng list: "['a','b']"
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [normalize_name(str(i)) for i in v if pd.notna(i) and str(i).strip()]
            except:
                pass
        # Nếu là chuỗi thường, normalize
        normalized = normalize_name(s)
        return [normalized] if normalized else []
    
    return []

def make_bert_text(row):
    """
    Tạo bert_text với cả format gốc và format đã normalize để search tốt hơn
    Ưu tiên title bằng cách thêm nhiều lần
    """
    parts = []
    
    # Title - thêm nhiều lần để ưu tiên khi search
    title = str(row.get("title", "")).strip()
    if title:
        # Thêm title gốc 3 lần để tăng weight
        parts.extend([title] * 3)
        # Thêm title lowercase
        parts.append(title.lower())
        # Thêm title normalized (loại bỏ ký tự đặc biệt)
        title_norm = normalize_title(title).lower()
        if title_norm and title_norm != title.lower():
            parts.append(title_norm)
        # Thêm từng từ trong title riêng lẻ (để search "Avatar" ra "Avatar" chứ không phải "Avatar 4")
        title_words = title.lower().split()
        parts.extend(title_words)
    
    # Genres - giữ nguyên format
    genres = row.get("genres", [])
    if isinstance(genres, list):
        parts.extend([str(g).strip() for g in genres if str(g).strip()])
    elif isinstance(genres, str):
        parts.append(genres.strip())
    
    # Keywords - giữ nguyên format
    keywords = row.get("keywords", [])
    if isinstance(keywords, list):
        parts.extend([str(k).strip() for k in keywords if str(k).strip()])
    elif isinstance(keywords, str):
        parts.append(keywords.strip())
    
    # Cast - normalize và thêm cả format gốc
    cast_list = row.get("cast_top5", [])
    if isinstance(cast_list, list):
        for actor in cast_list:
            actor_str = str(actor).strip()
            if actor_str:
                # Thêm format gốc (có thể có _)
                parts.append(actor_str)
                # Thêm format normalized (không có _, không có ngoặc)
                actor_norm = normalize_name(actor_str)
                if actor_norm and actor_norm != actor_str.lower():
                    parts.append(actor_norm)
    elif isinstance(cast_list, str):
        parts.append(cast_list.strip())
        actor_norm = normalize_name(cast_list)
        if actor_norm and actor_norm != cast_list.lower():
            parts.append(actor_norm)
    
    # Director - thêm cả format gốc và format normalized, thêm 2 lần để tăng weight
    director = str(row.get("director", "")).strip()
    if director:
        # Thêm format gốc (có thể có _) 2 lần
        parts.extend([director] * 2)
        # Thêm format normalized (không có _) 2 lần
        director_norm = normalize_name(director)
        if director_norm:
            parts.extend([director_norm] * 2)
            # Thêm từng từ trong tên director riêng lẻ
            director_words = director_norm.split()
            parts.extend(director_words)
    
    # Overview
    overview = str(row.get("overview", "")).strip()
    if overview:
        parts.append(overview)
    
    return " ".join(parts).strip()

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} movies")
    
    # Parse các cột list
    print("Parsing list columns...")
    for col in ["genres", "keywords", "cast_top5"]:
        if col in df.columns:
            df[col] = df[col].apply(to_list_normalized)
        else:
            df[col] = [[]] * len(df)
    
    # Normalize director
    print("Normalizing director...")
    df["director"] = df["director"].apply(lambda x: normalize_name(x) if pd.notna(x) else "")
    
    # Normalize title (tạo thêm cột title_normalized để search)
    print("Normalizing title...")
    df["title_normalized"] = df["title"].apply(normalize_title)
    
    # Tạo bert_text với cả format gốc và normalized
    print("Creating bert_text...")
    df["bert_text"] = df.apply(make_bert_text, axis=1)
    
    # Kiểm tra
    print("\nSample bert_text:")
    print(df[["title", "director", "bert_text"]].head(3))
    print(f"\nEmpty bert_text: {(df['bert_text'].str.len() == 0).sum()}")
    
    # Lưu file đã preprocess
    print(f"\nSaving preprocessed data to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("Done!")
    
    # Test một vài trường hợp
    print("\n=== Testing normalization ===")
    test_directors = ["rick_roessler", "denis_rovira_van_boekholt", "Rich Lee"]
    for d in test_directors:
        normalized = normalize_name(d)
        print(f"'{d}' -> '{normalized}'")
    
    test_actors = ["John_Doe", "Jane Smith (voice)", "rick_roessler"]
    for a in test_actors:
        normalized = normalize_name(a)
        print(f"'{a}' -> '{normalized}'")

if __name__ == "__main__":
    main()

