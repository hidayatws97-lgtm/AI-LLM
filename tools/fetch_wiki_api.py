#!/usr/bin/env python3
"""
Script untuk mengambil konten spesifik dari Wikipedia ID via API.
Bisa mencari berdasarkan topik atau mengambil judul artikel tertentu.
"""

import sys
import argparse
import requests
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import CONFIG

WIKI_API_URL = "https://id.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "HAI-LLM-Bot/1.0 (https://github.com/user/hai-project; your-email@example.com)"
}

def search_wikipedia(query, limit=5):
    """Mencari judul artikel berdasarkan keyword."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json"
    }
    
    response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    
    return [item["title"] for item in data.get("query", {}).get("search", [])]

def get_article_content(title):
    """Mengambil plain text dari artikel berdasarkan judul."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,  # Mengambil plain text, bukan HTML
        "titles": title,
        "format": "json"
    }
    
    response = requests.get(WIKI_API_URL, params=params, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    for page_id in pages:
        if page_id == "-1":
            print(f"  [!] Artikel '{title}' tidak ditemukan.")
            return None
        return pages[page_id].get("extract", "")
    return None

def clean_text(text):
    """Membersihkan teks dari sitasi [1], [edit], dll."""
    # Hapus sitasi seperti [1], [12], [atau 1]
    text = re.sub(r'\[\d+\]', '', text)
    # Hapus tag Wikipedia seperti [butuh rujukan], [edit], dll.
    text = re.sub(r'\[.*?\]', '', text) 
    # Hapus tag bahasa seperti code: en is deprecated
    text = re.sub(r'code:.*? is deprecated', '', text)
    # Hapus baris kosong berlebih
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Fetch Wikipedia ID content via API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic", type=str, help="Cari artikel berdasarkan topik/keyword")
    group.add_argument("--articles", type=str, nargs="+", help="Daftar judul artikel spesifik (pisahkan dengan spasi)")
    
    parser.add_argument("--limit", type=int, default=5, help="Jumlah artikel untuk dicari jika menggunakan --topic")
    parser.add_argument("--append", action="store_true", default=True, help="Tambahkan ke corpus.txt (default: True)")
    
    args = parser.parse_args()
    corpus_path = CONFIG.CORPUS_FILE
    
    titles_to_fetch = []
    
    if args.topic:
        print(f"[*] Mencari artikel dengan topik: '{args.topic}'...")
        titles_to_fetch = search_wikipedia(args.topic, args.limit)
    else:
        titles_to_fetch = args.articles

    if not titles_to_fetch:
        print("[!] Tidak ada artikel untuk diambil.")
        return

    print(f"[*] Menyiapkan pengambilan {len(titles_to_fetch)} artikel...")
    
    fetched_count = 0
    mode = "a" if args.append else "w"
    
    with open(corpus_path, mode, encoding="utf-8") as f:
        for title in titles_to_fetch:
            print(f"  [>] Mengambil: {title}...", end="", flush=True)
            content = get_article_content(title)
            
            if content:
                # Pembersihan dasar: hapus whitespace berlebih dan header yang tidak perlu
                cleaned_content = clean_text(content)
                
                # Tambahkan judul sebagai context (opsional, bagus untuk model)
                f.write(f"\n\n--- {title} ---\n")
                f.write(cleaned_content)
                f.write("\n")
                
                print(" [Selesai]")
                fetched_count += 1
            else:
                print(" [Gagal]")

    print(f"\n[OK] Berhasil menambahkan {fetched_count} artikel ke {corpus_path}")
    print(f"Ukuran file sekarang: {corpus_path.stat().st_size / 1024:.2f} KB")

if __name__ == "__main__":
    main()