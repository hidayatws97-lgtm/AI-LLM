#!/usr/bin/env python3
"""
Script untuk mengambil corpus dari Wikipedia Bahasa Indonesia menggunakan Hugging Face Datasets
dan menambahkannya ke corpus.txt yang sudah ada.
"""

import sys
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import CONFIG

def fetch_wikipedia():
    corpus_path = CONFIG.CORPUS_FILE
    print(f"Target file: {corpus_path}")

    print("Mengunduh dataset Wikipedia ID (versi 20220301.id)...")
    # Dataset ini berisi artikel bersih tanpa markup MediaWiki
    try:
        dataset = load_dataset("wikipedia", "20220301.id", trust_remote_code=True)
    except Exception as e:
        print(f"Gagal mengunduh: {e}")
        return

    articles = dataset['train']
    total = len(articles)
    print(f"Ditemukan {total} artikel.")

    # Gunakan mode 'a' (append) agar data lama tidak hilang
    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write("\n\n") # Pemisah antara data lama dan baru
        
        for i, article in enumerate(articles):
            text = article['text'].strip()
            if text:
                # Wikipedia dataset biasanya sudah bersih, 
                # tapi kita pastikan tidak ada baris kosong berlebih
                f.write(text + "\n")
            
            if (i + 1) % 5000 == 0:
                print(f"Progress: {i + 1}/{total} artikel tertulis...")

    print(f"Selesai! Corpus Wikipedia Indonesia telah ditambahkan ke {corpus_path}")
    print(f"Ukuran file sekarang: {corpus_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    fetch_wikipedia()