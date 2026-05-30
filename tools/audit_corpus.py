#!/usr/bin/env python3
import re
import sys
import os
from pathlib import Path

def audit_corpus(file_path):
    path = Path(file_path).resolve()
    if not path.exists():
        print(f"[!] File tidak ditemukan: {file_path}")
        print(f"[*] Lokasi absolut yang dicari: {path}")
        
        # Membantu mencari file jika ada typo atau masalah case-sensitivity
        print(f"[*] Memeriksa isi direktori: {path.parent}")
        if path.parent.exists():
            all_files = [f.name for f in path.parent.iterdir()]
            print(f"[*] File yang tersedia di folder data: {all_files}")
            
            # Cari file yang mirip namanya
            suggestion = [f for f in all_files if path.name.lower() in f.lower()]
            if suggestion:
                print(f"[?] Apakah maksud Anda salah satu dari ini? {suggestion}")
        else:
            print(f"[!] Direktori {path.parent} tidak ditemukan.")
        print("-" * 30)
        return

    # Cek izin akses (Permissions)
    if not os.access(path, os.R_OK):
        print(f"[!] File ditemukan tetapi TIDAK BISA DIBACA (Permission Denied).")
        print(f"[*] Coba jalankan perintah ini di terminal: chmod 644 {file_path}")
        print("-" * 30)
        return

    file_size = os.path.getsize(path)
    print(f"[*] Mengaudit: {file_path}")
    print(f"[*] Ukuran File: {file_size / (1024*1024*1024):.2f} GB")
    print("-" * 30)

    # Definisi Pola "Kotor"
    patterns = {
        "Sitasi Angka [12]": r'\[\d+\]',
        "Tag Wikipedia [edit/rujukan]": r'\[[a-zA-Z\s\?]+\]',
        "Header Wikitext (==)": r'==+ .*? ==+',
        "Template Wiki ({{...}})": r'\{\{.*?\}\}',
        "URL Mentah (http)": r'http[s]?://\S+',
        "Entitas HTML (&...;)": r'&\w+;',
        "Baris Kosong Berlebih (3+)": r'\n{3,}'
    }

    counts = {label: 0 for label in patterns}
    samples = {label: set() for label in patterns}
    is_clean = True

    # Gunakan chunk size 100MB agar aman di RAM 2GB
    CHUNK_SIZE = 100 * 1024 * 1024 

    try:
        with open(path, 'r', encoding='utf-8') as f:
            overlap = ""
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                # Gabungkan dengan sisa dari chunk sebelumnya
                text_to_scan = overlap + chunk
                
                for label, regex in patterns.items():
                    for match in re.finditer(regex, text_to_scan):
                        # Hindari double counting: hanya hitung jika match 
                        # berakhir di dalam chunk baru (bukan sepenuhnya di overlap)
                        if not overlap or match.end() > len(overlap):
                            counts[label] += 1
                            if len(samples[label]) < 3:
                                samples[label].add(match.group())
                
                # Simpan 1KB terakhir sebagai overlap untuk iterasi berikutnya
                overlap = chunk[-1024:] if len(chunk) > 1024 else chunk
                
    except Exception as e:
        print(f"[!] Terjadi kesalahan saat membaca file: {e}")
        return

    for label in patterns:
        if counts[label] > 0:
            is_clean = False
            print(f"[!] {label}: {counts[label]} temuan. Contoh: {list(samples[label])}")
        else:
            print(f"[OK] {label}: Bersih.")
    print("-" * 30)
    if is_clean:
        print("[HASIL] Dataset sudah BERSIH (identik dengan standar corpus.txt).")
    else:
        print("[HASIL] Dataset masih KOTOR. Anda perlu menjalankan skrip pembersih.")

if __name__ == "__main__":
    # Gunakan argumen pertama jika ada, jika tidak gunakan default corpus_ori.txt
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "/home/ubuntu/hai-project/data/corpus_ori.txt"
    
    audit_corpus(target_file)