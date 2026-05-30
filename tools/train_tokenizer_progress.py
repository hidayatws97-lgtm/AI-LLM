#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

# 1. Setup Path agar bisa load modul dari folder /apps
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config
from tokenizer2 import build_tokenizer_from_config

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Tokenizer with Alignment")
    parser.add_argument(
        "--env", 
        default="development", 
        choices=["development", "production", "testing"],
        help="Environment config yang digunakan (jalur file & hyperparams)."
    )
    args = parser.parse_args()

    # 2. Load Config Terpusat
    cfg = get_config(args.env)
    
    print(f"[*] Mode: {args.env}")
    print(f"[*] Target Vocab: {cfg.VOCAB_SIZE}")
    print(f"[*] Hybrid Params: Alpha={cfg.ALPHA}, Beta={cfg.BETA}, Gamma={cfg.GAMMA}")

    # 3. Path Alignment
    corpus_path = cfg.CORPUS_FILE
    if not corpus_path.exists():
        print(f"[!] Error: Corpus tidak ditemukan di {corpus_path}")
        sys.exit(1)

    # 4. Tokenizer Initialization (Aligned dengan config)
    # build_tokenizer_from_config harus menangani ALPHA, BETA, GAMMA, dll.
    tokenizer = build_tokenizer_from_config(cfg)

    print(f"[*] Melatih tokenizer menggunakan: {corpus_path}...")
    
    # Asumsi: tokenizer.train mendukung progress tracking
    tokenizer.train(str(corpus_path), vocab_size=cfg.VOCAB_SIZE, 
                    alpha=cfg.ALPHA, beta=cfg.BETA, gamma=cfg.GAMMA,
                    min_pair_freq=cfg.MIN_PAIR_FREQ, max_merges=cfg.MAX_MERGES, 
                    prune_every=cfg.PRUNE_EVERY)   

    # 5. Save Alignment
    # Pastikan output disimpan ke lokasi yang diharapkan oleh dataset_pretokenizer.py
    tokenizer.save_all(
        str(cfg.VOCAB_PATH),
        str(cfg.TOKEN2ID_PATH),
        str(cfg.ID2TOKEN_PATH)
    )
    
    print(f"\n[OK] Tokenizer berhasil disimpan di: {cfg.BASE_DIR}")

if __name__ == "__main__":
    main()