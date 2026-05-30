#!/usr/bin/env python3
"""
CLI helper untuk membuat token .bin dari corpus memakai tokenizer project.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APPS_DIR = PROJECT_ROOT / "apps"
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

from config import get_config  # noqa: E402
from dataset2 import pretokenize_corpus  # noqa: E402
from tokenizer2 import build_tokenizer_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretokenize corpus text menjadi token .bin untuk Dataset/DataLoader LLM.",
    )
    parser.add_argument(
        "--env",
        default="development",
        choices=["development", "production", "testing"],
        help="Config environment yang dipakai untuk default path dan dtype.",
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Path corpus .txt. Default: CONFIG.CORPUS_FILE.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path output .bin. Default: CONFIG.PRETOKENIZE_OUTPUT_PATH.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Dtype output, misalnya int32 atau int64. Default: CONFIG.PRETOKENIZE_DTYPE.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Cetak progress setiap N baris. Default: CONFIG.PRETOKENIZE_LOG_INTERVAL.",
    )
    parser.add_argument(
        "--vocab-path",
        default=None,
        help="Override path vocab JSON. Default: CONFIG.VOCAB_PATH.",
    )
    parser.add_argument(
        "--token2id-path",
        default=None,
        help="Override path token2id JSON. Default: CONFIG.TOKEN2ID_PATH.",
    )
    parser.add_argument(
        "--id2token-path",
        default=None,
        help="Override path id2token JSON. Default: CONFIG.ID2TOKEN_PATH.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=32,
        help="Jumlah token ID awal yang ditampilkan setelah selesai. 0 untuk disable.",
    )
    parser.add_argument(
        "--decode-preview",
        action="store_true",
        help="Decode token preview untuk sanity check singkat.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Izinkan overwrite output .bin jika sudah ada.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.env)

    corpus_path = Path(args.corpus_path or cfg.CORPUS_FILE)
    output_path = Path(args.output_path or cfg.PRETOKENIZE_OUTPUT_PATH)
    dtype = np.dtype(args.dtype or cfg.PRETOKENIZE_DTYPE)
    log_interval = args.log_interval or cfg.PRETOKENIZE_LOG_INTERVAL

    vocab_path = Path(args.vocab_path or cfg.VOCAB_PATH)
    token2id_path = Path(args.token2id_path or cfg.TOKEN2ID_PATH)
    id2token_path = Path(args.id2token_path or cfg.ID2TOKEN_PATH)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    tokenizer = build_tokenizer_from_config(cfg)
    tokenizer.load_all(str(vocab_path), str(token2id_path), str(id2token_path))

    print("Environment:", args.env)
    print("Corpus:", corpus_path)
    print("Output:", output_path)
    print("Dtype:", dtype)
    print("Vocab:", vocab_path)
    print("Token2ID:", token2id_path)
    print("ID2Token:", id2token_path)

    pretokenize_corpus(
        corpus_path=str(corpus_path),
        tokenizer=tokenizer,
        output_path=str(output_path),
        dtype=dtype,
        log_interval=log_interval,
    )

    arr = np.fromfile(output_path, dtype=dtype)
    print("Verification total tokens:", len(arr))

    if args.preview_tokens > 0:
        preview = arr[: args.preview_tokens].tolist()
        print(f"First {len(preview)} token ids:", preview)
        if args.decode_preview:
            print("Decoded preview:", tokenizer.decode(preview))


if __name__ == "__main__":
    main()
