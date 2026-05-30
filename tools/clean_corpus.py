#!/usr/bin/env python3
import argparse
import html
import os
import re
import sys
from pathlib import Path


DIRTY_PATTERNS = {
    "numeric_citation": re.compile(r"\[\d+\]"),
    "wiki_tag": re.compile(r"\[[a-zA-Z\s\?]+\]", re.DOTALL),
    "wiki_header": re.compile(r"==+\s.*?\s==+", re.DOTALL),
    "wiki_template": re.compile(r"\{\{.*?\}\}", re.DOTALL),
    "raw_url": re.compile(r"https?://\S+"),
    "html_entity": re.compile(r"&\w+;"),
}


def scrub_dirty_patterns(text: str) -> str:
    # Nested bracket artifacts like "[ [ ] ]" need more than one substitution pass.
    for _ in range(8):
        before = text
        text = DIRTY_PATTERNS["raw_url"].sub("", text)
        text = DIRTY_PATTERNS["wiki_template"].sub("", text)
        text = DIRTY_PATTERNS["wiki_header"].sub("", text)
        text = DIRTY_PATTERNS["numeric_citation"].sub("", text)
        text = DIRTY_PATTERNS["wiki_tag"].sub("", text)
        text = DIRTY_PATTERNS["html_entity"].sub("", text)
        if text == before:
            break
    return text


def clean_text(text: str) -> str:
    """Clean a text block using the same dirty-pattern family as audit_corpus.py."""
    text = html.unescape(text)
    text = scrub_dirty_patterns(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_line(line: str) -> str:
    return clean_text(line)


def write_cleaned_block(dst, block: str, blank_streak: int) -> tuple[int, int]:
    lines_written = 0
    cleaned = clean_text(block)

    for line in cleaned.splitlines():
        line = clean_line(line).strip()
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                dst.write("\n")
                lines_written += 1
            continue

        blank_streak = 0
        dst.write(line + "\n")
        lines_written += 1

    return lines_written, blank_streak


def clean_corpus(
    input_path: Path,
    output_path: Path,
    encoding: str = "utf-8",
    max_block_chars: int = 2_000_000,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output tidak boleh sama dengan input. Gunakan file sementara lalu rename manual jika perlu.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_read = 0
    lines_written = 0
    blank_streak = 0
    block_lines: list[str] = []
    block_chars = 0

    with input_path.open("r", encoding=encoding, errors="replace", newline="") as src, output_path.open(
        "w", encoding=encoding, newline="\n"
    ) as dst:
        for raw_line in src:
            lines_read += 1
            block_lines.append(raw_line)
            block_chars += len(raw_line)

            if raw_line.strip() == "" or block_chars >= max_block_chars:
                written, blank_streak = write_cleaned_block(dst, "".join(block_lines), blank_streak)
                lines_written += written
                block_lines.clear()
                block_chars = 0

            if lines_read % 1_000_000 == 0:
                print(f"[*] Diproses: {lines_read:,} baris...", flush=True)

        if block_lines:
            written, blank_streak = write_cleaned_block(dst, "".join(block_lines), blank_streak)
            lines_written += written

    return lines_read, lines_written


def default_output_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Membersihkan corpus berdasarkan pola yang dicek oleh tools/audit_corpus.py."
    )
    parser.add_argument("input", help="Path file corpus kotor, contoh: data/corpus_ori.txt")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path output. Default: <nama_input>_clean.txt di folder yang sama.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding file input/output. Default: utf-8.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=2_000_000,
        help="Batas karakter blok pembersihan untuk menangkap pola multi-line. Default: 2000000.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output_for(input_path).resolve()

    print(f"[*] Input : {input_path}")
    print(f"[*] Output: {output_path}")

    try:
        print(f"[*] Ukuran: {os.path.getsize(input_path) / (1024 * 1024 * 1024):.2f} GB")
        lines_read, lines_written = clean_corpus(input_path, output_path, args.encoding, args.max_block_chars)
    except Exception as exc:
        print(f"[!] Gagal membersihkan corpus: {exc}")
        return 1

    print("-" * 30)
    print(f"[OK] Baris dibaca  : {lines_read:,}")
    print(f"[OK] Baris ditulis : {lines_written:,}")
    print(f"[OK] File bersih   : {output_path}")
    print(f"[*] Audit ulang dengan: tools\\audit_corpus.py {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
