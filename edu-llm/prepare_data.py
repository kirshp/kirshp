"""Скачивает обучающий текст в data/input.txt.

    python prepare_data.py                 # Tiny Shakespeare (~1 МБ)
    python prepare_data.py --url URL       # любой .txt в UTF-8
"""

import argparse
import pathlib
import urllib.request

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="data/input.txt")
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {args.url} ...")
    with urllib.request.urlopen(args.url) as r:
        text = r.read().decode("utf-8", errors="replace")
    out.write_text(text, encoding="utf-8")
    print(f"Готово: {out} — {len(text):,} символов, {len(set(text))} уникальных")


if __name__ == "__main__":
    main()
