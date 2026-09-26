from __future__ import annotations

import argparse
from pathlib import Path

import requests


MODEL_ID = "openai/clip-vit-base-patch32"
FILES = [
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/clip-vit-base-patch32")
    parser.add_argument("--verify-ssl", action="store_true")
    return parser.parse_args()


def download_file(file_name: str, output_dir: Path, verify_ssl: bool) -> None:
    url = f"https://huggingface.co/{MODEL_ID}/resolve/main/{file_name}"
    output_path = output_dir / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"exists {output_path}")
        return

    print(f"download {file_name}")
    with requests.get(url, stream=True, verify=verify_ssl, timeout=300) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for file_name in FILES:
        download_file(file_name, output_dir, verify_ssl=args.verify_ssl)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
