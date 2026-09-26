from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import count_labels_from_parquet, prepare_balanced_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Priprema izbalansiran lokalni skup slika galaksija.")
    parser.add_argument("--output-dir", default="data/gz_hubble_500")
    parser.add_argument("--cache-root", default="data/gz_hubble_small/_parquet_cache",
                        help="Gde su (ili gde ce biti skinuti) parquet fajlovi")
    parser.add_argument("--train-per-class", type=int, default=450)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-ssl-verification", action="store_true")
    parser.add_argument("--no-download", action="store_true",
                        help="Ne skidaj nista, koristi samo postojeci kes")
    parser.add_argument("--count-only", action="store_true",
                        help="Samo ispisi raspodelu klasa u izvornom skupu i izadji")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_ssl_verification:
        os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["REQUESTS_CA_BUNDLE"] = ""

    if args.count_only:
        # Raspodela klasa u izvornom skupu - tabela za poglavlje 3.2.2 rada.
        report = {}
        for split in ("train", "test"):
            counts = count_labels_from_parquet(args.cache_root, split)
            report[split] = counts
            print(f"\n=== {split} ===")
            total = sum(counts.values())
            for label, count in counts.items():
                print(f"  {label:38s} {count:7d}  ({100 * count / max(total, 1):5.2f}%)")
            print(f"  {'UKUPNO OZNACENIH':38s} {total:7d}")
        Path("raspodela_klasa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nsacuvano: raspodela_klasa.json")
        return

    info = prepare_balanced_dataset(
        output_dir=args.output_dir,
        cache_root=args.cache_root,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        verify_ssl=not args.disable_ssl_verification,
        download_if_missing=not args.no_download,
    )
    print(json.dumps({k: v for k, v in info.items() if k != "available_in_source"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
