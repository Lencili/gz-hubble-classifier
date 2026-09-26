from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MULTIMODAL = ["original", "gamma", "contrast", "highpass", "equalize", "sharpen"]

# (naziv modela, familija, batch size, velicina slike)
DEFAULT_MODELS = [
    ("resnet18", "cnn", 32, 128),
    ("mobilenet_v3_small", "cnn", 32, 128),
    ("efficientnet_b0", "cnn", 24, 128),
    ("densenet121", "cnn", 24, 128),
    ("vit_tiny_patch16_224", "vit", 16, 224),
    ("deit_tiny_patch16_224", "vit", 16, 224),
    ("swin_tiny_patch4_window7_224", "vit", 8, 224),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/gz_hubble_500")
    parser.add_argument("--output-root", default="outputs_500_pretrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    # Pretrening je sada podrazumevan. Prenos ucenja sa ImageNet-a je standard
    # u klasifikaciji morfologije galaksija; trening od nule na nekoliko hiljada
    # slika daje znatno slabije i tesko branjive rezultate.
    parser.add_argument("--from-scratch", action="store_true",
                        help="Treniraj bez pretreniranih tezina (za poredjenje uticaja prenosa ucenja)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Podskup modela po imenu, npr. resnet18 vit_tiny_patch16_224")
    parser.add_argument("--conditions", nargs="+", default=["original", "multimodal"],
                        choices=["original", "multimodal"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Samo ispisi komande")
    return parser.parse_args()


def selected_models(args: argparse.Namespace):
    if not args.models:
        return DEFAULT_MODELS
    wanted = set(args.models)
    chosen = [entry for entry in DEFAULT_MODELS if entry[0] in wanted]
    missing = wanted - {entry[0] for entry in chosen}
    if missing:
        raise SystemExit(f"Nepoznati modeli: {', '.join(sorted(missing))}")
    return chosen


def run_training(model, family, condition, modalities, batch_size, image_size, args) -> Path:
    suffix = "" if not args.from_scratch else "_scratch"
    output_dir = Path(args.output_root) / f"{model}_{condition}{suffix}"

    done = (output_dir / "metrics.json").exists() and (output_dir / "best.ckpt").exists()
    if done and not args.overwrite:
        print(f"preskacem, vec postoji: {output_dir}")
        return output_dir

    command = [
        sys.executable, "-m", "src.gz_classifier.train",
        "--data-dir", args.data_dir,
        "--output-dir", str(output_dir),
        "--model", model,
        "--model-family", family,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(batch_size),
        "--image-size", str(image_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--modalities", *modalities,
    ]
    if args.amp:
        command.append("--amp")
    if args.from_scratch:
        command.append("--no-pretrained")

    print("\n" + " ".join(command))
    if args.dry_run:
        return output_dir
    subprocess.run(command, check=True)
    return output_dir


def collect_summary(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = metrics["config"]
        test = metrics.get("test", metrics.get("final_val", {}))
        row = {
            "run": metrics_path.parent.name,
            "model": config["model"],
            "model_family": config["model_family"],
            "condition": "multimodal" if len(config["modalities"]) > 1 else "original",
            "pretrained": not config.get("no_pretrained", False),
            "modalities": "+".join(config["modalities"]),
            "best_val_macro_f1": metrics.get("best_val_macro_f1", metrics.get("best_macro_f1")),
            "test_macro_f1": test.get("macro_f1"),
            "test_acc": test.get("acc"),
            "epochs_run": metrics["epochs_run"],
            "best_epoch": metrics.get("best_epoch"),
            "best_checkpoint": metrics["checkpoints"]["best"],
        }
        for name, value in (metrics.get("modality_reliance") or {}).items():
            row[f"rel_{name}"] = round(value, 4)
        rows.append(row)
    return rows


def add_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["model"], row["pretrained"])
        by_key.setdefault(key, {})[str(row["condition"])] = row

    for group in by_key.values():
        original = group.get("original")
        multimodal = group.get("multimodal")
        if not (original and multimodal):
            continue
        for metric in ("test_macro_f1", "test_acc", "best_val_macro_f1"):
            base = original.get(metric)
            other = multimodal.get(metric)
            if base is None or other is None:
                continue
            original[f"delta_{metric}"] = 0.0
            multimodal[f"delta_{metric}"] = float(other) - float(base)
    return rows


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = add_deltas(rows)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    models = selected_models(args)

    plan = [(m, f, c, b, s) for m, f, b, s in models for c in args.conditions]
    print(f"planirano {len(plan)} treninga | pretrained={not args.from_scratch} | izlaz: {output_root}")

    for model, family, condition, batch_size, image_size in plan:
        modalities = MULTIMODAL if condition == "multimodal" else ["original"]
        # ISTI batch size u oba uslova.
        #
        # Prva verzija je multimodalnom uslovu prepolovljavala batch "zbog
        # memorije". To je bilo i nepotrebno i stetno za poredjenje:
        # InputAdapter odmah svodi 18 kanala na 3, pa backbone u oba slucaja
        # vidi tenzor istog oblika i trosi istu memoriju. Razlikuje se samo
        # ulazni tenzor - pri batch 32 i 128x128 to je 38 MB naspram 6 MB,
        # zanemarljivo na kartici od 15 GB.
        #
        # Razlicit batch size menja dinamiku ucenja (broj koraka po epohi, sum
        # u gradijentu), pa bi eksperiment merio i tu razliku umesto samo
        # koristi od dodatnih modaliteta.
        run_training(model, family, condition, modalities, batch_size, image_size, args)

    if args.dry_run:
        return
    rows = collect_summary(output_root)
    write_summary(output_root, rows)
    print(f"\nzbirni rezultati: {(output_root / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()
