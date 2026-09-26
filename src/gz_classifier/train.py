from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    HubbleTorchDataset,
    LocalImageDataset,
    available_splits,
    build_label_info,
    build_local_label_info,
    build_synthetic_split,
    load_hubble_split,
)
from .modalities import DEFAULT_MODALITIES
from .models import ModelConfig, build_model, modality_reliance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--model-family", choices=("cnn", "vit"), default="cnn")
    parser.add_argument("--subset", choices=("full", "tiny"), default="full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--data-dir", default=None, help="Use a local prepared image dataset")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-identity-init", action="store_true",
                        help="Nasumicna inicijalizacija adaptera umesto propustanja prvog modaliteta")
    parser.add_argument("--synthetic", action="store_true", help="Use generated data for offline smoke tests")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-checkpoint", default=None, help="Initialize model weights from an existing checkpoint")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, label_info, args, split: str, train: bool) -> DataLoader:
    if isinstance(dataset, str):
        torch_dataset = LocalImageDataset(
            root=dataset,
            split=split,
            label_info=label_info,
            modalities=args.modalities,
            image_size=args.image_size,
            train=train,
        )
    else:
        torch_dataset = HubbleTorchDataset(
            dataset=dataset,
            label_info=label_info,
            modalities=args.modalities,
            image_size=args.image_size,
            train=train,
        )
    return DataLoader(
        torch_dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = False,
) -> dict:
    train = optimizer is not None
    model.train(train)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and train and torch.cuda.is_available())
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        losses.append(float(loss.detach().cpu()))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def scalars(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if not key.startswith("y_")}


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, label_names: Iterable[str]) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "model": args.model,
            "model_family": args.model_family,
            "image_size": args.image_size,
            "modalities": list(args.modalities),
            "pretrained": not args.no_pretrained,
        },
        "label_names": list(label_names),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ podaci
    # Tri odvojena skupa:
    #   train      - ucenje tezina
    #   validation - rano zaustavljanje i izbor najboljeg checkpointa
    #   test       - evaluira se jednom, na samom kraju, iz best.ckpt
    # Ako pripremljeni skup nema "val" split, pada na "test" uz upozorenje,
    # radi kompatibilnosti sa starije pripremljenim podacima.
    val_split = "val"
    test_split = "test"
    leakage_warning = False

    if args.data_dir:
        train_source = val_source = test_source = args.data_dir
        label_info = build_local_label_info(args.data_dir)
        splits = available_splits(args.data_dir)
        if "val" not in splits:
            val_split = "test"
            leakage_warning = True
            print("UPOZORENJE: skup nema 'val' split, rano zaustavljanje koristi test skup.")
    elif args.synthetic:
        train_source = build_synthetic_split(args.limit_train or 128, image_size=args.image_size)
        val_source = test_source = build_synthetic_split(args.limit_val or 64, image_size=args.image_size)
        label_info = build_label_info(train_source)
        leakage_warning = True
    else:
        subset = None if args.subset == "full" else args.subset
        train_source = load_hubble_split("train", subset=subset, limit=args.limit_train)
        val_source = test_source = load_hubble_split("test", subset=subset, limit=args.limit_val)
        label_info = build_label_info(train_source)
        leakage_warning = True

    train_loader = make_loader(train_source, label_info, args, split="train", train=True)
    val_loader = make_loader(val_source, label_info, args, split=val_split, train=False)
    test_loader = make_loader(test_source, label_info, args, split=test_split, train=False)

    # ------------------------------------------------------------------ model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig(
        model=args.model,
        model_family=args.model_family,
        num_classes=len(label_info.names),
        in_channels=3 * len(args.modalities),
        pretrained=not args.no_pretrained,
    )
    model = build_model(model_config, identity_init=not args.no_identity_init).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(payload["state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(
        f"model={args.model} family={args.model_family} pretrained={not args.no_pretrained} "
        f"modalities={len(args.modalities)} klase={len(label_info.names)} uredjaj={device}"
    )

    # ------------------------------------------------------------ trening petlja
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer, amp=args.amp)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, amp=False)
        row = {"epoch": epoch, "train": scalars(train_metrics), "val": scalars(val_metrics)}
        history.append(row)
        print(json.dumps(row))

        save_checkpoint(output_dir / "last.ckpt", model, args, label_info.names)
        if val_metrics["macro_f1"] > best_f1 + args.min_delta:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.ckpt", model, args, label_info.names)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Rano zaustavljanje posle {epoch} epoha (najbolja: {best_epoch}).")
                break

    # ------------------------------------------- finalna evaluacija iz best.ckpt
    # Sve prijavljene metrike opisuju ISTI model - onaj sacuvan kao best.ckpt -
    # a ne poslednju epohu, koja je posle ranog zaustavljanja losija.
    best_path = output_dir / "best.ckpt"
    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
    model.to(device)

    val_final = run_epoch(model, val_loader, criterion, device, optimizer=None, amp=False)
    test_final = run_epoch(model, test_loader, criterion, device, optimizer=None, amp=False)

    test_report = classification_report(
        test_final["y_true"], test_final["y_pred"],
        target_names=label_info.names, output_dict=True, zero_division=0,
    )
    test_confusion = confusion_matrix(test_final["y_true"], test_final["y_pred"]).tolist()

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "labels.json").write_text(json.dumps(label_info.names, indent=2), encoding="utf-8")

    summary = {
        # --- nove, jasne oznake ---
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "val": scalars(val_final),
        "test": {
            **scalars(test_final),
            "classification_report": test_report,
            "confusion_matrix": test_confusion,
        },
        "modality_reliance": modality_reliance(model, list(args.modalities)),
        "validation_is_test": leakage_warning,
        # --- zadrzano radi kompatibilnosti sa app.py i summary.csv ---
        "best_macro_f1": best_f1,
        "final_val": scalars(test_final),
        "classification_report": test_report,
        "confusion_matrix": test_confusion,
        "epochs_run": len(history),
        "early_stopping": {"patience": args.patience, "min_delta": args.min_delta},
        "labels": label_info.names,
        "config": vars(args),
        "checkpoints": {
            "best": str((output_dir / "best.ckpt").resolve()),
            "last": str((output_dir / "last.ckpt").resolve()),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"GOTOVO {output_dir.name}: best val macro-F1 {best_f1:.4f} (epoha {best_epoch}) | "
        f"test macro-F1 {test_final['macro_f1']:.4f} | test acc {test_final['acc']:.4f}"
    )
    if summary["modality_reliance"]:
        print("oslanjanje na modalitete:", json.dumps(summary["modality_reliance"], indent=2))


if __name__ == "__main__":
    main()
