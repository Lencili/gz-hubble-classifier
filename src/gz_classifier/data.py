from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import random
from pathlib import Path
import re
from io import BytesIO
from typing import Iterable

try:  # datasets i pyarrow trebaju samo za preuzimanje izvorne zbirke,
    from datasets import Dataset, load_dataset  # ne i za rad nad lokalnim slikama
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # okruzenje samo za inferencu (npr. Hugging Face Spaces)
    Dataset = object  # type: ignore[assignment,misc]
    load_dataset = pa = pq = None  # type: ignore[assignment]
import numpy as np
import requests
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset

from .modalities import ModalitiesTransform


DATASET_ID = "mwalmsley/gz_hubble"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class LabelInfo:
    names: list[str]
    to_id: dict[str, int]


def safe_label_name(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return cleaned.strip("_") or "unknown"


def load_hubble_split(split: str, subset: str | None = None, limit: int | None = None) -> Dataset:
    dataset = load_dataset(DATASET_ID, split=split)
    if subset == "tiny":
        max_rows = limit or 256
        return dataset.select(range(min(max_rows, len(dataset))))
    if limit is not None:
        return dataset.select(range(min(limit, len(dataset))))
    return dataset


def build_label_info(dataset: Dataset, label_column: str = "summary") -> LabelInfo:
    labels = sorted({str(label) for label in dataset[label_column] if str(label).strip()})
    return LabelInfo(names=labels, to_id={name: index for index, name in enumerate(labels)})


def parquet_urls(config: str = "default", verify_ssl: bool = True) -> dict[str, list[str]]:
    response = requests.get(f"https://huggingface.co/api/datasets/{DATASET_ID}/parquet", verify=verify_ssl, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return payload[config]


def download_file(url: str, path: Path, verify_ssl: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(url, verify=verify_ssl, stream=True, timeout=300) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def build_synthetic_split(rows: int, image_size: int = 128, num_classes: int = 7) -> Dataset:
    rng = np.random.default_rng(42)
    labels = [f"class_{index}" for index in range(num_classes)]
    images = []
    summaries = []
    for index in range(rows):
        label_id = index % num_classes
        base = rng.normal(loc=20 + label_id * 18, scale=12, size=(image_size, image_size, 3))
        yy, xx = np.ogrid[:image_size, :image_size]
        cy = image_size // 2 + rng.integers(-12, 13)
        cx = image_size // 2 + rng.integers(-12, 13)
        radius = 12 + label_id * 3
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
        base[mask] += 120
        if label_id % 2 == 0:
            base[:, image_size // 2 - 2 : image_size // 2 + 2, :] += 40
        images.append(Image.fromarray(np.uint8(np.clip(base, 0, 255))))
        summaries.append(labels[label_id])
    return Dataset.from_dict({"image": images, "summary": summaries})


# ---------------------------------------------------------------------------
# Priprema lokalnog skupa iz parquet fajlova
# ---------------------------------------------------------------------------

def _parquet_paths(cache_root: Path, split: str, config: str) -> list[Path]:
    directory = cache_root / config / split
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)


def count_labels_from_parquet(
    cache_root: str | Path,
    split: str,
    label_column: str = "summary",
    config: str = "default",
) -> dict[str, int]:
    """Prebrojava primere po klasi bez ucitavanja slika."""
    counts: dict[str, int] = {}
    for path in _parquet_paths(Path(cache_root), split, config):
        table = pq.read_table(path, columns=[label_column])
        for value in table.column(label_column).to_pylist():
            label = str(value).strip()
            if label:
                counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _index_labels(paths: list[Path], label_column: str) -> dict[str, list[tuple[int, int]]]:
    """Prvi prolaz: cita samo kolonu sa oznakama i pamti gde je koji primer."""
    index: dict[str, list[tuple[int, int]]] = {}
    for file_index, path in enumerate(paths):
        table = pq.read_table(path, columns=[label_column])
        for row_index, value in enumerate(table.column(label_column).to_pylist()):
            label = str(value).strip()
            if label:
                index.setdefault(label, []).append((file_index, row_index))
    return index


def _extract_images(
    paths: list[Path],
    picks: dict[int, list[tuple[int, str, str]]],
    output_path: Path,
    label_column: str,
    source: str,
    image_column: str = "image",
) -> list[dict[str, str]]:
    """Drugi prolaz: iz svakog fajla vadi samo izabrane redove i snima slike."""
    rows: list[dict[str, str]] = []
    per_label_counter: dict[tuple[str, str], int] = {}

    for file_index, path in enumerate(paths):
        wanted = picks.get(file_index)
        if not wanted:
            continue
        table = pq.read_table(path, columns=[image_column, label_column])
        indices = pa.array([row_index for row_index, _, _ in wanted])
        subset = table.take(indices).to_pylist()

        for (row_index, label, split_name), record in zip(wanted, subset):
            payload = record[image_column]
            image_bytes = payload["bytes"] if isinstance(payload, dict) else payload
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            safe_label = safe_label_name(label)
            key = (split_name, safe_label)
            position = per_label_counter.get(key, 0)
            per_label_counter[key] = position + 1

            class_dir = output_path / split_name / safe_label
            class_dir.mkdir(parents=True, exist_ok=True)
            relative_path = Path(split_name) / safe_label / f"{safe_label}_{position:04d}.jpg"
            image.save(output_path / relative_path, format="JPEG", quality=95)

            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "safe_label": safe_label,
                    "path": relative_path.as_posix(),
                    # izvorni split se upisuje jer numeracija fajlova pocinje od 0 u
                    # oba splita, pa bi "0:123" inace bio dvosmislen
                    "source_index": f"{source}:{file_index}:{row_index}",
                }
            )
        del table, subset
    return rows


def prepare_balanced_dataset(
    output_dir: str | Path,
    cache_root: str | Path | None = None,
    train_per_class: int = 450,
    val_per_class: int = 100,
    test_per_class: int = 150,
    label_column: str = "summary",
    config: str = "default",
    seed: int = 42,
    verify_ssl: bool = False,
    download_if_missing: bool = True,
) -> dict:
    """Pravi izbalansiran lokalni skup sa tri podele: train / val / test.

    Razlike u odnosu na prvu verziju:

    - NASUMICNO UZORKOVANJE. Ranije su uzimani prvi N primera koji naidju, sto
      moze biti sistematski pristrasno ako je izvorni skup ma kako sortiran.
      Sada se prvo indeksiraju svi primeri po klasi, pa se bira nasumican uzorak
      sa fiksnim seed-om radi reproduktivnosti.

    - ODVOJEN VALIDATION SKUP. Iz TRAIN dela izvornog skupa vadi se po klasi
      train_per_class + val_per_class primera odjednom, jednim uzorkovanjem bez
      ponavljanja, pa se prvi deo proglasi treningom a ostatak validacijom.
      Test se vadi iz TEST dela izvornog skupa i dodiruje se tek na kraju, pa
      ne ucestvuje ni u ranom zaustavljanju ni u izboru checkpointa.

    - FILTRIRANJE KLASA. Klase koje nemaju dovoljno primera se izostavljaju, a
      broj primera po klasi se belezi u dataset_info.json radi obrazlozenja u radu.

    - DVA PROLAZA. Prvi cita samo oznake, drugi vadi samo izabrane redove, pa se
      slike ne dekoduju bez potrebe.
    """
    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_root) if cache_root else output_path / "_parquet_cache"

    train_paths = _parquet_paths(cache_path, "train", config)
    eval_paths = _parquet_paths(cache_path, "test", config)

    if download_if_missing and (not train_paths or not eval_paths):
        urls_by_split = parquet_urls(config=config, verify_ssl=verify_ssl)
        for split in ("train", "test"):
            for file_index, url in enumerate(urls_by_split[split]):
                download_file(url, cache_path / config / split / f"{file_index}.parquet", verify_ssl=verify_ssl)
        train_paths = _parquet_paths(cache_path, "train", config)
        eval_paths = _parquet_paths(cache_path, "test", config)

    if not train_paths or not eval_paths:
        raise FileNotFoundError(f"Nema parquet fajlova u {cache_path}")

    print("indeksiranje train splita...")
    train_index = _index_labels(train_paths, label_column)
    print("indeksiranje test splita...")
    eval_index = _index_labels(eval_paths, label_column)

    available = {
        label: {"train": len(train_index.get(label, [])), "eval": len(eval_index.get(label, []))}
        for label in sorted(set(train_index) | set(eval_index))
    }

    # Validation se uzima iz TRAIN dela izvornog skupa, a test iz TEST dela.
    # Time test skup ostaje potpuno nedirnut do same evaluacije, a validation
    # ipak postoji za rano zaustavljanje i izbor checkpointa.
    needed_train = train_per_class + val_per_class
    kept = sorted(
        label for label, counts in available.items()
        if counts["train"] >= needed_train and counts["eval"] >= test_per_class
    )
    dropped = {label: counts for label, counts in available.items() if label not in kept}

    if not kept:
        raise ValueError(
            "Nijedna klasa nema dovoljno primera. Trazeno: "
            f"{needed_train} u train delu i {test_per_class} u test delu."
        )

    print(f"zadrzane klase ({len(kept)}): {', '.join(kept)}")
    if dropped:
        print(f"izostavljene klase ({len(dropped)}):")
        for label, counts in dropped.items():
            print(f"  {label}: train={counts['train']} test={counts['eval']}")

    train_picks: dict[int, list[tuple[int, str, str]]] = {}
    eval_picks: dict[int, list[tuple[int, str, str]]] = {}

    for label in kept:
        chosen = rng.sample(train_index[label], needed_train)
        for position, (file_index, row_index) in enumerate(chosen):
            split_name = "train" if position < train_per_class else "val"
            train_picks.setdefault(file_index, []).append((row_index, label, split_name))

        for file_index, row_index in rng.sample(eval_index[label], test_per_class):
            eval_picks.setdefault(file_index, []).append((row_index, label, "test"))

    for picks in (train_picks, eval_picks):
        for file_index in picks:
            picks[file_index].sort(key=lambda item: item[0])

    print("izdvajanje slika...")
    rows = _extract_images(train_paths, train_picks, output_path, label_column, source="train")
    rows += _extract_images(eval_paths, eval_picks, output_path, label_column, source="test")

    with (output_path / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "safe_label", "path", "source_index"])
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["split"], {})
        counts[row["split"]][row["label"]] = counts[row["split"]].get(row["label"], 0) + 1

    info = {
        "dataset_id": DATASET_ID,
        "source": "parquet_api",
        "config": config,
        "seed": seed,
        "per_class": {"train": train_per_class, "val": val_per_class, "test": test_per_class},
        "labels": kept,
        "counts": counts,
        "available_in_source": available,
        "dropped_labels": dropped,
    }
    (output_path / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


class LocalImageDataset(TorchDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        label_info: LabelInfo,
        modalities: Iterable[str],
        image_size: int,
        train: bool,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.label_info = label_info
        self.samples = load_local_samples(self.root, split)
        if not self.samples:
            raise ValueError(f"Split '{split}' je prazan u {self.root}")
        self.transform = ModalitiesTransform(modalities=modalities, image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(self.label_info.to_id[label], dtype=torch.long)
        return x, y


def load_local_metadata(root: str | Path) -> list[dict[str, str]]:
    metadata_path = Path(root) / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_local_label_info(root: str | Path) -> LabelInfo:
    rows = load_local_metadata(root)
    labels = sorted({row["label"] for row in rows})
    return LabelInfo(names=labels, to_id={name: index for index, name in enumerate(labels)})


def available_splits(root: str | Path) -> set[str]:
    return {row["split"] for row in load_local_metadata(root)}


def load_local_samples(root: str | Path, split: str) -> list[tuple[Path, str]]:
    root_path = Path(root)
    rows = load_local_metadata(root_path)
    samples = [(root_path / row["path"], row["label"]) for row in rows if row["split"] == split]
    return [(path, label) for path, label in samples if path.suffix.lower() in IMAGE_EXTENSIONS]


class HubbleTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        label_info: LabelInfo,
        modalities: Iterable[str],
        image_size: int,
        train: bool,
        image_column: str = "image",
        label_column: str = "summary",
    ) -> None:
        self.dataset = dataset
        self.label_info = label_info
        self.image_column = image_column
        self.label_column = label_column
        self.transform = ModalitiesTransform(modalities=modalities, image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset[index]
        image = row[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        x = self.transform(image)
        y = torch.tensor(self.label_info.to_id[str(row[self.label_column])], dtype=torch.long)
        return x, y
