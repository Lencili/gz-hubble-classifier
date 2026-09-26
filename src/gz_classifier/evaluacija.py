"""Zajednicke funkcije za evaluaciju vec istreniranih modela.

Koriste ih ablacija modaliteta i ispitivanje robusnosti. Nijedna od te dve
analize ne trenira nista - obe samo propustaju test skup kroz sacuvani model.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import LabelInfo, load_local_samples
from .modalities import ModalitiesTransform


class SkupZaOcenu(Dataset):
    """Test skup, uz mogucnost da se slika pre obrade namerno pokvari."""

    def __init__(self, root, split: str, label_info: LabelInfo,
                 modalities: Iterable[str], image_size: int,
                 degradacija: Callable[[Image.Image], Image.Image] | None = None,
                 max_slika: int | None = None) -> None:
        self.samples = load_local_samples(root, split)
        if max_slika:
            # Ravnomeran izbor po celom skupu, da raspodela klasa ostane slicna.
            korak = max(1, len(self.samples) // max_slika)
            self.samples = self.samples[::korak][:max_slika]
        self.label_info = label_info
        self.transform = ModalitiesTransform(modalities, image_size, train=False)
        self.degradacija = degradacija

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        putanja, oznaka = self.samples[index]
        slika = Image.open(putanja).convert("RGB")
        if self.degradacija is not None:
            slika = self.degradacija(slika)
        return self.transform(slika), torch.tensor(self.label_info.to_id[oznaka], dtype=torch.long)


def napravi_loader(root, split, label_info, modalities, image_size,
                   degradacija=None, batch_size: int = 32, num_workers: int = 2,
                   max_slika: int | None = None) -> DataLoader:
    skup = SkupZaOcenu(root, split, label_info, modalities, image_size, degradacija, max_slika)
    return DataLoader(skup, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def oceni(model, loader: DataLoader, device: torch.device, imena_klasa: list[str],
          maska: torch.Tensor | None = None) -> dict:
    """Propusta ceo skup kroz model i vraca metrike.

    `maska` je tenzor oblika [1, C, 1, 1] sa nulama i jedinicama. Mnozenjem
    ulaza njome iskljucuju se pojedini kanali. Posto je adapter linearan
    (izlaz je zbir tezina puta kanali), postavljanje kanala na nulu tacno
    uklanja doprinos tog modaliteta, bez ostatka.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if maska is not None:
            x = x * maska
        logits = model(x)
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    izvestaj = classification_report(y_true, y_pred, target_names=imena_klasa,
                                     output_dict=True, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_po_klasi": {k: float(izvestaj[k]["f1-score"]) for k in imena_klasa},
    }


def maska_bez_modaliteta(broj_modaliteta: int, iskljuceni: int, device: torch.device) -> torch.Tensor:
    maska = torch.ones(1, 3 * broj_modaliteta, 1, 1, device=device)
    maska[:, iskljuceni * 3:(iskljuceni + 1) * 3] = 0.0
    return maska


# ---------------------------------------------------------------------------
# Degradacije slike
# ---------------------------------------------------------------------------

def sum_gaus(sigma: float) -> Callable[[Image.Image], Image.Image]:
    def f(slika: Image.Image) -> Image.Image:
        arr = np.asarray(slika).astype(np.float32)
        arr = arr + np.random.normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    return f


def zamucenje(sigma: float) -> Callable[[Image.Image], Image.Image]:
    return lambda slika: slika.filter(ImageFilter.GaussianBlur(radius=sigma))


def kompresija(kvalitet: int) -> Callable[[Image.Image], Image.Image]:
    def f(slika: Image.Image) -> Image.Image:
        bafer = io.BytesIO()
        slika.save(bafer, format="JPEG", quality=kvalitet)
        bafer.seek(0)
        return Image.open(bafer).convert("RGB")
    return f


def smanjenje(faktor: float) -> Callable[[Image.Image], Image.Image]:
    def f(slika: Image.Image) -> Image.Image:
        w, h = slika.size
        mala = slika.resize((max(1, int(w * faktor)), max(1, int(h * faktor))), Image.BICUBIC)
        return mala.resize((w, h), Image.BICUBIC)
    return f


DEGRADACIJE = {
    "sum":        [("σ=5", sum_gaus(5)), ("σ=10", sum_gaus(10)), ("σ=20", sum_gaus(20)), ("σ=40", sum_gaus(40))],
    "zamucenje":  [("σ=1", zamucenje(1)), ("σ=2", zamucenje(2)), ("σ=3", zamucenje(3))],
    "kompresija": [("q=70", kompresija(70)), ("q=40", kompresija(40)), ("q=20", kompresija(20))],
    "smanjenje":  [("×0.75", smanjenje(0.75)), ("×0.5", smanjenje(0.5)), ("×0.25", smanjenje(0.25))],
}
