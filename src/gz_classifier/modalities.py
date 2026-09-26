from __future__ import annotations

import random
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


DEFAULT_MODALITIES = ("original",)
MULTIMODAL_PRESET = ("original", "gamma", "contrast", "highpass", "equalize", "sharpen")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def apply_modality(image: Image.Image, modality: str) -> Image.Image:
    """Generise jednu filtriranu verziju slike.

    Modaliteti su motivisani astronomskom praksom: slabe strukture (spoljasnji
    krakovi, precke, ljuske) cesto su jedva vidljive na sirovom snimku, pa se
    rutinski gledaju kroz transformacije kontrasta i prostorne filtere.
    """
    image = _to_rgb(image)
    modality = modality.lower()

    if modality == "original":
        return image
    if modality == "gamma":
        arr = np.asarray(image).astype(np.float32) / 255.0
        arr = np.power(arr, 0.65)
        return Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255)))
    if modality == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.9)
    if modality == "sharpen":
        return image.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
    if modality == "equalize":
        return ImageOps.equalize(image)
    if modality == "highpass":
        arr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
        high = cv2.addWeighted(arr, 1.7, blur, -0.7, 0)
        return Image.fromarray(cv2.cvtColor(np.clip(high, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    raise ValueError(f"Unknown modality: {modality}")


def build_image_transform(image_size: int, train: bool) -> transforms.Compose:
    """Zadrzano radi kompatibilnosti; ModalitiesTransform vise ne koristi ovu funkciju."""
    ops: list[object] = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(ops)


class ModalitiesTransform:
    """Pretvara jednu PIL sliku u tenzor oblika [3 * len(modalities), H, W].

    Redosled operacija je bitan i razlikuje se od prve verzije koda:

    1. Resize na radnu rezoluciju.
       Filteri se primenjuju na istoj rezoluciji i tokom treninga i tokom
       evaluacije, pa `highpass` i `sharpen` znace istu stvar bez obzira na
       velicinu izvorne slike.

    2. Generisanje modaliteta nad ciste (neaugmentovane) slike.
       Da histogram kod `equalize` i statistike kod `highpass` ne budu
       iskrivljene crnim uglovima koje ostavlja rotacija.

    3. Ista geometrijska augmentacija na svih 6 verzija.
       Parametri (flip, ugao) uzorkuju se JEDNOM po slici i primenjuju se
       identicno na sve modalitete, pa su kanali prostorno poravnati.

    U prvoj verziji koda augmentacija se pozivala zasebno za svaki modalitet,
    pa je svaki dobijao svoju nasumicnu rotaciju. Model je tada dobijao sest
    medjusobno pomerenih verzija iste galaksije.

    Flip i rotacija su ovde fizicki opravdani: galaksija nema "gore" ni "dole",
    orijentacija na snimku je posledica ugla teleskopa.
    """

    def __init__(self, modalities: Iterable[str], image_size: int, train: bool) -> None:
        self.modalities = tuple(modalities)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @property
    def in_channels(self) -> int:
        return 3 * len(self.modalities)

    def _sample_geometry(self) -> dict[str, float | bool]:
        if not self.train:
            return {"hflip": False, "vflip": False, "angle": 0.0}
        return {
            "hflip": random.random() < 0.5,
            "vflip": random.random() < 0.5,
            "angle": random.uniform(-180.0, 180.0),
        }

    def _apply_geometry(self, image: Image.Image, params: dict) -> Image.Image:
        if params["hflip"]:
            image = TF.hflip(image)
        if params["vflip"]:
            image = TF.vflip(image)
        if params["angle"]:
            image = TF.rotate(image, float(params["angle"]), interpolation=TF.InterpolationMode.BILINEAR)
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = _to_rgb(image)
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        params = self._sample_geometry()

        tensors = []
        for modality in self.modalities:
            view = apply_modality(image, modality)
            view = self._apply_geometry(view, params)
            tensors.append(self.normalize(TF.to_tensor(view)))
        return torch.cat(tensors, dim=0)
