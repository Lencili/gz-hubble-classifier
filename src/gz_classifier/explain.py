from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from .modalities import ModalitiesTransform


@dataclass
class Explanation:
    heatmap: np.ndarray
    overlay: Image.Image
    metod: str


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values - values.min()
    return values / (values.max() + 1e-8)


def _overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))).astype(np.uint8)
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(rgb, 1.0 - alpha, colored, alpha, 0))


def find_last_conv(module: nn.Module) -> nn.Module | None:
    last = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            last = submodule
    return last


def find_last_spatial_conv(model: nn.Module, x: torch.Tensor) -> nn.Module | None:
    """Poslednji konvolucioni sloj cija mapa ima vise od jedne pozicije.

    Redosled definisanja slojeva u kodu nije isto sto i redosled izvrsavanja,
    niti garantuje prostornu velicinu izlaza. Arhitekture poput EfficientNet-a
    i MobileNet-a sadrze squeeze-and-excitation blokove sa konvolucijama ciji
    je izlaz 1x1 - ako se takav sloj uzme za GradCAM, toplotna mapa je
    konstanta i ne govori nista.

    Zato se radi jedan prolaz unapred sa hookovima na svim konvolucijama, pa
    se bira poslednji IZVRSENI sloj cija je mapa prostorna.
    """
    kandidati: list[nn.Module] = []
    hookovi = []

    def napravi(modul: nn.Module):
        def hook(_m, _ulaz, izlaz):
            if isinstance(izlaz, torch.Tensor) and izlaz.ndim == 4 and izlaz.shape[-1] * izlaz.shape[-2] > 1:
                kandidati.append(modul)
        return hook

    for modul in model.modules():
        if isinstance(modul, nn.Conv2d):
            hookovi.append(modul.register_forward_hook(napravi(modul)))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hookovi:
            h.remove()
    return kandidati[-1] if kandidati else None


# ---------------------------------------------------------------------------
# Izbor ciljnog sloja
# ---------------------------------------------------------------------------

def _tokens_to_map(x: torch.Tensor, prefiks: int) -> torch.Tensor:
    """Niz tokena [B, T, D] vraca u prostorni oblik [B, D, H, W].

    Transformer izlaz nije slika nego niz zakrpa, pa se pre GradCAM racuna mora
    vratiti u dve dimenzije. Prvih `prefiks` tokena su [CLS] i slicni i nemaju
    polozaj na slici, pa se odbacuju.
    """
    b, t, d = x.shape
    n = t - prefiks
    h = int(round(n ** 0.5))
    while h > 1 and n % h:
        h -= 1
    return x[:, prefiks:, :].transpose(1, 2).reshape(b, d, h, n // h)


def _nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def select_target_layer(model: nn.Module, model_family: str = "cnn") -> tuple[nn.Module | None, Callable | None, str]:
    """Bira sloj nad kojim se racuna GradCAM i, po potrebi, preoblikovanje.

    Konvolucione mreze: poslednji konvolucioni sloj.
    Swin: norm1 poslednjeg bloka poslednjeg nivoa (izlaz je [B, H, W, C]).
    ViT i DeiT: norm1 poslednjeg bloka (izlaz je niz tokena).
    """
    backbone = getattr(model, "backbone", model)

    if model_family.lower() == "vit":
        if hasattr(backbone, "layers"):                      # Swin
            try:
                return backbone.layers[-1].blocks[-1].norm1, _nhwc_to_nchw, "GradCAM (Swin)"
            except (AttributeError, IndexError):
                pass
        if hasattr(backbone, "blocks"):                      # ViT / DeiT
            prefiks = int(getattr(backbone, "num_prefix_tokens", 1))
            try:
                return (backbone.blocks[-1].norm1,
                        lambda x: _tokens_to_map(x, prefiks),
                        "GradCAM (ViT)")
            except (AttributeError, IndexError):
                pass

    sloj = find_last_conv(backbone)
    return sloj, None, "GradCAM"


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 reshape: Callable | None = None) -> None:
        self.model = model
        self.target_layer = target_layer
        self.reshape = reshape
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(self._capture_activation),
            target_layer.register_full_backward_hook(self._capture_gradient),
        ]

    def _capture_activation(self, _m, _i, output) -> None:
        self.activations = (output[0] if isinstance(output, (tuple, list)) else output).detach()

    def _capture_gradient(self, _m, _gi, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor, class_id: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        cilj = logits.argmax(dim=1).item() if class_id is None else int(class_id)
        logits[:, cilj].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hookovi nisu uhvatili aktivacije i gradijente")

        a, g = self.activations, self.gradients
        if self.reshape is not None:
            a, g = self.reshape(a), self.reshape(g)

        tezine = g.mean(dim=(2, 3), keepdim=True)
        cam = (tezine * a).sum(dim=1).relu()[0].cpu().numpy()
        if cam.max() <= 0:                       # nista ne podrzava klasu
            cam = np.abs((tezine * a).sum(dim=1)[0].cpu().numpy())
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        return _normalize_map(cam)


def gradient_saliency(model: nn.Module, x: torch.Tensor, class_id: int | None = None) -> np.ndarray:
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    cilj = logits.argmax(dim=1).item() if class_id is None else int(class_id)
    logits[:, cilj].sum().backward()
    return _normalize_map(x.grad.detach().abs().mean(dim=1)[0].cpu().numpy())


def explain_image(
    model: nn.Module,
    image: Image.Image,
    transform: ModalitiesTransform,
    device: torch.device,
    class_id: int | None = None,
    model_family: str = "cnn",
) -> Explanation:
    x = transform(image).unsqueeze(0).to(device)
    sloj, reshape, naziv = select_target_layer(model, model_family)
    if model_family.lower() != "vit":
        # probni prolaz otkriva koji je sloj zaista prostoran
        bolji = find_last_spatial_conv(model, x)
        if bolji is not None:
            sloj = bolji

    if sloj is not None:
        cam = GradCAM(model, sloj, reshape)
        try:
            mapa = cam(x, class_id=class_id)
            return Explanation(heatmap=mapa, overlay=_overlay(image, mapa), metod=naziv)
        except Exception:
            pass
        finally:
            cam.close()

    mapa = gradient_saliency(model, x, class_id=class_id)
    return Explanation(heatmap=mapa, overlay=_overlay(image, mapa), metod="Gradijentna mapa")


# ---------------------------------------------------------------------------
# Doprinos pojedinih modaliteta ZA JEDNU SLIKU
# ---------------------------------------------------------------------------

def modality_attribution(
    model: nn.Module,
    image: Image.Image,
    transform: ModalitiesTransform,
    device: torch.device,
    modalities: Iterable[str],
    class_id: int | None = None,
) -> dict[str, float]:
    """Koliko je svaki modalitet doprineo odluci NA OVOJ SLICI.

    Tezine adaptera pokazuju globalnu meru - jedan recept za sve slike.
    Ova mera je drugacija: racuna se gradijent izlaza za izabranu klasu po
    ulaznim pikselima, pomnozi se samim ulazom i sabere po grupama od tri
    kanala koje pripadaju istom modalitetu.

    Time se dobija odgovor na pitanje "na sta se model oslonio bas kod ove
    galaksije", sto tezine adaptera ne mogu da daju.
    """
    modalities = list(modalities)
    if len(modalities) <= 1:
        return {}

    x = transform(image).unsqueeze(0).to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    cilj = logits.argmax(dim=1).item() if class_id is None else int(class_id)
    logits[:, cilj].sum().backward()

    doprinos = (x.grad * x).detach().abs()[0]          # [3n, H, W]
    udeli: dict[str, float] = {}
    for i, naziv in enumerate(modalities):
        udeli[naziv] = float(doprinos[i * 3:(i + 1) * 3].sum())
    ukupno = sum(udeli.values()) or 1.0
    return {k: v / ukupno for k, v in udeli.items()}
