from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn
from torchvision import models as tv_models


@dataclass(frozen=True)
class ModelConfig:
    model: str
    model_family: str
    num_classes: int
    in_channels: int = 3
    pretrained: bool = True


class InputAdapter(nn.Module):
    """Svodi multimodalni ulaz (3 * broj modaliteta kanala) na 3 kanala.

    Kada je ulaz vec 3-kanalni, adapter je identitet i mreza je identicna
    standardnoj. Kada ima vise modaliteta, koristi se konvolucija 1x1 koja na
    svakoj poziciji pravi naucenu linearnu kombinaciju svih ulaznih kanala.

    Inicijalizacija (`identity_init`) je vazna za postenje eksperimenta.
    Sa nasumicnim tezinama, multimodalni model bi u nultom koraku prosledjivao
    pretreniranom backbone-u nasumicnu mesavinu kanala - dakle nesto sto ne lici
    na sliku - dok bi model sa jednim modalitetom krenuo direktno od prave slike.
    Multimodalni uslov bi tako startovao sa hendikepom i poredjenje bi merilo i
    tu razliku, a ne samo korist od dodatnih modaliteta.

    Zato se adapter inicijalizuje tako da na pocetku propusta PRVI modalitet
    (po konvenciji `original`) nepromenjen, a ostalima daje tezinu nula. Oba
    uslova time krecu iz iste tacke, a trening moze samo da doda informaciju iz
    ostalih modaliteta ako mu koristi. Gradijenti po nultim tezinama nisu nula,
    pa se ostali modaliteti normalno ukljucuju tokom treninga.
    """

    def __init__(self, in_channels: int, identity_init: bool = True) -> None:
        super().__init__()
        if in_channels == 3:
            self.adapter: nn.Module = nn.Identity()
            return

        conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        if identity_init:
            with torch.no_grad():
                conv.weight.zero_()
                for channel in range(3):
                    conv.weight[channel, channel, 0, 0] = 1.0
                if conv.bias is not None:
                    conv.bias.zero_()
        self.adapter = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class AdaptedBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, identity_init: bool = True) -> None:
        super().__init__()
        self.input_adapter = InputAdapter(in_channels, identity_init=identity_init)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_adapter(x))


def _build_torchvision_cnn(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if not hasattr(tv_models, name):
        raise ValueError(f"Unknown torchvision model: {name}")
    weights = "DEFAULT" if pretrained else None
    model = getattr(tv_models, name)(weights=weights)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        classifier = list(model.classifier)
        for index in range(len(classifier) - 1, -1, -1):
            if isinstance(classifier[index], nn.Linear):
                classifier[index] = nn.Linear(classifier[index].in_features, num_classes)
                model.classifier = nn.Sequential(*classifier)
                break
        else:
            raise ValueError(f"Model {name} classifier has no Linear layer")
    else:
        raise ValueError(f"Model {name} has an unsupported classifier layout")
    return model


def build_model(config: ModelConfig, identity_init: bool = True) -> nn.Module:
    family = config.model_family.lower()
    if family == "cnn":
        backbone = _build_torchvision_cnn(config.model, config.num_classes, config.pretrained)
    elif family == "vit":
        backbone = timm.create_model(config.model, pretrained=config.pretrained, num_classes=config.num_classes)
    else:
        raise ValueError("model_family must be 'cnn' or 'vit'")
    return AdaptedBackbone(backbone=backbone, in_channels=config.in_channels, identity_init=identity_init)


def modality_reliance(model: nn.Module, modalities: list[str]) -> dict[str, float]:
    """Mera oslanjanja modela na svaki modalitet, iz tezina adaptera.

    Adapter je matrica oblika [3, 3 * broj_modaliteta, 1, 1]. Svaka grupa od tri
    ulazna kanala pripada jednom modalitetu, pa zbir apsolutnih vrednosti tezina
    po grupi pokazuje koliko model "vuce" iz tog modaliteta. Vrednosti se
    normalizuju tako da se sabiraju na 1.

    Koristi se u poglavlju o tome na koje se transformacije model oslanja.
    Vraca prazan recnik za modele sa jednim modalitetom (adapter je identitet).
    """
    adapter = getattr(getattr(model, "input_adapter", None), "adapter", None)
    if not isinstance(adapter, nn.Conv2d):
        return {}

    weight = adapter.weight.detach().abs().sum(dim=(0, 2, 3))
    scores = {}
    for index, name in enumerate(modalities):
        scores[name] = float(weight[index * 3 : (index + 1) * 3].sum())
    total = sum(scores.values()) or 1.0
    return {name: value / total for name, value in scores.items()}
