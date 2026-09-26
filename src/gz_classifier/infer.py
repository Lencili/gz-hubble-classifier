from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.nn import functional as F

from .modalities import ModalitiesTransform
from .models import ModelConfig, build_model


class GalaxyPredictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        self.label_names = list(payload["label_names"])
        cfg = payload["config"]
        self.image_size = int(cfg["image_size"])
        self.modalities = tuple(cfg["modalities"])
        self.model_name = str(cfg["model"])
        self.model_family = str(cfg["model_family"])
        model_config = ModelConfig(
            model=cfg["model"],
            model_family=cfg["model_family"],
            num_classes=len(self.label_names),
            in_channels=3 * len(self.modalities),
            pretrained=False,
        )
        self.model = build_model(model_config)
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = ModalitiesTransform(self.modalities, self.image_size, train=False)

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict[str, float | str]]:
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1)[0]
        values, indices = probs.topk(min(top_k, len(self.label_names)))
        return [
            {"label": self.label_names[int(index)], "probability": float(value)}
            for value, index in zip(values.detach().cpu(), indices.detach().cpu())
        ]


def load_predictor(checkpoint: str | Path | None,
                   device: str | None = None) -> GalaxyPredictor | None:
    """Ucitava sacuvani model. `device` se prosledjuje da bi analize mogle da
    se pokrenu i na procesoru, kad GPU nije dostupan."""
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    if not path.exists():
        return None
    return GalaxyPredictor(path, device=device)

