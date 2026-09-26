from __future__ import annotations

import argparse
import json

from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .data import build_label_info, load_hubble_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", choices=("full", "tiny"), default="full")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label-column", default="summary")
    parser.add_argument("--image-column", default="image")
    return parser.parse_args()


def prompts_for(label: str) -> list[str]:
    readable = label.replace("_", " ").replace("-", " ")
    return [
        f"a Hubble telescope image of a {readable} galaxy",
        f"an astronomy image showing a {readable} galaxy",
        f"a galaxy morphology class: {readable}",
    ]


def main() -> None:
    args = parse_args()
    subset = None if args.subset == "full" else args.subset
    dataset = load_hubble_split(args.split, subset=subset, limit=args.limit)
    label_info = build_label_info(dataset, label_column=args.label_column)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(args.model, local_files_only=args.local_files_only).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)

    text_prompts = [prompt for label in label_info.names for prompt in prompts_for(label)]
    label_ids = [index for index, label in enumerate(label_info.names) for _ in prompts_for(label)]
    text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    y_true: list[int] = []
    y_pred: list[int] = []
    for row in tqdm(dataset):
        image = row[args.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
        with torch.inference_mode():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).squeeze(0)
            per_prompt = torch.zeros(len(label_info.names), device=device)
            counts = torch.zeros(len(label_info.names), device=device)
            for score, label_id in zip(scores, label_ids):
                per_prompt[label_id] += score
                counts[label_id] += 1
            pred = int((per_prompt / counts).argmax().detach().cpu())
        y_true.append(label_info.to_id[str(row[args.label_column])])
        y_pred.append(pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": label_info.names,
        "report": classification_report(y_true, y_pred, target_names=label_info.names, zero_division=0),
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print(metrics["report"])


if __name__ == "__main__":
    main()
