"""Klasifikacija bez prethodnog ucenja (zero-shot) modelom CLIP, nad NASIM test skupom.

Zasto poseban skript, a ne src/gz_classifier/zeroshot.py: taj ucitava podatke
direktno sa Hugging Face-a, pa meri nad drugim skupom i nad svih sest klasa.
Ovde se koristi tacno onaj skup za testiranje na kome su merena i nasa
sedam modela - 150 slika po klasi, pet klasa - da bi poredjenje bilo posteno.

Pokretanje iz korena projekta:

    python zeroshot_lokalno.py

Model se ucitava lokalno iz models/clip-vit-base-patch32, bez pristupa mrezi.
Rezultat se snima u analize/zeroshot_clip.json i ispisuje na ekran.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import CLIPModel, CLIPProcessor

KLASE = ["barred_spiral", "unbarred_spiral", "edge_on_disk", "smooth_round", "smooth_cigar"]

# Vise formulacija po klasi; ocene se usrednjavaju, da rezultat ne zavisi od
# jedne srecno pogodjene recenice.
OPISI = {
    "barred_spiral":   ["a barred spiral galaxy",
                        "a Hubble telescope image of a spiral galaxy with a central bar",
                        "a galaxy with spiral arms and a straight bar through its centre"],
    "unbarred_spiral": ["an unbarred spiral galaxy",
                        "a Hubble telescope image of a spiral galaxy without a bar",
                        "a galaxy with spiral arms and no central bar"],
    "edge_on_disk":    ["an edge-on disk galaxy",
                        "a Hubble telescope image of a disk galaxy seen edge on",
                        "a thin flat galaxy seen from the side"],
    "smooth_round":    ["a smooth round elliptical galaxy",
                        "a Hubble telescope image of a round featureless galaxy",
                        "a smooth ball-shaped galaxy with no structure"],
    "smooth_cigar":    ["a smooth cigar-shaped galaxy",
                        "a Hubble telescope image of an elongated featureless galaxy",
                        "a smooth elongated galaxy with no spiral arms"],
}



def kao_tenzor(izlaz):
    """Neke verzije transformers-a vracaju objekat umesto tenzora.

    get_text_features/get_image_features u vecini verzija daju torch.Tensor, ali
    u nekima vrate BaseModelOutputWithPooling. Ovde se u oba slucaja izvlaci
    vektor obelezja, da skript radi bez obzira na verziju biblioteke.
    """
    if isinstance(izlaz, torch.Tensor):
        return izlaz
    for ime in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        v = getattr(izlaz, ime, None)
        if isinstance(v, torch.Tensor):
            return v[:, 0] if v.dim() == 3 else v
    raise TypeError(f"neocekivan tip izlaza: {type(izlaz)}")


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--model-dir", default="models/clip-vit-base-patch32")
    a.add_argument("--data-dir", default="data/gz_hubble_500/test")
    a.add_argument("--out", default="analize/zeroshot_clip.json")
    a.add_argument("--device", default=None)
    args = a.parse_args()

    uredjaj = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("uredjaj:", uredjaj)

    model = CLIPModel.from_pretrained(args.model_dir, local_files_only=True).to(uredjaj).eval()
    procesor = CLIPProcessor.from_pretrained(args.model_dir, local_files_only=True)

    # --- tekstualne ocene se racunaju jednom
    recenice, pripada = [], []
    for i, k in enumerate(KLASE):
        for r in OPISI[k]:
            recenice.append(r)
            pripada.append(i)
    ulaz = procesor(text=recenice, return_tensors="pt", padding=True).to(uredjaj)
    with torch.inference_mode():
        t = kao_tenzor(model.get_text_features(**ulaz))
        t = t / t.norm(dim=-1, keepdim=True)
    pripada_t = torch.tensor(pripada, device=uredjaj)

    y_true, y_pred = [], []
    koren = Path(args.data_dir)
    for i, klasa in enumerate(KLASE):
        putanje = sorted((koren / klasa).glob("*.jpg")) + sorted((koren / klasa).glob("*.png"))
        for p in putanje:
            slika = Image.open(p).convert("RGB")
            u = procesor(images=slika, return_tensors="pt").to(uredjaj)
            with torch.inference_mode():
                v = kao_tenzor(model.get_image_features(**u))
                v = v / v.norm(dim=-1, keepdim=True)
                ocene = (v @ t.T).squeeze(0)
                po_klasi = torch.zeros(len(KLASE), device=uredjaj)
                broj = torch.zeros(len(KLASE), device=uredjaj)
                po_klasi.index_add_(0, pripada_t, ocene)
                broj.index_add_(0, pripada_t, torch.ones_like(ocene))
                y_pred.append(int((po_klasi / broj).argmax()))
            y_true.append(i)
        print(f"  {klasa}: {len(putanje)} slika")

    izvestaj = classification_report(y_true, y_pred, target_names=KLASE,
                                     zero_division=0, output_dict=True)
    rezultat = {
        "model": "openai/clip-vit-base-patch32",
        "skup": str(koren),
        "broj_slika": len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "classification_report": izvestaj,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": KLASE,
        "opisi": OPISI,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rezultat, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"tacnost   : {rezultat['acc']:.4f}")
    print(f"macro F1  : {rezultat['macro_f1']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=KLASE, zero_division=0))
    print("snimljeno u", args.out)


if __name__ == "__main__":
    main()
