"""Ablacija modaliteta: koliko model izgubi kad mu se iskljuci po jedan modalitet.

Pokretanje:
    python ablacija.py --output-root /content/outputs_500_pretrained --out /content/analize

Ne trenira nista. Za svaki multimodalni model propusta test skup jednom sa svim
modalitetima, pa jos jednom za svaki modalitet koji se iskljucuje. Pad makro F1
mere pokazuje koliko je taj modalitet zaista doprinosio.

Ovo dopunjuje tezine adaptera: tezine govore koliko model modalitetu daje
prostora, a ablacija koliko ga zaista koristi za tacnost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.gz_classifier.data import build_local_label_info
from src.gz_classifier.evaluacija import maska_bez_modaliteta, napravi_loader, oceni
from src.gz_classifier.infer import load_predictor


def parse_args() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--output-root", default="/content/outputs_500_pretrained")
    a.add_argument("--data-dir", default="data/gz_hubble_500")
    a.add_argument("--split", default="test")
    a.add_argument("--out", default="analize")
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--num-workers", type=int, default=2)
    a.add_argument("--models", nargs="+", default=None, help="Podskup modela; podrazumevano svi")
    a.add_argument("--device", default=None, help="cuda ili cpu; podrazumevano automatski")
    a.add_argument("--max-slika", type=int, default=None,
                   help="Ograniciti broj test slika (za pokretanje na procesoru)")
    return a.parse_args()


def main() -> None:
    args = parse_args()
    koren, izlaz = Path(args.output_root), Path(args.out)
    izlaz.mkdir(parents=True, exist_ok=True)
    label_info = build_local_label_info(args.data_dir)
    klase = label_info.names
    print("uredjaj:", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    runovi = sorted(p for p in koren.glob("*_multimodal") if (p / "best.ckpt").exists())
    if args.models:
        runovi = [p for p in runovi if any(p.name.startswith(m) for m in args.models)]
    if not runovi:
        print("Nema multimodalnih modela sa best.ckpt u", koren)
        return

    svi = {}
    for run in runovi:
        model = run.name.replace("_multimodal", "")
        print(f"\n=== {model} ===")
        pred = load_predictor(run / "best.ckpt", device=args.device)
        modaliteti = list(pred.modalities)
        loader = napravi_loader(args.data_dir, args.split, label_info, modaliteti,
                                pred.image_size, batch_size=args.batch_size,
                                num_workers=args.num_workers, max_slika=args.max_slika)

        osnovna = oceni(pred.model, loader, pred.device, klase)
        print(f"  svi modaliteti: makro F1 {osnovna['macro_f1']:.4f}")

        rezultat = {"osnovna": osnovna, "bez": {}}
        for i, naziv in enumerate(modaliteti):
            maska = maska_bez_modaliteta(len(modaliteti), i, pred.device)
            bez = oceni(pred.model, loader, pred.device, klase, maska=maska)
            pad = osnovna["macro_f1"] - bez["macro_f1"]
            rezultat["bez"][naziv] = {**bez, "pad_macro_f1": pad}
            print(f"  bez {naziv:<10} makro F1 {bez['macro_f1']:.4f}   pad {pad:+.4f}")
        svi[model] = rezultat
        del pred
        torch.cuda.empty_cache()

    (izlaz / "ablacija.json").write_text(json.dumps(svi, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- grafik: pad makro F1 po modalitetu, za svaki model ---
    modeli = list(svi)
    modaliteti = list(svi[modeli[0]]["bez"])
    matrica = np.array([[svi[m]["bez"][mm]["pad_macro_f1"] for mm in modaliteti] for m in modeli])

    fig, ax = plt.subplots(figsize=(1.5 * len(modaliteti) + 4, 0.62 * len(modeli) + 2.4))
    granica = float(np.abs(matrica).max()) or 1.0
    im = ax.imshow(matrica, cmap="RdBu_r", vmin=-granica, vmax=granica, aspect="auto")
    ax.set_xticks(range(len(modaliteti)), modaliteti, fontsize=9)
    ax.set_yticks(range(len(modeli)), modeli, fontsize=9)
    for i in range(len(modeli)):
        for j in range(len(modaliteti)):
            ax.text(j, i, f"{matrica[i, j]:+.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(matrica[i, j]) > granica * 0.6 else "#14171E")
    ax.set_title("Пад макро F1 мере када се модалитет искључи", fontsize=10.5, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="пад (веће = модалитет је важнији)")
    fig.tight_layout()
    fig.savefig(izlaz / "ablacija.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsacuvano: {(izlaz / 'ablacija.json').resolve()}")
    print(f"          {(izlaz / 'ablacija.png').resolve()}")


if __name__ == "__main__":
    main()
