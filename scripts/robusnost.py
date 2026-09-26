"""Robusnost na degradaciju slike.

Pokretanje:
    python robusnost.py --output-root /content/outputs_500_pretrained --out /content/analize

Ne trenira nista. Test skup se namerno kvari - sumom, zamucenjem, JPEG
kompresijom i smanjenjem rezolucije - i mere se performanse istog modela na
svakom nivou. Poredjenje dva uslova ulaza odgovara na pitanje da li
multimodalni ulaz cini model otpornijim.

Hipoteza se moze braniti u oba smera: highpass i sharpen isticu ivice, ali
pojacavaju i sum, pa je moguce da multimodalni model bude OSETLJIVIJI. I to
je nalaz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.gz_classifier.data import build_local_label_info
from src.gz_classifier.evaluacija import DEGRADACIJE, napravi_loader, oceni
from src.gz_classifier.infer import load_predictor

BOJA = {"original": "#1D6FB8", "multimodal": "#C2680E"}
NASLOV = {"sum": "Гаусов шум", "zamucenje": "Замућење",
          "kompresija": "JPEG компресија", "smanjenje": "Смањена резолуција"}


def parse_args() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--output-root", default="/content/outputs_500_pretrained")
    a.add_argument("--data-dir", default="data/gz_hubble_500")
    a.add_argument("--split", default="test")
    a.add_argument("--out", default="analize")
    a.add_argument("--batch-size", type=int, default=32)
    a.add_argument("--num-workers", type=int, default=2)
    a.add_argument("--models", nargs="+", default=["densenet121", "swin_tiny_patch4_window7_224"])
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

    svi: dict = {}
    for model in args.models:
        svi[model] = {}
        for uslov in ("original", "multimodal"):
            ckpt = koren / f"{model}_{uslov}" / "best.ckpt"
            if not ckpt.exists():
                print(f"preskacem {model}_{uslov}: nema best.ckpt")
                continue
            print(f"\n=== {model} · {uslov} ===")
            pred = load_predictor(ckpt, device=args.device)
            mod = list(pred.modalities)

            def izmeri(deg=None) -> float:
                loader = napravi_loader(args.data_dir, args.split, label_info, mod,
                                        pred.image_size, degradacija=deg,
                                        batch_size=args.batch_size, num_workers=args.num_workers,
                                        max_slika=args.max_slika)
                return oceni(pred.model, loader, pred.device, klase)["macro_f1"]

            polazno = izmeri()
            print(f"  bez degradacije: {polazno:.4f}")
            svi[model][uslov] = {"bez_degradacije": polazno}

            for vrsta, nivoi in DEGRADACIJE.items():
                svi[model][uslov][vrsta] = {}
                for oznaka, funkcija in nivoi:
                    v = izmeri(funkcija)
                    svi[model][uslov][vrsta][oznaka] = v
                    print(f"  {vrsta:<11} {oznaka:<7} {v:.4f}   ({v - polazno:+.4f})")

            del pred
            torch.cuda.empty_cache()

    (izlaz / "robusnost.json").write_text(json.dumps(svi, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- grafik: po jedan panel za svaku vrstu degradacije ---
    for model, uslovi in svi.items():
        if len(uslovi) < 2:
            continue
        fig, ose = plt.subplots(1, len(DEGRADACIJE), figsize=(4.0 * len(DEGRADACIJE), 3.6), sharey=True)
        for ax, (vrsta, nivoi) in zip(ose, DEGRADACIJE.items()):
            oznake = ["полазно"] + [o for o, _ in nivoi]
            for uslov, podaci in uslovi.items():
                vrednosti = [podaci["bez_degradacije"]] + [podaci[vrsta][o] for o, _ in nivoi]
                ax.plot(oznake, vrednosti, marker="o", linewidth=2, markersize=6,
                        color=BOJA[uslov], label=uslov)
            ax.set_title(NASLOV[vrsta], fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(labelsize=8.5)
        ose[0].set_ylabel("макро F1", fontsize=9.5)
        ose[0].legend(fontsize=9)
        fig.suptitle(f"{model} · отпорност на деградацију слике", fontsize=11.5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(izlaz / f"robusnost_{model}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("  slika:", f"robusnost_{model}.png")

    print(f"\nsacuvano u: {izlaz.resolve()}")


if __name__ == "__main__":
    main()
