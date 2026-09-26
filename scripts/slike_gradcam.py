"""Pravi slike za rad iz vec istreniranih modela.

Pokretanje (u Colab-u, gde su checkpointi):

    python slike_gradcam.py --models densenet121 swin_tiny_patch4_window7_224

Proizvodi dve vrste izlaza:

1. UPOREDNE GradCAM SLIKE
   Za svaku klasu bira primere i prikazuje ih u tri kolone: originalna slika,
   toplotna mapa modela treniranog samo na originalu, i toplotna mapa modela
   treniranog na sest modaliteta. Prednost se daje primerima kod kojih se dva
   modela NE SLAZU, jer se tu najbolje vidi razlika.

2. DOPRINOS MODALITETA PO KLASAMA
   Tezine adaptera daju jedan recept za sve slike. Ovde se, za svaku sliku
   test skupa, racuna doprinos svakog modaliteta iz gradijenta po ulazu, pa se
   usrednjava po klasama. Time se dobija odgovor na pitanje koji modalitet
   koristi kojoj vrsti galaksija - sto tezine adaptera ne mogu da daju.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.gz_classifier.data import build_local_label_info, load_local_samples
from src.gz_classifier.explain import explain_image, modality_attribution
from src.gz_classifier.infer import load_predictor

SRPSKI = {
    "barred_spiral": "спирална са пречком",
    "edge_on_disk": "диск из профила",
    "smooth_cigar": "глатка издужена",
    "smooth_round": "глатка округла",
    "unbarred_spiral": "спирална без пречке",
}


def parse_args() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--output-root", default="/content/outputs_500_pretrained")
    a.add_argument("--data-dir", default="data/gz_hubble_500")
    a.add_argument("--models", nargs="+", default=["densenet121"])
    a.add_argument("--out", default="slike")
    a.add_argument("--per-class", type=int, default=2)
    a.add_argument("--split", default="test")
    a.add_argument("--limit-attr", type=int, default=None,
                   help="Koliko slika po klasi za racun doprinosa (podrazumevano sve)")
    a.add_argument("--device", default=None, help="cuda ili cpu; podrazumevano automatski")
    return a.parse_args()


def ucitaj_par(koren: Path, model: str, device: str | None = None):
    org = load_predictor(koren / f"{model}_original" / "best.ckpt", device=device)
    mm = load_predictor(koren / f"{model}_multimodal" / "best.ckpt", device=device)
    if org is None or mm is None:
        nedostaje = [u for u, p in (("original", org), ("multimodal", mm)) if p is None]
        raise FileNotFoundError(f"{model}: nedostaje best.ckpt za {', '.join(nedostaje)}")
    return org, mm


def predvidi(predictor, slika: Image.Image) -> str:
    return str(predictor.predict(slika, top_k=1)[0]["label"])


# ---------------------------------------------------------------------------

def uporedne_slike(model: str, org, mm, uzorci, klase, izlaz: Path, po_klasi: int) -> None:
    for klasa in klase:
        putanje = [p for p, l in uzorci if l == klasa]
        if not putanje:
            continue

        neslaganje, slaganje = [], []
        for p in putanje[:60]:                      # dovoljno da se nadje izbor
            slika = Image.open(p).convert("RGB")
            po, pm = predvidi(org, slika), predvidi(mm, slika)
            (neslaganje if po != pm else slaganje).append((p, po, pm))
            if len(neslaganje) >= po_klasi:
                break

        izabrani = (neslaganje + slaganje)[:po_klasi]
        if not izabrani:
            continue

        fig, ose = plt.subplots(len(izabrani), 3, figsize=(9.6, 3.4 * len(izabrani)))
        ose = np.atleast_2d(ose)
        TACNO, NETACNO = "#2C6742", "#8F3220"

        for red, (putanja, po, pm) in enumerate(izabrani):
            slika = Image.open(putanja).convert("RGB")
            # Обе мапе се рачунају за ИСТУ, тачну класу - да се пореди где
            # сваки модел тражи одлику те класе, без обзира на то шта је погодио.
            cid = org.label_names.index(klasa) if klasa in org.label_names else None
            e_org = explain_image(org.model, slika, org.transform, org.device,
                                  class_id=cid, model_family=org.model_family)
            e_mm = explain_image(mm.model, slika, mm.transform, mm.device,
                                 class_id=cid, model_family=mm.model_family)

            polja = [
                (slika, "снимак", None),
                (e_org.overlay, f"само оригинал → {po}", po == klasa),
                (e_mm.overlay, f"мултимодал → {pm}", pm == klasa),
            ]
            for kol, (sadrzaj, naslov, pogodak) in enumerate(polja):
                ax = ose[red, kol]
                ax.imshow(sadrzaj)
                boja = "#14171E" if pogodak is None else (TACNO if pogodak else NETACNO)
                ax.set_title(naslov, fontsize=9, color=boja)
                ax.axis("off")

        fig.suptitle(f"{model} · {SRPSKI.get(klasa, klasa)}  ({klasa})", fontsize=11, y=0.995)
        # razmak izmedju redova mora da bude izricit: bez njega naslovi drugog
        # reda naleze na slike prvog, sto se vidi tek u gotovom dokumentu
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.6)
        fig.subplots_adjust(hspace=0.16)
        put = izlaz / f"gradcam_{model}_{klasa}.png"
        fig.savefig(put, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("  ", put.name)


# ---------------------------------------------------------------------------

def doprinos_po_klasama(model: str, mm, uzorci, klase, izlaz: Path, granica: int | None):
    modaliteti = list(mm.modalities)
    zbir = {k: defaultdict(float) for k in klase}
    broj = defaultdict(int)

    for klasa in klase:
        putanje = [p for p, l in uzorci if l == klasa]
        if granica:
            putanje = putanje[:granica]
        for p in putanje:
            slika = Image.open(p).convert("RGB")
            udeli = modality_attribution(mm.model, slika, mm.transform, mm.device, modaliteti)
            for m, v in udeli.items():
                zbir[klasa][m] += v
            broj[klasa] += 1
        print(f"   {klasa}: {broj[klasa]} slika")

    matrica = np.array([[zbir[k][m] / max(broj[k], 1) for m in modaliteti] for k in klase])

    fig, ax = plt.subplots(figsize=(1.35 * len(modaliteti) + 3.4, 0.62 * len(klase) + 2.2))
    im = ax.imshow(matrica, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(modaliteti)), modaliteti, fontsize=9)
    ax.set_yticks(range(len(klase)), [SRPSKI.get(k, k) for k in klase], fontsize=9)
    for i in range(len(klase)):
        for j in range(len(modaliteti)):
            v = matrica[i, j]
            ax.text(j, i, f"{100*v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > matrica.max() * 0.6 else "#14171E")
    ax.set_title(f"{model} · допринос модалитета по класама (%)", fontsize=10.5, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="просечан удео")
    fig.tight_layout()
    put = izlaz / f"doprinos_{model}.png"
    fig.savefig(put, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ", put.name)

    tabela = {k: {m: round(zbir[k][m] / max(broj[k], 1), 5) for m in modaliteti} for k in klase}
    (izlaz / f"doprinos_{model}.json").write_text(
        json.dumps(tabela, indent=2, ensure_ascii=False), encoding="utf-8")
    return tabela


def main() -> None:
    args = parse_args()
    koren = Path(args.output_root)
    izlaz = Path(args.out)
    izlaz.mkdir(parents=True, exist_ok=True)

    import torch
    print("uredjaj:", args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    klase = build_local_label_info(args.data_dir).names
    uzorci = load_local_samples(args.data_dir, args.split)

    for model in args.models:
        print(f"\n=== {model} ===")
        try:
            org, mm = ucitaj_par(koren, model, args.device)
        except FileNotFoundError as exc:
            print("  preskacem:", exc)
            continue
        print(" uporedne GradCAM slike:")
        uporedne_slike(model, org, mm, uzorci, klase, izlaz, args.per_class)
        print(" doprinos modaliteta po klasama:")
        doprinos_po_klasama(model, mm, uzorci, klase, izlaz, args.limit_attr)

    print(f"\nsve u: {izlaz.resolve()}")


if __name__ == "__main__":
    main()
