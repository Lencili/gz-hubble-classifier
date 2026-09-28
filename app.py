from __future__ import annotations

import argparse
import io
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.gz_classifier.data import build_local_label_info, load_local_samples
from src.gz_classifier.explain import explain_image, modality_attribution
from src.gz_classifier.infer import GalaxyPredictor, load_predictor
from src.gz_classifier.modalities import apply_modality

# transformers, sklearn i zeroshot se uvoze tek kad se CLIP zaista koristi, da
# se na serveru bez CLIP modela ti paketi uopste ne instaliraju.


DEFAULT_LABELS = [
    "smooth",
    "featured_or_disk",
    "edge_on_disk",
    "barred_spiral",
    "unbarred_spiral",
    "merger",
    "artifact",
]


@st.cache_resource
def cached_predictor(checkpoint: str | None) -> GalaxyPredictor | None:
    return load_predictor(checkpoint)


@st.cache_resource
def cached_clip(model_path: str) -> tuple["CLIPModel", "CLIPProcessor", torch.device]:
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor, device


def clip_predict(image: Image.Image, labels: list[str], model_path: str, top_k: int = 5) -> list[dict[str, float | str]]:
    from src.gz_classifier.zeroshot import prompts_for

    model, processor, device = cached_clip(model_path)
    prompts = [prompt for label in labels for prompt in prompts_for(label)]
    label_ids = [index for index, label in enumerate(labels) for _ in prompts_for(label)]
    inputs = processor(text=prompts, images=image.convert("RGB"), return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.softmax(dim=1)[0]
        per_label = torch.zeros(len(labels), device=device)
        for score, label_id in zip(scores, label_ids):
            per_label[label_id] += score
        per_label = per_label / per_label.sum()
        values, indices = per_label.topk(min(top_k, len(labels)))
    return [{"label": labels[int(i)], "probability": float(v)} for v, i in zip(values.cpu(), indices.cpu())]


def parse_cli() -> argparse.Namespace:
    """Argumenti komandne linije, sa podrazumevanim vrednostima iz okruzenja.

    Lokalno se i dalje pokrece sa `--checkpoint ... --data-dir ...`, a na
    serveru (Hugging Face Spaces) se ista podesavanja zadaju kao promenljive
    okruzenja. Izricit argument uvek ima prednost nad promenljivom.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", default=os.environ.get("GZ_CHECKPOINT", ""))
    parser.add_argument("--data-dir", default=os.environ.get("GZ_DATA_DIR", "data/gz_hubble_500"))
    parser.add_argument("--report-dir", default=os.environ.get("GZ_REPORT_DIR", ""))
    parser.add_argument("--clip-model",
                        default=os.environ.get("GZ_CLIP_MODEL", "models/clip-vit-base-patch32"))
    return parser.parse_known_args()[0]


@st.cache_resource(show_spinner="Preuzimam modele sa Hugging Face Hub-a...")
def preuzmi_model_sa_huba() -> str:
    """Na serveru modeli stizu sa Hub-a; lokalno se ne radi nista.

    GZ_MODEL_REPO zadaje repozitorijum na Hub-u, a GZ_RUNS spisak runova u njemu,
    razdvojen zarezima (svaki run je podfolder, npr. "densenet121_multimodal").
    Fajlovi se smestaju u outputs_space/<run>/, pa ih zatim nalazi discover_runs().
    Ako GZ_RUNS nije zadat, uzima se jedan run iz korena repozitorijuma.
    """
    repo = os.environ.get("GZ_MODEL_REPO", "").strip()
    if not repo:
        return ""

    from huggingface_hub import hf_hub_download

    runovi = [r.strip() for r in os.environ.get("GZ_RUNS", "").split(",") if r.strip()]
    u_podfolderima = bool(runovi)
    if not runovi:
        runovi = [os.environ.get("GZ_RUN_NAME", "densenet121_multimodal")]

    prvi = ""
    for run in runovi:
        cilj = Path("outputs_space") / run
        cilj.mkdir(parents=True, exist_ok=True)
        for ime, obavezan in (("best.ckpt", True), ("metrics.json", True),
                              ("history.json", False), ("labels.json", False)):
            odrediste = cilj / ime
            if odrediste.exists():
                continue
            putanja = f"{run}/{ime}" if u_podfolderima else ime
            try:
                shutil.copy(hf_hub_download(repo_id=repo, filename=putanja), odrediste)
            except Exception as greska:
                if obavezan:
                    st.error(f"Ne mogu da preuzmem {putanja} iz repozitorijuma {repo}: {greska}")
                    break
        prvi = prvi or str(cilj / "best.ckpt")
    return prvi


BEZ_CKPT = "  (bez checkpointa)"


def output_roots() -> list[Path]:
    """Svi folderi sa rezultatima u korenu projekta, najskoriji prvi.

    Trazi se po obrascu umesto po tacnom imenu, da preimenovanje foldera sa
    rezultatima ne zahteva izmenu koda.
    """
    koreni = [p for p in Path(".").glob("output*") if p.is_dir()]
    return sorted(koreni, key=lambda p: p.stat().st_mtime, reverse=True)


def discover_runs(roots: list[Path] | None = None) -> dict[str, Path]:
    """Runovi se prepoznaju po metrics.json, a ne po checkpointu.

    Kada je trening radjen u oblaku, cesto se lokalno preuzmu samo metrike, bez
    tezina modela. Takvi runovi i dalje treba da budu vidljivi na stranicama
    Trening i Eksperiment; samo predikcija zahteva checkpoint.
    """
    runs: dict[str, Path] = {}
    for root in roots or output_roots():
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "metrics.json").exists():
                oznaka = "" if (child / "best.ckpt").exists() else BEZ_CKPT
                runs[f"{root.name} / {child.name}{oznaka}"] = child
    return runs


def load_experiment_summary(root: Path) -> pd.DataFrame:
    summary_path = root / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)

    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = metrics["config"]
        rows.append(
            {
                "run": metrics_path.parent.name,
                "model": config["model"],
                "condition": "multimodal" if len(config["modalities"]) > 1 else "original",
                "modalities": "+".join(config["modalities"]),
                "best_val_macro_f1": metrics.get("best_val_macro_f1", metrics.get("best_macro_f1")),
                "test_macro_f1": metrics.get("test", metrics.get("final_val", {})).get("macro_f1"),
                "test_acc": metrics.get("test", metrics.get("final_val", {})).get("acc"),
                "epochs_run": metrics["epochs_run"],
            }
        )
    return pd.DataFrame(rows)


def dataset_labels(data_dir: Path, predictor: GalaxyPredictor | None) -> list[str]:
    if data_dir.exists() and (data_dir / "metadata.csv").exists():
        return build_local_label_info(data_dir).names
    if predictor:
        return predictor.label_names
    return DEFAULT_LABELS


def grouped_samples(data_dir: Path, split: str) -> dict[str, list[Path]]:
    if not (data_dir / "metadata.csv").exists():
        return {}
    groups: dict[str, list[Path]] = {}
    for path, label in load_local_samples(data_dir, split):
        groups.setdefault(label, []).append(path)
    return groups


def par_run(naziv: str) -> str | None:
    """Naziv suprotnog uslova za isti model: original <-> multimodal."""
    if naziv.endswith("_multimodal"):
        return naziv[: -len("_multimodal")] + "_original"
    if naziv.endswith("_original"):
        return naziv[: -len("_original")] + "_multimodal"
    return None


def dugme_za_snimanje(slika: Image.Image, naziv: str, kljuc: str) -> None:
    bafer = io.BytesIO()
    slika.save(bafer, format="PNG")
    st.download_button("Sacuvaj sliku", bafer.getvalue(), file_name=naziv,
                       mime="image/png", key=kljuc, use_container_width=True)


def izaberi_sliku(data_dir: Path):
    """Vraca (slika, opis) ili (None, '') ako nema izbora."""
    source = st.radio("Izvor slike", ["Lokalni dataset", "Upload"], horizontal=True)
    if source == "Lokalni dataset":
        if not (data_dir / "metadata.csv").exists():
            st.warning("Lokalni dataset nije pronadjen. Pokreni prepare_dataset komandu iz README-a.")
            return None, ""
        splits = sorted(set(pd.read_csv(data_dir / "metadata.csv")["split"]))
        split = st.segmented_control("Split", splits, default="test" if "test" in splits else splits[0])
        groups = grouped_samples(data_dir, split or splits[0])
        if not groups:
            st.warning("Izabrani split je prazan.")
            return None, ""
        label = st.selectbox("Klasa", sorted(groups))
        selected = st.selectbox("Slika", groups[label], format_func=lambda path: path.name)
        return Image.open(selected).convert("RGB"), f"{split} / {label} / {selected.name}"

    uploaded = st.file_uploader("Upload galaxy image", type=("jpg", "jpeg", "png", "webp"))
    if uploaded is None:
        st.info("Uploaduj sliku za predikciju.")
        return None, ""
    return Image.open(uploaded).convert("RGB"), uploaded.name


def prikazi_objasnjenje(predictor: GalaxyPredictor, image: Image.Image, class_id: int | None,
                        naslov: str, kljuc: str) -> None:
    try:
        obj = explain_image(predictor.model, image, predictor.transform, predictor.device,
                            class_id=class_id, model_family=predictor.model_family)
        st.image(obj.overlay, caption=f"{naslov} — {obj.metod}", use_container_width=True)
        dugme_za_snimanje(obj.overlay, f"gradcam_{kljuc}.png", f"dl_{kljuc}")
    except Exception as exc:
        st.warning(f"Objasnjenje nije dostupno: {exc}")


def prediction_page(predictor: GalaxyPredictor | None, data_dir: Path, labels: list[str],
                    clip_model_path: str, runs: dict[str, Path] | None = None,
                    izabrani: str | None = None) -> None:
    st.header("Predikcija")
    image, caption = izaberi_sliku(data_dir)
    if image is None:
        return

    if predictor is None:
        st.warning("Checkpoint nije ucitan, pa nema ni predikcije ni GradCAM-a. "
                   "Preuzmi best.ckpt sa Drive-a u odgovarajuci podfolder.")
        st.image(image, caption=caption, width=320)
        with st.expander("Zero-shot CLIP baseline"):
            if Path(clip_model_path).exists():
                st.dataframe(clip_predict(image, labels, clip_model_path),
                             use_container_width=True, hide_index=True)
            else:
                st.warning(f"CLIP model nije pronadjen lokalno: {clip_model_path}")
        return

    predictions = predictor.predict(image, top_k=len(predictor.label_names))
    najbolja = str(predictions[0]["label"])

    # --- izbor klase koja se objasnjava -------------------------------------
    st.subheader("Objasnjenje odluke")
    c1, c2 = st.columns([2, 3])
    with c1:
        izbor = st.selectbox(
            "Objasni klasu",
            [f"predvidjena ({najbolja})"] + list(predictor.label_names),
            help="GradCAM moze da odgovori i na pitanje 'zasto NIJE ova druga klasa'. "
                 "Poredjenje dve mape za isti snimak je najkorisniji prikaz.",
        )
    class_id = None if izbor.startswith("predvidjena") else predictor.label_names.index(izbor)
    with c2:
        uporedi = st.checkbox(
            "Uporedi sa suprotnim uslovom ulaza",
            help="Prikazuje istu sliku kroz model iste arhitekture treniran u drugom uslovu "
                 "(original naspram multimodalnog), da se vidi da li gledaju na isto mesto.",
        )

    kolone = st.columns(3 if uporedi else 2)
    kolone[0].image(image, caption=caption, use_container_width=True)
    with kolone[1]:
        prikazi_objasnjenje(predictor, image, class_id, "ovaj model", "glavni")

    if uporedi:
        with kolone[2]:
            par_ime, drugi = None, None
            if runs and izabrani:
                koren, _, run = izabrani.partition(" / ")
                cilj = par_run(run.replace(BEZ_CKPT, ""))
                for ime, putanja in (runs or {}).items():
                    if cilj and ime.startswith(f"{koren} / {cilj}") and (putanja / "best.ckpt").exists():
                        par_ime = ime
                        drugi = cached_predictor(str(putanja / "best.ckpt"))
                        break
            if par_ime and drugi:
                prikazi_objasnjenje(drugi, image, class_id, par_ime.split(" / ")[-1], "par")
            else:
                st.info("Suprotni uslov nije dostupan lokalno (nedostaje njegov best.ckpt).")

    # --- predikcije ---------------------------------------------------------
    st.subheader("Predikcije treniranog modela")
    st.dataframe(predictions, use_container_width=True, hide_index=True)

    # --- doprinos modaliteta za OVU sliku ------------------------------------
    if len(predictor.modalities) > 1:
        st.subheader("Na sta se model oslonio kod ove slike")
        st.caption(
            "Tezine adaptera daju jedan recept za sve slike. Ova mera je drugacija — "
            "racuna se iz gradijenta po ulaznim pikselima, pa pokazuje doprinos svakog "
            "modaliteta bas kod ovog snimka."
        )
        try:
            udeli = modality_attribution(predictor.model, image, predictor.transform,
                                         predictor.device, predictor.modalities, class_id=class_id)
            df = pd.DataFrame({"modalitet": list(udeli), "udeo": list(udeli.values())}).set_index("modalitet")
            st.bar_chart(df)
            st.dataframe((df * 100).round(2).rename(columns={"udeo": "udeo (%)"}),
                         use_container_width=True)
        except Exception as exc:
            st.warning(f"Racun doprinosa nije uspeo: {exc}")

    with st.expander("Kako izgleda svih sest modaliteta ove slike"):
        st.image([apply_modality(image, m) for m in predictor.modalities],
                 caption=list(predictor.modalities), width=180)

    with st.expander("Zero-shot CLIP baseline"):
        if Path(clip_model_path).exists():
            st.dataframe(clip_predict(image, labels, clip_model_path),
                         use_container_width=True, hide_index=True)
        else:
            st.warning(f"CLIP model nije pronadjen lokalno: {clip_model_path}")


def dataset_page(data_dir: Path) -> None:
    st.header("Dataset")
    metadata_path = data_dir / "metadata.csv"
    info_path = data_dir / "dataset_info.json"
    if not metadata_path.exists():
        st.warning("Dataset jos nije pripremljen lokalno.")
        return

    df = pd.read_csv(metadata_path)
    st.write(f"Lokalna putanja: `{data_dir}`")
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        st.json(info, expanded=False)

    counts = df.groupby(["split", "label"]).size().reset_index(name="count")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    all_splits = sorted(set(df["split"]))
    split = st.segmented_control("Primeri iz splita", all_splits, default=all_splits[0], key="dataset_split")
    sample_df = df[df["split"] == split].head(24)
    image_paths = [data_dir / row["path"] for _, row in sample_df.iterrows()]
    captions = [row["label"] for _, row in sample_df.iterrows()]
    st.image(image_paths, caption=captions, width=140)


def training_page(report_dir: Path) -> None:
    st.header("Tok treninga i performanse")
    history_path = report_dir / "history.json"
    metrics_path = report_dir / "metrics.json"
    if not history_path.exists():
        st.warning("Jos nema izvestaja treninga. Pokreni trening komandu iz README-a.")
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    rows = []
    for item in history:
        rows.append(
            {
                "epoch": item["epoch"],
                "train_loss": item["train"]["loss"],
                "train_acc": item["train"]["acc"],
                "train_macro_f1": item["train"]["macro_f1"],
                "val_loss": item["val"]["loss"],
                "val_acc": item["val"]["acc"],
                "val_macro_f1": item["val"]["macro_f1"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.line_chart(df.set_index("epoch")[["train_loss", "val_loss"]])
    st.line_chart(df.set_index("epoch")[["train_macro_f1", "val_macro_f1"]])

    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1 (validacija)", f"{metrics['best_macro_f1']:.3f}")
    c2.metric("Macro F1 (test)", f"{metrics['test']['macro_f1']:.3f}")
    c3.metric("Tacnost (test)", f"{metrics['test']['acc']:.3f}")
    c4.metric("Epoha", metrics["epochs_run"])

    st.subheader("Early stopping")
    st.write(
        f"Trening prati `macro_f1` na test/validation splitu. "
        f"Zaustavljanje: patience={metrics['early_stopping']['patience']}, "
        f"min_delta={metrics['early_stopping']['min_delta']}."
    )

    report = metrics.get("classification_report", {})
    labels = metrics.get("labels", [])
    class_rows = []
    for label in labels:
        if label in report:
            class_rows.append({"label": label, **report[label]})
    if class_rows:
        st.subheader("Performanse po klasi (test skup)")
        st.caption(
            "Ovo su brojevi iz kojih nastaje macro F1: prosek kolone f1-score "
            "preko pet klasa, svaka sa istom tezinom."
        )
        st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)

    if "confusion_matrix" in metrics:
        st.subheader("Confusion matrix")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels), use_container_width=True)


def experiment_page(root: Path) -> None:
    st.header("Eksperiment: original vs vise modaliteta")
    st.caption(f"Rezultati iz: `{root}`")
    df = load_experiment_summary(root)
    if df.empty:
        st.warning("Jos nema rezultata eksperimenta. Pokreni `python -m src.gz_classifier.run_modality_experiment`.")
        return

    display_cols = [
        "model",
        "condition",
        "pretrained",
        "best_val_macro_f1",
        "test_macro_f1",
        "test_acc",
        "epochs_run",
        "delta_test_macro_f1",
        "delta_test_acc",
        "modalities",
    ]
    available_cols = [col for col in display_cols if col in df.columns]
    st.dataframe(df[available_cols].sort_values(["model", "condition"]), use_container_width=True, hide_index=True)

    metric = "test_macro_f1" if "test_macro_f1" in df.columns else "best_macro_f1"
    if {"model", "condition", metric}.issubset(df.columns):
        st.subheader("Macro F1 na test skupu")
        # stack=False: stubici jedan pored drugog. Sa podrazumevanim slaganjem
        # dva uslova se sabiraju, pa se cita 1,6 umesto dve vrednosti oko 0,8.
        st.bar_chart(df.pivot_table(index="model", columns="condition", values=metric),
                     stack=False, y_label="macro F1 (test)")

    delta_col = "delta_test_macro_f1" if "delta_test_macro_f1" in df.columns else "delta_best_macro_f1_vs_original"
    if delta_col in df.columns:
        delta_df = df[df["condition"] == "multimodal"][["model", delta_col]].set_index("model")
        st.subheader("Dobitak/gubitak multimodalnog ulaza")
        st.caption("Razlika macro F1 mere u odnosu na isti model treniran samo na originalnoj slici.")
        st.bar_chart(delta_df, y_label="promena macro F1")

    reliance_cols = [c for c in df.columns if c.startswith("rel_")]
    if reliance_cols:
        st.subheader("Oslanjanje na modalitete (tezine adaptera)")
        rel = df[df["condition"] == "multimodal"].set_index("model")[reliance_cols]
        rel.columns = [c.removeprefix("rel_") for c in rel.columns]
        st.caption("Udeo svakog modaliteta u tezinama adaptera; po modelu se sabiraju na 1.")
        st.bar_chart(rel, stack=False, y_label="udeo u tezinama adaptera")
        st.dataframe(rel, use_container_width=True)


def main() -> None:
    args = parse_cli()
    st.set_page_config(page_title="Galaxy Classifier", page_icon="*", layout="wide")
    st.title("Galaxy Classifier")

    preuzmi_model_sa_huba()
    runs = discover_runs()
    selected_run: str | None = None
    checkpoint = args.checkpoint or ""
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    clip_model_path = args.clip_model

    if runs:
        default_index = 0
        for index, (_name, run_dir) in enumerate(runs.items()):
            if (run_dir / "best.ckpt").as_posix() == Path(checkpoint).as_posix():
                default_index = index
                break
        selected_run = st.sidebar.selectbox("Model", list(runs), index=default_index)
        run_dir = runs[selected_run]
        ckpt_path = run_dir / "best.ckpt"
        checkpoint = str(ckpt_path) if ckpt_path.exists() else ""
        report_dir = run_dir
        if not ckpt_path.exists():
            st.sidebar.info(
                "Za ovaj run lokalno postoje samo metrike. Stranice Trening i "
                "Eksperiment rade normalno; za predikciju preuzmi best.ckpt sa Drive-a."
            )
    else:
        st.sidebar.warning("Nema rezultata ni u jednom output folderu.")

    page = st.sidebar.radio("Stranica", ["Predikcija", "Dataset", "Trening", "Eksperiment"])

    with st.sidebar.expander("Napredna podesavanja", expanded=False):
        st.caption("Ove putanje se obicno ne menjaju.")
        checkpoint = st.text_input("Checkpoint", value=checkpoint)
        data_dir = Path(st.text_input("Data dir", value=str(data_dir)))
        report_dir = Path(st.text_input("Report dir", value=str(report_dir)))
        clip_model_path = st.text_input("CLIP model", value=clip_model_path)

    predictor = cached_predictor(checkpoint or None)
    labels = dataset_labels(data_dir, predictor)

    if page == "Predikcija":
        prediction_page(predictor, data_dir, labels, clip_model_path,
                        runs=runs, izabrani=selected_run)
    elif page == "Dataset":
        dataset_page(data_dir)
    elif page == "Trening":
        training_page(report_dir)
    else:
        experiment_page(report_dir.parent)


if __name__ == "__main__":
    main()
