# Uputstvo za pokretanje

## Šta je promenjeno u kodu


| Fajl | Izmena |
|---|---|
| `modalities.py` | Augmentacija se uzorkuje jednom po slici i primenjuje identično na svih 6 modaliteta, pa su kanali prostorno poravnati. Filteri se primenjuju posle resize-a, na čistoj slici. |
| `models.py` | `InputAdapter` se inicijalizuje tako da na startu propušta prvi modalitet nepromenjen. Dodata funkcija `modality_reliance` koja iz težina adaptera računa oslanjanje na svaki modalitet. |
| `train.py` | Odvojen validation skup za rano zaustavljanje; finalna evaluacija se radi iz `best.ckpt` na test skupu; dodat `--seed`. |
| `data.py` | Nasumično uzorkovanje umesto prvih N; automatsko izostavljanje klasa sa premalo primera; podela na train/val/test; dva prolaza kroz parquet. |
| `prepare_dataset.py` | Novi CLI, plus `--count-only` za tabelu raspodele klasa. |
| `run_modality_experiment.py` | Pretrenirane težine su podrazumevane; `--from-scratch` za poređenje; `--models` za podskup; bogatiji `summary.csv`. |
| `app.py` | Prepoznaje više outputs foldera; prikazuje test metrike i oslanjanje na modalitete. |

---

## Korak 1 — Raspodela klasa (tabela za rad)

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.gz_classifier.prepare_dataset --count-only
```

Ispisuje broj primera po klasi i snima `raspodela_klasa.json`. To je Tabela 1 u poglavlju 3.2.2.

## Korak 2 — Priprema novog skupa

```powershell
python -m src.gz_classifier.prepare_dataset --output-dir data/gz_hubble_500 --no-download
```

Koristi parquet fajlove koje već imaš, ne skida ništa novo. Traje nekoliko minuta.

Rezultat: **5 klasa × (450 train + 100 validation + 150 test)** = 3 500 slika.
Klasa `featured_without_bar_or_spiral` se automatski izostavlja (ima samo 61 primer u celom skupu).

## Korak 3 — Priprema za Colab

```powershell
Compress-Archive -Path src, app.py, requirements.txt, proba_pipeline.py -DestinationPath kod.zip -Force
Compress-Archive -Path data\gz_hubble_500 -DestinationPath podaci.zip -Force
```

`kod.zip` je oko 40 KB, `podaci.zip` oko 160 MB (JPEG se ne sabija dodatno).

Napravi na Google Drive-u folder `diplomski` i otpremi u njega:
- `kod.zip`
- `podaci.zip`
- `colab_trening.ipynb`

Zatim otvori `colab_trening.ipynb` sa Drive-a (desni klik → Open with → Google Colaboratory) i pokreni ćelije redom.

## Korak 4 — Posle treninga

Rezultati se vraćaju na Drive u `diplomski/outputs_500_pretrained/`. Skini taj folder u projekat i pokreni aplikaciju:

```powershell
streamlit run app.py
```
