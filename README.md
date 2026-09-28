---
title: Galaxy Morphology Classifier
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Galaxy Morphology Classifier

Multimodalna klasifikacija morfologije galaksija u pet klasa (Galaxy Zoo: Hubble).
DenseNet121 sa multimodalnim ulazom, macro F1 = 0,844 na skupu za testiranje.

Ulaz modela nije jedna slika, nego sest verzija iste slike (original + gamma,
contrast, highpass, equalize, sharpen), spojenih po kanalima u 18 kanala.
Konvolucija 1x1 sa 57 parametara svodi ih na 3 kanala koje ocekuje mreza
prethodno obucena na ImageNet-u.

Aplikacija prikazuje predikciju sa verovatnocama i Grad-CAM objasnjenje odluke,
metrike treninga i poredjenje svih 14 treninga iz rada.

Model se preuzima sa Hub-a pri pokretanju (promenljiva okruzenja `GZ_MODEL_REPO`).

Kod: https://github.com/Lencili/gz-hubble-classifier
