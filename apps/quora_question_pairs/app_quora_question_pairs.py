"""Quora Question Pairs - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_quora_question_pairs.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_quora_question_pairs.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_quora_question_pairs.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


SORU_KELIMELERI = ("what", "how", "why", "which", "who", "whom", "where",
                   "when", "is", "are", "can", "do", "does", "should", "will")


def soru_tipi(metin):
    ilk = metin.lower().split()[0] if metin.strip() else ""
    return ilk if ilk in SORU_KELIMELERI else "diger"


def ortak_parca_orani(a, b, sinir=100):
    import difflib
    a, b = a[:sinir].lower(), b[:sinir].lower()
    if not a or not b:
        return 0.0
    eslesme = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b))
    return eslesme.size / max(min(len(a), len(b)), 1)


def tahmin(soru1, soru2):
    if BUNDLE is None:
        return "Once train_quora_question_pairs.ipynb calistirilip model uretilmeli."
    k1, k2 = set(soru1.lower().split()), set(soru2.lower().split())
    v1 = BUNDLE["vectorizer"].transform([soru1])
    v2 = BUNDLE["vectorizer"].transform([soru2])
    # Notebook'un 3. bolumundeki cift ozelliklerinin birebir karsiliklari.
    # Soru sikligi demoda bilinemez: derlemde bir kez gecmis gibi davranilir.
    degerler = {
        "ortak_kelime_orani": len(k1 & k2) / max(len(k1 | k2), 1),
        "uzunluk_farki": abs(len(soru1) - len(soru2)),
        "soru_tipi_ayni": float(soru_tipi(soru1) == soru_tipi(soru2)),
        "ortak_parca_orani": ortak_parca_orani(soru1, soru2),
        "soru1_sikligi": float(np.log1p(1)),
        "soru2_sikligi": float(np.log1p(1)),
        "tfidf_kosinus": float(v1.multiply(v2).sum()),
    }
    # Kolon listesi bundle'dan geliyor; eski (2 ozellikli) dosyayla da calisir.
    kolonlar = BUNDLE.get("cift_kolonlari", ["ortak_kelime_orani", "uzunluk_farki"])
    ozellik = np.array([[degerler.get(kolon, 0.0) for kolon in kolonlar]])
    X = hstack([BUNDLE["vectorizer"].transform([soru1 + " " + soru2]), ozellik])
    olasilik = BUNDLE["model"].predict_proba(X)[0, 1]
    etiket = "DUPLICATE / AYNI soru" if olasilik > 0.5 else "different / farkli sorular"
    return f"Same-question probability / Ayni soru olasiligi: %{olasilik * 100:.1f} - {etiket}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Same Question? / Ayni Soru mu?"
ALTBASLIK = ("Predicts whether two questions ask the same thing.  \n"
             "Iki sorunun ayni seyi sorup sormadigini tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Detect whether two Quora questions **ask the same thing** (e.g. "How can I learn
Python?" vs "What is the best way to learn Python?"). Quora uses this to merge
duplicates. Binary classification, scored by **log-loss** -- confidence matters.

Quora'ya sorulan iki sorunun **ayni seyi mi sordugunu** tespit etmek. Quora bunu
yinelenen sorulari birlestirmek icin kullanir. Ikili siniflandirma; olcut
**log-loss** -- tahminin ne kadar *emin* oldugu da onemli.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Why hand-made pair features / Neden elle cift ozellikleri:** TF-IDF sees the pair as
  one concatenated document and cannot compute the overlap *between* the questions.
  *(TF-IDF cifti tek belge olarak goruyor; aralarindaki iliskiyi hesaplayamiyor.)*
- **Pair features / Cift ozellikleri:** shared-word ratio (the strongest single signal),
  length difference, same question type, longest common chunk, and TF-IDF cosine
  similarity, which rewards overlap on *rare* words.
  *(Jaccard her kelimeyi esit sayar; kosinus nadir kelimeye agirlik verir.)*
- **Question frequency: structural, not leakage / Yapisal sinyal, sizinti degil:** how
  often a question recurs is counted without ever looking at a label.
  *(Etiketi gormek sizintidir, veri kumesinin seklini gormek degildir.)*
- **What the scores say next / Skorlarin soyledigi:** log loss ~0.60 vs a ~0.66 baseline
  but ROC-AUC ~0.87 -- the probabilities are under-confident, so **calibration** is the
  obvious next step.
  *(Siralama kayiptan daha iyi; sonraki adim kalibrasyon.)*
"""

ORNEK_KOD = '''\
# 1) Pair features that describe the questions "relative to each other"
ortak = len(k1 & k2) / len(k1 | k2)                  # shared-word ratio
kosinus = tfidf(q1).multiply(tfidf(q2)).sum()         # TF-IDF similarity
uzunluk_farki = abs(len(q1) - len(q2))

# 2) Combine text n-grams + pair features
X = hstack([tfidf.transform([q1 + " " + q2]), [[ortak, kosinus, uzunluk_farki]]])

# 3) CV
skor = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
'''


with gr.Blocks(title="Same Question? / Ayni Soru mu?") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            soru1 = gr.Textbox(label="Question 1 / Soru 1", placeholder="How can I learn Python?")
            soru2 = gr.Textbox(label="Question 2 / Soru 2", placeholder="What is the best way to learn Python?")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [soru1, soru2], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_quora_question_pairs.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/quora-question-pairs)")


if __name__ == "__main__":
    demo.launch()
