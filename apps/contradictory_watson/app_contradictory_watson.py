"""Contradictory, My Dear Watson - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

import joblib
from scipy.sparse import hstack, csr_matrix
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_contradictory_watson.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_contradictory_watson.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_contradictory_watson.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


ETIKETLER = {0: "ENTAILMENT - hypothesis supports the premise / hipotez onermeyi destekliyor",
             1: "NEUTRAL - relationship unclear / iliski belirsiz",
             2: "CONTRADICTION - hypothesis contradicts the premise / hipotez celisiyor"}


def tahmin(onerme, hipotez):
    if BUNDLE is None:
        return "Once train_contradictory_watson.ipynb calistirilip model uretilmeli."
    vec = BUNDLE["vectorizer"].transform([onerme + " || " + hipotez])
    # Yeni model dosyasi n-gramlarin yanina kelime ortusme oranini bekliyor;
    # eski dosyada bu sutun yok — beklenen genislige bakarak ikisiyle de calis.
    beklenen = getattr(BUNDLE["model"], "n_features_in_", vec.shape[1])
    if beklenen == vec.shape[1] + 1:
        p = set(onerme.lower().split())
        h = set(hipotez.lower().split())
        oran = len(p & h) / max(len(h), 1)
        vec = hstack([vec, csr_matrix([[oran]])])
    etiket = int(BUNDLE["model"].predict(vec)[0])
    return ETIKETLER[etiket]


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Contradiction Detection (15 languages) / Celiski Tespiti (15 dil)"
ALTBASLIK = ("Predicts the logical relation (entailment / contradiction / neutral) "
             "between a premise and a hypothesis.  \n"
             "Bir onerme ile hipotez arasindaki mantik iliskisini tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Given a **premise** and a **hypothesis**, find their logical relation: does the
hypothesis *support* (entailment), *contradict*, or stay *neutral*? This is an NLI
task, and the data spans **15 languages**. Multi-class classification, scored by accuracy.

Bir **onerme** (premise) ile **hipotez** (hypothesis) verildiginde mantik iliskisini
bulmak: hipotez onermeyi *destekliyor mu*, *celisiyor mu*, yoksa *belirsiz mi*? NLI
gorevi ve veri **15 dilde**. Cok sinifli siniflandirma; olcut dogruluk.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Character n-grams, not words / Kelime degil karakter n-gram:** with 15 languages a
  word vocabulary fragments, and **Chinese and Japanese have no whitespace words** -- a
  word tokenizer learns nothing for them.
  *(Cince ve Japonca'da bosluklu kelime kavrami yok; kelime tokenizer'i coker.)*
- **The honest limit / Durust sinir:** TF-IDF captures **surface overlap, not the logical
  relation** -- where "not" sits changes everything logically and nothing for TF-IDF.
  *(Bu yaklasim mantiksal iliskiyi modelleyemez, yalnizca yuzeysel ortusmeyi yakalar.)*
- **So the race measured the floor / Yaris tabani olctu:** all models clustered near the
  33% baseline (~44% for the winner) -- the problem is the **approach**, not the model.
  *(Sorun model seciminde degil, yaklasimin kendisinde.)*
- **XLM-RoBERTa broke it / XLM-R siniri kirdi:** encoding both sentences **together**
  took accuracy from **~44% to ~70%** -- a change of representation, not tuning.
  *(Fark ayarlamada degil temsilde; hicbir hiperparametre aramasi bunu kapatamazdi.)*
"""

ORNEK_KOD = '''\
# 1) Join the two sentences with an explicit separator, so the premise's last
#    word and the hypothesis's first cannot form a fake n-gram at the seam.
metin = premise + " || " + hypothesis

# Character n-grams, NOT words: with 15 languages a word vocabulary fragments,
# and Chinese/Japanese have no whitespace-delimited words at all.
tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                        max_features=50000, min_df=3)
X = tfidf.transform([metin])

# 2) Add the overlap signal: shared-word ratio
oran = len(set(premise.split()) & set(hypothesis.split())) / len(set(hypothesis.split()))
X = hstack([X, csr_matrix([[oran]])])

# 3) CV over 3 classes (entail / neutral / contradiction)
skor = cross_val_score(model, X, y, cv=5, scoring="f1_macro").mean()
'''


with gr.Blocks(title="Contradiction Detection (15 languages) / Celiski Tespiti (15 dil)") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            onerme = gr.Textbox(lines=2, label="Premise / Onerme",
                                placeholder="He is sleeping on the couch.")
            hipotez = gr.Textbox(lines=2, label="Hypothesis / Hipotez",
                                 placeholder="He is wide awake.")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [onerme, hipotez], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_contradictory_watson.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/contradictory-my-dear-watson)")


if __name__ == "__main__":
    demo.launch()
