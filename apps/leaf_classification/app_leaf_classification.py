"""Leaf Classification - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

import joblib
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_leaf_classification.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_leaf_classification.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_leaf_classification.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


def tahmin(ornek_no):
    if BUNDLE is None:
        return "Once train_leaf_classification.ipynb calistirilip model uretilmeli."
    ornek = BUNDLE["ornekler"].iloc[int(ornek_no)]
    X = BUNDLE["scaler"].transform(
        ornek.drop(labels=["id", "species"]).to_frame().T.astype(float))
    tahmin_tur = BUNDLE["encoder"].inverse_transform(BUNDLE["model"].predict(X))[0]
    return (f"Predicted / Model tahmini: {tahmin_tur}\n"
            f"Actual / Gercek tur:       {ornek['species']}\n"
            f"{'CORRECT / DOGRU' if tahmin_tur == ornek['species'] else 'WRONG / YANLIS'}")


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# 99-Species Leaf Classification / 99 Tur Yaprak Siniflandirmasi"
ALTBASLIK = ("Predicts the plant species from 192 shape/margin/texture features. "
             "Features can't be typed by hand, so the demo runs on 10 saved real "
             "samples.  \n192 sekil/kenar/doku olcusunden bitki turunu tahmin eder. "
             "Ozellikler elle girilemez; demo 10 kayitli gercek ornek uzerinde calisir.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Identify **which of 99 plant species** a leaf is, from features extracted from its
image. Multi-class classification. Kaggle ranks by **log-loss** -- being confident
in the right class matters, not just being right.

Bir yaprak goruntusunden cikarilan olculere gore **99 bitki turunden hangisi**
oldugunu tahmin etmek. Cok sinifli siniflandirma. Kaggle sirasi **log-loss** ile --
dogru siniftan ne kadar *emin* oldugun da onemli.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Ready-made features / Hazir ozellikler:** 192 numeric measures per leaf (shape,
  margin, texture); the whole question is whether they separate 99 species from 10
  examples each.
  *(Sinif basina 10 ornekle 99 sinif; ozelliklerin kalitesi veri miktarini telafi ediyor.)*
- **Log loss, not accuracy / Dogruluk degil log loss:** the metric scores the
  **confidence** behind a prediction. Here accuracy ~0.99 **and** log loss ~0.05.
  *(Onemli olan dogru tahmin degil, dogru guven.)*
- **Column order is the silent failure / Sessiz hata sutun sirasi:** if the submission's
  99 columns shift, the file still looks valid and Kaggle reports nothing -- every
  probability lands on the wrong species.
  *(Sira kayarsa dosya gecerli gorunur ama skor cop cikar.)*
- **Three ideas measured and rejected / Ucu olculup reddedildi:** probability clipping,
  a per-family committee and frozen ImageNet features all failed to beat the simple
  model -- sometimes the most instructive result is that nothing you added helped.
  *(Bir teknigin ne zaman gereksiz oldugunu bilmek, nasil uygulanacagini bilmek kadar degerli.)*
"""

ORNEK_KOD = '''\
# 1) Scaling + linear model in one Pipeline (clean in every fold)
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, C=10)),
])

# 2) CV over 99 classes (log-loss-aligned; f1_macro reported)
skor = cross_val_score(pipe, X, y, cv=5, scoring="f1_macro").mean()
'''


with gr.Blocks(title="99-Species Leaf Classification / 99 Tur Yaprak Siniflandirmasi") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            ornek_no = gr.Slider(0, 9, value=0, step=1,
                                 label="Demo sample / Demo ornegi (10 leaves from train set)")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=3)
    calistir.click(tahmin, [ornek_no], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_leaf_classification.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/leaf-classification)")


if __name__ == "__main__":
    demo.launch()
