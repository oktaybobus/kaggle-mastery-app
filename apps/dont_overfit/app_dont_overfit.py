"""Don't Overfit II - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_dont_overfit.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_dont_overfit.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_dont_overfit.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']} CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']} capraz dogrulama:** `{metrik['cv_skor']:.4f}`")


def cekirdek(model):
    """Pipeline ise son adımı (asıl modeli), değilse modelin kendisini döndür.

    Notebook'taki yarış ölçeğe duyarlı modelleri StandardScaler ile sarıyor;
    sarılı modelde `coef_` Pipeline'ın üstünde görünmez.
    """
    return model.steps[-1][1] if hasattr(model, "steps") else model


def tahmin(tohum):
    if BUNDLE is None:
        return "Once train_dont_overfit.ipynb calistirilip model uretilmeli."
    rng = np.random.default_rng(int(tohum))
    satir = pd.DataFrame(rng.normal(0, 1, size=(1, len(BUNDLE["columns"]))),
                         columns=BUNDLE["columns"])
    olasilik = BUNDLE["model"].predict_proba(satir)[0, 1]

    # Kazanan modeli notebook'taki yaris seciyor; katsayi veren bir model
    # cikmayabilir (orn. RandomForest). O yuzden katsayi notunu kosula bagladim.
    son = cekirdek(BUNDLE["model"])
    if hasattr(son, "coef_"):
        aktif = int((np.asarray(son.coef_) != 0).sum())
        not_satiri = (f"\n(Model uses {aktif} of {len(BUNDLE['columns'])} features / "
                      f"{len(BUNDLE['columns'])} ozellikten {aktif} tanesini kullaniyor)")
    else:
        not_satiri = ""
    return f"Positive-class probability / Pozitif sinif olasiligi: %{olasilik * 100:.1f}{not_satiri}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Don't Overfit! II"
ALTBASLIK = ("250 rows, 300 anonymous features: generalizing without memorizing. "
             "The demo generates a random profile and shows the model's call.  \n"
             "250 satir, 300 anonim ozellik: ezber yapmadan genelleme. Demo rastgele "
             "bir profil uretip modelin tahminini gosterir.")

# Bu yarismada CV, standart 5-kat degil "5x5 tekrarli" -- veri cok az oldugu icin
# bir kez bolmek gurultulu; not bu yuzden biraz daha uzun.
CV_NOT = ("<sub>**CV (cross-validation):** the way to test a model on data it hasn't seen. "
          "Here the data is tiny (250 rows), so a single split is luck-driven; a **5x5 "
          "repeated** CV is used -- a 5-fold split repeated 5 times with different shuffles, "
          "averaging 25 tests, so the score is robust to noise.  \n"
          "**CV (capraz dogrulama):** modeli hic gormedigi veride sinama yolu. Veri cok az "
          "(250 satir), tek bolme sansa bagli; **5x5 tekrarli** kullanildi -- 5-kat bolme "
          "farkli karistirmalarla 5 kez tekrar edilir, 25 sinamanin ortalamasi alinir.</sub>")

AMAC = """
### Goal / Amac
A deliberate trap: only **250 training rows, 300 anonymous features** and 19,750
test rows. The point is to build a model that generalizes **without memorizing**.
Binary classification, scored by **ROC-AUC** (ranking quality).

Kasitli bir tuzak: **sadece 250 egitim satiri, 300 anonim ozellik** ve 19.750 test
satiri. Amac, az veriye **ezber yapmadan** genelleyen bir model kurmak. Ikili
siniflandirma; olcut **ROC-AUC** (siralama kalitesi).
"""

NASIL = """
### How it was done / Nasil yapildi
- **Main danger is overfitting / Ana tehlike asiri ogrenme:** 300 features on 250 rows;
  training AUC glued itself to **1.000** while validation stayed near **0.69**. The cure
  is strong regularization.
  *(Egitim AUC 1.000'e yapisirken validasyon 0.69'da kaldi -- boyut tuzaginin kaniti.)*
- **L1 does the selection structurally / L1 secimi yapisal yapar:** at `C=0.05` only
  **four** features survived, and that four-column model generalized better than the race
  winner using all 300 (~0.73 vs ~0.69).
  *(300 ozellikten 4'e inen model, hepsini kullanandan daha iyi genelledi.)*
- **Robust validation / Saglam dogrulama:** a single split misleads at 250 rows; **5x5
  repeated CV** (25 measurements) denoises the score.
  *(250 satirda tek bir CV, olctugu sey kadar rastgeleligi de olcer.)*
- **A reported negative result / Raporlanan negatif sonuc:** the soft-voting blend lost
  to plain **ElasticNet** (~0.74), which already balances L1 and L2 in one model.
  *(Harman kaybetti ve bu gizlenmedi.)*
"""

ORNEK_KOD = '''\
# 1) ElasticNet logistic: L1 zeros most coefficients (kills overfitting)
model = LogisticRegression(penalty="elasticnet", solver="saga",
                           l1_ratio=0.5, C=0.1, max_iter=5000)

# 2) Robust on little data: repeat 5-fold CV 5 times (25 tests)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
skor = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

# 3) How many features survived?
print((model.fit(X, y).coef_ != 0).sum(), "/ 300 features used")
'''


with gr.Blocks(title="Don't Overfit! II") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            tohum = gr.Slider(0, 9999, value=42, step=1, label="Random sample seed / Rastgele ornek tohumu")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [tohum], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_dont_overfit.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/dont-overfit-ii)")


if __name__ == "__main__":
    demo.launch()
