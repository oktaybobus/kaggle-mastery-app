"""Porto Seguro - Gradio demosu (iki dilli: EN + TR, anonim ozellikli veri).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_porto_seguro.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_porto_seguro.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_porto_seguro.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


def tahmin(gurultu, tohum):
    if BUNDLE is None:
        return "Once train_porto_seguro.ipynb calistirilip model uretilmeli."
    rng = np.random.default_rng(int(tohum))
    satir = pd.Series(BUNDLE["defaults"]).reindex(BUNDLE["columns"]).fillna(0)
    satir = satir + rng.normal(0, gurultu, size=len(satir)) * satir.abs().clip(lower=0.1)
    olasilik = BUNDLE["model"].predict_proba(satir.to_frame().T)[0, 1]
    return f"Claim probability / Sigorta talebi olasiligi: %{olasilik * 100:.2f}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Porto Seguro - Driver Claim Risk / Sigorta Talebi Riski"
ALTBASLIK = ("Estimates a driver's yearly claim probability. Features are anonymous, "
             "so the demo builds a scenario as median profile + noise.  \n"
             "Surucu profilinden yillik kaza talebi olasiligini tahmin eder. Ozellikler "
             "anonim; demo medyan profil + gurultu ile senaryo uretir.")

# Bu yarisma cok dengesiz; dogruluk yaniltir, o yuzden not siralama metrigini vurgular.
CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. Here "
          "accuracy would mislead (positives are ~3.6%), so the metric is **Gini/AUC** -- "
          "ranking quality, not right/wrong.  \n**CV (capraz dogrulama):** egitim verisi 5 "
          "parcaya bolunur, model 4'uyle ogrenip 1'inde sinanir, 5 kez doner. Dengesiz "
          "veride dogruluk yaniltir; metrik **Gini/AUC** (siralama kalitesi).</sub>")

AMAC = """
### Goal / Amac
For Brazilian insurer Porto Seguro, predict **whether a driver will file a claim**
next year. Highly imbalanced binary classification (~3.6% positives). Scored by
**normalized Gini** -- the ability to **rank** risky drivers, not label them.

Brezilyali sigortaci Porto Seguro icin bir surucunun **onumuzdeki yil kaza talebi
acip acmayacagini** tahmin etmek. Cok dengesiz ikili siniflandirma (~%3.6 pozitif).
Olcut **normalized Gini** -- riskli suruculeri dogru **siralama** yetenegi.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Imbalance reality / Dengesizlik gercegi:** ~4 in 100 drivers file a claim; "always
  no" scores 96% accuracy, so the metric is **Gini/AUC** (ranking).
  *(Dogruluk yaniltici; "hep hayir" bile %96 cikar. Metrik Gini/AUC.)*
- **Hidden missing values / Gizli eksikler:** gaps are stored as **`-1`**, so `isna()`
  returns zero; converted to `NaN` and median-imputed.
  *(`isna()` sifir doner ama veri temiz degil: eksikler -1 olarak saklanmis.)*
- **Leak-free target encoding / Sizintisiz hedef kodlamasi:** `_cat` levels are replaced
  by their claim rate, encoded **out-of-fold** so no row sees its own label.
  *(Kat-disi kodlama olmadan hedef kodlamasi, hedefi ozellige kopyalamaktir.)*
- **Demo note / Demo notu:** no real profile can be entered, so the demo adds tunable
  noise on top of the median driver.
  *(Demo, medyan surucunun uzerine ayarlanabilir gurultu ekler.)*
"""

ORNEK_KOD = '''\
# 1) The hidden trap: isna() returns 0, yet the data is not clean.
#    Missing values are stored as -1, and pandas reads them as ordinary numbers.
df = df.replace(-1, np.nan)            # make the hidden gaps visible
df = df.drop(columns=calc_kolonlari)   # ps_calc_* correlates ~0 with the target

# 2) Target-encode the _cat columns -- but out-of-fold, or it IS leakage.
#    Each row is encoded from the folds it does NOT belong to, so no row
#    ever sees its own label inside its own feature. Small levels are
#    smoothed toward the global rate (m=20), so a 3-driver level with one
#    accidental claim does not report a rate nine times the average.
oran = df.groupby(kol)["target"].mean()
sayim = df.groupby(kol)["target"].size()
puruzlu = (oran * sayim + genel_oran * 20) / (sayim + 20)

# 3) Right metric on imbalanced data: normalized Gini (= 2*AUC - 1).
#    It scores the RANKING, not a threshold -- in insurance the question is
#    "who is riskier?", not "will this driver claim?".
def gini_normalized(y_true, y_prob):
    return 2 * roc_auc_score(y_true, y_prob) - 1
'''


with gr.Blocks(title="Porto Seguro - Driver Claim Risk / Sigorta Talebi Riski") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            gurultu = gr.Slider(0, 1, value=0.2, label="Profile deviation / Profil sapmasi (noise from median)")
            tohum = gr.Slider(0, 999, value=42, step=1, label="Random seed / Rastgele tohum")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [gurultu, tohum], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_porto_seguro.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)")


if __name__ == "__main__":
    demo.launch()
