"""House Prices - Gradio demosu (iki dilli: EN + TR).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_house_prices.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_house_prices.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_house_prices.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}, log price) / "
            f"{metrik['cv_kat']}-katlama CV (log fiyat):** `{metrik['cv_skor']:.4f}`")


def tahmin(kalite, alan, yil, bodrum, garaj, banyo):
    if BUNDLE is None:
        return "Once train_house_prices.ipynb calistirilip model uretilmeli."
    satir = pd.Series(BUNDLE["defaults"]).reindex(BUNDLE["columns"]).fillna(0)
    satir["OverallQual"] = kalite
    satir["GrLivArea"] = alan
    satir["YearBuilt"] = yil
    satir["TotalBsmtSF"] = bodrum
    satir["GarageCars"] = garaj
    satir["FullBath"] = banyo
    # Yeni model dosyasindaki bilesik alan ozelligi; kaydiricilarla tutarli kalsin.
    if "TotalSF" in satir.index:
        kat1 = float(satir.get("1stFlrSF", 0))
        kat2 = float(satir.get("2ndFlrSF", 0))
        satir["TotalSF"] = bodrum + kat1 + kat2
    girdi = satir.to_frame().T
    if "model_ridge" in BUNDLE:
        # Harman kazandiysa tahmin iki ayagin log uzayindaki ortalamasi.
        log_fiyat = (BUNDLE["model_ridge"].predict(girdi)[0]
                     + BUNDLE["model_agac"].predict(girdi)[0]) / 2
    else:
        log_fiyat = BUNDLE["model"].predict(girdi)[0]
    fiyat = float(np.expm1(log_fiyat))
    return f"Estimated sale price / Tahmini satis fiyati: {fiyat:,.0f} $"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# House Prices - Sale Price Prediction / Ev Fiyati Tahmini"
ALTBASLIK = ("Estimates a house's sale price from a few core features; unset ones "
             "use the training median.  \n"
             "Birkac temel ozellikten evin satis fiyatini tahmin eder; girilmeyenler "
             "egitim medyanindan gelir.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Predict the **sale price** of homes in Ames, Iowa from 79 features. A classic
regression contest. Ranked by **RMSE on the log price**, so errors on cheap and
expensive houses count proportionally the same.

Iowa, Ames'teki evlerin **satis fiyatini** 79 ozellikten tahmin etmek. Klasik
regresyon yarismasi. Siralama **log fiyat uzerinde RMSE** ile -- pahali ve ucuz
evlerdeki hatalar oransal olarak esit.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Log target / Log hedef:** price is right-skewed; `log1p` makes the error
  proportional, exactly what Kaggle's log-RMSE metric asks for. Predicted back with
  `expm1`.
  *(Metrik zaten log-RMSE; model metrikle hizalandi.)*
- **Missingness read as meaning / Eksiklik bilgi olarak okundu:** an empty `PoolQC`
  means **there is no pool**, so it gets an explicit "none" level instead of a mode fill.
  *(PoolQC bossa havuz yok demek; modla doldurmak sinyali yok ederdi.)*
- **Imputation moved into the Pipeline / Doldurma Pipeline'a tasindi:** `SimpleImputer`
  learns from the fitted training fold only -- the earlier train+test median was a
  small but real leak.
  *(Her katlama yalnizca kendi egitim payinin medyanini goruyor.)*
- **Blend / Harman:** Ridge spreads small contributions across ~300 one-hot columns, the
  tree captures thresholds; the log-space average (~0.12) beat either alone (~0.13).
  *(Hatalari iliskisiz iki modelin ortalamasi ikisini de gecti.)*
"""

ORNEK_KOD = '''\
# 1) Log target (aligns with RMSE-of-log)
y = np.log1p(train["SalePrice"])

# 2) Composite area feature
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 3) Ridge + tree blend, back-transformed to dollars
log_fiyat = (ridge.predict(X) + agac.predict(X)) / 2
fiyat = np.expm1(log_fiyat)
'''


with gr.Blocks(title="House Prices - Sale Price Prediction / Ev Fiyati Tahmini") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            kalite = gr.Slider(1, 10, value=6, step=1, label="Overall Quality / Genel Kalite (OverallQual)")
            alan = gr.Slider(400, 5000, value=1500, label="Living Area (sqft) / Yasam Alani (GrLivArea)")
            yil = gr.Slider(1880, 2010, value=1990, step=1, label="Year Built / Yapim Yili")
            bodrum = gr.Slider(0, 3000, value=1000, label="Basement Area (sqft) / Bodrum Alani")
            garaj = gr.Slider(0, 4, value=2, step=1, label="Garage Cars / Garaj Kapasitesi")
            banyo = gr.Slider(0, 4, value=2, step=1, label="Full Baths / Banyo Sayisi")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [kalite, alan, yil, bodrum, garaj, banyo], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_house_prices.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)")


if __name__ == "__main__":
    demo.launch()
