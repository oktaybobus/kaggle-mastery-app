"""Bike Sharing Demand - Gradio demosu (iki dilli: EN + TR).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_bike_sharing.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_bike_sharing.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_bike_sharing.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}, log space) / "
            f"{metrik['cv_kat']}-katlama CV (log uzayinda):** `{metrik['cv_skor']:.4f}`")


def tahmin(saat, gun, ay, sicaklik, nem, ruzgar, hava, is_gunu):
    if BUNDLE is None:
        return "Once train_bike_sharing.ipynb calistirilip model uretilmeli."
    mevsim = {12: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1,
              6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[int(ay)]
    ozellikler = {
        "saat": int(saat), "haftanin_gunu": int(gun), "ay": int(ay), "yil": 2012,
        "season": mevsim, "holiday": 0, "workingday": int(is_gunu),
        "weather": int(hava), "temp": sicaklik, "atemp": sicaklik,
        "humidity": nem, "windspeed": ruzgar,
    }
    # Notebook'un 3. bolumundeki turetilmis ozellikler; eski model dosyasinda
    # bu sutunlar yoksa columns listesi onlari zaten dislar.
    ozellikler["saat_sin"] = np.sin(2 * np.pi * int(saat) / 24)
    ozellikler["saat_cos"] = np.cos(2 * np.pi * int(saat) / 24)
    ozellikler["saat_x_isgunu"] = int(saat) * int(is_gunu)
    satir = pd.DataFrame([ozellikler])[BUNDLE["columns"]]
    if "model_casual" in BUNDLE:
        # Ayrik modelleme kazandiysa tahmin iki modelin adet toplami.
        talep = float(np.expm1(max(BUNDLE["model_casual"].predict(satir)[0], 0))
                      + np.expm1(max(BUNDLE["model_registered"].predict(satir)[0], 0)))
    else:
        talep = float(np.expm1(max(BUNDLE["model"].predict(satir)[0], 0)))
    return f"Hourly rentals / Tahmini saatlik kiralama: {talep:,.0f}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Hourly Bike Demand / Saatlik Bisiklet Talebi"
ALTBASLIK = ("Predicts hourly rental count from date and weather.  \n"
             "Tarih ve hava durumundan bir saatlik kiralama adedini tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Predict **how many bikes** will be rented in a given hour of the Washington D.C.
bike-share system. A regression task driven by date, hour and weather. Kaggle
scores it with **RMSLE** -- proportional error, so mistakes on large counts are
not punished more than on small ones.

Washington D.C. bisiklet paylasim sisteminde bir saat icin **kac bisikletin
kiralanacagini** tahmin etmek. Girdiler tarih, saat, hava. Kaggle **RMSLE** ile
olcer -- oransal hata, buyuk taleplerdeki mutlak hata degil.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Chronological split / Kronolojik bolme:** a time series trains on the past and is
  tested on the future (`shuffle=False`, `TimeSeriesSplit`); a random split leaks
  tomorrow into training.
  *(Rastgele bolme gelecegi gecmise sizdirir; skor tavan yapar, leaderboard'da coker.)*
- **Cyclical time / Zaman dongusu:** hour is circular (23 is next to 0), so it is encoded
  with `sin`/`cos` plus an **hour x working-day** interaction.
  *(Saat dairesel; sin/cos cifti bunu modele dogru anlatiyor.)*
- **Log target / Log hedef:** demand is right-skewed; trained on `log1p(count)`,
  predicted back with `expm1` (matches RMSLE).
  *(Talep carpik; log1p uzerinde egitilip expm1 ile geri cevrildi.)*
- **Split modelling / Ayrik modelleme:** `casual` and `registered` behave differently;
  modelled separately and summed **in count space**. It won by a whisker (~0.325 either
  way) and that thin margin is reported, not hidden.
  *(Fark binde birlikti ve bu gizlenmedi.)*
"""

ORNEK_KOD = '''\
# 1) Cyclical hour + working-day interaction
df["saat_sin"] = np.sin(2 * np.pi * df["saat"] / 24)
df["saat_cos"] = np.cos(2 * np.pi * df["saat"] / 24)
df["saat_x_isgunu"] = df["saat"] * df["workingday"]

# 2) Model casual and registered separately, in log space
model_casual.fit(X, np.log1p(df["casual"]))
model_registered.fit(X, np.log1p(df["registered"]))

# 3) Prediction = sum of the two back-transformed legs
talep = np.expm1(model_casual.predict(X)) + np.expm1(model_registered.predict(X))
'''


with gr.Blocks(title="Hourly Bike Demand / Saatlik Bisiklet Talebi") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            saat = gr.Slider(0, 23, value=8, step=1, label="Hour / Saat")
            gun = gr.Slider(0, 6, value=1, step=1, label="Weekday / Haftanin Gunu (0=Mon)")
            ay = gr.Slider(1, 12, value=6, step=1, label="Month / Ay")
            sicaklik = gr.Slider(-5, 40, value=22, label="Temperature (C) / Sicaklik")
            nem = gr.Slider(0, 100, value=55, label="Humidity (%) / Nem")
            ruzgar = gr.Slider(0, 50, value=10, label="Wind speed / Ruzgar Hizi")
            hava = gr.Dropdown([1, 2, 3, 4], value=1, label="Weather / Hava (1=clear ... 4=storm)")
            is_gunu = gr.Checkbox(value=True, label="Working day? / Is gunu mu?")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [saat, gun, ay, sicaklik, nem, ruzgar, hava, is_gunu], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_bike_sharing.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/bike-sharing-demand)")


if __name__ == "__main__":
    demo.launch()
