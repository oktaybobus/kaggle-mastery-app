"""Store Sales - Gradio demosu (iki dilli: EN + TR).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_store_sales.pkl"))
except Exception:
    pass


def sonuc_notu():
    """Bu yarismada final tek kronolojik dogrulama (cv_kat=1) ile olculur; eski
    dosya katlamali CV tasiyordu -- iki bicimi de okuyabilmeli."""
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_store_sales.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_store_sales.ipynb` calistirilmali.")
    if metrik.get("cv_kat") == 1:
        return (f"**Final model / Final model:** `{metrik['model']}`  \n"
                f"**{metrik['cv_olcu']}:** `{metrik['cv_skor']:.4f}`")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}):** `{metrik['cv_skor']:.4f}`")


def tahmin(tarih, magaza, aile, promosyon):
    if BUNDLE is None:
        return "Once train_store_sales.ipynb calistirilip model uretilmeli."
    t = pd.Timestamp(tarih)
    # Yeni model dosyasinda gecikme/petrol/hiyerarsi sutunlari da var; demo
    # bunlari bilemez, medyan profilden (defaults) doldurur. Eski dosyada
    # defaults yok — sadece bilinen 8 sutun kalir.
    satir = pd.Series(BUNDLE.get("defaults", {})).reindex(
        BUNDLE.get("columns", ["store_nbr", "family", "onpromotion", "yil", "ay",
                               "gun", "haftanin_gunu", "ayin_basi_sonu"])).fillna(0)
    satir["store_nbr"] = int(magaza)
    satir["family"] = int(BUNDLE["aile_encoder"].transform([aile])[0])
    satir["onpromotion"] = int(promosyon)
    satir["yil"], satir["ay"], satir["gun"] = t.year, t.month, t.day
    satir["haftanin_gunu"] = t.dayofweek
    satir["ayin_basi_sonu"] = int(t.day in (1, 2, 15, 16, 30, 31))
    satis = float(np.expm1(max(BUNDLE["model"].predict(satir.to_frame().T)[0], 0)))
    return f"Daily sales / Tahmini gunluk satis: {satis:,.1f}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Store Sales Forecast / Market Satis Tahmini"
ALTBASLIK = ("Predicts daily sales for a store x product family.  \n"
             "Bir magaza x urun ailesi icin gunluk satis adedini tahmin eder.")

# Bu bir zaman serisi; dogrulama rastgele katlama degil KRONOLOJIK bolme ile.
CV_NOT = ("<sub>**Chronological validation:** on time series, random folds cheat -- using "
          "the future to predict the past is unrealistic. Instead the model trains on "
          "*earlier* days and is tested on *later* days; the reported score is this "
          "forward test, mimicking live use.  \n"
          "**Kronolojik dogrulama:** zaman serisinde rastgele katlama hile olur; model "
          "*once*ki gunlerle egitilip *sonra*ki gunlerde sinanir. Rapordaki skor bu ileri "
          "sinamanin sonucu -- canli kullanimi taklit eder.</sub>")

AMAC = """
### Goal / Amac
Predict **daily unit sales** for each **store x product family** in Ecuador's
Favorita grocery chain. Time-series regression. Kaggle scores it with **RMSLE**
(log-space error) -- proportional across large and small sales.

Ekvador'daki Favorita market zincirinde her **magaza x urun ailesi** icin **gunluk
satis adedini** tahmin etmek. Zaman serisi regresyonu. Olcut **RMSLE** (log uzayinda
hata) -- buyuk ve kucuk satislarda oransal.
"""

NASIL = """
### How it was done / Nasil yapildi
- **1,782 parallel series in one table / Tek tabloda 1.782 seri:** 54 stores x 33
  families learned **together**, with store/family means as features -- one model shares
  strength across sparse series.
  *(Tek model gucu seriler arasinda paylastiriyor.)*
- **Lags, and the 16-day subtlety / Gecikmeler ve 16 gun inceligi:** for the later test
  days the 7- and 14-day lags reach into the unknown, so those cells always use the
  **16-day lag**. Miss this and validation inflates while the leaderboard collapses.
  *(Bu detay atlanirsa skor siser ve leaderboard'da coker; hata mesaji cikmaz.)*
- **Oil forward-filled, never back-filled / Ileri doldurma:** `bfill` would drag future
  prices into the past -- in a time series every imputation choice is a leakage choice.
  *(Zaman serisinde her doldurma karari ayni zamanda bir sizinti karari.)*
- **Scalability as a selection criterion / Olceklenebilirlik secim kriteri:** the race
  winner never finished the 3-million-row final fit; a histogram-based booster finished
  in minutes **and scored better** (~0.50 vs ~0.51).
  *(Yaristaki skor tek kriter degil.)*
"""

ORNEK_KOD = '''\
# 1) Calendar + lag features
df["haftanin_gunu"] = df["date"].dt.dayofweek
df["ayin_basi_sonu"] = df["date"].dt.day.isin([1, 2, 15, 16, 30, 31]).astype(int)
df["satis_lag7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)

# 2) Log target
y = np.log1p(df["sales"])

# 3) Chronological validation: train on past, test on future (NOT random CV)
egit = df[df["date"] < kesim];  dogrula = df[df["date"] >= kesim]
model.fit(egit[kolonlar], np.log1p(egit["sales"]))
'''


with gr.Blocks(title="Store Sales Forecast / Market Satis Tahmini") as demo:
    aileler = BUNDLE["aileler"] if BUNDLE else ["GROCERY I"]
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            tarih = gr.Textbox(value="2017-08-20", label="Date (YYYY-MM-DD) / Tarih")
            magaza = gr.Slider(1, 54, value=1, step=1, label="Store no / Magaza No")
            aile = gr.Dropdown(aileler, value=aileler[0], label="Product family / Urun Ailesi")
            promosyon = gr.Slider(0, 200, value=0, step=1, label="Items on promo / Promosyondaki Urun")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [tarih, magaza, aile, promosyon], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_store_sales.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting)")


if __name__ == "__main__":
    demo.launch()
