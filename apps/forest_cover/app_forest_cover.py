"""Forest Cover Type - Gradio demosu (iki dilli: EN + TR).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_forest_cover.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_forest_cover.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_forest_cover.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


TIPLER = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
          5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}


def tahmin(yukseklik, egim, su_mesafe, yol_mesafe, golge_ogle):
    if BUNDLE is None:
        return "Once train_forest_cover.ipynb calistirilip model uretilmeli."
    satir = pd.Series(BUNDLE["defaults"]).reindex(BUNDLE["columns"]).fillna(0)
    satir["Elevation"] = yukseklik
    satir["Slope"] = egim
    satir["Horizontal_Distance_To_Hydrology"] = su_mesafe
    satir["Horizontal_Distance_To_Roadways"] = yol_mesafe
    satir["Hillshade_Noon"] = golge_ogle
    # Yeni model dosyasinda turetilmis Oklit su uzakligi var; kaydiriciyla
    # tutarli kalsin diye yeniden hesapliyorum (eski dosyada sutun yok).
    if "Su_Uzaklik_Oklit" in satir.index:
        dikey = float(satir.get("Vertical_Distance_To_Hydrology", 0))
        satir["Su_Uzaklik_Oklit"] = float(np.sqrt(su_mesafe ** 2 + dikey ** 2))
    tip = int(BUNDLE["model"].predict(satir.to_frame().T)[0])
    return f"Cover type / Orman ortusu: Type {tip} - {TIPLER.get(tip, '?')}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Forest Cover Type Prediction / Orman Ortusu Tipi Tahmini"
ALTBASLIK = ("Predicts one of 7 forest cover types from terrain measurements.  \n"
             "Arazi olculerinden 7 orman ortusu tipinden birini tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Classify each 30x30 m patch of Colorado's Roosevelt National Forest into one of
**7 cover types** from terrain and soil data. Multi-class classification, scored
by **accuracy**.

Colorado Roosevelt Ulusal Ormani'ndaki 30x30 m parseller icin **7 orman ortusu
tipinden hangisi** oldugunu arazi ve toprak verisinden tahmin etmek. Cok sinifli
siniflandirma; olcut **dogruluk (accuracy)**.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Cyclical aspect / Dairesel baki:** `Aspect` is an angle -- 359 and 1 are neighbours,
  yet numerically 358 apart. A sine/cosine pair fixes the representation.
  *(359 derece ile 1 derece komsu; sin/cos cifti bunu modele dogru anlatiyor.)*
- **Geo feature / Cografi ozellik:** true Euclidean distance to water is computed from
  its horizontal and vertical parts -- trees can only approximate a square root.
  *(Agaclar karekoku ancak cok sayida bolmeyle yaklasik yapabilir.)*
- **Tree ensemble won / Agac toplulugu kazandi:** the class boundaries are axis-parallel
  thresholds ("above 2,500 m"), exactly what trees learn natively; Random Forest reached
  ~0.86 macro-F1.
  *(Sinirlar esik kurallari seklinde; agaclarin dogal ogrendigi bicim.)*
- **A caveat worth knowing / Bilinmesi gereken uyari:** training is balanced **by
  design** while the test set follows the real forest distribution -- validation above
  the leaderboard is expected, not a bug.
  *(Egitim dengeli, test degil; iki skor dogrudan karsilastirilamaz.)*
"""

ORNEK_KOD = '''\
# 1) True (Euclidean) distance to water: combine horizontal + vertical legs
df["Su_Uzaklik_Oklit"] = np.sqrt(
    df["Horizontal_Distance_To_Hydrology"] ** 2
    + df["Vertical_Distance_To_Hydrology"] ** 2)

# 2) Race the tree ensemble under the same CV
skor = cross_val_score(
    RandomForestClassifier(n_estimators=300, n_jobs=-1),
    X, y, cv=5, scoring="accuracy").mean()
'''


with gr.Blocks(title="Forest Cover Type Prediction / Orman Ortusu Tipi Tahmini") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            yukseklik = gr.Slider(1800, 4000, value=2800, label="Elevation (m) / Yukseklik")
            egim = gr.Slider(0, 60, value=15, label="Slope (deg) / Egim")
            su_mesafe = gr.Slider(0, 1400, value=250, label="Horiz. dist. to water (m) / Suya Yatay Mesafe")
            yol_mesafe = gr.Slider(0, 7000, value=2000, label="Horiz. dist. to road (m) / Yola Yatay Mesafe")
            golge_ogle = gr.Slider(0, 255, value=220, label="Noon hillshade / Ogle Golgelenmesi")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [yukseklik, egim, su_mesafe, yol_mesafe, golge_ogle], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_forest_cover.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/forest-cover-type-prediction)")


if __name__ == "__main__":
    demo.launch()
