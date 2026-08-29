"""Titanic hayatta kalma tahmini - Gradio demosu (iki dilli: EN + TR).

Sayfa artik cizik bir test kutusu degil: amaci, sonucu, nasil yapildigini ve
ornek kodu ayni ekranda EN + TR anlatan bir vaka calismasi. Interaktif "Try it /
Dene" bolumu ortada durur.
"""

import os

import joblib
import pandas as pd
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_titanic.pkl"))
except Exception:
    pass


def sonuc_notu():
    """Notebook'un modelle kaydettigi ic dogrulama skorunu iki dilli satira cevir."""
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_titanic.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_titanic.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama capraz dogrulama:** `{metrik['cv_skor']:.4f}`")


def tahmin(pclass, cinsiyet, yas, ucret, sibsp, parch, liman):
    if BUNDLE is None:
        return "Once train_titanic.ipynb calistirilip model_titanic.pkl uretilmeli."
    satir = pd.Series(0.0, index=BUNDLE["columns"])
    satir["Sex"] = 1 if cinsiyet == "Kadin" else 0
    satir["Age"] = yas
    satir["Fare"] = ucret
    satir["SibSp"] = sibsp
    satir["Parch"] = parch
    satir["AileBoyu"] = sibsp + parch + 1
    satir["Yalniz"] = 1 if satir["AileBoyu"] == 1 else 0
    # Yeni model dosyasindaki unvan ve bilet grubu ozellikleri (eski dosyada
    # bu sutunlar yok, guardlar sayesinde iki surumle de calisir). Unvan demoda
    # cinsiyet + yastan turetiliyor: 13 yasindan kucuk erkek = Master.
    if cinsiyet == "Kadin":
        unvan = "Miss" if yas < 18 else "Mrs"
    else:
        unvan = "Master" if yas < 13 else "Mr"
    if "Unvan_" + unvan in satir.index:
        satir["Unvan_" + unvan] = 1
    if "BiletGrupBoyu" in satir.index:
        grup = sibsp + parch + 1          # bilet bilinmiyor: aile boyu varsayimi
        satir["BiletGrupBoyu"] = grup
        satir["KisiBasiUcret"] = ucret / max(grup, 1)
    for anahtar in ("Pclass_" + str(pclass), "Embarked_" + liman):
        if anahtar in satir.index:
            satir[anahtar] = 1
    olasilik = BUNDLE["model"].predict_proba(satir.to_frame().T)[0, 1]
    return f"Survival probability / Hayatta kalma olasiligi: %{olasilik * 100:.1f}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Titanic - Survival Prediction / Hayatta Kalma Tahmini"
ALTBASLIK = ("Predicts a passenger's survival probability from their profile; the "
             "notebook's race picks the model.  \n"
             "Yolcu profilinden hayatta kalma olasiligini tahmin eder; modeli "
             "notebook'taki yaris secer.")

# Tum demolarda ayni: "CV skoru" ne demek, okuyan bilmeyebilir diye kisa not (EN + TR).
CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- the model's expected "
          "performance on data it has *never* seen (real learning, not memorization).  \n"
          "**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 parcayla "
          "ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV skoru** bu "
          "5 sinamanin ortalamasi -- modelin *hic gormedigi* veride beklenen basarisi.</sub>")

AMAC = """
### Goal / Amac
Predict **who survived** the 1912 Titanic disaster from the passenger manifest.
A binary classification task; Kaggle's classic starter competition. Scored by
**accuracy** -- the share of test passengers labelled correctly.

1912'de batan Titanic'in yolcu manifestosundan **kimin hayatta kaldigini** tahmin
etmek. Ikili siniflandirma; Kaggle'in klasik giris yarismasi. Basari olcusu
**dogruluk (accuracy)** -- test yolcularinin yuzde kacini dogru etiketledigin.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Feature engineering / Ozellik uretimi:** title from the name, family size, is-alone,
  ticket group size and fare per person.
  *(Grup bileti toplam yazildigi icin kisi basi ucret gercek bilet seviyesini olcuyor.)*
- **Why the title matters / Unvan neden onemli:** `Master` meant a **boy**; the overall
  median age (28) would turn him into an adult, the title median (~5) keeps the child
  signal.
  *(Master erkek cocuk demek; genel medyan onu yetiskin yapardi.)*
- **Imputation statistics from training ONLY / Doldurma yalnizca egitimden:** the same
  training-set numbers fill the test set; per-set filling shifts the distribution and
  corrupts the score without raising an error.
  *(Her set kendi medyaniyla doldurulsa fark sessizce skora yansirdi.)*
- **Model race / Model yarisi:** several models compete under the same split;
  scale-sensitive ones get their own `StandardScaler`. Logistic regression won at ~0.83
  CV accuracy.
  *(Olcege duyarli modeller kendi StandardScaler'iyla sarildi.)*
"""

ORNEK_KOD = '''\
# 1) Title + family features from the raw columns
df["Unvan"] = df["Name"].str.extract(r",\\s*([^\\.]+)\\.")
df["AileBoyu"] = df["SibSp"] + df["Parch"] + 1
df["Yalniz"] = (df["AileBoyu"] == 1).astype(int)

# 2) Impute + scale + model in one Pipeline (prevents leakage)
pipe = Pipeline([
    ("prep", on_isleme),             # impute + one-hot + scale
    ("model", LogisticRegression(max_iter=1000)),
])

# 3) Race the candidates under the same CV, keep the best
for ad, aday in adaylar.items():
    skor = cross_val_score(aday, X, y, cv=5, scoring="accuracy").mean()
    print(f"{ad}: {skor:.4f}")
'''


with gr.Blocks(title="Titanic - Survival Prediction / Hayatta Kalma Tahmini") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            pclass = gr.Dropdown([1, 2, 3], value=3, label="Ticket Class / Bilet Sinifi (Pclass)")
            cinsiyet = gr.Radio(["Kadin", "Erkek"], value="Erkek", label="Sex / Cinsiyet (Kadin=F, Erkek=M)")
            yas = gr.Slider(0, 80, value=30, label="Age / Yas")
            ucret = gr.Slider(0, 300, value=32, label="Fare ($) / Bilet Ucreti")
            sibsp = gr.Slider(0, 8, value=0, step=1, label="Siblings/Spouse / Kardes-Es (SibSp)")
            parch = gr.Slider(0, 6, value=0, step=1, label="Parents/Children / Ebeveyn-Cocuk (Parch)")
            liman = gr.Dropdown(["S", "C", "Q"], value="S", label="Embark Port / Binis Limani")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin,
                   [pclass, cinsiyet, yas, ucret, sibsp, parch, liman], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_titanic.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/titanic)")


if __name__ == "__main__":
    demo.launch()
