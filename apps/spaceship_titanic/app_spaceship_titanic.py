"""Spaceship Titanic - Gradio demosu (iki dilli: EN + TR).

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
    BUNDLE = joblib.load(os.path.join(KOK, "model_spaceship_titanic.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_spaceship_titanic.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_spaceship_titanic.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


HARCAMA = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def tahmin(gezegen, cryo, guverte, hedef, yas, vip, harcama):
    if BUNDLE is None:
        return "Once train_spaceship_titanic.ipynb calistirilip model uretilmeli."

    # CryoSleep'teki yolcular kabinlerinden cikamiyor: veride uykudaki her
    # yolcunun harcamasi istisnasiz 0. Kullanici ikisini birden girerse modele
    # egitimde hic gormedigi bir satir gitmis olur ve tahmin anlamsizlasir,
    # o yuzden harcamayi sifirlayip durumu ekranda soyluyorum.
    uyari = ""
    if cryo and harcama > 0:
        uyari = ("\n(CryoSleep on: sleeping passengers always spend 0 in the data, "
                 "so spending was reset. / CryoSleep isaretli: harcama sifirlandi.)")
        harcama = 0.0

    satir = pd.Series(0.0, index=BUNDLE["columns"])
    satir["Age"] = yas
    satir["CryoSleep"] = 1 if cryo else 0
    satir["VIP"] = 1 if vip else 0
    # Yeni model dosyasi harcamalari log olcekte ve grup ozellikleriyle bekliyor;
    # sema, GrupBoyu sutununun varligindan anlasiliyor (eski dosya ham olcek).
    yeni_sema = "GrupBoyu" in satir.index
    for kalem in HARCAMA:
        pay = harcama / len(HARCAMA)
        satir[kalem] = np.log1p(pay) if yeni_sema else pay
    satir["ToplamHarcama"] = np.log1p(harcama) if yeni_sema else harcama
    if yeni_sema:
        satir["GrupBoyu"] = 1     # demo tek yolcu kurar: yalniz seyahat varsayimi
        satir["YalnizMi"] = 1
    for anahtar in ("HomePlanet_" + gezegen, "Destination_" + hedef, "Guverte_" + guverte):
        if anahtar in satir.index:
            satir[anahtar] = 1
    olasilik = BUNDLE["model"].predict_proba(satir.to_frame().T)[0, 1]
    return f"Transport probability / Isinlanma olasiligi: %{olasilik * 100:.1f}{uyari}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Spaceship Titanic - Transport Prediction / Isinlanma Tahmini"
ALTBASLIK = ("Predicts whether a passenger was transported to another dimension.  \n"
             "Yolcunun baska bir boyuta isinlanip isinlanmadigini tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Predict **who was transported** to another dimension after a spacetime anomaly --
a sci-fi remake of Titanic. Binary classification, scored by **accuracy**.

Bir carpismada uzay-zaman anomalisine yakalanan yolculardan **kimin baska bir
boyuta isinlandigini** tahmin etmek -- Titanic'in sci-fi yeniden yorumu. Ikili
siniflandirma; olcut **dogruluk (accuracy)**.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Rule as a feature / Mantik kurali ozellige:** sleeping (CryoSleep) passengers always
  spend 0 and are transported far more often -- the signal lives in that relation.
  *(Sinyal tek sutunda degil, uyku ile harcama arasindaki iliskide.)*
- **Spending gaps filled with 0, not the median / Medyan degil 0:** a sleeping
  passenger's spending genuinely does not exist; a median would fabricate it and break
  the relation above.
  *(Doldurma stratejisi istatistiksel kolaylikla degil, alan bilgisiyle secilir.)*
- **Group features / Grup ozellikleri:** the `gggg_pp` ID yields group size and is-alone;
  identifier structure carries no label information, so this is not leakage.
  *(Kimlik yapisi etiket tasimaz; bu yuzden sizinti degil.)*
- **Missingness as its own category / Eksiklik ayri kategori:** dropping incomplete rows
  would cost about a quarter of the data -- and absence itself may be informative.
  *(Eksik satiri atmak egitim verisinin dortte birini gotururdu.)*
"""

ORNEK_KOD = '''\
# 1) CryoSleep rule: a sleeping passenger spends 0
uyku = df["CryoSleep"] == 1
df.loc[uyku, harcama_kalemleri] = 0

# 2) Log-scale the spends + total
for kalem in harcama_kalemleri:
    df[kalem] = np.log1p(df[kalem])
df["ToplamHarcama"] = np.log1p(df[harcama_kalemleri].sum(axis=1))

# 3) Group features from the ID
df["Grup"] = df["PassengerId"].str.split("_").str[0]
df["GrupBoyu"] = df.groupby("Grup")["Grup"].transform("size")
'''


with gr.Blocks(title="Spaceship Titanic - Transport Prediction / Isinlanma Tahmini") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            gezegen = gr.Dropdown(["Earth", "Europa", "Mars"], value="Earth", label="Home Planet / Ana Gezegen")
            cryo = gr.Checkbox(label="CryoSleep (asleep?) / (uykuda mi?)")
            guverte = gr.Dropdown(list("ABCDEFGT"), value="F", label="Cabin Deck / Kabin Guvertesi")
            hedef = gr.Dropdown(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"],
                                value="TRAPPIST-1e", label="Destination / Hedef")
            yas = gr.Slider(0, 80, value=27, label="Age / Yas")
            vip = gr.Checkbox(label="VIP")
            harcama = gr.Slider(0, 10000, value=700, label="Total spend / Toplam Harcama")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=3)
    calistir.click(tahmin, [gezegen, cryo, guverte, hedef, yas, vip, harcama], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_spaceship_titanic.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/spaceship-titanic)")


if __name__ == "__main__":
    demo.launch()
