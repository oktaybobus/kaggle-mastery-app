"""Ghouls, Goblins & Ghosts - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
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
    BUNDLE = joblib.load(os.path.join(KOK, "model_ghouls_goblins_ghosts.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_ghouls_goblins_ghosts.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_ghouls_goblins_ghosts.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


def tahmin(kemik, curume, sac, ruh, renk):
    if BUNDLE is None:
        return "Once train_ghouls_goblins_ghosts.ipynb calistirilip model uretilmeli."
    satir = pd.Series(0.0, index=BUNDLE["columns"])
    satir["bone_length"] = kemik
    satir["rotting_flesh"] = curume
    satir["hair_length"] = sac
    satir["has_soul"] = ruh
    if "color_" + renk in satir.index:
        satir["color_" + renk] = 1
    # Yeni model dosyasindaki carpim ozellikleri; eski dosyada sutun yoksa atlanir.
    if "sac_x_ruh" in satir.index:
        satir["sac_x_ruh"] = sac * ruh
    if "kemik_x_sac" in satir.index:
        satir["kemik_x_sac"] = kemik * sac
    tur = BUNDLE["model"].predict(satir.to_frame().T)[0]
    return f"Most likely a {tur}! / Buyuk ihtimalle bir {tur}!"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Creature Classifier / Yaratik Siniflandirici"
ALTBASLIK = ("Predicts a creature's type from graveyard measurements.  \n"
             "Mezarliktan toplanan olculere gore yaratik turunu tahmin eder.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Predict whether a creature is a **Ghoul, Goblin or Ghost** from measurements -- a
Halloween-themed multi-class task. The data is tiny (~370 rows), so it overfits
easily. Scored by **accuracy**.

Mezarliktan toplanan olculere gore bir yaratigin **Ghoul, Goblin mu yoksa Ghost
mu** oldugunu tahmin etmek -- Cadilar Bayrami temali cok sinifli gorev. Veri
kucuk (~370 satir), asiri ogrenmeye acik. Olcut **dogruluk (accuracy)**.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Interaction features / Etkilesim ozellikleri:** products of weak lone signals
  (hair x soul) compress a **diagonal** class boundary into one column.
  *(Ayrim kosegen boyunca; carpim onu tek sutuna sikistiriyor.)*
- **Colour: decided by measurement / Renk karari olcumle:** the question went to 5-fold
  CV, not intuition; result **~0.74 vs ~0.74** -- colour does no harm, and an
  inconclusive experiment is still a finding worth reporting.
  *(Sonucsuz cikan bir deney de raporlanmasi gereken bir bulgu.)*
- **Regularization confirmed by the data / Duzenlilestirme veriden dogrulandi:** the
  search chose `min_samples_leaf=4` on its own; the default of 1 memorizes single
  creatures.
  *(Arama freni kendiliginden secti; ihtiyac disaridan dayatilmadi.)*
- **An honest caveat / Durust bir uyari:** Random Forest's one-or-two-point edge over
  logistic regression is indistinguishable from split luck at 371 rows.
  *(Bu olcekte model secimi belirleyici degil.)*
"""

ORNEK_KOD = '''\
# 1) Interaction (product) features -- big lift on small data
df["sac_x_ruh"]   = df["hair_length"] * df["has_soul"]
df["kemik_x_sac"] = df["bone_length"] * df["hair_length"]

# 2) Color one-hot + model race
X = pd.get_dummies(df, columns=["color"])
skor = cross_val_score(RandomForestClassifier(max_depth=5),
                       X, y, cv=5, scoring="accuracy").mean()
'''


with gr.Blocks(title="Creature Classifier / Yaratik Siniflandirici") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            kemik = gr.Slider(0, 1, value=0.4, label="Bone length (norm) / Kemik Uzunlugu")
            curume = gr.Slider(0, 1, value=0.5, label="Rotting flesh / Curume Orani")
            sac = gr.Slider(0, 1, value=0.5, label="Hair length / Sac Uzunlugu")
            ruh = gr.Slider(0, 1, value=0.5, label="Has soul / Ruh Orani")
            renk = gr.Dropdown(["white", "black", "clear", "blue", "green", "blood"],
                               value="white", label="Color / Renk")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [kemik, curume, sac, ruh, renk], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_ghouls_goblins_ghosts.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo)")


if __name__ == "__main__":
    demo.launch()
