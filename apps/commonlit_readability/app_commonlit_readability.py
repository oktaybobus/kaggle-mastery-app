"""CommonLit Readability - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, sonucu, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os
import re

import joblib
import pandas as pd
from scipy.sparse import hstack
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_commonlit_readability.pkl"))
except Exception:
    pass


def sonuc_notu():
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ("> Run `train_commonlit_readability.ipynb` first to produce the score.  \n"
                "> Skor icin once `train_commonlit_readability.ipynb` calistirilmali.")
    return (f"**Winner model / Kazanan model:** `{metrik['model']}`  \n"
            f"**{metrik['cv_kat']}-fold CV ({metrik['cv_olcu']}) / "
            f"{metrik['cv_kat']}-katlama CV:** `{metrik['cv_skor']:.4f}`")


# Notebook'un 3. bolumundeki dil ozelliklerinin birebir kopyasi; sik kelime
# listesi model dosyasindan geliyor (egitim derleminden cikarildi).
SIK_KELIMELER = (BUNDLE or {}).get("sik_kelimeler") or set()


def hece_sayisi(kelime):
    kelime = kelime.lower()
    sayi = len(re.findall(r"[aeiouy]+", kelime))
    if kelime.endswith("e") and not kelime.endswith(("le", "ee", "ye")) and sayi > 1:
        sayi -= 1
    return max(sayi, 1)


def metin_heceleri(metin):
    return sum(hece_sayisi(k) for k in re.findall(r"[A-Za-z]+", metin))


def nadir_kelime_orani(metin):
    kelimeler = re.findall(r"[a-z]+", metin.lower())
    if not kelimeler or not SIK_KELIMELER:
        return 0.0
    return sum(k not in SIK_KELIMELER for k in kelimeler) / len(kelimeler)


def dil_ozellikleri(metinler):
    ozellik = pd.DataFrame()
    ozellik["karakter"] = metinler.str.len()
    ozellik["kelime"] = metinler.str.split().str.len()
    ozellik["cumle"] = metinler.str.count(r"[.!?]") + 1
    ozellik["ort_kelime_uzunlugu"] = ozellik["karakter"] / ozellik["kelime"]
    ozellik["ort_cumle_uzunlugu"] = ozellik["kelime"] / ozellik["cumle"]
    ozellik["ort_hece"] = metinler.map(metin_heceleri) / ozellik["kelime"]
    ozellik["flesch_kincaid"] = (0.39 * ozellik["ort_cumle_uzunlugu"]
                                 + 11.8 * ozellik["ort_hece"] - 15.59)
    ozellik["nadir_oran"] = metinler.map(nadir_kelime_orani)
    return ozellik


def tahmin(metin):
    if BUNDLE is None:
        return "Once train_commonlit_readability.ipynb calistirilip model uretilmeli."
    seri = pd.Series([metin])
    # Eski model dosyasi 5 ozellikle egitilmis olabilir: kolon listesi ve
    # (varsa) olcekleyici bundle'dan geliyor, boylece iki surumle de calisir.
    dil = dil_ozellikleri(seri)[BUNDLE.get("dil_kolonlari", ["karakter", "kelime", "cumle",
                                                            "ort_kelime_uzunlugu",
                                                            "ort_cumle_uzunlugu"])]
    dil_degerler = dil.values
    if BUNDLE.get("dil_scaler") is not None:
        dil_degerler = BUNDLE["dil_scaler"].transform(dil)
    X = hstack([BUNDLE["vectorizer"].transform(seri), dil_degerler])
    puan = float(BUNDLE["model"].predict(X)[0])
    if puan > -0.5:
        yorum = "easy to read / kolay okunur"
    elif puan > -1.8:
        yorum = "medium / orta zorlukta"
    else:
        yorum = "hard / zor bir metin"
    return f"Readability score / Okunabilirlik puani: {puan:.2f} ({yorum}; lower = harder)"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

BASLIK = "# Text Readability Score / Metin Okunabilirlik Puani"
ALTBASLIK = ("Scores an English reading passage; lower score = harder text.  \n"
             "Ingilizce bir okuma parcasina puan verir; dusuk puan = daha zor metin.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- expected performance on unseen "
          "data.  \n**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 "
          "parcayla ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV "
          "skoru** bu sinamalarin ortalamasi -- modelin *hic gormedigi* veride basarisi.</sub>")

AMAC = """
### Goal / Amac
Give a **readability score** to grade 3-12 classroom texts: low score = hard,
high score = easy. A regression task. Kaggle ranks by **RMSE** (average deviation
from the true score).

3-12. sinif ders metinlerine bir **okunabilirlik puani** vermek: dusuk = zor,
yuksek = kolay. Regresyon problemi. Kaggle sirasi **RMSE** ile (tahminin gercek
puandan ortalama sapmasi).
"""

NASIL = """
### How it was done / Nasil yapildi
- **Two sources, not one / Iki kaynak:** linguistic features measure the text's **form**
  (Flesch-Kincaid and friends), TF-IDF its **content**; stacked, they beat either alone.
  *(Dil ozellikleri bicimi, TF-IDF icerigi olcuyor; biri digerinin yerini tutmuyor.)*
- **Scaling BEFORE stacking / Yapistirmadan ONCE olcekleme:** raw counts next to 0-1
  TF-IDF weights would cripple the linear family in the race.
  *(Olcekleme olmadan Ridge cezali terimde ezilir ve yarisa sakat girer.)*
- **The baseline is the reference / Referans taban cizgisi:** predicting the mean scores
  the target's standard deviation (~1.01); the real result is ~30% below that floor.
  *(Skorun mutlak degeri degil, tabana olan mesafesi anlamli.)*
- **Ridge won, then meaning won / Once Ridge, sonra anlam:** on 20,000 sparse columns
  Ridge beat the trees (~0.71); fine-tuned **RoBERTa** reached **~0.61** by reading
  meaning instead of counting words.
  *(RoBERTa kelime saymak yerine anlami okuyarak sinari asti.)*
"""

ORNEK_KOD = '''\
# 1) Classic readability feature: Flesch-Kincaid
fk = 0.39 * (kelime / cumle) + 11.8 * (hece / kelime) - 15.59

# 2) Combine TF-IDF (text) + numeric language features
from scipy.sparse import hstack
X = hstack([tfidf.transform(metinler), scaler.transform(dil_ozellik)])

# 3) Ridge regression, CV with RMSE
skor = -cross_val_score(Ridge(alpha=1.0), X, y, cv=5,
                        scoring="neg_root_mean_squared_error").mean()
'''


with gr.Blocks(title="Text Readability Score / Metin Okunabilirlik Puani") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            metin = gr.Textbox(lines=6, label="English reading passage / Ingilizce okuma parcasi")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [metin], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_commonlit_readability.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/commonlitreadabilityprize)")


if __name__ == "__main__":
    demo.launch()
