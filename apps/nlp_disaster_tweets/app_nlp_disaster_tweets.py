"""Disaster Tweets - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

import re

import joblib
import numpy as np
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

BUNDLE = None
try:
    BUNDLE = joblib.load(os.path.join(KOK, "model_nlp_disaster_tweets.pkl"))
except Exception:
    pass


def metrik_notu():
    """Notebook'un modelle birlikte kaydettiği skoru tek satırlık not olarak ver.

    Model dosyası eski sürümse ya da notebook henüz çalıştırılmadıysa boş
    döner; demo bu yüzden metrik olmadan da açılır.
    """
    metrik = (BUNDLE or {}).get("metrikler") or {}
    if metrik.get("cv_skor") is None:
        return ""
    return (f"\n\nKazanan model: {metrik['model']} - "
            f"{metrik['cv_kat']}-katlama CV {metrik['cv_olcu']}: {metrik['cv_skor']:.4f}")


def temizle(metin):
    metin = str(metin).lower()
    metin = re.sub(r"https?://\S+", " ", metin)
    metin = re.sub(r"[^a-z#@\s]", " ", metin)
    return re.sub(r"\s+", " ", metin).strip()


# Notebook'un 3.1'de sectigi temizleme varyanti bundle'dan geliyor.
try:
    from nltk.stem import PorterStemmer
    _govdeleyici = PorterStemmer()
except Exception:
    _govdeleyici = None

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:
    ENGLISH_STOP_WORDS = frozenset()


def varyanti_uygula(metin):
    ayar = (BUNDLE or {}).get("temizleme") or {}
    if ayar.get("durak"):
        metin = " ".join(k for k in metin.split() if k not in ENGLISH_STOP_WORDS)
    if ayar.get("kok") and _govdeleyici is not None:
        metin = " ".join(k if k.startswith(("#", "@")) else _govdeleyici.stem(k)
                         for k in metin.split())
    return metin


def tahmin(tweet):
    if BUNDLE is None:
        return "Once train_nlp_disaster_tweets.ipynb calistirilip model uretilmeli."
    metin = varyanti_uygula(temizle(tweet))
    vec = BUNDLE["vectorizer"].transform([metin])
    # Yeni model dosyasi kelime + karakter TF-IDF + keyword blogu bekliyor;
    # demoda keyword bilinmedigi icin 'yok' seviyesi isaretlenir.
    if "vectorizer_karakter" in BUNDLE:
        from scipy.sparse import hstack, csr_matrix
        parcalar = [vec, BUNDLE["vectorizer_karakter"].transform([metin])]
        anahtarlar = list(BUNDLE.get("anahtar_kolonlari") or [])
        if anahtarlar:
            anahtar_vek = np.zeros((1, len(anahtarlar)))
            if "yok" in anahtarlar:
                anahtar_vek[0, anahtarlar.index("yok")] = 1.0
            parcalar.append(csr_matrix(anahtar_vek))
        vec = hstack(parcalar).tocsr()
    olasilik = BUNDLE["model"].predict_proba(vec)[0, 1]
    etiket = "GERCEK AFET / REAL DISASTER" if olasilik > 0.5 else "normal / mecazi"
    return f"Disaster probability / Afet olasiligi: %{olasilik * 100:.1f} - {etiket}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

def sonuc_notu():
    """Notebook'un kaydettigi metrikleri goster; yoksa nazikce uyar."""
    if BUNDLE is None:
        return ("> Run `train_nlp_disaster_tweets.ipynb` first to produce the model.  \n"
                "> Model icin once `train_nlp_disaster_tweets.ipynb` calistirilmali.")
    not_ = metrik_notu().strip()
    if not not_:
        return ("**Model:** TF-IDF (word + character) with a classic classifier.  \n"
                "Exact F1 is reported in the notebook output. / "
                "Kesin F1 skoru notebook ciktisinda.")
    return ("**Metric / Olcut:** F1 (positive class / pozitif sinif)  \n" + not_)


BASLIK = "# Disaster Tweets / Afet Tweeti Tespiti"
ALTBASLIK = ("Decides whether a tweet reports a real disaster or speaks figuratively.  \n"
             "Bir tweetin gercek bir afeti mi haber verdigini yoksa mecazi mi "
             "konustugunu belirler.")

CV_NOT = ("<sub>**CV (cross-validation):** the training data is split into 5 parts; the "
          "model learns on 4 and is tested on the held-out 1, repeated 5 times. The "
          "**CV score** is the average of those 5 tests -- the model's expected "
          "performance on data it has *never* seen (real learning, not memorization).  \n"
          "**CV (capraz dogrulama):** egitim verisi 5 parcaya bolunur; model 4 parcayla "
          "ogrenip disarida biraktigi 1 parcada sinanir, bu 5 kez doner. **CV skoru** bu "
          "5 sinamanin ortalamasi -- modelin *hic gormedigi* veride beklenen basarisi.</sub>")

AMAC = """
### Goal / Amac
Decide whether a tweet **reports a real disaster** or merely **speaks figuratively** --
"this exam is a disaster" is not an emergency. Real-time filtering of this kind is what
lets disaster-response teams watch social media without drowning in noise. Binary text
classification, scored by **F1** on the positive class, so both catching real events
(recall) and avoiding false alarms (precision) count.

Bir tweetin **gercek bir afeti mi haber verdigini** yoksa **mecazi mi konustugunu**
belirlemek -- "bu sinav tam bir felaket" bir acil durum degildir. Bu tur bir gercek
zamanli filtreleme, afet ekiplerinin sosyal medyayi gurultude bogulmadan izlemesini
saglar. Ikili metin siniflandirmasi; olcut **F1**, yani hem yakalama (recall) hem
isabet (precision) birlikte onemli.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Hashtags and mentions are kept / Hashtag ve mention korunur:** links are stripped but
  `#` and `@` survive -- `#earthquake` is a tagged topic, not just a word.
  *(Baglantilar atilir ama `#` ve `@` kalir: `#earthquake` bir etikettir, kelime degil.)*
- **Word + character TF-IDF / Kelime + karakter TF-IDF:** Twitter is full of misspellings
  (`earthquak`, `fiiire`); character fragments catch the shared core.
  *(Karakter parcalari yazim hatalarina dayaniklilik saglar.)*
- **Bigrams / Ikili n-gram:** `not fire` and `fire` are different things, which single
  words cannot express.
  *(Tek kelime olumsuzlamayi goremez.)*
- **Output / Cikti:** a probability of "real disaster"; the competition wants a hard 0/1
  label, so 0.5 is the threshold.
  *(Yarisma kesin etiket ister; esik 0.5.)*
"""

ORNEK_KOD = '''\
# Cleaning: strip links, keep hashtags and mentions
def temizle(metin):
    metin = metin.lower()
    metin = re.sub(r"https?://\\S+", " ", metin)      # links are unique noise
    metin = re.sub(r"[^a-z#@\\s]", " ", metin)        # keep # and @
    return re.sub(r"\\s+", " ", metin).strip()

# Word TF-IDF: bigrams so "not fire" differs from "fire"
kelime = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)

# Character TF-IDF: robust to Twitter misspellings (earthquak / fiiire)
karakter = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                           max_features=20000, min_df=2)

# Stack both, plus a one-hot of the keyword column
X = hstack([kelime.fit_transform(metinler),
            karakter.fit_transform(metinler),
            anahtar_onehot]).tocsr()

model = BernoulliNB()          # presence/absence fits 280-character texts
model.fit(X, y)
'''


with gr.Blocks(title="Disaster Tweets / Afet Tweeti Tespiti") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(CV_NOT)
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            tweet = gr.Textbox(lines=3, label="Tweet",
                               placeholder="Forest fire near La Ronge Sask. Canada")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [tweet], cikti)

    gr.Examples(
        examples=[["Forest fire near La Ronge Sask. Canada"],
                  ["this exam is a disaster, I am ablaze with stress"],
                  ["Just got sent this photo from Ruby #Alaska as smoke from wildfires"],
                  ["My new playlist is fire, whole room is ablaze"]],
        inputs=[tweet],
        label="Examples / Ornekler",
    )

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: "
                "`train_nlp_disaster_tweets.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/nlp-getting-started)")


if __name__ == "__main__":
    demo.launch()
