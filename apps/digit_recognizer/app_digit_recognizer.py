"""Digit Recognizer - Gradio demosu (iki dilli: EN + TR).

Sayfa amaci, nasil yapildigini ve ornek kodu EN + TR anlatir; interaktif
"Try it / Dene" bolumu ortada durur.
"""

import os

# TensorFlow'u numpy/gradio'dan ÖNCE içe aktarıyorum. macOS'ta ters sırada
# yüklenince tahmin çağrısı kilitleniyor (notebook'taki aynı OpenMP sorunu).
try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np
import gradio as gr

# Model dosyasını bu dosyanın yanından oku: demo hangi dizinden
# çalıştırılırsa çalıştırılsın modeli bulsun (notebook onu buraya yazıyor).
KOK = os.path.dirname(os.path.abspath(__file__))

MODEL = None
if tf is not None:
    try:
        MODEL = tf.keras.models.load_model(os.path.join(KOK, "model_digit_recognizer.keras"))
    except Exception:
        pass


def tahmin(goruntu):
    if MODEL is None:
        return {"model yok - once notebook calistirilmali": 1.0}
    if goruntu is None:
        return {}
    from PIL import Image
    img = Image.fromarray(goruntu).convert("L").resize((28, 28))
    dizi = np.array(img).astype("float32")
    if dizi.mean() > 127:  # beyaz zemin ise MNIST gibi ters cevir
        dizi = 255 - dizi
    dizi = dizi.reshape(1, 28, 28, 1) / 255.0
    prob = MODEL.predict(dizi, verbose=0)[0]
    return {str(i): float(p) for i, p in enumerate(prob)}


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

def sonuc_notu():
    """Keras modelinin ozeti; skor yerine mimari, cunku sayi notebook ciktisinda."""
    if MODEL is None:
        return ("> Run `train_digit_recognizer.ipynb` first to produce the model.  \n"
                "> Model icin once `train_digit_recognizer.ipynb` calistirilmali.")
    try:
        katman = len(MODEL.layers)
        parametre = MODEL.count_params()
        return (f"**Model:** convolutional neural network (CNN) / evrisimli sinir agi  \n"
                f"**Layers / Katman:** {katman} - **params / parametre:** {parametre:,}  \n"
                f"Exact accuracy is reported in the notebook output. / "
                f"Kesin dogruluk skoru notebook ciktisinda.")
    except Exception:
        return "**Model:** convolutional neural network (CNN) / evrisimli sinir agi."


BASLIK = "# Handwritten Digit Recognition / El Yazisi Rakam Tanima"
ALTBASLIK = ("Recognizes a handwritten digit (0-9). Resized to 28x28; white-background "
             "images are auto-inverted.  \n"
             "El yazisi bir rakami (0-9) tanir. 28x28'e kucultulur; beyaz zeminli "
             "goruntuler otomatik ters cevrilir.")

AMAC = """
### Goal / Amac
Correctly classify 28x28 grayscale images of handwritten digits (0-9) -- the famous
**MNIST** dataset. A computer-vision starter contest. Multi-class classification,
scored by **accuracy**.

El yazisi rakamlarin (0-9) 28x28 gri tonlamali goruntulerini dogru siniflandirmak --
unlu **MNIST** veri seti. Bilgisayarli goruye giris yarismasi. Cok sinifli
siniflandirma; olcut **dogruluk (accuracy)**.
"""

NASIL = """
### How it was done / Nasil yapildi
- **CNN:** convolution layers learn local patterns (curves, corners) regardless of where
  they sit in the image.
  *(Evrisim katmanlari yerel desenleri konumdan bagimsiz ogrenir.)*
- **Augmentation WITHOUT flips / Cevirmesiz veri artirma:** rotation, zoom and shift
  preserve a digit's identity; flipping does not (a mirrored 6 becomes a 9).
  *(Cevirme etiketi bozar: ters cevrilmis bir 6, 9'a benzer.)*
- **Preprocessing / On isleme:** pixels scaled to 0-1; white-background drawings are
  auto-inverted to match MNIST.
  *(Piksel 0-1'e; beyaz zeminli goruntuler otomatik ters cevrildi.)*
- **Test-time augmentation / Tahminde veri artirma:** predictions averaged over several
  augmented passes lifted accuracy from ~0.9936 to ~0.9962 -- the notebook's one large
  measurable jump. The demo shows the top 3 digits.
  *(TTA dogrulukta olculebilir tek buyuk sicramayi sagladi; demo ilk 3 rakami gosterir.)*
"""

ORNEK_KOD = '''\
# Augmentation lives inside the model: active in training, bypassed at predict time.
# NOTE: no RandomFlip -- a mirrored 2 is not a 2, and a flipped 6 becomes a 9.
artirma = keras.Sequential([
    layers.RandomRotation(0.08),      # ~+/-29 degrees; more starts turning 6 into 9
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
], name="veri_artirma")

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    artirma,
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),          # 7x7x128 -> 128, not Flatten
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
'''


with gr.Blocks(title="Handwritten Digit Recognition / El Yazisi Rakam Tanima") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            goruntu = gr.Image(label="Handwritten digit / El yazisi rakam (photo or drawing)")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Label(num_top_classes=3, label="Prediction / Tahmin")
    calistir.click(tahmin, [goruntu], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_digit_recognizer.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/digit-recognizer)")


if __name__ == "__main__":
    demo.launch()
