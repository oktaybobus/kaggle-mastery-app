"""Aerial Cactus - Gradio demosu (iki dilli: EN + TR).

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
        MODEL = tf.keras.models.load_model(os.path.join(KOK, "model_aerial_cactus.keras"))
    except Exception:
        pass


def tahmin(goruntu):
    if MODEL is None:
        return "Once train_aerial_cactus.ipynb calistirilip model uretilmeli."
    from PIL import Image
    img = Image.fromarray(goruntu).convert("RGB").resize((32, 32))
    dizi = np.array(img).astype("float32").reshape(1, 32, 32, 3) / 255.0
    olasilik = float(MODEL.predict(dizi, verbose=0)[0, 0])
    etiket = "CACTUS / KAKTUS VAR" if olasilik > 0.5 else "no cactus / kaktus yok"
    return f"Cactus probability / Kaktus olasiligi: %{olasilik * 100:.1f} - {etiket}"


# --- Sayfa metinleri: yarisma bilgisi burada, mantik yukarida ---------------

def sonuc_notu():
    """Keras modelinin ozeti; skor yerine mimari, cunku sayi notebook ciktisinda."""
    if MODEL is None:
        return ("> Run `train_aerial_cactus.ipynb` first to produce the model.  \n"
                "> Model icin once `train_aerial_cactus.ipynb` calistirilmali.")
    try:
        katman = len(MODEL.layers)
        parametre = MODEL.count_params()
        return (f"**Model:** small convolutional neural network (CNN) / kucuk CNN  \n"
                f"**Layers / Katman:** {katman} - **params / parametre:** {parametre:,}  \n"
                f"Exact ROC-AUC is reported in the notebook output. / "
                f"Kesin ROC-AUC skoru notebook ciktisinda.")
    except Exception:
        return "**Model:** small convolutional neural network (CNN) / kucuk CNN."


BASLIK = "# Aerial Cactus Identification / Havadan Kaktus Tespiti"
ALTBASLIK = ("Predicts whether an aerial photo contains a cactus.  \n"
             "Bir hava fotografinda kaktus olup olmadigini tahmin eder.")

AMAC = """
### Goal / Amac
Detect **whether a columnar cactus is present** in a 32x32 aerial photo -- automated
plant tracking for environmental monitoring. Binary image classification. Kaggle
ranks by **ROC-AUC** (ranking quality).

32x32 piksellik hava fotograflarinda **sutun kaktusu var mi yok mu** tespit etmek --
cevre gozlemi icin otomatik bitki takibi. Ikili goruntu siniflandirmasi. Siralama
**ROC-AUC** ile.
"""

NASIL = """
### How it was done / Nasil yapildi
- **Two architectures raced / Iki mimari yaristirildi:** a scratch-built CNN against an
  ImageNet-pretrained **MobileNetV2**; transfer learning won (0.9999 vs 0.9962 ROC-AUC).
  *(Sifirdan CNN ile MobileNetV2 yaristirildi; transfer ogrenme kazandi.)*
- **In-graph resizing / Graf ici boyutlandirma:** a `Resizing(96, 96)` layer lifts the
  32x32 input inside the model -- 17,500 files on disk stay untouched.
  *(Model icindeki `Resizing` katmani 32x32'yi 96x96'ya cikariyor.)*
- **Flips are allowed here / Cevirme burada serbest:** an aerial photo has no canonical
  orientation, so rotation and flipping preserve the label.
  *(Hava fotografinda yon keyfi oldugu icin cevirme etiketi bozmuyor.)*
- **Output / Cikti:** a single sigmoid neuron -> cactus probability, averaged over a
  3-seed ensemble with test-time augmentation.
  *(3 tohumlu topluluk + tahminde veri artirma ortalamasi.)*
"""

ORNEK_KOD = '''\
# Winning architecture: MobileNetV2 transfer learning, adapted to 32x32 input
omurga = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights="imagenet", alpha=0.35)

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    artirma,                              # flips/rotation/zoom, training-time only
    layers.Resizing(96, 96),              # upscale inside the graph
    layers.Rescaling(2.0, offset=-1.0),   # 0-1 -> [-1, 1], what MobileNetV2 expects
    omurga,
    layers.GlobalAveragePooling2D(),      # 8x8x128 -> 128, not Flatten
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

# Two-phase training: warm up the head, then fine-tune the whole backbone
omurga.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
model.fit(x_train, y_train, epochs=3)

omurga.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="binary_crossentropy",
              metrics=[keras.metrics.AUC(name="auc")])
model.fit(x_train, y_train, epochs=60, callbacks=geri_cagrilar)
'''


with gr.Blocks(title="Aerial Cactus Identification / Havadan Kaktus Tespiti") as demo:
    gr.Markdown(BASLIK + "\n" + ALTBASLIK)
    with gr.Row():
        gr.Markdown(AMAC)
        gr.Markdown("### Result / Sonuc\n" + sonuc_notu())
    gr.Markdown(NASIL)

    gr.Markdown("---\n### Try it / Dene")
    with gr.Row():
        with gr.Column():
            goruntu = gr.Image(label="Aerial photo / Hava fotografi")
            calistir = gr.Button("Predict / Tahmin et", variant="primary")
        cikti = gr.Textbox(label="Result / Sonuc", lines=2)
    calistir.click(tahmin, [goruntu], cikti)

    with gr.Accordion("Example code / Ornek kod (from the notebook)", open=False):
        gr.Code(ORNEK_KOD, language="python")

    gr.Markdown("---\nTraining notebook / Egitim notebook'u: `train_aerial_cactus.ipynb` - "
                "[Kaggle](https://www.kaggle.com/c/aerial-cactus-identification)")


if __name__ == "__main__":
    demo.launch()
