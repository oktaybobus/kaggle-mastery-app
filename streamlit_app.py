"""Kaggle Mastery - one interactive Streamlit app for all 16 competitions.

Each competition reuses the exact `tahmin()` from its Gradio `app_<proj>.py`.
Gradio itself is never imported (a stub stands in); trained models are pulled from
the Hugging Face model repo OKTAYBBS/kaggle-mastery-models at runtime and cached.
"""
import importlib
import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _compat_gradio
_compat_gradio.install()

MODELS_REPO = "OKTAYBBS/kaggle-mastery-models"
APPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps")

# competition -> display title + Kaggle url
TITLES = {
    "titanic": ("Titanic - Survival", "https://www.kaggle.com/c/titanic"),
    "spaceship_titanic": ("Spaceship Titanic", "https://www.kaggle.com/c/spaceship-titanic"),
    "house_prices": ("House Prices", "https://www.kaggle.com/c/house-prices-advanced-regression-techniques"),
    "bike_sharing": ("Bike Sharing Demand", "https://www.kaggle.com/c/bike-sharing-demand"),
    "forest_cover": ("Forest Cover Type", "https://www.kaggle.com/c/forest-cover-type-prediction"),
    "porto_seguro": ("Porto Seguro Driver", "https://www.kaggle.com/c/porto-seguro-safe-driver-prediction"),
    "ghouls_goblins_ghosts": ("Ghouls, Goblins & Ghosts", "https://www.kaggle.com/c/ghouls-goblins-and-ghosts-boo"),
    "leaf_classification": ("Leaf Classification", "https://www.kaggle.com/c/leaf-classification"),
    "dont_overfit": ("Don't Overfit! II", "https://www.kaggle.com/c/dont-overfit-ii"),
    "store_sales": ("Store Sales Forecasting", "https://www.kaggle.com/c/store-sales-time-series-forecasting"),
    "commonlit_readability": ("CommonLit Readability", "https://www.kaggle.com/c/commonlitreadabilityprize"),
    "contradictory_watson": ("Contradictory, My Dear Watson", "https://www.kaggle.com/c/contradictory-my-dear-watson"),
    "nlp_disaster_tweets": ("Disaster Tweets NLP", "https://www.kaggle.com/c/nlp-getting-started"),
    "quora_question_pairs": ("Quora Question Pairs", "https://www.kaggle.com/c/quora-question-pairs"),
    "aerial_cactus": ("Aerial Cactus Identification", "https://www.kaggle.com/c/aerial-cactus-identification"),
    "digit_recognizer": ("Digit Recognizer (MNIST)", "https://www.kaggle.com/c/digit-recognizer"),
}

# widget specs, in the order each tahmin() expects its positional args.
# kinds: sl=slider(min,max,default,step) | se=selectbox(choices,default) |
#        rb=radio(choices,default) | cb=checkbox(default) | tx=text(default) |
#        ta=textarea(default) | img=image upload | dyn=populated from module attr
S = {
    "titanic": [
        ("Ticket class (Pclass)", "se", [[1, 2, 3], 3]),
        ("Sex", "rb", [["Kadin", "Erkek"], "Erkek"]),
        ("Age", "sl", [0, 80, 30, 1]),
        ("Fare ($)", "sl", [0, 300, 32, 1]),
        ("Siblings/Spouse (SibSp)", "sl", [0, 8, 0, 1]),
        ("Parents/Children (Parch)", "sl", [0, 6, 0, 1]),
        ("Embark port", "se", [["S", "C", "Q"], "S"]),
    ],
    "spaceship_titanic": [
        ("Home planet", "se", [["Earth", "Europa", "Mars"], "Earth"]),
        ("CryoSleep (asleep?)", "cb", [False]),
        ("Cabin deck", "se", [list("ABCDEFGT"), "F"]),
        ("Destination", "se", [["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"], "TRAPPIST-1e"]),
        ("Age", "sl", [0, 80, 27, 1]),
        ("VIP", "cb", [False]),
        ("Total spend", "sl", [0, 10000, 700, 50]),
    ],
    "house_prices": [
        ("Overall quality", "sl", [1, 10, 6, 1]),
        ("Living area (sqft)", "sl", [400, 5000, 1500, 10]),
        ("Year built", "sl", [1880, 2010, 1990, 1]),
        ("Basement area (sqft)", "sl", [0, 3000, 1000, 10]),
        ("Garage cars", "sl", [0, 4, 2, 1]),
        ("Full baths", "sl", [0, 4, 2, 1]),
    ],
    "bike_sharing": [
        ("Hour", "sl", [0, 23, 8, 1]),
        ("Weekday (0=Mon)", "sl", [0, 6, 1, 1]),
        ("Month", "sl", [1, 12, 6, 1]),
        ("Temperature (C)", "sl", [-5, 40, 22, 1]),
        ("Humidity (%)", "sl", [0, 100, 55, 1]),
        ("Wind speed", "sl", [0, 50, 10, 1]),
        ("Weather (1=clear..4=storm)", "se", [[1, 2, 3, 4], 1]),
        ("Working day?", "cb", [True]),
    ],
    "forest_cover": [
        ("Elevation (m)", "sl", [1800, 4000, 2800, 10]),
        ("Slope (deg)", "sl", [0, 60, 15, 1]),
        ("Horiz. dist. to water (m)", "sl", [0, 1400, 250, 10]),
        ("Horiz. dist. to road (m)", "sl", [0, 7000, 2000, 50]),
        ("Noon hillshade", "sl", [0, 255, 220, 1]),
    ],
    "porto_seguro": [
        ("Profile deviation (noise)", "sl", [0.0, 1.0, 0.2, 0.01]),
        ("Random seed", "sl", [0, 999, 42, 1]),
    ],
    "ghouls_goblins_ghosts": [
        ("Bone length (norm)", "sl", [0.0, 1.0, 0.4, 0.01]),
        ("Rotting flesh", "sl", [0.0, 1.0, 0.5, 0.01]),
        ("Hair length", "sl", [0.0, 1.0, 0.5, 0.01]),
        ("Has soul", "sl", [0.0, 1.0, 0.5, 0.01]),
        ("Color", "se", [["white", "black", "clear", "blue", "green", "blood"], "white"]),
    ],
    "leaf_classification": [
        ("Test sample #", "sl", [0, 9, 0, 1]),
    ],
    "dont_overfit": [
        ("Random sample seed", "sl", [0, 9999, 42, 1]),
    ],
    "store_sales": [
        ("Date (YYYY-MM-DD)", "tx", ["2017-08-20"]),
        ("Store no", "sl", [1, 54, 1, 1]),
        ("Product family", "dyn", ["aileler"]),
        ("Items on promo", "sl", [0, 200, 0, 1]),
    ],
    "commonlit_readability": [
        ("Passage", "ta", ["The cat sat on the mat. It was a warm and sunny day."]),
    ],
    "contradictory_watson": [
        ("Premise", "ta", ["A man is playing a guitar on stage."]),
        ("Hypothesis", "ta", ["A person is performing music."]),
    ],
    "nlp_disaster_tweets": [
        ("Tweet", "ta", ["Forest fire near La Ronge Sask. Canada"]),
    ],
    "quora_question_pairs": [
        ("Question 1", "tx", ["How can I learn Python?"]),
        ("Question 2", "tx", ["What is the best way to learn Python?"]),
    ],
    "aerial_cactus": [("Aerial photo (32x32)", "img", [])],
    "digit_recognizer": [("Handwritten digit image", "img", [])],
}

CATEGORY = {
    "Tabular": ["titanic", "spaceship_titanic", "house_prices", "bike_sharing",
                "forest_cover", "porto_seguro", "ghouls_goblins_ghosts",
                "leaf_classification", "dont_overfit", "store_sales"],
    "NLP": ["commonlit_readability", "contradictory_watson",
            "nlp_disaster_tweets", "quora_question_pairs"],
    "Computer Vision": ["aerial_cactus", "digit_recognizer"],
}


@st.cache_resource(show_spinner=False)
def get_module(project):
    """Download the model into apps/<project>/, then import the app module (gradio-stubbed)."""
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(APPS_DIR, project)
    ext = "keras" if project in ("aerial_cactus", "digit_recognizer") else "pkl"
    fname = f"model_{project}.{ext}"
    local = os.path.join(dest_dir, fname)
    if not os.path.exists(local):
        path = hf_hub_download(MODELS_REPO, f"{project}/{fname}", repo_type="model")
        os.symlink(path, local) if hasattr(os, "symlink") else None
        if not os.path.exists(local):
            import shutil
            shutil.copyfile(path, local)
    mod = importlib.import_module(f"apps.{project}.app_{project}")
    return mod


def render_inputs(project, mod):
    vals = []
    for i, (label, kind, params) in enumerate(S[project]):
        key = f"{project}_{i}"
        if kind == "sl":
            lo, hi, dv, step = params
            vals.append(st.slider(label, float(lo), float(hi), float(dv), float(step), key=key)
                        if isinstance(step, float) or isinstance(dv, float)
                        else st.slider(label, lo, hi, dv, step, key=key))
        elif kind == "se":
            choices, dv = params
            vals.append(st.selectbox(label, choices, index=choices.index(dv), key=key))
        elif kind == "rb":
            choices, dv = params
            vals.append(st.radio(label, choices, index=choices.index(dv), horizontal=True, key=key))
        elif kind == "cb":
            vals.append(st.checkbox(label, value=params[0], key=key))
        elif kind == "tx":
            vals.append(st.text_input(label, value=params[0], key=key))
        elif kind == "ta":
            vals.append(st.text_area(label, value=params[0], key=key))
        elif kind == "dyn":
            choices = list(getattr(mod, params[0], []))
            vals.append(st.selectbox(label, choices, key=key) if choices
                        else st.text_input(label, key=key))
        elif kind == "img":
            up = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=key)
            if up is not None:
                from PIL import Image
                img = Image.open(up).convert("RGB")
                st.image(img, width=160)
                vals.append(np.array(img))
            else:
                vals.append(None)
    return vals


st.set_page_config(page_title="Kaggle Mastery", layout="centered")
st.title("Kaggle Mastery")
st.caption("Interactive demos for 16 Kaggle competitions - reusing each project's real trained model.")

with st.sidebar:
    st.header("Competition")
    cat = st.radio("Category", list(CATEGORY.keys()))
    project = st.selectbox("Pick a competition",
                           CATEGORY[cat],
                           format_func=lambda p: TITLES[p][0])
    st.markdown("---")
    st.markdown("[Models](https://huggingface.co/OKTAYBBS/kaggle-mastery-models) - "
                "[GitHub](https://github.com/oktaybobus)")

# The two CNN demos need TensorFlow, which is too heavy for the free tier;
# they are showcased on their static page + notebook instead of run live here.
CV_PROJECTS = {"aerial_cactus", "digit_recognizer"}

title, kurl = TITLES[project]
st.subheader(title)
st.markdown(f"[View competition on Kaggle]({kurl})")

if project in CV_PROJECTS:
    space = f"https://huggingface.co/spaces/OKTAYBBS/kaggle-mastery-{project.replace('_', '-')}"
    st.info(
        "This is a convolutional neural network (Keras) demo. To keep the live app "
        "fast and light, image models are not run here - explore the approach and "
        "results on the project page, or run the notebook to try it end to end."
    )
    st.markdown(f"- [Project page / showcase]({space})\n- [Competition on Kaggle]({kurl})")
    st.stop()

with st.spinner("Loading model..."):
    try:
        mod = get_module(project)
    except Exception as e:
        st.error(f"Could not load the model for {title}: {e}")
        st.stop()

# result note from the notebook's stored metrics
note_fn = getattr(mod, "sonuc_notu", None) or getattr(mod, "metrik_notu", None)
if callable(note_fn):
    try:
        note = note_fn()
        if note:
            st.info(note)
    except Exception:
        pass

st.markdown("#### Try it")
vals = render_inputs(project, mod)

if st.button("Predict", type="primary"):
    if any(v is None for v in vals):
        st.warning("Please provide all inputs (upload an image for vision demos).")
    else:
        with st.spinner("Predicting..."):
            try:
                out = mod.tahmin(*vals)
                if isinstance(out, dict):
                    ranked = sorted(out.items(), key=lambda kv: kv[1], reverse=True)
                    top_label, top_prob = ranked[0]
                    st.success(f"Prediction: {top_label}  ({float(top_prob) * 100:.1f}%)")
                    st.bar_chart({k: float(v) for k, v in ranked[:10]})
                else:
                    st.success(out)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Kaggle Mastery - 16-competition portfolio by OKTAYBBS. "
           "Predictions run the same model saved by each training notebook.")
