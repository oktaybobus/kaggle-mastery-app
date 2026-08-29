# Kaggle Mastery - Interactive Streamlit App

One interactive demo for all 16 Kaggle Mastery competitions. Each competition reuses
the exact prediction logic from its Gradio `app_*.py`; trained models are pulled at
runtime from the Hugging Face model repo
[OKTAYBBS/kaggle-mastery-models](https://huggingface.co/OKTAYBBS/kaggle-mastery-models)
(so this repo stays small - no model files committed).

## Run locally
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io -> New app -> pick the repo, branch, and
   `streamlit_app.py` as the main file.
3. (Optional) Set the app URL subdomain to `oktaybobus-kaggle-mastery` so it matches
   the links on the Hugging Face static Spaces.
