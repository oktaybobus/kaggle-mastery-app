"""Minimal gradio stub: lets the Gradio app_*.py modules import without gradio.
Only their pure-python tahmin()/sonuc_notu() helpers are used by Streamlit; every
gr.* UI call becomes an inert Dummy (callable, context-manager, any-attribute)."""
import sys, types

class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Dummy()
    def __enter__(self): return _Dummy()
    def __exit__(self, *a): return False
    def __getattr__(self, name): return _Dummy()
    def __getitem__(self, k): return _Dummy()
    def __iter__(self): return iter(())

def install():
    if "gradio" in sys.modules:
        return
    m = types.ModuleType("gradio")
    def _any(*a, **k): return _Dummy()
    m.__getattr__ = lambda name: _any
    themes = types.ModuleType("gradio.themes")
    themes.__getattr__ = lambda name: _any
    m.themes = themes
    sys.modules["gradio"] = m
    sys.modules["gradio.themes"] = themes
