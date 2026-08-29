"""Minimal gradio stub: lets the Gradio app_*.py modules import without gradio.
Only their pure-python tahmin()/sonuc_notu() helpers are used by Streamlit; every
gr.* UI call becomes an inert Dummy (callable, context-manager, any-attribute).

Important: the fake module must expose real dunder attributes (__file__, __spec__)
and its __getattr__ must NOT intercept dunder names, otherwise inspect.getmodule /
inspect.stack (used by Streamlit) crash when they read module.__file__.
"""
import sys
import types


class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Dummy()
    def __enter__(self): return _Dummy()
    def __exit__(self, *a): return False
    def __getattr__(self, name): return _Dummy()
    def __getitem__(self, k): return _Dummy()
    def __iter__(self): return iter(())


def _any(*a, **k):
    return _Dummy()


def _make_module(name):
    m = types.ModuleType(name)
    m.__file__ = __file__          # a real string path -> inspect stays happy
    m.__spec__ = None
    m.__path__ = []                # mark as package so submodule access is fine

    def _getattr(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return _any
    m.__getattr__ = _getattr
    return m


def install():
    if "gradio" in sys.modules and getattr(sys.modules["gradio"], "__file__", None) == __file__:
        return
    m = _make_module("gradio")
    themes = _make_module("gradio.themes")
    m.themes = themes
    sys.modules["gradio"] = m
    sys.modules["gradio.themes"] = themes
