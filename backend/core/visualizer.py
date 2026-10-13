# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.visualization.visualizer for clean domain separation.
"""
from core.visualization.visualizer import *
import importlib
_mod = importlib.import_module("core.visualization.visualizer")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
