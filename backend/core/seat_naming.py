# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.spatial.seat_naming for clean domain separation.
"""
from core.spatial.seat_naming import *
import importlib
_mod = importlib.import_module("core.spatial.seat_naming")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
