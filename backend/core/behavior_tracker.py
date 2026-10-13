# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.behavior.behavior_tracker for clean domain separation.
"""
from core.behavior.behavior_tracker import *
import importlib
_mod = importlib.import_module("core.behavior.behavior_tracker")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
