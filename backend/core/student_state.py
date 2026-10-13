# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.behavior.student_state for clean domain separation.
"""
from core.behavior.student_state import *
import importlib
_mod = importlib.import_module("core.behavior.student_state")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
