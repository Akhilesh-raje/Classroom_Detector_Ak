# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.attendance.attendance_engine for clean domain separation.
"""
from core.attendance.attendance_engine import *
import importlib
_mod = importlib.import_module("core.attendance.attendance_engine")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
