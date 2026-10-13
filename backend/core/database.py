# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.storage.database for clean domain separation.
"""
import importlib
_mod = importlib.import_module("core.storage.database")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
