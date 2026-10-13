# -*- coding: utf-8 -*-
"""
SmartClass AI — Compatibility Proxy
Redirects to core.analytics.report_generator for clean domain separation.
"""
import importlib
_mod = importlib.import_module("core.analytics.report_generator")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
