"""
SmartClass AI — Detection Compatibility Proxy
Redirects to core.detection.detector for clean domain separation.
"""
from core.detection.detector import *
import importlib
_mod = importlib.import_module("core.detection.detector")
for _k, _v in _mod.__dict__.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
