"""Concrete PII detector adapters.

Each adapter wraps an optional backend and is imported directly from its module
so that installing only one extra is enough (the import site raises a clear
ImportError when its backend is missing, per FR-017):

    from gaussia.detectors.presidio import PresidioDetector          # gaussia[privacy-presidio]
    from gaussia.detectors.huggingface import HuggingFacePIIDetector  # gaussia[privacy-huggingface]
"""
