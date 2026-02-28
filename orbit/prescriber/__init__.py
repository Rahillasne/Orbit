def __getattr__(name: str):
    """Lazy-import to avoid pulling in heavy ML dependencies at import time."""
    _imports = {
        "Prescriber": "orbit.prescriber.prescriber",
        "Prescription": "orbit.prescriber.prescriber",
        "PrescriptionReport": "orbit.prescriber.prescriber",
    }
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Prescriber", "Prescription", "PrescriptionReport"]
