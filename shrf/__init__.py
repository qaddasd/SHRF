"""SHRF: Shrinkage Heap Regression Forest — attention-free memory."""
from .config import SHRFConfig

__all__ = ["SHRFConfig", "SHRFLayer"]


def __getattr__(name):
    if name == "SHRFLayer":
        from .layer import SHRFLayer
        return SHRFLayer
    raise AttributeError(name)