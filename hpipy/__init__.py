"""hpiPy: House Price Indices in Python."""

import os

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

from . import extensions, price_index, utils

__all__ = ["price_index", "utils", "extensions"]
