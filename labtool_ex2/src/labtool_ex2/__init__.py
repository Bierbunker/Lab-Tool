__all__ = ["Project"]
__version__ = "0.0.2"

from .Project import *
from .Project import Project as Project  # noqa
from . import monkeypatch_uncertainties as mpatch

mpatch.display()
