# __init__.py
# pylint: disable=unused-import

import sys
# from .constants import
from .import_tools import get_plot_fx
from .class_scrna import Rna
# from .class_spatial import Spatial
from . import utils as tl
from . import processing as pp
from . import analysis as ax
from . import visualization as pl
# from . import class_scrna, class_spatial, constants
from . import class_scrna, class_spatial, constants

# mod = ["ax", "pl", "pp", "tl", "Rna", "Spatial"]
mod = ["ax", "pl", "pp", "tl", "Rna", "get_plot_fx"]
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in mod})

__all__ = [
    "ax", "pl", "pp", "tl",
    "processing", "analysis", "visualization", "utils",
    "class_scrna", "get_plot_fx", "constants", "get_plot_fx"
    # "class_spatial"
]
