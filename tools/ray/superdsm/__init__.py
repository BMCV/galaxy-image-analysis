from .version import *
__version__ = VERSION

# By-pass this bug: https://github.com/flatironinstitute/sparse_dot/issues/7
from . import _mkl

from . import pipeline
from . import config

