from .core import *

try:
    from .swift import *
except ModuleNotFoundError as exc:
    if exc.name != "swift":
        raise
