__version__ = "1.0.0"  # Effort-based versioning. Single source of truth, read by pyproject.toml and the docs.
__author__ = "David Braun"
from .core import AudioTree
from .core import SaliencyParams
from .writer import AudioWriter
from .tree_writer import TreeWriter

__all__ = [
    "AudioTree",
    "SaliencyParams",
    "AudioWriter",
    "TreeWriter",
    "sources",
    "transforms",
]


def __getattr__(name):
    # ``sources`` and ``transforms`` are the only modules that import grain;
    # load them lazily so that `import audiotree` (AudioTree, TreeWriter,
    # AudioWriter) works in environments without grain — e.g. downstream
    # packages that depend on audiotree only for the AudioTree container.
    if name in ("sources", "transforms"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
