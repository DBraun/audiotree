"""Invariants on the public API surface.

Nothing in the suite pinned the public surface, which is why parameter naming
and calling conventions drifted apart across the library without anything going
red. These are structural checks, not a signature snapshot: a snapshot of every
default would need updating on each ordinary change and would train people to
regenerate it without reading the diff.
"""

import inspect

import pytest

import audiotree
import audiotree.sources
import audiotree.transforms
import audiotree.transforms.jax
from audiotree import AudioTree, AudioWriter, TreeWriter
from audiotree.sources import ManifestDataSource, TreeDataSource

# Public callables and the number of leading positional parameters each is
# allowed. Everything past that must be keyword-only, so adding, reordering or
# renaming a parameter cannot silently change what a positional call means.
POSITIONAL_BUDGET = {
    "audiotree.sources.create_audio_dataset": 2,
    "audiotree.sources.create_balanced_audio_dataset": 3,
    "audiotree.sources.create_windowed_audio_dataset": 2,
    "audiotree.sources.find_audio_files": 2,
    "audiotree.AudioTree.create": 2,
    "audiotree.AudioTree.from_file": 1,
    "audiotree.AudioTree.from_manifest": 1,
    "audiotree.AudioTree.write": 1,
    "audiotree.AudioTree.resample": 1,
    "audiotree.AudioWriter": 1,
    "audiotree.TreeWriter": 2,
    "audiotree.sources.ManifestDataSource": 1,
    "audiotree.sources.TreeDataSource": 1,
}


def _resolve(dotted: str):
    obj = audiotree
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj.__init__ if inspect.isclass(obj) else obj


@pytest.mark.parametrize("dotted,budget", sorted(POSITIONAL_BUDGET.items()))
def test_positional_parameter_budget(dotted, budget):
    """Public callables expose at most a couple of positional parameters."""
    parameters = [
        p
        for p in inspect.signature(_resolve(dotted)).parameters.values()
        if p.name not in ("self", "cls")
        and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert len(parameters) <= budget, (
        f"{dotted} exposes {len(parameters)} positional parameters "
        f"({[p.name for p in parameters]}); budget is {budget}. Put a `*` after "
        f"the ones that are genuinely positional."
    )


def test_transform_namespaces_expose_the_same_names():
    """The NumPy and JAX transform namespaces must not drift apart.

    A name in one but not the other is a portability trap: a pipeline written
    against one backend fails at import when switched to the other.
    """
    # `choose` decides in Python which sub-transforms to run, so it cannot be
    # traced; a JAX version would be a jit-incompatible lie. It is the only
    # sanctioned asymmetry -- anything else here is drift.
    backend_specific = {"choose"}
    numpy_names = set(audiotree.transforms.__all__) - backend_specific
    jax_names = set(audiotree.transforms.jax.__all__) - backend_specific
    assert numpy_names == jax_names, {
        "numpy_only": sorted(numpy_names - jax_names),
        "jax_only": sorted(jax_names - numpy_names),
    }


@pytest.mark.parametrize(
    "module",
    [audiotree, audiotree.sources, audiotree.transforms, audiotree.transforms.jax],
    ids=lambda m: m.__name__,
)
def test_all_entries_resolve(module):
    """Every name in `__all__` actually exists on its module."""
    for name in module.__all__:
        assert hasattr(module, name), f"{module.__name__}.__all__ names missing {name}"


@pytest.mark.parametrize(
    "cls", [AudioTree, AudioWriter, TreeWriter, ManifestDataSource, TreeDataSource]
)
def test_public_methods_are_documented(cls):
    """Public methods carry a docstring, so autodoc has something to render."""
    undocumented = [
        name
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_")
        and callable(member)
        and getattr(member, "__doc__", None) in (None, "")
        and getattr(member, "__module__", "").startswith("audiotree")
    ]
    assert not undocumented, (
        f"{cls.__name__} has undocumented public methods: {undocumented}"
    )


def test_transform_constructors_reject_unknown_parameters():
    """A misspelled transform parameter raises instead of being ignored."""
    for namespace in (audiotree.transforms, audiotree.transforms.jax):
        with pytest.raises(TypeError, match="unexpected parameter"):
            namespace.volume_change(min_dB=6.0)
