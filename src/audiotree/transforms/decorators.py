"""Decorators for creating transforms from functions."""

import difflib
import inspect
from typing import Any, Callable, Dict, Optional

from audiotree.transforms.base import BaseMapTransform, BaseRandomTransform

# Parameter names the decorators own. A decorated function may not use them,
# since the wrapper would shadow them and the value would never reach the
# function.
_RESERVED = ("prob", "split_seed", "scope", "output_key")

_RESERVED_DOCS = """
    prob: Probability of applying the transform, drawn independently per batch
        item. Defaults to ``1.0`` (always).
    split_seed: Give each AudioTree leaf its own RNG split. Defaults to ``True``.
    scope: Which leaves of a dict-of-AudioTree element to transform. Defaults to
        ``None`` (all of them). See :ref:`dict_batches`.
    output_key: Write the result under a new key instead of replacing the input.
"""

_MAP_RESERVED_DOCS = """
    scope: Which leaves of a dict-of-AudioTree element to transform. Defaults to
        ``None`` (all of them). See :ref:`dict_batches`.
    output_key: Write the result under a new key instead of replacing the input.
"""


def _transform_parameters(fn: Callable, drop: tuple) -> Dict[str, Any]:
    """The decorated function's own parameters and their defaults."""
    parameters = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in drop:
            continue
        if name in _RESERVED:
            raise TypeError(
                f"{fn.__name__} declares a parameter named {name!r}, which the "
                f"transform decorator reserves. Rename it."
            )
        if param.default is not inspect.Parameter.empty:
            parameters[name] = param.default
    return parameters


def _check_parameter_names(fn_name: str, given, known) -> None:
    """Reject misspelled or unknown transform parameters.

    Unknown keys used to be accepted and silently dropped, so
    ``volume_change(min_dB=40)`` (capital B) ran with the defaults and the
    augmentation the caller configured simply never happened.
    """
    unknown = [name for name in given if name not in known]
    if not unknown:
        return
    hints = []
    for name in unknown:
        close = difflib.get_close_matches(name, known, n=1)
        hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise TypeError(
        f"{fn_name}() got unexpected parameter(s): {', '.join(hints)}. "
        f"Valid parameters: {', '.join(sorted(known)) or '(none)'}, "
        f"plus {', '.join(_RESERVED)}."
    )


def _synthesize_doc(fn: Callable, drop: tuple, reserved_docs: str) -> Optional[str]:
    """Adapt the function's docstring to the constructor it becomes.

    The decorated function documents ``audio_tree``/``rng``, which the
    constructor does not accept, and cannot document ``prob``/``scope``/... ,
    which it does. Drop the former and append the latter.
    """
    doc = fn.__doc__
    if not doc:
        return doc
    lines = doc.splitlines()
    kept, skipping = [], False
    for line in lines:
        stripped = line.strip()
        starts_dropped = any(
            stripped.startswith(f"{name}:") or stripped.startswith(f"{name} (")
            for name in drop
        )
        if starts_dropped:
            skipping = True
            continue
        # A continuation line of a dropped entry is indented further than it.
        if skipping and stripped and not line.startswith(" " * 12):
            skipping = False
        if skipping and stripped:
            continue
        skipping = False
        kept.append(line)

    doc = "\n".join(kept)
    if "Args:" in doc:
        head, _, tail = doc.partition("Args:")
        return f"{head}Args:{tail.rstrip()}\n{reserved_docs}"
    return f"{doc.rstrip()}\n\n    Args:\n{reserved_docs}"


def _build_wrapper(fn, base_class, drop, reserved_docs, make_transform):
    """Shared machinery for both decorators."""
    param_defaults = _transform_parameters(fn, drop)
    own_parameters = [
        param
        for param in inspect.signature(fn).parameters.values()
        if param.name not in drop
    ]

    # Defined once, at decoration time. Building it inside the wrapper minted a
    # fresh local type per call, so instances had no stable identity, no useful
    # repr, and could not be pickled -- which grain needs, since its worker
    # processes are spawned, not forked.
    class FunctionBasedTransform(base_class):
        """Transform created from a decorated function."""

        @staticmethod
        def get_default_config():
            return param_defaults.copy()

        def __repr__(self):
            settings = ", ".join(
                f"{k}={v!r}" for k, v in sorted(self.config_dict.items())
            )
            return f"{fn.__name__}({settings})"

    make_transform(FunctionBasedTransform, fn)
    FunctionBasedTransform.__name__ = f"{fn.__name__}_Transform"
    # Pickle resolves this path: after decoration the wrapper is bound to
    # `fn.__name__` in `fn.__module__`, and the class hangs off it.
    FunctionBasedTransform.__qualname__ = f"{fn.__name__}.Transform"
    FunctionBasedTransform.__module__ = fn.__module__

    def wrapper(
        *args, prob=1.0, split_seed=True, scope=None, output_key=None, **kwargs
    ):
        """Create a transform instance with given parameters."""
        # Bind positional args against the *function's* parameters, so
        # `roll(0.5, 1.0)` means what the rendered signature says it means. The
        # reserved parameters used to sit first and positional, so those two
        # values landed on `prob` and `split_seed`.
        if args:
            bound = inspect.Signature(own_parameters).bind_partial(*args)
            overlap = set(bound.arguments) & set(kwargs)
            if overlap:
                raise TypeError(
                    f"{fn.__name__}() got multiple values for "
                    f"{', '.join(sorted(overlap))}"
                )
            kwargs = {**bound.arguments, **kwargs}

        _check_parameter_names(fn.__name__, kwargs, param_defaults)
        config = {**param_defaults, **kwargs}

        instance = FunctionBasedTransform.__new__(FunctionBasedTransform)
        _init_transform(
            instance, base_class, config, prob, split_seed, scope, output_key
        )
        instance.config_dict = dict(kwargs)
        return instance

    reserved_signature = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
        for name, default in (
            ("prob", 1.0),
            ("split_seed", True),
            ("scope", None),
            ("output_key", None),
        )
        if name in _reserved_for(base_class)
    ]
    wrapper.__signature__ = inspect.Signature(own_parameters + reserved_signature)
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__module__ = fn.__module__
    wrapper.__doc__ = _synthesize_doc(fn, drop, reserved_docs)
    wrapper.Transform = FunctionBasedTransform
    wrapper.__wrapped__ = fn
    return wrapper


def _reserved_for(base_class) -> tuple:
    """Which reserved parameters a given base class actually accepts."""
    if issubclass(base_class, BaseRandomTransform):
        return _RESERVED
    return ("scope", "output_key")


def _init_transform(instance, base_class, config, prob, split_seed, scope, output_key):
    if issubclass(base_class, BaseRandomTransform):
        base_class.__init__(
            instance,
            config=config,
            prob=prob,
            split_seed=split_seed,
            scope=scope,
            output_key=output_key,
        )
    else:
        base_class.__init__(instance, config=config, scope=scope, output_key=output_key)


def random_transform(fn: Callable) -> Callable:
    """Decorator to create a RandomMapTransform from a function.

    The decorated function should have signature::

        fn(audio_tree: AudioTree, rng, **params) -> AudioTree

    The returned callable constructs the transform. It takes the function's own
    parameters (positionally or by keyword) plus the keyword-only ``prob``,
    ``split_seed``, ``scope`` and ``output_key``. A misspelled parameter raises
    ``TypeError`` rather than being silently ignored.

    Usage::

        @random_transform
        def volume_norm(audio_tree, rng, min_db=-20.0, max_db=-15.0):
            ...
            return audio_tree

        transform = volume_norm(min_db=-30, max_db=-10, prob=0.9)
        ds = ds.random_map(transform, seed=42)
    """
    return _build_wrapper(
        fn,
        BaseRandomTransform,
        drop=("audio_tree", "rng"),
        reserved_docs=_RESERVED_DOCS,
        make_transform=lambda cls, f: setattr(
            cls,
            "_apply_transform",
            staticmethod(
                lambda audio_tree, rng, **params: f(audio_tree, rng, **params)
            ),
        ),
    )


def map_transform(fn: Callable) -> Callable:
    """Decorator to create a MapTransform from a function.

    The decorated function should have signature::

        fn(audio_tree: AudioTree, **params) -> AudioTree

    The returned callable constructs the transform. It takes the function's own
    parameters (positionally or by keyword) plus the keyword-only ``scope`` and
    ``output_key``. A misspelled parameter raises ``TypeError`` rather than
    being silently ignored.

    Usage::

        @map_transform
        def trim(audio_tree, length=1.0):
            ...
            return audio_tree

        transform = trim(length=3.0)
        ds = ds.map(transform)
    """
    return _build_wrapper(
        fn,
        BaseMapTransform,
        drop=("audio_tree",),
        reserved_docs=_MAP_RESERVED_DOCS,
        make_transform=lambda cls, f: setattr(
            cls,
            "_apply_transform",
            staticmethod(lambda audio_tree, **params: f(audio_tree, **params)),
        ),
    )
