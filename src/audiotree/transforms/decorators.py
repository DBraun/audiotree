"""Decorators for creating transforms from functions."""

import difflib
import inspect
import textwrap
from typing import Any, Callable, Dict, Optional, Sequence

from audiotree.transforms.base import BaseMapTransform, BaseRandomTransform

# Parameter names the decorators own. A decorated function may not use them,
# since the wrapper would shadow them and the value would never reach the
# function.
_RESERVED = ("prob", "split_seed", "scope", "output_key")

# Distinguishes "not passed" from an explicit value, so a map transform can
# reject `prob` rather than silently defaulting it.
_UNSET = object()


class _Required:
    """Placeholder default for a parameter the caller must supply."""

    def __repr__(self) -> str:
        return "<required>"


#: Stands in for a decorated function's no-default parameters in the config, so
#: they are still advertised, still accepted, and still checked for.
_REQUIRED = _Required()

#: Google-style ``Args:`` entries for the parameters the decorators add, keyed
#: by name. Written unindented; ``_reserved_docs`` indents them to match the
#: docstring they are spliced into.
_RESERVED_DOC_ENTRIES = {
    "prob": (
        "prob: Probability of applying the transform, drawn independently per\n"
        "    batch item. Defaults to ``1.0`` (always)."
    ),
    "split_seed": (
        "split_seed: Give each AudioTree leaf its own RNG split. With ``False``\n"
        "    every leaf draws identically, which keeps a dry/wet pair in\n"
        "    lockstep. Defaults to ``True``."
    ),
    "scope": (
        "scope: Which leaves of a dict-of-AudioTree element to transform.\n"
        "    Defaults to ``None`` (all of them). See :ref:`dict_batches`."
    ),
    "output_key": (
        "output_key: Write the result under a new key instead of replacing the\n"
        "    input. A plain string names exactly one output, so it requires a\n"
        "    single in-scope leaf; with several, pass a callable that maps each\n"
        "    leaf's path to a distinct name."
    ),
}


def _reserved_docs(names: Sequence[str], indent: int) -> str:
    """Render the reserved ``Args:`` entries at the given indentation."""
    entries = "\n".join(_RESERVED_DOC_ENTRIES[name] for name in names)
    return textwrap.indent(entries, " " * indent)


def _transform_parameters(fn: Callable, drop: tuple) -> Dict[str, Any]:
    """The decorated function's own parameters and their defaults.

    A parameter without a default is recorded as ``_REQUIRED`` rather than
    skipped: it is part of the signature the wrapper publishes, so it has to be
    accepted, and the wrapper checks that the sentinel was replaced.
    """
    variadic = {
        inspect.Parameter.VAR_POSITIONAL: "*",
        inspect.Parameter.VAR_KEYWORD: "**",
    }
    parameters = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name in drop:
            continue
        if name in _RESERVED:
            raise TypeError(
                f"{fn.__name__} declares a parameter named {name!r}, which the "
                f"transform decorator reserves. Rename it."
            )
        if param.kind in variadic:
            # A transform's parameters are configured by name, so a variadic
            # one could never receive anything.
            raise TypeError(
                f"{fn.__name__} declares {variadic[param.kind]}{name}, which a "
                f"transform cannot have: its parameters are configured by name. "
                f"Declare them explicitly."
            )
        if param.default is inspect.Parameter.empty:
            parameters[name] = _REQUIRED
        else:
            parameters[name] = param.default
    return parameters


def _check_parameter_names(fn_name: str, given, known, reserved: Sequence[str]) -> None:
    """Reject misspelled or unknown transform parameters.

    Unknown keys used to be accepted and silently dropped, so
    ``volume_change(min_dB=40)`` (capital B) ran with the defaults and the
    augmentation the caller configured simply never happened.

    ``reserved`` is the subset of the reserved parameters this transform kind
    actually accepts: suggesting ``prob`` to a map transform, which rejects it,
    would send the caller straight into the next TypeError.
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
        f"plus {', '.join(reserved)}."
    )


#: Google-style section headers, used to find where the ``Args:`` block ends.
_SECTION_HEADERS = frozenset(
    (
        "args",
        "arguments",
        "attributes",
        "example",
        "examples",
        "note",
        "notes",
        "parameters",
        "raises",
        "references",
        "returns",
        "see also",
        "todo",
        "warning",
        "warnings",
        "warns",
        "yields",
    )
)


def _header_name(line: str) -> Optional[str]:
    """The section name if ``line`` is a Google-style header, else ``None``."""
    stripped = line.strip()
    if not stripped.endswith(":"):
        return None
    name = stripped.rstrip(":").strip().lower()
    return name if name in _SECTION_HEADERS else None


def _synthesize_doc(
    fn: Callable, drop: tuple, reserved: Sequence[str]
) -> Optional[str]:
    """Adapt the function's docstring to the constructor it becomes.

    The decorated function documents ``audio_tree``/``rng``, which the
    constructor does not accept, and cannot document ``prob``/``scope``/... ,
    which it does. Drop the former and splice the latter into the ``Args:``
    block — appending them to the end of the docstring instead put them after
    ``Returns:`` and ``Example:``, where Napoleon reads them as prose glued to
    the example rather than as parameters.
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

    # Python 3.13 dedents docstrings at compile time and earlier versions do
    # not, so the indentation of the sections has to be measured, not assumed.
    body = [line for line in kept[1:] if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in body), default=4)

    args_index = next(
        (i for i, line in enumerate(kept) if _header_name(line) == "args"), None
    )
    if args_index is None:
        # No Args block of its own: open one just before the first other
        # section, or at the end if there is none.
        args_index = next(
            (i for i, line in enumerate(kept) if _header_name(line) is not None),
            len(kept),
        )
        kept[args_index:args_index] = ["", " " * indent + "Args:"]
        args_index += 1
        end = args_index + 1
    else:
        # The Args block runs until the next section header.
        end = next(
            (
                i
                for i in range(args_index + 1, len(kept))
                if _header_name(kept[i]) is not None
            ),
            len(kept),
        )
        while end > args_index + 1 and not kept[end - 1].strip():
            end -= 1

    entries = _reserved_docs(reserved, indent + 4).splitlines()
    tail = kept[end:]
    # A section header needs a blank line before it, but not two.
    separator = [] if (tail and not tail[0].strip()) else [""]
    return "\n".join(kept[:end] + entries + separator + tail).rstrip() + "\n"


def _build_wrapper(fn, base_class, drop, make_transform):
    """Shared machinery for both decorators."""
    param_defaults = _transform_parameters(fn, drop)
    own_parameters = [
        param
        for param in inspect.signature(fn).parameters.values()
        if param.name in param_defaults
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
            # The function's own parameters, then the reserved settings that
            # differ from their defaults -- so the repr reads as a constructor
            # call that rebuilds this transform, not a different one.
            settings = sorted(self.config_dict.items())
            settings += [
                (name, self.reserved_dict[name])
                for name in _RESERVED
                if name in self.reserved_dict
            ]
            rendered = ", ".join(f"{k}={v!r}" for k, v in settings)
            return f"{fn.__name__}({rendered})"

    make_transform(FunctionBasedTransform, fn)
    FunctionBasedTransform.__name__ = f"{fn.__name__}_Transform"
    # Pickle resolves this path: after decoration the wrapper is bound to
    # `fn.__name__` in `fn.__module__`, and the class hangs off it.
    FunctionBasedTransform.__qualname__ = f"{fn.__name__}.Transform"
    FunctionBasedTransform.__module__ = fn.__module__

    accepted_reserved = _reserved_for(base_class)

    def wrapper(
        *args, prob=_UNSET, split_seed=_UNSET, scope=None, output_key=None, **kwargs
    ):
        """Create a transform instance with given parameters."""
        # A map transform applies unconditionally, so `prob`/`split_seed` are
        # meaningless for it. They were still swallowed by the wrapper and
        # dropped, so `trim(length=1.0, prob=0.5)` trimmed every time with no
        # sign that half the request was ignored.
        for reserved_name, value in (("prob", prob), ("split_seed", split_seed)):
            if value is not _UNSET and reserved_name not in accepted_reserved:
                raise TypeError(
                    f"{fn.__name__}() got unexpected parameter {reserved_name!r}: "
                    f"it is a map transform and applies unconditionally. Only "
                    f"random transforms take {reserved_name!r}."
                )
        prob = 1.0 if prob is _UNSET else prob
        split_seed = True if split_seed is _UNSET else split_seed
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

        _check_parameter_names(fn.__name__, kwargs, param_defaults, accepted_reserved)
        config = {**param_defaults, **kwargs}
        # A parameter with no default has to be supplied here; leaving the
        # sentinel in the config would defer the failure to the first batch,
        # inside a grain worker.
        missing = [name for name, value in config.items() if value is _REQUIRED]
        if missing:
            raise TypeError(
                f"{fn.__name__}() missing required parameter(s): "
                f"{', '.join(repr(name) for name in missing)}."
            )

        instance = FunctionBasedTransform.__new__(FunctionBasedTransform)
        _init_transform(
            instance, base_class, config, prob, split_seed, scope, output_key
        )
        instance.config_dict = dict(kwargs)
        # Reserved settings that differ from their defaults, as passed (before
        # any normalization), for the repr.
        instance.reserved_dict = {
            name: value
            for name, value, default in (
                ("prob", prob, 1.0),
                ("split_seed", split_seed, True),
                ("scope", scope, None),
                ("output_key", output_key, None),
            )
            if name in accepted_reserved and value != default
        }
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
    wrapper.__doc__ = _synthesize_doc(fn, drop, accepted_reserved)
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
    ``TypeError`` rather than being silently ignored, as does omitting one that
    has no default. Parameter values are opaque, so a dict, a list or ``None``
    is passed through to the function unchanged.

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
    being silently ignored, as does omitting one that has no default. Parameter
    values are opaque, so a dict, a list or ``None`` is passed through to the
    function unchanged.

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
        make_transform=lambda cls, f: setattr(
            cls,
            "_apply_transform",
            staticmethod(lambda audio_tree, **params: f(audio_tree, **params)),
        ),
    )
