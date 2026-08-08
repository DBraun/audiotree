from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    TypedDict,
    TYPE_CHECKING,
    Union,
    Unpack,
)

from absl import logging
from flax import struct
import jax
from jax import numpy as jnp, tree_util
import librosa
import loudness
import numpy as np
import soundfile

from . import _manifest
from ._fs import safe_join
from .loudness import (
    _jit_integrated_loudness,
    _jit_windowed_loudness,
    _numpy_windowed_lufs,
    _window_samples,
    _windowed_num_windows,
    pad_to_gating_block,
    safe_gain_db,
    shift_lufs,
    shift_lufs_windows,
)
from .resample import resample

if TYPE_CHECKING:
    # An array field that may hold either a NumPy or a JAX array. ``jax`` is
    # already imported at module top (``replace_lufs(device=...)`` needs
    # ``jax.devices`` / ``jax.device_put``), so this annotation resolves against
    # it without a duplicate import.
    ArrayLike = Union[np.ndarray, jax.Array]
else:
    # Runtime fallback so the name still resolves (e.g. for
    # ``typing.get_type_hints``) without JAX installed.
    ArrayLike = np.ndarray


def search_uniform(
    rng: np.random.Generator,
    offset: float,
    duration: float,
    total_duration: float,
    attempt: int,
    max_attempts: int,
) -> float:
    """Draw an excerpt offset uniformly over the file."""
    lower_bound = max(0.0, offset)
    upper_bound = max(total_duration - duration, lower_bound)
    return rng.uniform(lower_bound, upper_bound)


def search_bias_early(
    rng: np.random.Generator,
    offset: float,
    duration: float,
    total_duration: float,
    attempt: int,
    max_attempts: int,
) -> float:
    """Draw an offset that concentrates earlier in the file as attempts mount."""
    lower_bound = max(0.0, offset)
    upper_bound1 = max(total_duration - duration, lower_bound)
    # linearly interpolate the upper bound based on number of attempts so far
    alpha = attempt / (max_attempts - 1) if max_attempts > 1 else 0
    upper_bound2 = min(upper_bound1, lower_bound + duration)
    upper_bound = upper_bound1 * (1 - alpha) + upper_bound2 * alpha
    return rng.uniform(lower_bound, upper_bound)


#: Offset-search functions selectable by name in a config file.
_SEARCH_FUNCTIONS = {
    "uniform": search_uniform,
    "bias_early": search_bias_early,
}


def _resolve_search_function(spec) -> Callable:
    """Resolve an excerpt ``search`` to a callable.

    Accepts a callable, a registered name (``"uniform"``, ``"bias_early"``), or
    a dotted path to an importable function (``"mypkg.offsets.my_search"``).

    Deliberately *not* ``eval``: :class:`ExcerptConfig` is bound from YAML, so an
    ``eval`` here would make a config file arbitrary code execution.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"search must be a name or a callable, got {type(spec).__name__}."
        )
    if spec in _SEARCH_FUNCTIONS:
        return _SEARCH_FUNCTIONS[spec]

    known = ", ".join(repr(k) for k in _SEARCH_FUNCTIONS)
    if "." not in spec:
        raise ValueError(
            f"Unknown search {spec!r}. Valid names: {known}, or a dotted path to "
            f"an importable function (e.g. 'mypkg.offsets.my_search')."
        )

    module_name, _, attribute = spec.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"Could not import module {module_name!r} for search {spec!r}. "
            f"Valid names: {known}, or a dotted path to an importable function."
        ) from exc
    try:
        function = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"Module {module_name!r} has no attribute {attribute!r} "
            f"(from search {spec!r})."
        ) from exc
    if not callable(function):
        raise ValueError(f"search {spec!r} resolved to a non-callable.")
    return function


@dataclasses.dataclass(frozen=True)
class ExcerptConfig:
    """How to choose which part of a file an excerpt comes from.

    One named ``strategy`` rather than a set of interacting flags:

    ``"start"``
        Always offset 0. Deterministic, and the right choice for validation sets
        where every epoch should see identical audio.
    ``"random"``
        A uniformly random offset (the default). No audio is measured, so this
        costs one read per item.
    ``"loudest"``
        Draw up to ``num_tries`` candidate offsets and keep the loudest, stopping
        early once one exceeds ``lufs_cutoff``. Use it on corpora with long quiet
        stretches; it costs up to ``num_tries`` reads and loudness measurements
        per item.

    Note that ``"loudest"`` is a best-of-``num_tries`` search, not a filter: on a
    file where nothing clears the cutoff it still has to return something, and
    ``on_failure`` decides what.

    Attributes:
        strategy: Which of the three above.
        num_tries: Maximum candidate offsets to try (``"loudest"`` only).
        lufs_cutoff: Integrated loudness (LUFS) that ends the search early
            (``"loudest"`` only).
        search: How each candidate offset is drawn (``"loudest"`` only) — a
            registered name (``"uniform"``, ``"bias_early"``), a dotted path to
            an importable function, or a callable.
        on_failure: What to do when no candidate clears ``lufs_cutoff``
            (``"loudest"`` only). ``"keep"`` returns the loudest excerpt found,
            ``"skip"`` returns ``None`` (grain drops it at
            ``to_iter_dataset()``), ``"raise"`` raises naming the file.

    Example:
        >>> from audiotree import ExcerptConfig
        >>> ExcerptConfig().strategy                       # random offset
        'random'
        >>> ExcerptConfig(strategy="start").strategy       # deterministic
        'start'
        >>> loud = ExcerptConfig(
        ...     strategy="loudest", lufs_cutoff=-30, on_failure="skip"
        ... )
        >>> loud.num_tries, loud.on_failure
        (8, 'skip')
    """

    strategy: Literal["start", "random", "loudest"] = "random"
    num_tries: int = 8
    lufs_cutoff: float = -40.0
    # Annotated as a union even though argbind binds it from YAML: the registry
    # handles the string case, and claiming `str` while accepting a callable was
    # a lie.
    search: Union[str, Callable] = "uniform"
    on_failure: Literal["keep", "skip", "raise"] = "keep"

    #: Parameters that only mean anything under ``strategy="loudest"``.
    _LOUDEST_ONLY = ("num_tries", "lufs_cutoff", "search", "on_failure")

    def __post_init__(self):
        strategies = ("start", "random", "loudest")
        if self.strategy not in strategies:
            raise ValueError(
                f"strategy must be one of {strategies}, got {self.strategy!r}."
            )
        if self.on_failure not in ("keep", "skip", "raise"):
            raise ValueError(
                f"on_failure must be 'keep', 'skip' or 'raise', got "
                f"{self.on_failure!r}."
            )
        if self.num_tries < 1:
            raise ValueError(f"num_tries must be >= 1, got {self.num_tries}.")

        # Setting a loudest-only knob under another strategy silently did nothing
        # before, which is exactly how "I set lufs_cutoff and nothing happened"
        # goes unnoticed.
        if self.strategy != "loudest":
            defaults = {
                f.name: f.default
                for f in dataclasses.fields(self)
                if f.name in self._LOUDEST_ONLY
            }
            ignored = [
                name
                for name, default in defaults.items()
                if getattr(self, name) != default
            ]
            if ignored:
                raise ValueError(
                    f"{', '.join(sorted(ignored))} only appl"
                    f"{'ies' if len(ignored) == 1 else 'y'} to "
                    f"strategy='loudest', but strategy is {self.strategy!r}."
                )

        # Resolve eagerly so a bad name fails when the config is built, not deep
        # inside a data worker several minutes into training.
        _resolve_search_function(self.search)

    @property
    def resolved_search(self) -> Callable:
        """The ``search`` spec as a callable."""
        return _resolve_search_function(self.search)


# Fixed width (in Unicode code points) for filepath/source strings encoded into
# extras arrays, so they can be batched and stored in fixed-width int32 arrays.
# Strings longer than this raise in ``_encode_string`` rather than being
# truncated, to avoid silently corrupting paths.
_str_max_length = 1024


def _is_integer_scalar(key) -> bool:
    """Whether *key* indexes a single batch item.

    True for the builtin ``int``, for NumPy integer scalars (``np.argmax`` and
    friends), and for 0-d integer arrays (NumPy or JAX) -- all of which would
    otherwise scalar-index every leaf and silently drop the batch axis. ``bool``
    is excluded even though it subclasses ``int``, so ``tree[True]`` is a mask
    rather than item 1.
    """
    if isinstance(key, bool):
        return False
    if isinstance(key, (int, np.integer)):
        return True
    if isinstance(key, (np.ndarray, jax.Array)):
        return key.ndim == 0 and np.issubdtype(key.dtype, np.integer)
    return False


def _is_string_list(x) -> bool:
    """Whether *x* is a non-empty list of strings.

    Such lists are supported ``extras`` leaves (see ``tree_writer``), so the
    batch-axis operations treat them as a single leaf and index them
    element-wise rather than letting ``jax.tree_util`` descend into the list and
    slice each string's characters.
    """
    return isinstance(x, list) and bool(x) and all(isinstance(s, str) for s in x)


def _is_string_leaf(x) -> bool:
    """Whether *x* is a string ``extras`` leaf, in any of its batch forms.

    The forms track the tree's leading axes (see ``tree_writer``): a bare
    ``str`` is a batch of 1, a list of strings holds one per batch item, a
    list of such lists is the mini-batched (rank-4) nesting, and an empty
    list is the ``batch_size == 0`` tree (e.g. after a :meth:`AudioTree.filter`
    that kept nothing). Every batch-axis operation uses this as its
    ``is_leaf`` so ``jax.tree_util`` never descends into the list and slices
    the strings themselves.
    """
    if isinstance(x, str):
        return True
    if not isinstance(x, list):
        return False
    if not x:
        return True
    return all(isinstance(s, str) for s in x) or all(_is_string_list(s) for s in x)


def _as_string_list(x) -> list:
    """Normalize a string leaf to list form (a bare ``str`` is a batch of 1)."""
    return [x] if isinstance(x, str) else x


def _index_string_list(strings: list, key) -> list:
    """Index a string list along the batch axis with any ``__getitem__`` key.

    Mirrors NumPy's batch-axis semantics for the sub-batch key forms: slices,
    integer sequences/arrays (negative indices included), and boolean masks.
    On a mini-batched tree the elements are themselves lists, which select
    whole mini-batches exactly like rows of an array.
    """
    if isinstance(key, slice):
        return strings[key]
    key = np.asarray(key)
    if key.dtype == np.bool_:
        if key.shape != (len(strings),):
            raise IndexError(
                f"boolean mask of shape {tuple(key.shape)} does not match "
                f"string leaf of length {len(strings)}."
            )
        return [s for s, keep in zip(strings, key) if keep]
    if not np.issubdtype(key.dtype, np.integer):
        raise IndexError(
            f"string leaves can only be indexed with slices, integers, or "
            f"boolean masks, got key dtype {key.dtype}."
        )
    return [strings[int(i)] for i in key]


def _index_batch_axis(x, key):
    """Index a leaf along the batch axis, matching :meth:`AudioTree.__getitem__`.

    Arrays and string leaves are indexed with *key*; every other leaf
    (scalars, ...) is passed through unchanged. String leaves always come back
    in list form, so indexing normalizes a bare ``str`` to the equivalent
    one-item list.
    """
    if isinstance(x, (np.ndarray, jax.Array)):
        return x[key]
    if _is_string_leaf(x):
        return _index_string_list(_as_string_list(x), key)
    return x


_XLA_PLATFORMS = ("cpu", "gpu", "tpu")


def _resolve_device(device) -> tuple[Optional["jax.Device"], Optional[str]]:
    """Normalize a ``device=`` argument to ``(jax.Device | None, platform | None)``.

    Accepts an XLA platform name (``"cpu"`` / ``"gpu"`` / ``"tpu"``), a concrete
    :class:`jax.Device`, or ``None`` for "wherever the data already is".
    """
    if device is None:
        return None, None
    if isinstance(device, str):
        if device not in _XLA_PLATFORMS:
            raise ValueError(
                f"device must be None, one of {_XLA_PLATFORMS}, or a jax.Device, "
                f"got {device!r}."
            )
        return jax.devices(device)[0], device
    if isinstance(device, jax.Device):
        return device, device.platform
    raise TypeError(
        f"device must be None, one of {_XLA_PLATFORMS}, or a jax.Device, got "
        f"{type(device).__name__}."
    )


def _resolve_lufs_engine(
    engine: Optional[str], platform: Optional[str], input_is_numpy: bool
) -> str:
    """Pick the loudness kernel from an ``engine=``/``device=`` pair.

    ``engine=None`` follows the waveform's own array library, except on a
    non-CPU device where only the JAX kernel exists. The NumPy engine is the
    exact BS.1770 IIR meter and is CPU-only, so pairing it with an accelerator
    is a contradiction rather than a silent downgrade.
    """
    if engine is None:
        on_accelerator = platform is not None and platform != "cpu"
        return "jax" if (on_accelerator or not input_is_numpy) else "numpy"
    if engine not in ("numpy", "jax"):
        raise ValueError(f"engine must be None, 'numpy', or 'jax', got {engine!r}.")
    if engine == "numpy" and platform not in (None, "cpu"):
        raise ValueError(
            f"engine='numpy' is the exact BS.1770 IIR meter and runs on the CPU "
            f"only, so it cannot be combined with device={platform!r}. Use "
            f"engine='jax' for the accelerator, or drop device= to measure on "
            f"the CPU."
        )
    return engine


def _require_batched_rank(waveform, operation: str) -> None:
    """Reject a mini-batched waveform for an *operation* that needs one batch axis.

    :meth:`AudioTree.reshape_mini_batches` gives a tree *two* leading axes
    ``(Mini, Batch, Channels, Samples)``. Most operations are written over the
    trailing axes and do not care, but the ones that mean "per batch item" have
    no rank-4 reading -- so they say which rank they got rather than failing
    somewhere downstream with a shape nobody can trace back.
    """
    if waveform is None or waveform.ndim == 3:
        return
    raise ValueError(
        f"{operation} needs a rank-3 (Batch, Channels, Samples) waveform, got "
        f"rank {waveform.ndim} {tuple(waveform.shape)}. A mini-batched tree "
        f"(rank 4, from reshape_mini_batches) has two leading axes; call "
        f"flatten_mini_batches() first."
    )


def _leading_axis_size(*candidates) -> Optional[int]:
    """Batch size implied by the first non-``None`` array in *candidates*.

    ``None`` when nothing was given (a tree with neither audio nor tokens has no
    batch axis to broadcast provenance over).
    """
    for value in candidates:
        if value is not None:
            return value.shape[0]
    return None


#: Fields whose value describes one particular waveform, at one particular
#: length, sample rate, channel count and level. Any operation that changes the
#: audio in one of those ways must clear them, or a downstream consumer reads
#: loudness or codec tokens that describe the *previous* audio -- silently, and
#: with nothing downstream able to detect it. See
#: :meth:`AudioTree._invalidate_derived`.
DERIVED_FIELDS: tuple = ("lufs", "lufs_windows", "codes", "latents")

#: ``extras`` keys that are derived in the same way. ``"codec_scale"`` is
#: written by ``encode_with_codec`` alongside ``codes`` and is meaningless once
#: those tokens are gone.
DERIVED_EXTRAS_KEYS: tuple = ("codec_scale",)


class _AudioTreeFields(TypedDict, total=False):
    """The ``AudioTree`` fields, as keyword arguments to :meth:`AudioTree.replace`.

    ``flax.struct.dataclass`` gives every tree an untyped ``replace(**updates)``,
    which leaves the library's primary mutation API invisible to type checkers
    and editors despite ``py.typed``. Unpacking this keeps the field names and
    their types in the signature. ``tests/test_core.py`` asserts it stays in
    sync with the dataclass fields.
    """

    waveform: ArrayLike | None
    sample_rate: int
    lufs: ArrayLike | None
    lufs_windows: ArrayLike | None
    pitch: ArrayLike | None
    velocity: ArrayLike | None
    note_duration: ArrayLike | None
    codes: ArrayLike | None
    latents: ArrayLike | None
    extras: dict


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and extras.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    The constructor stores its arguments verbatim -- it neither reshapes the waveform nor encodes
    provenance. Use :meth:`create` (which accepts a ``(Samples,)`` or ``(Channels, Samples)``
    waveform and ``filepath`` / ``source`` strings) unless you already hold batched arrays.

    Args:
        waveform (np.ndarray or jax.Array): Audio waveform data shaped ``(Batch, Channels, Samples)``,
            or ``None`` for token-only trees (``codes`` / ``latents`` without audio).
        sample_rate (int): Sample rate of ``waveform``, such as 44100 Hz.
        lufs (np.ndarray or jax.Array, optional): Integrated loudness of the audio waveform in LUFS, shaped ``(Batch,)``.
            You may not need to set this when initializing. Instead, use ``replace_lufs()`` to create a new AudioTree with
            ``lufs`` (and ``lufs_windows``) calculated.
        lufs_windows (np.ndarray or jax.Array, optional): Per-window integrated loudness in LUFS, shaped
            ``(Batch, Windows)`` — one value per non-overlapping analysis window. Populated alongside ``lufs`` by
            ``replace_lufs()``.
        pitch (np.ndarray or jax.Array, optional): The MIDI pitch where 60 is middle C. The shape is ``(Batch,)``.
        velocity (np.ndarray or jax.Array, optional): The MIDI velocity between 0 and 127. The shape is ``(Batch,)``.
        note_duration (np.ndarray or jax.Array, optional): A note duration in units of your choice.
            The value is not necessarily the same as the duration of the audio data. The shape is ``(Batch,)``.
        codes (np.ndarray or jax.Array, optional): The neural audio codec tokens for the audio.
        latents (np.ndarray or jax.Array, optional): The latent representations of the audio.
        extras (dict): Any extra per-item data can be placed here. Provenance lives here too, under the
            ``"filepath"`` and ``"source"`` keys (encoded arrays, read back via the :attr:`filepath`
            and :attr:`source` properties); pass ``filepath=`` / ``source=`` to :meth:`create` rather
            than encoding them by hand.

    Example:
        >>> audio = AudioTree.create(jnp.zeros((2, 44100)), 44100)  # stereo, 1 s
        >>> audio.waveform.shape
        (1, 2, 44100)
        >>> audio.sample_rate
        44100

    Note:
        Every consumer that enumerates these fields derives its list from
        ``dataclasses.fields`` (see ``PYTREE_FIELDS`` below), so adding a field
        here is picked up automatically.
    """

    waveform: ArrayLike | None
    sample_rate: int = struct.field(pytree_node=False)
    lufs: ArrayLike | None = None
    lufs_windows: ArrayLike | None = None
    pitch: ArrayLike | None = None
    velocity: ArrayLike | None = None
    note_duration: ArrayLike | None = None
    codes: ArrayLike | None = None
    latents: ArrayLike | None = None
    extras: dict = struct.field(pytree_node=True, default_factory=dict)

    def replace(self, **updates: Unpack[_AudioTreeFields]) -> Self:
        """Return a new ``AudioTree`` with the given fields replaced.

        Args:
            **updates: Any subset of the fields above; everything else is carried
                over from ``self``, which is never mutated.

        Returns:
            AudioTree: A copy of ``self`` with ``updates`` applied.

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)
            >>> quiet = audio.replace(waveform=audio.waveform * 0.5)
            >>> quiet.sample_rate
            44100

        Note:
            ``flax.struct.dataclass`` installs its own ``replace`` over this one
            at class-creation time. That implementation is this one verbatim
            (``dataclasses.replace``) minus the typing, which is the whole
            reason to spell it out here.
        """
        return dataclasses.replace(self, **updates)

    #: Alias holding the typed ``replace`` above, so it can be put back after
    #: ``flax.struct.dataclass`` overwrites the attribute (see below the class).
    _typed_replace = replace

    @classmethod
    def create(
        cls,
        waveform: ArrayLike | None,
        sample_rate: int,
        *,
        lufs: ArrayLike | None = None,
        lufs_windows: ArrayLike | None = None,
        pitch: ArrayLike | None = None,
        velocity: ArrayLike | None = None,
        note_duration: ArrayLike | None = None,
        codes: ArrayLike | None = None,
        latents: ArrayLike | None = None,
        extras: dict | None = None,
        filepath: Union[str, Path, List[Union[str, Path]]] | None = None,
        source: Union[str, List[str]] | None = None,
    ) -> Self:
        """Create an ``AudioTree``, normalizing the waveform to ``(Batch, Channels, Samples)``.

        A bare ``(Samples,)`` or ``(Channels, Samples)`` waveform gains the missing leading axes, so
        you don't have to reshape by hand. ``filepath`` and ``source`` are encoded into ``extras``.

        Args:
            waveform: Audio of shape ``(Samples)``, ``(Channels, Samples)``, or
                ``(Batch, Channels, Samples)``, or ``None`` for token-only trees
                (e.g. ``codes`` / ``latents`` without audio).
            sample_rate: Sample rate of ``waveform`` in Hz (e.g. 44100).
            lufs: Optional precomputed integrated loudness (LUFS); usually left ``None`` and filled by
                :meth:`replace_lufs`.
            lufs_windows: Optional precomputed per-window loudness ``(Batch, Windows)``; usually left ``None``
                and filled by :meth:`replace_lufs`.
            pitch: Optional MIDI pitch ``(Batch,)`` (60 = middle C).
            velocity: Optional MIDI velocity ``(Batch,)`` in ``[0, 127]``.
            note_duration: Optional per-note duration ``(Batch,)`` (not the audio duration).
            codes: Optional neural-codec tokens.
            latents: Optional latent representations.
            extras: Optional extras dict of additional per-item leaves (copied, not mutated).
            filepath: Optional path(s) for the batch; encoded into ``extras["filepath"]`` and read
                back via the :attr:`filepath` property. Pass a single path to tag the whole batch
                (it is repeated for every item), or a list with one path per batch item.
            source: Optional source-group name(s) (e.g. ``"music"``), encoded into
                ``extras["source"]`` and read back via the :attr:`source` property. Pass a
                single string to tag the whole batch, or a list with one name per batch item
                (unlike :meth:`from_file`, which only accepts a single string).

        Returns:
            AudioTree: A new ``AudioTree`` whose waveform is ``(Batch, Channels, Samples)``.

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)  # 1 s mono
            >>> audio.waveform.shape
            (1, 1, 44100)
            >>> audio.sample_rate
            44100
        """
        # Handle audio dimensionality - ensure it's (Batch, Channels, Samples).
        # ``waveform`` may be None for token-only trees (codes/latents only).
        if waveform is not None:
            if waveform.ndim == 1:
                waveform = waveform[None, None, :]  # Add batch and channel dimension
            elif waveform.ndim == 2:
                waveform = waveform[None, :, :]  # Add batch dimension
            elif waveform.ndim > 3:
                # ``filepath`` / ``source`` are encoded one row per batch item,
                # broadcast over a *single* leading axis. A mini-batched
                # waveform has two, so the provenance would silently come out
                # one row per mini-batch -- fewer strings than items.
                raise ValueError(
                    f"AudioTree.create normalizes to (Batch, Channels, Samples) "
                    f"and got a rank-{waveform.ndim} waveform "
                    f"{tuple(waveform.shape)}. Create the tree at rank 3 and "
                    f"call reshape_mini_batches() to add a mini-batch axis."
                )

        # Handle extras and filepath
        if extras is None:
            extras = {}
        else:
            extras = extras.copy()  # Don't modify the original dict

        # A single string tags the whole batch: encode once and repeat, so that
        # ``tree[2].filepath`` and any filtered sub-batch keep their provenance
        # instead of only item 0 carrying it.
        batch_size = _leading_axis_size(waveform, codes, latents)

        def _encode_provenance(field, value):
            encoded = cls._encode_filepaths(value)
            if isinstance(value, (str, Path)):
                if batch_size is not None:
                    encoded = np.repeat(encoded, batch_size, axis=0)
            elif batch_size is not None and len(encoded) != batch_size:
                # A list encodes one row per item, so a wrong-length list would
                # silently misalign provenance with the batch -- fewer (or more)
                # strings than items. Same failure the rank-4 guard rejects.
                raise ValueError(
                    f"AudioTree.create got {len(encoded)} {field} for a batch of "
                    f"{batch_size}. Pass a single value to tag the whole batch, "
                    f"or a list with one per batch item."
                )
            return encoded

        if filepath is not None:
            extras["filepath"] = _encode_provenance("filepath", filepath)

        if source is not None:
            extras["source"] = _encode_provenance("source", source)

        return cls(
            waveform=waveform,
            sample_rate=sample_rate,
            lufs=lufs,
            lufs_windows=lufs_windows,
            pitch=pitch,
            velocity=velocity,
            note_duration=note_duration,
            codes=codes,
            latents=latents,
            extras=extras,
        )

    def _invalidate_derived(
        self, *, keep: Sequence[str] = (), **updates: Unpack[_AudioTreeFields]
    ) -> Self:
        """Apply *updates* and drop every field derived from the old waveform.

        The single place that decides what "the audio changed" invalidates.
        ``lufs``, ``lufs_windows``, ``codes``, ``latents`` and
        ``extras["codec_scale"]`` (:data:`DERIVED_FIELDS` /
        :data:`DERIVED_EXTRAS_KEYS`) all describe one specific waveform; every
        length-, rate-, channel- or energy-changing operation routes through
        here so none of them can be forgotten one method at a time.

        Args:
            keep: Derived field names the caller has itself kept correct, and so
                does not want cleared -- e.g. :meth:`normalize_lufs` shifts
                ``lufs``/``lufs_windows`` by the gain it applied rather than
                discarding them. Keeping ``"codes"`` also keeps
                ``extras["codec_scale"]``, which is only meaningful with them.
            **updates: Forwarded to :meth:`replace`, and applied *after* the
                invalidation so a caller can supply a fresh value for a derived
                field.

        Returns:
            AudioTree: A copy with the stale derived fields set to ``None``,
            stale derived ``extras`` keys removed, and ``updates`` applied.

        Raises:
            ValueError: If ``keep`` names something that is not a derived field.
        """
        unknown = sorted(set(keep) - set(DERIVED_FIELDS))
        if unknown:
            raise ValueError(
                f"keep must name derived fields {DERIVED_FIELDS}, got {unknown}."
            )
        cleared: dict = {name: None for name in DERIVED_FIELDS if name not in keep}

        if "codes" not in keep:
            extras = updates.get("extras", self.extras)
            if any(key in extras for key in DERIVED_EXTRAS_KEYS):
                updates["extras"] = {
                    key: value
                    for key, value in extras.items()
                    if key not in DERIVED_EXTRAS_KEYS
                }

        return self.replace(**(cleared | dict(updates)))

    def replace_extras(self, **kwargs) -> Self:
        """Return a new ``AudioTree`` with ``kwargs`` merged into ``extras``.

        Syntactic sugar for ``self.replace(extras={**self.extras, **kwargs})``.
        Keys in ``kwargs`` overwrite existing ``extras`` keys with the same name;
        all other keys are kept. Neither the original tree nor its ``extras``
        dict is mutated. For a key that isn't a valid Python identifier, use the
        ``self.replace(extras=...)`` form directly.

        Args:
            **kwargs: Entries to merge into ``extras``. Values should be arrays
                (or pytrees of arrays) so the result stays batchable and jittable.

        Returns:
            AudioTree: A new ``AudioTree`` with the merged ``extras``.

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)
            >>> audio = audio.replace_extras(tempo=np.array([120.0]))
            >>> audio.extras["tempo"]
            array([120.])
        """
        return self.replace(extras=self.extras | kwargs)

    def replace_lufs(
        self,
        lufs_window_sec: float = 0.4,
        lufs_hop_sec: float | None = None,
        *,
        device: Optional[Union[Literal["cpu", "gpu", "tpu"], "jax.Device"]] = None,
        engine: Optional[Literal["numpy", "jax"]] = None,
    ) -> Self:
        """Compute and set the integrated and per-window loudness (LUFS).

        Returns a new AudioTree with both ``lufs`` and ``lufs_windows`` populated:

        * ``lufs`` — the **gated** integrated loudness of each batch item, per the
          ITU-R BS.1770-4 standard (the standard "program loudness"). Measured on
          the CPU with the ``loudness`` library for NumPy waveforms, or a vmapped
          ``jaxloudnorm`` meter for JAX waveforms.
        * ``lufs_windows`` — the **ungated** K-weighted loudness of each window (a
          loudness-over-time curve). The signal is K-weighted once and each window
          reports the K-weighted mean-square of its samples in LUFS, so windows are
          directly comparable and a fully silent window is ``-inf``. NumPy uses
          exact IIR biquads (``scipy``); JAX uses ``jaxloudnorm``'s FIR-approximated
          filters on the accelerator.

        The two engines are not bit-identical.

        *Where* the measurement runs and *which* kernel runs are separate
        choices, so ``device=`` and ``engine=`` are separate arguments: asking
        for the exact IIR meter does not commit you to a device, and moving the
        work to an accelerator does not silently swap in the FIR approximation.

        Args:
            lufs_window_sec: Length in seconds of each ``lufs_windows`` window.
                Must be at least 0.4s (the EBU momentary integration time).
            lufs_hop_sec: Step in seconds between window starts. Defaults to
                ``lufs_window_sec`` (non-overlapping windows); a smaller value
                overlaps them, but it must still span at least one sample at
                the tree's sample rate. The trailing partial window is dropped,
                so an excerpt shorter than one window yields an empty
                ``lufs_windows``.
            device: *Where* to compute — an XLA platform name (``"cpu"`` /
                ``"gpu"`` / ``"tpu"``, mirroring ``jax.jit``'s ``backend``) or a
                :class:`jax.Device`. ``None`` (default) leaves the waveform where
                it is. ``device="gpu"`` is much faster for a large batch, since
                the NumPy ``lufs`` path measures one item at a time.
            engine: *Which* kernel to run. ``"numpy"`` is the exact
                ITU-R BS.1770 IIR meter (the ``loudness`` C++ library + ``scipy``
                biquads); it is CPU-only, so it cannot be combined with a
                non-CPU ``device``. ``"jax"`` is the vmapped ``jaxloudnorm``
                kernel with FIR-approximated K-weighting, and runs on whatever
                ``device`` says. ``None`` (default) follows the waveform's own
                array library — NumPy waveform to ``"numpy"``, JAX waveform to
                ``"jax"`` — except that a non-CPU ``device`` implies ``"jax"``,
                the only engine that can run there.

                The **returned** ``lufs`` / ``lufs_windows`` always match the
                waveform's array type (a NumPy waveform yields NumPy loudness
                whatever the engine), so you never need a manual
                ``jax.device_put`` / ``jax.device_get`` round-trip.

        Returns:
            AudioTree with ``lufs`` shaped ``(*batch,)`` and ``lufs_windows`` shaped
            ``(*batch, num_windows)``.

        Note:
            **Channel Limitations**: Supports up to 5 channels:

            - Mono (1 channel): Single channel
            - Stereo (2 channels): [Left, Right]
            - 5.0/5.1 Surround (5 channels): [Left, Right, Center, Left Surround, Right Surround]

            Will raise ValueError if audio has more than 5 channels.
        """
        if lufs_window_sec < 0.4:
            raise ValueError(
                f"lufs_window_sec must be at least 0.4s (the EBU momentary "
                f"integration time), got {lufs_window_sec}."
            )
        if lufs_hop_sec is None:
            lufs_hop_sec = lufs_window_sec
        if lufs_hop_sec <= 0:
            raise ValueError(f"lufs_hop_sec must be positive, got {lufs_hop_sec}.")
        # Both engines share the 5-channel BS.1770 layouts; the NumPy meter
        # would otherwise silently compute a value for 6+ channels while the
        # JAX one raised.
        if self.num_channels > 5:
            raise ValueError(
                f"Audio must have five channels or less (the BS.1770 "
                f"mono/stereo/5.x layouts), got {self.num_channels}."
            )
        # ``engine`` chooses the kernel and ``device`` chooses where it runs; the
        # output stays in the waveform's own array library so the tree does not go
        # heterogeneous. Resolved up front so a bad pairing fails before any work.
        jax_device, platform = _resolve_device(device)
        input_is_numpy = isinstance(self.waveform, np.ndarray)
        engine = _resolve_lufs_engine(engine, platform, input_is_numpy)

        # Flatten any leading axes (e.g. after reshape_mini_batches) to a single
        # batch axis, then restore them on the computed loudness.
        leading_shape = self.waveform.shape[:-2]
        waveform = self.waveform.reshape(-1, *self.waveform.shape[-2:])
        ws = _window_samples(lufs_window_sec, self.sample_rate)
        hs = _window_samples(lufs_hop_sec, self.sample_rate)
        if hs < 1:
            # A positive hop can still round to 0 samples, which would step
            # the window nowhere (and divide by zero counting windows).
            raise ValueError(
                f"lufs_hop_sec={lufs_hop_sec} spans {hs} samples at "
                f"{self.sample_rate} Hz; the hop must span at least 1 sample."
            )
        num_windows = _windowed_num_windows(waveform.shape[-1], ws, hs)

        if engine == "jax":
            compute_waveform = (
                jnp.asarray(waveform)
                if jax_device is None
                else jax.device_put(waveform, jax_device)
            )
            # ``zeros`` is left to the default so the FIR tap count scales with
            # the sample rate; a fixed 512 cannot realize the 38 Hz high-pass
            # much above 16 kHz.
            lufs_array = _jit_integrated_loudness(compute_waveform, self.sample_rate)
            if num_windows == 0:
                lufs_windows_array = jnp.zeros(
                    (compute_waveform.shape[0], 0), dtype=jnp.float32
                )
            else:
                lufs_windows_array = _jit_windowed_loudness(
                    compute_waveform,
                    self.sample_rate,
                    lufs_window_sec,
                    lufs_hop_sec,
                )
        else:
            compute_waveform = np.asarray(waveform)
            lufs_array = _numpy_integrated_lufs(compute_waveform, self.sample_rate)
            if num_windows == 0:
                lufs_windows_array = np.zeros(
                    (compute_waveform.shape[0], 0), dtype=np.float32
                )
            else:
                lufs_windows_array = _numpy_windowed_lufs(
                    compute_waveform,
                    self.sample_rate,
                    lufs_window_sec,
                    lufs_hop_sec,
                )

        # Coerce results back to the waveform's array library (a no-op when the
        # kernel already ran there; a device transfer when ``engine`` selected the
        # other one).
        out_np = np if input_is_numpy else jnp
        lufs_array = out_np.asarray(lufs_array)
        lufs_windows_array = out_np.asarray(lufs_windows_array)

        return self.replace(
            lufs=lufs_array.reshape(leading_shape),
            lufs_windows=lufs_windows_array.reshape(*leading_shape, num_windows),
        )

    def normalize_lufs(
        self,
        target_lufs: float,
        *,
        max_gain_db: Optional[float] = None,
        device: Optional[Union[Literal["cpu", "gpu", "tpu"], "jax.Device"]] = None,
        engine: Optional[Literal["numpy", "jax"]] = None,
    ) -> Self:
        """Normalize audio to a target LUFS level.

        Computes the current loudness (if not already set), then scales the audio
        to achieve the target LUFS. The returned AudioTree has updated
        ``waveform``, ``lufs``, and ``lufs_windows`` fields (a constant gain shifts
        every window's LUFS by the same amount). Changing the level invalidates
        ``codes``, ``latents`` and ``extras["codec_scale"]``, which describe the
        audio at its previous level.

        Items whose loudness is not finite are **passed through unscaled**. Digital
        silence, and any excerpt below the BS.1770 absolute gate, measure ``-inf``
        LUFS, for which no gain reaches the target; scaling by the implied ``+inf``
        would produce an all-``NaN`` waveform. Their ``lufs`` stays ``-inf``, so a
        silent item is still identifiable afterwards.

        Args:
            target_lufs: Target loudness in LUFS (e.g., -18.0 for broadcast standard).
            max_gain_db: Optional ceiling on the applied gain, so a very quiet (but
                still measurable) item is not amplified without bound. ``None`` (the
                default) applies whatever gain the target implies; a capped item
                lands at ``lufs + max_gain_db`` rather than at ``target_lufs``.
            device: Where to compute the loudness when it is not already set,
                forwarded to :meth:`replace_lufs` (see there). Ignored when
                ``lufs`` is already populated.
            engine: Which loudness kernel to use when it is not already set,
                forwarded to :meth:`replace_lufs` (see there). Ignored when
                ``lufs`` is already populated.

        Returns:
            AudioTree with audio scaled to target LUFS and loudness updated.

        Example:
            >>> t = jnp.arange(44100) / 44100  # 1 s at 44.1 kHz
            >>> tree = AudioTree.create(0.5 * jnp.sin(2 * jnp.pi * 1000 * t), 44100)
            >>> normalized = tree.normalize_lufs(-18.0)
            >>> float(normalized.lufs[0])  # now at the target LUFS
            -18.0

            Silence is left alone instead of becoming ``NaN``:

            >>> silent = AudioTree.create(jnp.zeros((1, 1, 44100)), 44100)
            >>> out = silent.normalize_lufs(-18.0)
            >>> bool(jnp.all(out.waveform == 0.0)), float(out.lufs[0])
            (True, -inf)
        """
        # Ensure loudness is computed
        if self.lufs is None:
            tree = self.replace_lufs(device=device, engine=engine)
        else:
            tree = self

        numpy = np if isinstance(self.waveform, np.ndarray) else jnp
        gain_db = safe_gain_db(tree.lufs, target_lufs, max_gain_db, xp=numpy)
        linear_gain = numpy.power(10.0, gain_db / 20.0)
        # Cast to audio dtype to avoid float64 promotion
        linear_gain = linear_gain.astype(tree.waveform.dtype)
        # Expand gain for broadcasting: [..., 1, 1] over the channel and
        # sample axes, for any number of leading batch axes.
        linear_gain = linear_gain[..., None, None]
        scaled_waveform = tree.waveform * linear_gain

        # A constant gain shifts every window's LUFS by the same dB as the
        # integrated value, so both move by ``gain_db`` (kept aligned rather than
        # left stale). Adding the gain rather than assigning ``target_lufs`` is
        # what keeps a skipped (non-finite) or capped item honest. ``codes`` and
        # ``latents`` have no such closed form, so they are invalidated.
        return tree._invalidate_derived(
            keep=("lufs", "lufs_windows"),
            waveform=scaled_waveform,
            lufs=shift_lufs(tree.lufs, gain_db, xp=numpy),
            lufs_windows=shift_lufs_windows(tree.lufs_windows, gain_db, xp=numpy),
        )

    @staticmethod
    def _encode_string(s: str) -> np.ndarray:
        """Encode a single filepath *s* to an array of Unicode code points.

        The returned array is shaped ``(1, _str_max_length)`` so that multiple
        rows (filepath) can be concatenated along *axis=0*.

        Raises:
            ValueError: If *s* is longer than ``_str_max_length`` code points.
                The string is not truncated, to avoid silently corrupting paths.
        """
        s = str(s)
        if len(s) > _str_max_length:
            raise ValueError(
                f"String of length {len(s)} exceeds the extras encoding limit "
                f"of {_str_max_length} characters and would be truncated: {s!r}. "
                "Increase audiotree.core._str_max_length to store longer strings."
            )
        encoded = [ord(char) for char in s]
        encoded += [0] * (_str_max_length - len(encoded))
        return np.array([encoded], dtype=np.int32)  # [1, _str_max_length]

    @classmethod
    def _encode_filepaths(
        cls, paths: Union[str, Path, List[Union[str, Path]]]
    ) -> np.ndarray:
        """Vectorized helper to encode one or more *paths*.

        Args:
            paths: A single filepath or an iterable of filepath.

        Returns
        -------
        np.ndarray
            An array shaped ``(N, _str_max_length)`` where *N* is the number of
            paths provided.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        encoded_rows = [cls._encode_string(p)[0] for p in paths]
        return np.stack(encoded_rows, axis=0)

    @staticmethod
    def _decode_string(encoded_array: np.ndarray) -> str:
        """Decode a single encoded filepath back to *str*."""
        decoded = "".join(chr(int(val)) for val in encoded_array if val != 0)
        return decoded

    @classmethod
    def _decode_strings(cls, encoded: ArrayLike, key: str) -> list:
        """Decode a provenance array, nesting to match its leading axes.

        The last axis is the encoded string, so ``(Batch, _str_max_length)``
        decodes to a flat list of ``Batch`` strings and a mini-batched tree's
        ``(Mini, Batch, _str_max_length)`` to ``Mini`` lists of ``Batch``
        strings. Recursing rather than assuming the leading axis is the batch is
        what keeps :attr:`filepath` and :attr:`source` working at rank 4 --
        indexing one row of the latter used to hand ``_decode_string`` a whole
        sub-array, which failed with NumPy's "truth value of an array is
        ambiguous" and named neither the field nor the rank.

        Raises:
            ValueError: If *encoded* has no leading axis at all (a bare
                ``(_str_max_length,)`` row), which cannot say how many items it
                describes.
        """
        if encoded.ndim < 2:
            raise ValueError(
                f"extras[{key!r}] must have a leading batch axis, i.e. shape "
                f"(Batch, {_str_max_length}), got rank {encoded.ndim} "
                f"{tuple(encoded.shape)}. Encode it with "
                f"AudioTree.create({key}=...) rather than by hand."
            )
        if encoded.ndim == 2:
            return [cls._decode_string(row) for row in encoded]
        return [cls._decode_strings(sub, key) for sub in encoded]

    @property
    def filepath(self) -> Union[List[str], List[list]]:
        """Return the decoded filepaths stored in ``extras['filepath']``.

        One string per batch item. A mini-batched tree (rank 4, from
        :meth:`reshape_mini_batches`) has two leading axes, so it returns one
        list per mini-batch — the nesting always matches the tree's leading
        axes. Call :meth:`flatten_mini_batches` first for a flat list.

        An empty list is returned if the AudioTree does not contain any filepath
        extras.
        """
        if "filepath" not in self.extras:
            return []
        return self._decode_strings(self.extras["filepath"], "filepath")

    @property
    def source(self) -> Union[List[str], List[list]]:
        """Return the decoded source names stored in ``extras['source']``.

        Source names indicate which data source group each item in the batch came from.
        For example, if an AudioDataSimpleSource was created with
        ``sources={"music": [...], "speech": [...]}``, this property might return
        ``["music", "music", "speech", "music"]`` for a batch of 4 items. Like
        :attr:`filepath`, a mini-batched (rank-4) tree nests one list per
        mini-batch.

        An empty list is returned if the AudioTree does not contain any source
        extras.
        """
        if "source" not in self.extras:
            return []
        return self._decode_strings(self.extras["source"], "source")

    @property
    def samples(self) -> int:
        """Return the number of samples in the ``waveform`` (its last dimension)."""
        return self.waveform.shape[-1]

    @property
    def backend(self) -> str:
        """Which array library this tree's arrays belong to.

        One of ``"numpy"``, ``"jax"``, or ``"mixed"``. A tree is not required to
        be homogeneous and quietly stops being so more often than you would
        expect: applying a NumPy-namespace transform to a JAX tree converts the
        fields it touches, so ``audiotree.transforms.trim`` on a JAX tree hands
        back a NumPy ``waveform``. Nothing is wrong with that until something
        downstream assumes otherwise -- ``jax.jit`` on a NumPy leaf silently
        re-uploads it every call -- which is what this property is for.

        ``"mixed"`` is reported rather than raised, so it is safe to log.
        Convert with :func:`jax.device_put` / :func:`jax.device_get`.

        Examples:
            >>> import numpy as np, jax.numpy as jnp
            >>> from audiotree import AudioTree
            >>> AudioTree.create(np.zeros((1, 1, 8)), 16000).backend
            'numpy'
            >>> AudioTree.create(jnp.zeros((1, 1, 8)), 16000).backend
            'jax'
        """
        kinds = {
            "numpy" if isinstance(leaf, np.ndarray) else "jax"
            for leaf in self._array_leaves()
        }
        if not kinds:
            return "numpy"  # nothing to disagree about
        return kinds.pop() if len(kinds) == 1 else "mixed"

    @property
    def device(self) -> Optional["jax.Device"]:
        """The device this tree's arrays live on, or ``None`` on NumPy.

        ``jax.device_put`` and ``jax.device_get`` already move a tree; this is
        the missing half -- asking where it currently is without reaching into a
        leaf and hoping the rest agree.

        Returns:
            The :class:`jax.Device` shared by every array leaf, or ``None`` if
            the tree is NumPy-backed (host memory, no JAX device).

        Raises:
            ValueError: If the leaves do not agree on one device -- a mixed
                NumPy/JAX tree, or JAX arrays committed to different devices.
                Unlike :attr:`backend` this raises, because there is no honest
                single answer and returning one of them would be a guess.
        """
        leaves = list(self._array_leaves())
        if not leaves:
            return None
        devices = {
            None if isinstance(leaf, np.ndarray) else leaf.device for leaf in leaves
        }
        if len(devices) == 1:
            return devices.pop()
        raise ValueError(
            f"AudioTree spans more than one device: {sorted(map(str, devices))}. "
            f"Move it with jax.device_put(tree, device) or jax.device_get(tree) "
            f"before asking."
        )

    def _array_leaves(self):
        """Every non-``None`` array field, waveform first. Skips ``extras``."""
        for name in ARRAY_FIELDS:
            value = getattr(self, name, None)
            if value is not None:
                yield value

    @property
    def batch_size(self) -> int:
        """Return the size of the leading (batch) axis.

        Derived from ``waveform``, falling back to ``codes`` / ``latents`` for
        audio-less trees (e.g. token-only training examples).

        This is the *leading* axis, not the item count: on a mini-batched tree
        (rank 4, from :meth:`reshape_mini_batches`) it is the number of
        mini-batches, which is also what ``len()``, iteration and indexing walk
        over. :meth:`flatten_mini_batches` first if you want items.
        """
        for value in (self.waveform, self.codes, self.latents):
            if value is not None:
                return value.shape[0]
        raise ValueError(
            "AudioTree has no waveform, codes, or latents to infer a batch size from."
        )

    @property
    def num_channels(self) -> int:
        """Return the number of audio channels (``waveform.shape[-2]``)."""
        return self.waveform.shape[-2]

    def __len__(self) -> int:
        """Number of items in the batch (the leading axis).

        Together with ``__getitem__`` this makes an AudioTree iterable over
        its batch items, e.g., ``for item in tree: ...`` — each ``item`` is a
        batch-of-1 AudioTree.
        """
        return self.batch_size

    def __getitem__(
        self, key: Union[int, np.integer, slice, Sequence[int], ArrayLike]
    ) -> Self:
        """Index the batch axis, returning an AudioTree of the selected item(s).

        An integer key selects a single item but keeps the leading batch axis
        (a batch of 1); a slice, a list of indices, or a boolean mask selects a
        sub-batch. Every array field — including ``codes``, ``latents``, and the
        ``extras`` arrays — is indexed along the same axis so the fields stay
        rank-aligned. String ``extras`` leaves are selected element-wise with
        the same key and always come back as a list (a bare ``str``, the
        batch-of-1 form, becomes a one-item list).

        Any integer scalar counts as an integer key, not just the builtin
        ``int``: ``tree[np.argmax(tree.lufs)]`` keeps the batch axis exactly
        like ``tree[0]`` does.

        On a mini-batched (rank-4) tree the leading axis is the mini-batch axis,
        so ``tree[0]`` selects one mini-batch (still rank 4) rather than one
        item; :meth:`flatten_mini_batches` first to index items.
        """
        if _is_integer_scalar(key):
            key = int(key)
            n = self.batch_size
            if key < -n or key >= n:
                # Required for the sequence-iteration protocol: `for item in
                # tree` calls __getitem__(0), (1), ... and stops only on
                # IndexError (it does NOT consult __len__).
                raise IndexError(f"batch index {key} out of range for batch_size {n}")
            # Use a length-1 slice rather than a scalar index so the batch
            # axis survives on every field.
            key = slice(key, key + 1 or None)

        return tree_util.tree_map(
            lambda x: _index_batch_axis(x, key), self, is_leaf=_is_string_leaf
        )

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the batch axis, yielding a batch-of-1 AudioTree each.

        This makes AudioTree a proper ``collections.abc.Iterable`` (the
        sequence protocol via ``__getitem__`` already allowed ``for`` loops,
        but ``isinstance(tree, Iterable)`` was ``False`` without ``__iter__``).
        Each yielded item keeps the leading batch axis, e.g., iterating a
        batch-16 tree yields 16 trees of ``batch_size == 1``.
        """
        for i in range(self.batch_size):
            yield self[i]

    @classmethod
    def from_file(
        cls,
        audio_path: Union[str, Path],
        *,
        sample_rate: int | None = None,
        offset: float = 0.0,
        duration: float | None = None,
        mono: bool = False,
        pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
        | None = "constant",
        filepath: Union[str, Path, List[Union[str, Path]]] | None = None,
        source: str | None = None,
        extras: Optional[Dict[str, Any]] = None,
        # AudioTree properties
        lufs: Optional[ArrayLike] = None,
        lufs_windows: Optional[ArrayLike] = None,
        pitch: Optional[ArrayLike] = None,
        velocity: Optional[ArrayLike] = None,
        note_duration: Optional[ArrayLike] = None,
        codes: Optional[ArrayLike] = None,
        latents: Optional[ArrayLike] = None,
    ):
        """Create an AudioTree from an audio file path.

        Args:
            audio_path (str): Path to audio file.
            sample_rate (int, optional): Sample rate of audio data, such as 44100 Hz. If left as ``None``, the file's
                original sample rate will be used.
            offset (float, optional): Offset in seconds to audio data.
            duration (float, optional): Duration in seconds of audio data. The audio data will be trimmed or extended as
                necessary.
            mono (bool, optional): Whether to force the audio data to be single-channel.
            pad_mode (Literal): If duration is not None, and duration is less than the length of the audio, then
                ``pad_mode`` controls how the audio is right-padded (numpy.pad modes). Options:
                "constant" (zeros, default), "edge" (repeat edge), "reflect" (mirror), "symmetric" (mirror with edge),
                "wrap" (circular/loop), or None (no padding).
            filepath (Union[str, Path, List[str | Path]], optional): One or more paths to store in the returned
                ``AudioTree``'s extras. If *None* (default) the provided ``audio_path`` will be used.
            source (str, optional): The source group name for this audio file (e.g., "music", "speech").
                This is stored in extras and accessible via the ``source`` property.
            extras (dict, optional): Additional extras to include in the AudioTree. These are merged with
                automatically generated entries (offset, note_duration, filepath).
            lufs (np.ndarray or jax.Array, optional): Integrated loudness (LUFS) values to assign to the AudioTree.
            lufs_windows (np.ndarray or jax.Array, optional): Per-window loudness (LUFS) values to assign to the AudioTree.
            pitch (np.ndarray or jax.Array, optional): Pitch values to assign to the AudioTree.
            velocity (np.ndarray or jax.Array, optional): Velocity values to assign to the AudioTree.
            note_duration (np.ndarray or jax.Array, optional): Note note_duration values to assign to the AudioTree.
            codes (np.ndarray or jax.Array, optional): The neural audio codec tokens for the audio.
            latents (np.ndarray or jax.Array, optional): The latent representations of the audio.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        audio_path = Path(audio_path)

        data, sample_rate = librosa.load(
            str(audio_path), sr=sample_rate, offset=offset, duration=duration, mono=mono
        )

        # Compute the target length from the *effective* rate librosa loaded at.
        # When ``sample_rate`` is None the returned ``sample_rate`` is the file's
        # native rate, so ``duration`` is still honored (a short file is padded)
        # even without an explicit target rate -- the docstring promises the
        # audio is trimmed or extended unconditionally.
        target_length = None
        if duration is not None:
            target_length = round(duration * sample_rate)

        if data.ndim == 1:
            data = data[None, None, :]  # Add batch and channel dimension
        elif data.ndim == 2:
            data = data[None, :, :]  # Add batch dimension

        if (
            target_length is not None
            and pad_mode is not None
            and data.shape[-1] < target_length
        ):
            pad_right = target_length - data.shape[-1]
            # Modes like "wrap", "reflect", "edge" require non-empty data.
            # Fall back to "constant" (zero-pad) when the time axis is empty.
            effective_pad_mode = "constant" if data.shape[-1] == 0 else pad_mode
            data = np.pad(
                data,
                pad_width=((0, 0), (0, 0), (0, pad_right)),
                mode=effective_pad_mode,
            )

        # ``librosa.load`` with a target ``sample_rate`` resamples internally, and for a non-integer
        # rate ratio (e.g. 44100->48000) the resampled length can overshoot ``round(duration*sr)`` by
        # a sample: librosa sizes the output as ``ceil(n * new_sr / old_sr)`` with a float ratio, so
        # an exact value like 192000.0 evaluates to 192000.0000000003 and ceils to 192001. The pad
        # block above only extends *short* reads up to ``target_length``; this matching trim keeps
        # every excerpt of the same requested duration identical in length (without it, an over-long
        # resample leaks through and breaks batching on a 1-sample mismatch).
        if target_length is not None and data.shape[-1] > target_length:
            data = data[..., :target_length]

        # Start with user-provided extras or empty dict
        if extras is None:
            combined_extras = {}
        else:
            combined_extras = extras.copy()  # Don't modify the original

        # Add automatic extras (these override user extras to ensure correctness)
        combined_extras["offset"] = np.array([offset])

        if filepath is None:
            paths_to_store = [audio_path]
        else:
            # Normalize to list
            if isinstance(filepath, (str, Path)):
                paths_to_store = [filepath]
            else:
                paths_to_store = list(filepath)

        combined_extras["filepath"] = cls._encode_filepaths(paths_to_store)

        if source is not None:
            combined_extras["source"] = cls._encode_filepaths([source])

        # Wrap scalar properties in arrays with batch dimension
        # This ensures consistency - all AudioTree properties should have batch dimension
        def wrap_if_scalar(val, dtype=None):
            if val is None:
                return None
            if np.isscalar(val):
                return np.array([val], dtype=dtype)
            elif isinstance(val, np.ndarray) and val.ndim == 0:
                return np.array([val.item()], dtype=dtype)
            else:
                return val

        return cls(
            waveform=data,
            sample_rate=sample_rate,
            extras=combined_extras,
            lufs=wrap_if_scalar(lufs, np.float32),
            lufs_windows=lufs_windows,
            pitch=wrap_if_scalar(pitch, np.float32),
            velocity=wrap_if_scalar(velocity, np.int16),
            note_duration=wrap_if_scalar(note_duration, np.float32),
            codes=wrap_if_scalar(codes),
            latents=wrap_if_scalar(latents),
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        *,
        audio_dir: Optional[Union[str, Path]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Self:
        """Create an AudioTree by loading all items from a manifest file.

        This loads all entries from a manifest file created by AudioWriter and
        creates a single AudioTree with all items in the batch dimension.

        Parsing the manifest -- header check, presence masks, string decoding --
        is :func:`audiotree._manifest.read_entries`, the one reader of that
        format; nothing here re-derives the file's rules.

        Args:
            manifest_path: Path to the manifest file (NPZ format)
            audio_dir: Optional directory containing audio files. If None, uses manifest directory
            filter_fn: Optional predicate called with one manifest entry, a
                ``dict`` keyed by column name (``"filename"``, ``"sample_rate"``,
                the AudioTree label fields, ``"extras_*"``, plus ``"tags"``);
                return ``True`` to load that entry. These are the entries of
                :func:`audiotree._manifest.read_entries`, the same ones
                :class:`~audiotree.sources.AudioDataSource` passes *its*
                ``filter_fn``, so one predicate serves both (that source
                additionally demotes bookkeeping numbers such as
                ``sample_rate`` to plain Python scalars). A column with no
                value for an entry is absent from that entry's dict.

        Returns:
            AudioTree with all manifest entries concatenated along batch dimension

        Raises:
            ValueError: If the manifest is unreadable, holds no entries, names an
                audio file outside ``audio_dir``, or ``filter_fn`` matches
                nothing.

        Example:
            First, write a small manifest with :class:`~audiotree.AudioWriter`
            (here, 100 one-second stereo items; the first 60 are loud, the rest
            quiet, so the lufs field is recorded for filtering):

            >>> import tempfile
            >>> from audiotree import AudioWriter
            >>> out_dir = tempfile.mkdtemp()
            >>> lufs = np.where(np.arange(100) < 60, -10.0, -30.0).astype(np.float32)
            >>> batch = AudioTree.create(jnp.zeros((100, 2, 44100)), 44100, lufs=lufs)
            >>> with AudioWriter(out_dir) as writer:
            ...     _ = writer.write(batch)
            >>> manifest_path = f"{out_dir}/manifest.npz"

            Load every item into one batched ``AudioTree``:

            >>> tree = AudioTree.from_manifest(manifest_path)
            >>> tree.waveform.shape  # 100 items, stereo, 1 second each
            (100, 2, 44100)

            Load only entries that pass a filter on the manifest (the 60 loud
            items):

            >>> tree = AudioTree.from_manifest(
            ...     manifest_path,
            ...     filter_fn=lambda entry: entry.get('lufs', -float('inf')) > -20
            ... )
            >>> tree.waveform.shape
            (60, 2, 44100)
        """
        manifest_path = Path(manifest_path)
        entries = _manifest.read_entries(manifest_path)

        # Determine audio directory
        if audio_dir is None:
            audio_dir = manifest_path.parent
        else:
            audio_dir = Path(audio_dir)

        if filter_fn is not None:
            entries = [entry for entry in entries if filter_fn(entry)]
            if not entries:
                raise ValueError(
                    f"No entries match filter in manifest: {manifest_path}"
                )
        elif not entries:
            raise ValueError(f"Manifest holds no entries: {manifest_path}")

        def stack_column(name: str) -> Optional[np.ndarray]:
            """Gather one manifest column across the selected entries.

            Returns ``None`` when no selected entry has the column. A column
            that only *some* of them have is an error rather than a hole filled
            in with a sentinel: an AudioTree field carries exactly one value per
            batch item and has no way to say "this item has none".
            """
            values = [entry[name] for entry in entries if name in entry]
            if not values:
                return None
            if len(values) != len(entries):
                raise ValueError(
                    f"Manifest column {name!r} has a value for {len(values)} of "
                    f"the {len(entries)} selected entries. An AudioTree field "
                    f"holds one value per batch item, so a partly-present "
                    f"column cannot be loaded -- filter out the entries that "
                    f"lack it."
                )
            return np.stack(values)

        # Every item of a manifest shares one sample rate, and a manifest-only
        # write shares one shape, so the first selected entry describes them all.
        first = entries[0]
        sample_rate = int(first["sample_rate"])
        channels = int(first["channels"])
        samples = int(first["samples"])
        files_written = bool(first.get("files_written", True))

        # Check if audio files exist
        if files_written:
            # Load audio from files
            waveform = []
            for entry in entries:
                # A manifest travels with the data it describes, so its filenames
                # are untrusted: an absolute path or a `..` would otherwise read
                # any file this process can, and hand it back as `waveform`.
                audio_path = safe_join(
                    audio_dir, str(entry["filename"]), description="audio file"
                )

                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                data, _ = librosa.load(str(audio_path), sr=sample_rate, mono=False)

                if data.ndim == 1:
                    data = data[None, :]  # Add channel dimension
                elif data.ndim == 2:
                    pass  # Already (channels, samples)

                waveform.append(data)

            waveform = np.stack(waveform, axis=0)  # (batch, channels, samples)
        else:
            # No audio files - create zeros
            waveform = np.zeros((len(entries), channels, samples), dtype=np.float32)

        # Build extras dictionary
        extras = {}
        extras_columns = sorted(
            {key for entry in entries for key in entry if key.startswith("extras_")}
        )
        for key in extras_columns:
            extras[key[len("extras_") :]] = stack_column(key)

        # Build AudioTree kwargs
        tree_kwargs = {
            "sample_rate": sample_rate,
            "extras": extras,
        }

        # Restore the source filepath. AudioWriter stores them as a top-level
        # ``filepath`` column of decoded strings (not under an ``extras_``
        # prefix), so passing them back through ``filepath=`` re-encodes them
        # into ``extras['filepath']`` and makes the ``.filepath`` property work.
        filepaths = stack_column("filepath")
        if filepaths is not None:
            tree_kwargs["filepath"] = [str(p) for p in filepaths]

        # Add AudioTree fields from manifest
        for field_name in LABEL_FIELDS:
            values = stack_column(field_name)
            if values is not None:
                tree_kwargs[field_name] = values

        return cls.create(waveform, **tree_kwargs)

    @classmethod
    def excerpt(
        cls,
        audio_path: Union[str, Path],
        rng: np.random.Generator,
        duration: float,
        offset: float = 0.0,
        excerpt: Optional["ExcerptConfig"] = None,
        **kwargs,
    ) -> Optional[Self]:
        """Create an AudioTree from one section of an audio file.

        Which section is up to ``excerpt.strategy``: ``"start"`` takes the audio
        at ``offset``, ``"random"`` (the default) draws a single offset, and
        ``"loudest"`` defers to :meth:`loudest_excerpt`.

        Args:
            audio_path (str or Path): Path to audio file.
            rng (np.random.Generator): Random number generator such as ``np.random.default_rng(42)``.
            duration (float): Duration in seconds of audio data; must be positive. The audio data
                will be trimmed or lengthened as necessary.
            offset (float, optional): Earliest offset in seconds the excerpt may start at.
            excerpt (ExcerptConfig, optional): How to choose the offset; defaults to a uniformly
                random one. See :class:`ExcerptConfig`.
            **kwargs: Keyword arguments passed to ``AudioTree.from_file``.

        Returns:
            AudioTree, or ``None`` when ``excerpt`` searched for the loudest
            section, found nothing above the cutoff, and says ``on_failure="skip"``.
        """
        if duration <= 0:
            raise ValueError(f"excerpt needs a positive duration, got {duration!r}.")
        if excerpt is None:
            excerpt = ExcerptConfig()

        if excerpt.strategy == "loudest":
            if offset:
                raise ValueError(
                    "strategy='loudest' searches the whole file and cannot be "
                    f"combined with offset={offset!r}."
                )
            return cls.loudest_excerpt(
                audio_path, rng, excerpt=excerpt, duration=duration, **kwargs
            )

        if excerpt.strategy == "start":
            excerpt_offset = offset
        else:
            total_duration = soundfile.info(audio_path).duration  # seconds
            # One draw, so ``attempt``/``max_attempts`` are the degenerate case;
            # they exist because the loudest search takes many.
            excerpt_offset = excerpt.resolved_search(
                rng, offset, duration, total_duration, attempt=0, max_attempts=1
            )

        return cls.from_file(
            audio_path=audio_path, offset=excerpt_offset, duration=duration, **kwargs
        )

    @classmethod
    def loudest_excerpt(
        cls,
        audio_path: Union[str, Path],
        rng: np.random.Generator,
        excerpt: "ExcerptConfig",
        **kwargs,
    ) -> Optional[Self]:
        """Create an AudioTree from the loudest of several candidate excerpts.

        Draws up to ``excerpt.num_tries`` offsets and keeps the loudest, stopping
        early once one exceeds ``excerpt.lufs_cutoff``. This is a best-of-k
        search rather than a filter, so on a file where nothing clears the cutoff
        it still has to return something -- ``excerpt.on_failure`` decides what.

        Args:
            audio_path (str): Path to audio file.
            rng (np.random.Generator): Random number generator such as ``np.random.default_rng(42)``.
            excerpt (ExcerptConfig): How to search. ``strategy`` must be ``"loudest"``.
            **kwargs: Keyword arguments passed to ``AudioTree.from_file``.

        Returns:
            AudioTree, or ``None`` when nothing cleared the cutoff and
            ``on_failure="skip"``.
        """
        if "offset" in kwargs:
            raise ValueError(
                "``loudest_excerpt`` cannot be used with kwarg ``offset``."
            )
        if "duration" not in kwargs:
            raise ValueError(
                "``loudest_excerpt`` must be used with kwarg ``duration``."
            )
        if excerpt.strategy != "loudest":
            raise ValueError(
                f"loudest_excerpt needs strategy='loudest', got {excerpt.strategy!r}."
            )

        # Read the header once rather than per attempt.
        file_duration = soundfile.info(audio_path).duration
        duration = kwargs["duration"]

        best = None
        best_lufs = -np.inf
        for attempt in range(excerpt.num_tries):
            offset = excerpt.resolved_search(
                rng,
                0.0,
                duration,
                file_duration,
                attempt=attempt,
                max_attempts=excerpt.num_tries,
            )
            candidate = cls.from_file(audio_path=audio_path, offset=offset, **kwargs)
            if candidate.waveform.shape[-1] == 0:
                logging.warning(
                    f"Empty audio loaded from {audio_path} at offset "
                    f"{offset:.2f}s (file_duration={file_duration:.2f}s)"
                )
            candidate = candidate.replace_lufs()
            # One file, so a batch of one; keep it scalar for the comparison and
            # for the failure message.
            candidate_lufs = float(np.asarray(candidate.lufs).reshape(-1)[0])
            if best is None or candidate_lufs > best_lufs:
                best, best_lufs = candidate, candidate_lufs
            if best_lufs > excerpt.lufs_cutoff:
                return best

        # Nothing cleared the cutoff. Returning the quietest thing we found is
        # indistinguishable from success, which is why this is configurable.
        if excerpt.on_failure == "raise":
            raise RuntimeError(
                f"No excerpt of {audio_path} reached {excerpt.lufs_cutoff} LUFS in "
                f"{excerpt.num_tries} tries (loudest was {best_lufs:.1f}). "
                f"Pass on_failure='skip' to drop such files, or lower lufs_cutoff."
            )
        if excerpt.on_failure == "skip":
            return None
        return best

    def to_mono(
        self, strategy: Literal["average", "left", "right"] = "average"
    ) -> Self:
        """Reduce the ``waveform`` to mono.

        Changing the channel count changes the audio, so every derived field
        (``lufs``, ``lufs_windows``, ``codes``, ``latents``,
        ``extras["codec_scale"]``) is invalidated. A waveform that is already
        mono is returned unchanged, derived fields and all.

        Args:
            strategy: ``"average"`` mixes all channels down (default);
                ``"left"`` / ``"right"`` select the corresponding channel of a
                stereo waveform.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        # Validated before the mono short-circuit, so a typo'd strategy is not
        # silently accepted on whichever items happen to be mono already.
        if strategy not in ("average", "left", "right"):
            raise ValueError(
                f"Unsupported to_mono strategy {strategy!r}; expected "
                f"'average', 'left' or 'right'."
            )
        waveform = self.waveform
        C = self.num_channels
        if C == 1:
            return self
        if strategy == "average":
            waveform = waveform.mean(axis=-2, keepdims=True)
        elif strategy in ("left", "right") and C == 2:
            idx = 0 if strategy == "left" else 1
            waveform = waveform[..., idx : idx + 1, :]
        else:
            raise ValueError(
                f"Unsupported to_mono strategy {strategy!r} for {C} channels."
            )
        return self._invalidate_derived(waveform=waveform)

    def to_stereo(self) -> Self:
        """Make the ``waveform`` stereo.

        Changing the channel count changes the audio, so every derived field
        (``lufs``, ``lufs_windows``, ``codes``, ``latents``,
        ``extras["codec_scale"]``) is invalidated. A waveform that is already
        stereo is returned unchanged, derived fields and all.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        C = self.num_channels
        if C == 1:
            waveform = _concatenate([waveform, waveform], axis=-2)
            # Duplicating the channel changes the integrated loudness (BS.1770
            # sums per-channel energy), so the cached value is no longer valid.
            return self._invalidate_derived(waveform=waveform)
        elif C == 2:
            return self
        else:
            raise ValueError(f"Cannot make AudioTree stereo if it has {C} channels.")

    def write(
        self,
        filepath: Union[str, Path],
        *,
        subtype: str | None = None,
        format: str | None = None,
        endian: str | None = None,
    ) -> Path:
        """Write the ``waveform`` to an audio file using ``soundfile``.

        This is the inverse of :meth:`from_file`. The AudioTree must contain a
        single item (``batch_size == 1``); index or iterate the batch first
        (e.g. ``tree[0]`` or ``for item in tree``) to write each item. The
        sample rate is taken from ``self.sample_rate`` — call :meth:`resample`
        beforehand if you want a different one.

        Args:
            filepath: Output path. The file format is inferred from the
                extension (e.g. ``.wav``, ``.flac``, ``.ogg``) unless overridden
                by ``format``.
            subtype: soundfile subtype, e.g., ``"PCM_16"``, ``"PCM_24"``,
                ``"FLOAT"``. When ``None`` (default) soundfile picks the format
                default (``PCM_16`` for WAV).
            format: Major format override (e.g. ``"WAV"``, ``"FLAC"``). When
                ``None`` it is inferred from the filepath extension.
            endian: Endianness override (e.g. ``"FILE"``, ``"LITTLE"``,
                ``"BIG"``).

        Returns:
            Path: The path that was written.

        Raises:
            ValueError: If the tree has no ``waveform``, if it is mini-batched
                (rank 4 — ``batch_size`` then counts mini-batches, not items),
                or if ``batch_size != 1``.
        """
        if self.waveform is None:
            raise ValueError(
                "AudioTree.write needs a waveform, but this tree has none "
                "(a token-only tree holds codes/latents; decode them first)."
            )
        # Checked before ``batch_size``, which on a rank-4 tree counts
        # mini-batches: a (1, Batch, Channels, Samples) tree would otherwise
        # pass the batch-of-1 check and reach soundfile, whose "Invalid shape"
        # complaint is about the transposed array and names neither the tree's
        # rank nor the fix.
        _require_batched_rank(self.waveform, "AudioTree.write")
        if self.batch_size != 1:
            raise ValueError(
                f"AudioTree.write requires batch_size == 1, got {self.batch_size}. "
                f"Index or iterate the batch first (e.g. tree[0])."
            )
        filepath = Path(filepath)
        # soundfile expects (samples, channels); waveform is (1, channels, samples).
        audio = np.asarray(self.waveform[0].T)
        soundfile.write(
            str(filepath),
            audio,
            self.sample_rate,
            subtype=subtype,
            format=format,
            endian=endian,
        )
        return filepath

    def resample(
        self,
        sample_rate: int,
        *,
        zeros: int = 24,
        rolloff: float = 0.945,
        output_length: Optional[int] = None,
        full: bool = False,
    ) -> Self:
        """
        Resample the AudioTree's ``waveform`` to a new sample rate. NumPy-backed
        waveforms are resampled on CPU with `librosa`_ (soxr); JAX-backed waveforms
        use a JAX port of ``ResampleFrac`` from the PyTorch library `Julius`_. The
        two backends are not bit-identical.

        .. _librosa: https://librosa.org/
        .. _Julius: https://github.com/adefossez/julius/blob/main/julius/resample.py

        Args:
            sample_rate (int): The new sample rate of audio data, such as 44100 Hz.
            zeros (int, optional): number of zero crossing to keep in the sinc filter.
                JAX backend only.
            rolloff (float): use a lowpass filter that is ``rolloff * sample_rate / 2``,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce antialiasing, but will reduce some of the
                highest frequencies. JAX backend only.
            output_length (None or int): This can be set to the desired output length (last dimension).
                Allowed values are between 0 and ``ceil(length * sample_rate / old_sr)``. When ``None`` (default) is
                specified, the floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the ``output_length`` only
                for the last one, while passing ``full=True`` to all the other ones. JAX backend only.

        Changing the sample rate changes the audio's length and its samples, so
        every derived field (``lufs``, ``lufs_windows``, ``codes``, ``latents``,
        ``extras["codec_scale"]``) is invalidated. Resampling to the rate the
        tree already has returns it unchanged, derived fields and all.

        Returns:
            AudioTree: A new ``AudioTree`` resampled to ``sample_rate`` (the original is unchanged).

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)  # 1 s at 44.1 kHz
            >>> resampled = audio.resample(22050)
            >>> resampled.waveform.shape
            (1, 1, 22050)
            >>> resampled.sample_rate
            22050
        """
        if sample_rate == self.sample_rate:
            return self
        if isinstance(self.waveform, np.ndarray):
            # CPU backend: librosa (soxr). ``zeros``, ``rolloff``, and ``full``
            # are JAX-only knobs and do not apply here. librosa resamples along
            # the time axis directly, so no 3-D flattening is needed.
            waveform = librosa.resample(
                self.waveform,
                orig_sr=self.sample_rate,
                target_sr=sample_rate,
                axis=-1,
            ).astype(self.waveform.dtype)
            # Pin the output length to what the JAX/Julius backend produces
            # (``floor(T * new / old)``) so the two backends agree on shape;
            # soxr's length can differ by a sample. ``output_length`` overrides.
            if output_length is None:
                output_length = (
                    self.waveform.shape[-1] * sample_rate // self.sample_rate
                )
            waveform = librosa.util.fix_length(waveform, size=output_length, axis=-1)
        else:
            # JAX backend: the Julius port. Its kernel is strictly 3-D, so
            # flatten any leading axes (e.g. after reshape_mini_batches) and
            # restore them afterwards.
            leading_shape = self.waveform.shape[:-2]
            flat = self.waveform.reshape(-1, *self.waveform.shape[-2:])
            waveform = resample(
                flat,
                self.sample_rate,
                sample_rate,
                zeros=zeros,
                rolloff=rolloff,
                output_length=output_length,
                full=full,
            )
            waveform = waveform.reshape(*leading_shape, *waveform.shape[-2:])
        return self._invalidate_derived(waveform=waveform, sample_rate=sample_rate)

    def split(self, n_splits: int) -> List[Self]:
        """Split batch dimension into a list of smaller AudioTree objects.

        Divides the batch dimension evenly into n_splits separate AudioTree objects,
        each containing a portion of the original batch. Like indexing, this
        works on the *leading* axis, which on a mini-batched (rank-4) tree is
        the mini-batch axis rather than the item axis.

        Args:
            n_splits: Number of AudioTree objects to create. The batch size must be
                evenly divisible by this value.

        Returns:
            List of AudioTree objects, each with batch_size = original_batch_size / n_splits.

        Raises:
            ValueError: If ``n_splits`` is not positive, or if the batch size is
                not divisible by ``n_splits``.

        Example:
            >>> big_tree = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> big_tree.waveform.shape
            (12, 1, 44100)
            >>> split_trees = big_tree.split(2)
            >>> len(split_trees)
            2
            >>> split_trees[0].waveform.shape  # each tree has half the original batch size
            (6, 1, 44100)
        """
        if n_splits <= 0:
            raise ValueError(f"n_splits must be positive, got {n_splits}.")
        total_batch_size = self.batch_size
        if total_batch_size % n_splits != 0:
            raise ValueError(
                f"Total batch size {total_batch_size} must be divisible by the "
                f"number of splits {n_splits}."
            )

        split_batch_size = total_batch_size // n_splits

        # Slice the batch axis with the same leaf definition ``__getitem__``
        # uses, so a string extras leaf is sliced element-wise
        # (``names[i*s:(i+1)*s]``) instead of ``jax.tree_util`` descending into
        # it and slicing each string's characters.
        return [
            tree_util.tree_map(
                lambda x, i=i: _index_batch_axis(
                    x, slice(i * split_batch_size, (i + 1) * split_batch_size)
                ),
                self,
                is_leaf=_is_string_leaf,
            )
            for i in range(n_splits)
        ]

    def reshape_mini_batches(self, mini_batch_size: int) -> Self:
        """Reshape batch dimension into mini-batches by adding a new leading axis.

        Transforms audio data from shape (B, C, T) to (num_mini_batches, mini_batch_size, C, T),
        where B must be evenly divisible by mini_batch_size. String extras
        leaves nest the same way: a list of B strings becomes num_mini_batches
        lists of mini_batch_size strings, so indexing a mini-batch keeps them
        aligned with the arrays.

        Args:
            mini_batch_size: Number of samples per mini-batch. The total batch size must be
                evenly divisible by this value.

        Returns:
            AudioTree with an additional mini-batch dimension as the first axis.

        Raises:
            ValueError: If the batch size is not divisible by ``mini_batch_size``,
                or if the tree is already mini-batched (rank 4) — nothing in this
                library reads the rank-5 tree that would produce.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape  # 4 mini-batches of size 3
            (4, 3, 1, 44100)
        """
        _require_batched_rank(self.waveform, "reshape_mini_batches")
        B = self.batch_size

        if B % mini_batch_size != 0:
            raise ValueError(
                f"Batch size {B} must be divisible by mini_batch_size "
                f"{mini_batch_size}."
            )
        num_mini_batches = B // mini_batch_size

        # Reshape AudioTree to have leading mini-batch dimension
        # From (B, C, T) to (num_mini_batches, mini_batch_size, C, T)
        # Only reshape array-like objects since extras can contain non-arrays
        def reshape_leaf(x):
            if _is_string_leaf(x):
                # Nest to match the new leading axes, like the encoded
                # provenance arrays (see ``_decode_strings``): one list of
                # ``mini_batch_size`` strings per mini-batch.
                strings = _as_string_list(x)
                if len(strings) != B:
                    raise ValueError(
                        f"String extras leaf has {len(strings)} items but "
                        f"the batch size is {B}."
                    )
                return [
                    strings[i * mini_batch_size : (i + 1) * mini_batch_size]
                    for i in range(num_mini_batches)
                ]
            return (
                x.reshape(num_mini_batches, mini_batch_size, *x.shape[1:])
                if hasattr(x, "shape")
                else x
            )

        return tree_util.tree_map(reshape_leaf, self, is_leaf=_is_string_leaf)

    def flatten_mini_batches(self) -> Self:
        """Flatten mini-batches back into a single batch dimension.

        Undoes the operation performed by reshape_mini_batches(), transforming
        audio data from shape (num_mini_batches, mini_batch_size, C, T) back to
        (B, C, T). String extras leaves lose their per-mini-batch nesting the
        same way, back to one string per item.

        Returns:
            AudioTree with the mini-batch dimension flattened into the batch dimension.

        Raises:
            ValueError: If the waveform has fewer than 4 dimensions, i.e. it was
                never mini-batched.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape  # 4 mini-batches of size 3
            (4, 3, 1, 44100)
            >>> x_unbatched = x_batched.flatten_mini_batches()
            >>> x_unbatched.waveform.shape  # back to original shape
            (12, 1, 44100)
        """
        # Assuming the waveform has shape (num_mini_batches, mini_batch_size, C, T)
        # We want to reshape to (num_mini_batches * mini_batch_size, C, T)

        # Read the rank from whichever leaf is present (waveform first, then
        # codes / latents for a token-only tree).
        leaf = next(self._array_leaves(), None)
        shape = () if leaf is None else leaf.shape

        # We expect at least 4 dimensions for mini-batched data
        if len(shape) < 4:
            raise ValueError(
                f"Expected at least 4 dimensions for mini-batched data, got "
                f"{len(shape)}. Shape: {shape}"
            )

        # Flatten the first two dimensions
        # From (num_mini_batches, mini_batch_size, C, T) to (B, C, T)
        # Only reshape array-like objects since extras can contain non-arrays
        def flatten_leaf(x):
            if _is_string_leaf(x):
                # Undo ``reshape_mini_batches``'s nesting: one list per
                # mini-batch flattens back to one string per item. A flat
                # list has no mini-batch axis to remove.
                if isinstance(x, list) and x and isinstance(x[0], list):
                    return [s for sub in x for s in sub]
                return x
            return x.reshape(-1, *x.shape[2:]) if hasattr(x, "shape") else x

        return tree_util.tree_map(flatten_leaf, self, is_leaf=_is_string_leaf)

    def filter(self, predicate: Callable[[Self], bool]) -> Self:
        """Keep only the batch items for which ``predicate`` is true.

        The batch is split into one tree per item, ``predicate`` is called on
        each, and the survivors are concatenated back into a single tree.

        Args:
            predicate: Called with a batch-of-1 ``AudioTree``; return ``True`` to
                keep that item.

        Returns:
            AudioTree: A tree holding the kept items. When nothing is kept, the
            result has ``batch_size == 0`` (every array field is empty along the
            batch axis, every string extras leaf is ``[]``) rather than being
            ``None``. Filtering an already-empty tree returns such an empty
            tree without calling ``predicate``, so chained filters compose.

        Raises:
            ValueError: If the tree is mini-batched (rank 4). Dropping items
                independently within each mini-batch would leave the
                mini-batches ragged, so there is no rank-4 answer; flatten,
                filter, and reshape again.

        Example:
            >>> waveform = jnp.stack([jnp.zeros((1, 8)), jnp.ones((1, 8))])
            >>> tree = AudioTree.create(waveform, 16000)
            >>> loud = tree.filter(lambda item: bool(item.waveform.max() > 0.5))
            >>> loud.batch_size
            1
        """
        _require_batched_rank(self.waveform, "AudioTree.filter")
        B = self.batch_size
        # An already-empty tree (the documented result of a filter that kept
        # nothing) has no items for the predicate to see; skip straight to the
        # empty result instead of asking split(0) for one tree per item.
        audio_trees = [] if B == 0 else [t for t in self.split(B) if predicate(t)]

        if len(audio_trees) == 0:
            return tree_util.tree_map(
                lambda x: (
                    [] if _is_string_leaf(x) else x[:0] if hasattr(x, "shape") else x
                ),
                self,
                is_leaf=_is_string_leaf,
            )

        return _batch_audiotrees(audio_trees)

    @staticmethod
    def batch(items: Sequence[Any]) -> Any:
        """Batch function for use with grain's IterDataset.batch().

        Concatenates AudioTree objects along the batch axis (axis 0).
        Use this instead of grain's default batching, which would add
        an extra dimension since AudioTree already has shape (batch, channels, samples).

        Supports arbitrary nested structures containing AudioTrees. All arrays
        (including AudioTrees) are concatenated along axis 0, so data should have
        a leading batch dimension.

        Concatenation dispatches on the leaves' array library, so JAX in gives
        JAX out: batching a tree of ``jax.Array`` leaves stays on device instead
        of forcing a blocking host sync and silently returning NumPy.

        Args:
            items: Sequence of AudioTree objects, or structures (dicts, lists, etc.)
                containing AudioTree objects. Must be non-empty — there is no
                array library, sample rate or structure to infer from nothing.

        Returns:
            Batched structure with the same shape as the input items.

        Raises:
            ValueError: If ``items`` is empty.

        Example:
            >>> a = AudioTree.create(jnp.zeros((1, 1, 16000)), 16000)
            >>> batched = AudioTree.batch([a, a, a])  # concatenate along the batch axis
            >>> batched.waveform.shape
            (3, 1, 16000)
            >>> isinstance(batched.waveform, jax.Array)  # JAX in, JAX out
            True

            With Grain, pass it as the ``batch_fn`` (each item already has a leading batch axis)::

                ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)
        """
        items = list(items)
        if not items:
            raise ValueError(
                "AudioTree.batch needs at least one item; got an empty sequence."
            )

        def batching_function(*args):
            first_arg = args[0]
            if isinstance(first_arg, AudioTree):
                return _batch_audiotrees(args)
            elif isinstance(first_arg, (np.ndarray, jax.Array)) or _is_string_leaf(
                first_arg
            ):
                return _concatenate(args)
            else:
                return list(args)

        return tree_util.tree_map(
            batching_function,
            items[0],
            *items[1:],
            is_leaf=lambda x: isinstance(x, AudioTree) or _is_string_leaf(x),
        )


# ``flax.struct.dataclass`` unconditionally overwrites ``replace`` with an
# untyped ``**updates`` wrapper. Put the typed one back -- identical behaviour,
# but ``help()``, ``inspect.signature`` and the docs keep the field names.
AudioTree.replace = AudioTree._typed_replace


# --- Field lists ------------------------------------------------------------
# Derived from the dataclass rather than restated, because these were three
# hand-maintained copies that had to agree with each other and with the class.
# Order is declaration order, which is also the pytree flatten order and
# therefore the on-disk leaf order that TreeWriter records -- do not sort.

#: Fields that are pytree nodes (i.e. everything but the static ``sample_rate``).
PYTREE_FIELDS: tuple = tuple(
    f.name for f in dataclasses.fields(AudioTree) if f.metadata.get("pytree_node", True)
)

#: Pytree fields holding a single array, so ``extras`` (a dict) is excluded.
ARRAY_FIELDS: tuple = tuple(f for f in PYTREE_FIELDS if f != "extras")

#: Per-item labels: the array fields other than the waveform itself. These are
#: what the writers record as manifest columns.
LABEL_FIELDS: tuple = tuple(f for f in ARRAY_FIELDS if f != "waveform")


def _concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> ArrayLike:
    """Concatenate *arrays* with the array library their leaves already use.

    ``np.concatenate`` on JAX arrays is a blocking host sync *and* a silent type
    change (JAX in, NumPy out), which is exactly what batching a device-resident
    tree must not do. A single JAX array anywhere in *arrays* makes the whole
    concatenation JAX -- ``jnp.concatenate`` accepts NumPy operands, so a mixed
    sequence still works and lands on device.

    String extras leaves (a supported ``extras`` type) are joined as Python
    lists, since ``np``/``jnp.concatenate`` cannot concatenate bare strings.
    A bare ``str`` counts as a batch of 1 (the form ``TreeDataSource`` yields
    per item), so the result is always a flat list with one string per item.
    """
    if _is_string_leaf(arrays[0]):
        return [s for leaf in arrays for s in _as_string_list(leaf)]
    xp = jnp if any(isinstance(array, jax.Array) for array in arrays) else np
    return xp.concatenate(arrays, axis=axis)


def _batch_audiotrees(audio_trees: Sequence[AudioTree]) -> AudioTree:
    """Batch a list of AudioTrees into a single AudioTree.

    Concatenates all array fields along the batch axis (axis 0), using whichever
    array library the leaves already use (see :func:`_concatenate`). Requires all
    AudioTrees to have the same sample_rate and compatible shapes.

    Prefer using ``AudioTree.batch`` instead, which handles mixed-type
    structures (dicts with AudioTrees, arrays, strings, etc.) and rejects an
    empty sequence with a message.

    Args:
        audio_trees: Non-empty sequence of AudioTree objects to batch together.

    Returns:
        Single AudioTree with all items batched along axis 0.
    """
    return tree_util.tree_map(
        lambda *xs: _concatenate(xs), *audio_trees, is_leaf=_is_string_leaf
    )


def _numpy_integrated_lufs(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Integrated loudness (LUFS) per item of a ``(batch, channels, samples)`` waveform.

    Measured on the CPU with ``loudness.integrated_loudness``. Excerpts shorter
    than the BS.1770 gating block (400ms) are padded and level-compensated by
    :func:`~audiotree.loudness.pad_to_gating_block`, the same helper the JAX
    meter uses, so the two backends agree on short excerpts.
    """
    waveform = pad_to_gating_block(waveform, sample_rate, xp=np)
    audio_transposed = np.transpose(waveform, (0, 2, 1))  # [B, T, C]
    values = [
        loudness.integrated_loudness(np.ascontiguousarray(item), sample_rate)
        for item in audio_transposed
    ]
    return np.array(values, dtype=np.float32)
