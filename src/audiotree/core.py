from __future__ import annotations

import dataclasses
from functools import partial
import importlib
import json
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
    TYPE_CHECKING,
    Union,
)

from absl import logging
from flax import struct
import jax
from jax import numpy as jnp, tree_util
import librosa
import loudness
import numpy as np
import soundfile

from . import _format
from .loudness import (
    _jit_integrated_loudness,
    _jit_windowed_loudness,
    _numpy_windowed_lufs,
    _window_samples,
    _windowed_num_windows,
    safe_gain_db,
    shift_lufs,
    shift_lufs_windows,
)
from .resample import resample

if TYPE_CHECKING:
    # An array field that may hold either a NumPy or a JAX array. ``jax`` is
    # already imported at module top (``replace_lufs(backend=...)`` needs
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
# metadata arrays, so they can be batched and stored in fixed-width int32 arrays.
# Strings longer than this raise in ``_encode_string`` rather than being
# truncated, to avoid silently corrupting paths.
_str_max_length = 1024


@struct.dataclass
class AudioTree:
    """
    A `flax.struct.dataclass`_ for holding audio information including a waveform, sample rate, and metadata.

    The ``AudioTree`` class is inspired by Descript AudioTools's `AudioSignal`_.
        .. _AudioSignal: https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        .. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass

    Args:
        waveform (np.ndarray or jax.Array): Audio waveform data shaped ``(Samples)``, ``(Channels, Samples)``, or ``(Batch, Channels, Samples)``
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
        metadata (dict): Any extra metadata can be placed here.
        filepaths (Union[str, Path, List[Union[str, Path]]] | None): List of filepaths for the batch of audio.

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
    metadata: dict = struct.field(pytree_node=True, default_factory=dict)

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
        metadata: dict | None = None,
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
        source: Union[str, List[str]] | None = None,
    ) -> Self:
        """Create an ``AudioTree``, normalizing the waveform to ``(Batch, Channels, Samples)``.

        A bare ``(Samples,)`` or ``(Channels, Samples)`` waveform gains the missing leading axes, so
        you don't have to reshape by hand. ``filepaths`` and ``source`` are encoded into ``metadata``.

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
            metadata: Optional extra metadata dict (copied, not mutated).
            filepaths: Optional path(s) for the batch; encoded into ``metadata["filepath"]``.
            source: Optional source-group name(s) (e.g. ``"music"``), encoded into
                ``metadata["source"]`` and read back via the :attr:`source` property. Pass a
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

        # Handle metadata and filepaths
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()  # Don't modify the original dict

        if filepaths is not None:
            metadata["filepath"] = cls._encode_filepaths(filepaths)

        if source is not None:
            metadata["source"] = cls._encode_filepaths(source)

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
            metadata=metadata,
        )

    def replace_metadata(self, **kwargs) -> Self:
        """Return a new ``AudioTree`` with ``kwargs`` merged into ``metadata``.

        Syntactic sugar for ``self.replace(metadata={**self.metadata, **kwargs})``.
        Keys in ``kwargs`` overwrite existing ``metadata`` keys with the same name;
        all other keys are kept. Neither the original tree nor its ``metadata``
        dict is mutated. For a key that isn't a valid Python identifier, use the
        ``self.replace(metadata=...)`` form directly.

        Args:
            **kwargs: Entries to merge into ``metadata``. Values should be arrays
                (or pytrees of arrays) so the result stays batchable and jittable.

        Returns:
            AudioTree: A new ``AudioTree`` with the merged ``metadata``.

        Example:
            >>> audio = AudioTree.create(jnp.zeros((44100,)), 44100)
            >>> audio = audio.replace_metadata(tempo=np.array([120.0]))
            >>> audio.metadata["tempo"]
            array([120.])
        """
        return self.replace(metadata=self.metadata | kwargs)

    def replace_lufs(
        self,
        lufs_window_sec: float = 0.4,
        lufs_hop_sec: float | None = None,
        *,
        backend: Optional[Literal["cpu", "gpu", "tpu"]] = None,
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

        The two backends are not bit-identical.

        Args:
            lufs_window_sec: Length in seconds of each ``lufs_windows`` window.
                Must be at least 0.4s (the EBU momentary integration time).
            lufs_hop_sec: Step in seconds between window starts. Defaults to
                ``lufs_window_sec`` (non-overlapping windows); a smaller value
                overlaps them. The trailing partial window is dropped, so an
                excerpt shorter than one window yields an empty ``lufs_windows``.
            backend: XLA backend for the computation, mirroring ``jax.jit``'s
                ``backend`` argument. ``None`` (default) uses the waveform's own
                array library on its current device — the NumPy/CPU path (the
                ``loudness`` C++ library + ``scipy``, no JAX) for NumPy waveforms,
                and ``jaxloudnorm`` on the current device for JAX waveforms. A
                string ``"cpu"`` / ``"gpu"`` / ``"tpu"`` instead forces the vmapped
                ``jaxloudnorm`` kernel onto that XLA backend even for a NumPy
                waveform — e.g., ``backend="gpu"`` is much faster for a large batch,
                since the NumPy ``lufs`` path measures one item at a time. The
                **returned** ``lufs`` / ``lufs_windows`` always match the waveform's
                array type (a NumPy waveform yields NumPy loudness regardless of
                ``backend``), so you never need a manual ``jax.device_put`` /
                ``jax.device_get`` round-trip.

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
        if backend is not None and backend not in ("cpu", "gpu", "tpu"):
            raise ValueError(
                f"backend must be None, 'cpu', 'gpu', or 'tpu', got {backend!r}."
            )
        # Flatten any leading axes (e.g. after reshape_mini_batches) to a single
        # batch axis, then restore them on the computed loudness.
        leading_shape = self.waveform.shape[:-2]
        waveform = self.waveform.reshape(-1, *self.waveform.shape[-2:])
        ws = _window_samples(lufs_window_sec, self.sample_rate)
        hs = _window_samples(lufs_hop_sec, self.sample_rate)
        num_windows = _windowed_num_windows(waveform.shape[-1], ws, hs)

        # ``backend`` chooses the compute kernel/device; the output stays in the
        # waveform's own array library so the tree does not go heterogeneous. A
        # ``None`` backend follows the waveform's array type; an explicit XLA
        # backend forces the ``jaxloudnorm`` kernel onto that device.
        input_is_numpy = isinstance(waveform, np.ndarray)
        use_jax = not input_is_numpy if backend is None else True
        device = None if backend is None else jax.devices(backend)[0]

        if use_jax:
            compute_waveform = (
                jnp.asarray(waveform)
                if device is None
                else jax.device_put(waveform, device)
            )
            lufs_array = _jit_integrated_loudness(
                compute_waveform, self.sample_rate, zeros=512
            )
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
                    zeros=512,
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
        # kernel already ran there; a device transfer when ``backend`` forced the
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
        backend: Optional[Literal["cpu", "gpu", "tpu"]] = None,
    ) -> Self:
        """Normalize audio to a target LUFS level.

        Computes the current loudness (if not already set), then scales the audio
        to achieve the target LUFS. The returned AudioTree has updated
        ``waveform``, ``lufs``, and ``lufs_windows`` fields (a constant gain shifts
        every window's LUFS by the same amount).

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
            backend: XLA backend for computing the loudness when it is not already
                set, forwarded to :meth:`replace_lufs` (see there). Ignored when
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
            tree = self.replace_lufs(backend=backend)
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
        # what keeps a skipped (non-finite) or capped item honest.
        return tree.replace(
            waveform=scaled_waveform,
            lufs=shift_lufs(tree.lufs, gain_db, xp=numpy),
            lufs_windows=shift_lufs_windows(tree.lufs_windows, gain_db, xp=numpy),
        )

    @staticmethod
    def _encode_string(s: str) -> np.ndarray:
        """Encode a single filepath *s* to an array of Unicode code points.

        The returned array is shaped ``(1, _str_max_length)`` so that multiple
        rows (filepaths) can be concatenated along *axis=0*.

        Raises:
            ValueError: If *s* is longer than ``_str_max_length`` code points.
                The string is not truncated, to avoid silently corrupting paths.
        """
        s = str(s)
        if len(s) > _str_max_length:
            raise ValueError(
                f"String of length {len(s)} exceeds the metadata encoding limit "
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
            paths: A single filepath or an iterable of filepaths.

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

    @property
    def filepath(self) -> List[str]:
        """Return the decoded filepaths stored in ``metadata['filepath']``.

        An empty list is returned if the AudioTree does not contain any filepath
        metadata.
        """
        if "filepath" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["filepath"]]

    @property
    def source(self) -> List[str]:
        """Return the decoded source names stored in ``metadata['source']``.

        Source names indicate which data source group each item in the batch came from.
        For example, if an AudioDataSimpleSource was created with
        ``sources={"music": [...], "speech": [...]}``, this property might return
        ``["music", "music", "speech", "music"]`` for a batch of 4 items.

        An empty list is returned if the AudioTree does not contain any source
        metadata.
        """
        if "source" not in self.metadata:
            return []
        return [self._decode_string(data) for data in self.metadata["source"]]

    @property
    def samples(self) -> int:
        """Return the number of samples in the ``waveform`` (its last dimension)."""
        return self.waveform.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return the size of the leading (batch) axis.

        Derived from ``waveform``, falling back to ``codes`` / ``latents`` for
        audio-less trees (e.g. token-only training examples).
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

    def __getitem__(self, key: Union[int, slice]) -> Self:
        """Index the batch axis, returning an AudioTree of the selected item(s).

        An integer key selects a single item but keeps the leading batch axis
        (a batch of 1); a slice selects a sub-batch. Every array field —
        including ``codes``, ``latents``, and the ``metadata`` arrays — is
        indexed along the same axis so the fields stay rank-aligned.
        """
        if isinstance(key, int):
            n = self.batch_size
            if key < -n or key >= n:
                # Required for the sequence-iteration protocol: `for item in
                # tree` calls __getitem__(0), (1), ... and stops only on
                # IndexError (it does NOT consult __len__).
                raise IndexError(f"batch index {key} out of range for batch_size {n}")
            # Use a length-1 slice rather than a scalar index so the batch
            # axis survives on every field.
            key = slice(key, key + 1 or None)

        def _is_string_list(x) -> bool:
            return (
                isinstance(x, list) and bool(x) and all(isinstance(s, str) for s in x)
            )

        def _index(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)) or _is_string_list(x):
                return x[key]
            return x

        return tree_util.tree_map(_index, self, is_leaf=_is_string_list)

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
        filepaths: Union[str, Path, List[Union[str, Path]]] | None = None,
        source: str | None = None,
        metadata: Optional[Dict[str, Any]] = None,
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
            filepaths (Union[str, Path, List[str | Path]], optional): One or more filepaths to store in the returned
                ``AudioTree``'s metadata. If *None* (default) the provided ``audio_path`` will be used.
            source (str, optional): The source group name for this audio file (e.g., "music", "speech").
                This is stored in metadata and accessible via the ``source`` property.
            metadata (dict, optional): Additional metadata to include in the AudioTree. This metadata is merged with
                automatically generated fields (offset, note_duration, filepath).
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

        target_length = None
        if duration is not None and sample_rate is not None:
            target_length = round(duration * sample_rate)

        data, sample_rate = librosa.load(
            str(audio_path), sr=sample_rate, offset=offset, duration=duration, mono=mono
        )

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

        # Start with user-provided metadata or empty dict
        if metadata is None:
            combined_metadata = {}
        else:
            combined_metadata = metadata.copy()  # Don't modify the original

        # Add automatic metadata (these override user metadata to ensure correctness)
        combined_metadata["offset"] = np.array([offset])

        if filepaths is None:
            filepaths_to_store = [audio_path]
        else:
            # Normalize to list
            if isinstance(filepaths, (str, Path)):
                filepaths_to_store = [filepaths]
            else:
                filepaths_to_store = list(filepaths)

        combined_metadata["filepath"] = cls._encode_filepaths(filepaths_to_store)

        if source is not None:
            combined_metadata["source"] = cls._encode_filepaths([source])

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
            metadata=combined_metadata,
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
        filter_fn: Optional[Callable[[Any], bool]] = None,
    ) -> Self:
        """Create an AudioTree by loading all items from a manifest file.

        This loads all entries from a manifest file created by AudioWriter and
        creates a single AudioTree with all items in the batch dimension.

        Args:
            manifest_path: Path to the manifest file (NPZ format)
            audio_dir: Optional directory containing audio files. If None, uses manifest directory
            filter_fn: Optional function to filter entries. Should accept a dict entry and return bool.

        Returns:
            AudioTree with all manifest entries concatenated along batch dimension

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

        # Load manifest data
        manifest_data = np.load(manifest_path, allow_pickle=True)
        _format.check(
            {
                key[len(_format.NPZ_HEADER_PREFIX) :]: json.loads(
                    str(manifest_data[key])
                )
                for key in manifest_data.files
                if key.startswith(_format.NPZ_HEADER_PREFIX)
            },
            _format.MANIFEST,
            source=str(manifest_path),
        )

        # Determine audio directory
        if audio_dir is None:
            audio_dir = manifest_path.parent
        else:
            audio_dir = Path(audio_dir)

        # Convert to list of entry dictionaries for filtering
        num_entries = len(manifest_data["index"])

        if filter_fn is not None:
            # Create a lazy dict-like object for filtering
            class LazyEntry:
                def __init__(self, data, idx):
                    self.data = data
                    self.idx = idx

                def get(self, key, default=None):
                    if key in self.data:
                        return self.data[key][self.idx]
                    return default

                def __getitem__(self, key):
                    return self.data[key][self.idx]

            # Build mask using filter function
            mask = np.array(
                [filter_fn(LazyEntry(manifest_data, i)) for i in range(num_entries)]
            )
            indices = np.where(mask)[0]

            if len(indices) == 0:
                raise ValueError(
                    f"No entries match filter in manifest: {manifest_path}"
                )
        else:
            indices = np.arange(num_entries)

        # Get metadata for reconstruction
        sample_rate = int(manifest_data["sample_rate"][indices[0]])
        channels = int(manifest_data["channels"][indices[0]])
        samples = int(manifest_data["samples"][indices[0]])
        files_written = manifest_data.get(
            "files_written", np.ones(num_entries, dtype=bool)
        )[indices[0]]

        # Check if audio files exist
        if files_written:
            # Load audio from files
            waveform = []
            for idx in indices:
                filename = manifest_data["filename"][idx]
                audio_path = audio_dir / filename

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
            waveform = np.zeros((len(indices), channels, samples), dtype=np.float32)

        # Build metadata dictionary
        metadata = {}
        for key in manifest_data.keys():
            if key.startswith("metadata_"):
                # Extract metadata field
                metadata_key = key[9:]  # Remove 'metadata_' prefix
                metadata[metadata_key] = manifest_data[key][indices]

        # Build AudioTree kwargs
        tree_kwargs = {
            "sample_rate": sample_rate,
            "metadata": metadata,
        }

        # Restore the source filepaths. AudioWriter stores them as a top-level
        # ``filepath`` column of decoded strings (not under a ``metadata_``
        # prefix), so passing them back through ``filepaths=`` re-encodes them
        # into ``metadata['filepath']`` and makes the ``.filepath`` property work.
        if "filepath" in manifest_data:
            tree_kwargs["filepaths"] = [
                str(p) for p in manifest_data["filepath"][indices]
            ]

        # Add AudioTree fields from manifest
        for field_name in LABEL_FIELDS:
            if field_name in manifest_data:
                tree_kwargs[field_name] = manifest_data[field_name][indices]

        return cls.create(waveform, **tree_kwargs)

    @classmethod
    def excerpt(
        cls,
        audio_path: str,
        rng: np.random.Generator,
        offset: float = 0.0,
        duration: Optional[float] = None,
        search_function: Optional[Callable] = None,
        **kwargs,
    ) -> Self:
        """Create an AudioTree from a random section of audio from a file path.

        Args:
            audio_path (str): Path to audio file.
            rng (np.random.Generator): Random number generator.
            offset (float, optional): Offset in seconds to audio data.
            duration (float, optional): Duration in seconds of audio data. The audio data will be trimmed or lengthened
                as necessary.
            search_function (Callable, optional): A function that determines the random offset.
            **kwargs: Keyword arguments passed to ``AudioTree.__init__``.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        assert duration is not None and duration > 0
        info = soundfile.info(audio_path)
        total_duration = info.duration  # seconds

        if search_function is None:
            search_function = partial(search_uniform, attempt=0, max_attempts=1)

        random_offset = search_function(rng, offset, duration, total_duration)

        audio_signal = cls.from_file(
            audio_path=audio_path, offset=random_offset, duration=duration, **kwargs
        )

        return audio_signal

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

        Args:
            strategy: ``"average"`` mixes all channels down (default);
                ``"left"`` / ``"right"`` select the corresponding channel of a
                stereo waveform.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
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
        return self.replace(waveform=waveform, lufs=None, lufs_windows=None)

    def to_stereo(self) -> Self:
        """Make the ``waveform`` stereo.

        Returns:
            AudioTree: An instance of ``AudioTree``.
        """
        waveform = self.waveform
        C = self.num_channels
        if C == 1:
            numpy = np if isinstance(waveform, np.ndarray) else jnp
            waveform = numpy.concatenate([waveform, waveform], axis=-2)
            # Duplicating the channel changes the integrated loudness (BS.1770
            # sums per-channel energy), so the cached value is no longer valid.
            return self.replace(waveform=waveform, lufs=None, lufs_windows=None)
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
        """
        assert self.batch_size == 1, (
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
        return self.replace(
            waveform=waveform, sample_rate=sample_rate, lufs=None, lufs_windows=None
        )

    def split(self, n_splits: int) -> List[Self]:
        """Split batch dimension into a list of smaller AudioTree objects.

        Divides the batch dimension evenly into n_splits separate AudioTree objects,
        each containing a portion of the original batch.

        Args:
            n_splits: Number of AudioTree objects to create. The batch size must be
                evenly divisible by this value.

        Returns:
            List of AudioTree objects, each with batch_size = original_batch_size / n_splits.

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
        total_batch_size = self.waveform.shape[0]
        assert total_batch_size % n_splits == 0, (
            f"Total batch size {total_batch_size} must be divisible by number of splits {n_splits}"
        )

        split_batch_size = total_batch_size // n_splits

        return [
            tree_util.tree_map(
                lambda x: x[i * split_batch_size : (i + 1) * split_batch_size], self
            )
            for i in range(n_splits)
        ]

    def reshape_mini_batches(self, mini_batch_size: int) -> Self:
        """Reshape batch dimension into mini-batches by adding a new leading axis.

        Transforms audio data from shape (B, C, T) to (num_mini_batches, mini_batch_size, C, T),
        where B must be evenly divisible by mini_batch_size.

        Args:
            mini_batch_size: Number of samples per mini-batch. The total batch size must be
                evenly divisible by this value.

        Returns:
            AudioTree with an additional mini-batch dimension as the first axis.

        Example:
            >>> x = AudioTree(np.zeros((12, 1, 44100)), 44100)
            >>> x_batched = x.reshape_mini_batches(3)
            >>> x_batched.waveform.shape  # 4 mini-batches of size 3
            (4, 3, 1, 44100)
        """
        B = self.waveform.shape[0]

        # Calculate number of mini-batches (assuming B is evenly divisible)
        assert B % mini_batch_size == 0
        num_mini_batches = B // mini_batch_size

        # Reshape AudioTree to have leading mini-batch dimension
        # From (B, C, T) to (num_mini_batches, mini_batch_size, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        reshaped_audio_tree = tree_util.tree_map(
            lambda x: (
                x.reshape(num_mini_batches, mini_batch_size, *x.shape[1:])
                if hasattr(x, "shape")
                else x
            ),
            self,
        )
        return reshaped_audio_tree

    def flatten_mini_batches(self) -> Self:
        """Flatten mini-batches back into a single batch dimension.

        Undoes the operation performed by reshape_mini_batches(), transforming
        audio data from shape (num_mini_batches, mini_batch_size, C, T) back to
        (B, C, T).

        Returns:
            AudioTree with the mini-batch dimension flattened into the batch dimension.

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

        # Get the current shape
        shape = self.waveform.shape

        # We expect at least 4 dimensions for mini-batched data
        assert len(shape) >= 4, (
            f"Expected at least 4 dimensions for mini-batched data, got {len(shape)}. "
            f"Shape: {shape}"
        )

        # Flatten the first two dimensions
        # From (num_mini_batches, mini_batch_size, C, T) to (B, C, T)
        # Only reshape array-like objects since metadata can contain non-arrays
        flattened_audio_tree = tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]) if hasattr(x, "shape") else x,
            self,
        )
        return flattened_audio_tree

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
            batch axis) rather than being ``None``.

        Example:
            >>> waveform = jnp.stack([jnp.zeros((1, 8)), jnp.ones((1, 8))])
            >>> tree = AudioTree.create(waveform, 16000)
            >>> loud = tree.filter(lambda item: bool(item.waveform.max() > 0.5))
            >>> loud.batch_size
            1
        """
        filter_fn = predicate
        B = self.waveform.shape[0]
        audio_trees = self.split(B)
        audio_trees = list(filter(filter_fn, audio_trees))

        numpy = np if isinstance(self.waveform, np.ndarray) else jnp

        if len(audio_trees) == 0:
            return tree_util.tree_map(
                lambda x: x[:0] if hasattr(x, "shape") else x, self
            )

        audio_trees = tree_util.tree_map(
            lambda *xs: numpy.concatenate(xs, axis=0), *audio_trees
        )
        return audio_trees

    @staticmethod
    def batch(items: Sequence[Any]) -> Any:
        """Batch function for use with grain's IterDataset.batch().

        Concatenates AudioTree objects along the batch axis (axis 0).
        Use this instead of grain's default batching, which would add
        an extra dimension since AudioTree already has shape (batch, channels, samples).

        Supports arbitrary nested structures containing AudioTrees. All arrays
        (including AudioTrees) are concatenated along axis 0, so data should have
        a leading batch dimension.

        Args:
            items: Sequence of AudioTree objects, or structures (dicts, lists, etc.)
                containing AudioTree objects.

        Returns:
            Batched structure with the same shape as the input items.

        Example:
            >>> a = AudioTree.create(jnp.zeros((1, 1, 16000)), 16000)
            >>> batched = AudioTree.batch([a, a, a])  # concatenate along the batch axis
            >>> batched.waveform.shape
            (3, 1, 16000)

            With Grain, pass it as the ``batch_fn`` (each item already has a leading batch axis)::

                ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)
        """
        items = list(items)

        def batching_function(*args):
            first_arg = args[0]
            if isinstance(first_arg, AudioTree):
                return _batch_audiotrees(args)
            elif isinstance(first_arg, (np.ndarray, jnp.ndarray)):
                return np.concatenate(args, axis=0)
            else:
                return list(args)

        return tree_util.tree_map(
            batching_function,
            items[0],
            *items[1:],
            is_leaf=lambda x: isinstance(x, AudioTree),
        )


# --- Field lists ------------------------------------------------------------
# Derived from the dataclass rather than restated, because these were three
# hand-maintained copies that had to agree with each other and with the class.
# Order is declaration order, which is also the pytree flatten order and
# therefore the on-disk leaf order that TreeWriter records -- do not sort.

#: Fields that are pytree nodes (i.e. everything but the static ``sample_rate``).
PYTREE_FIELDS: tuple = tuple(
    f.name for f in dataclasses.fields(AudioTree) if f.metadata.get("pytree_node", True)
)

#: Pytree fields holding a single array, so ``metadata`` (a dict) is excluded.
ARRAY_FIELDS: tuple = tuple(f for f in PYTREE_FIELDS if f != "metadata")

#: Per-item labels: the array fields other than the waveform itself. These are
#: what the writers record as manifest columns.
LABEL_FIELDS: tuple = tuple(f for f in ARRAY_FIELDS if f != "waveform")


def _batch_audiotrees(audio_trees: Sequence[AudioTree]) -> AudioTree:
    """Batch a list of AudioTrees into a single AudioTree.

    Concatenates all array fields along the batch axis (axis 0) using NumPy.
    Requires all AudioTrees to have the same sample_rate and compatible shapes.

    Prefer using ``AudioTree.batch`` instead, which handles mixed-type
    structures (dicts with AudioTrees, arrays, strings, etc.).

    Args:
        audio_trees: List of AudioTree objects to batch together.

    Returns:
        Single AudioTree with all items batched along axis 0.
    """
    if not audio_trees:
        raise ValueError("Cannot batch empty list of AudioTrees")

    return tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *audio_trees)


def _numpy_integrated_lufs(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Integrated loudness (LUFS) per item of a ``(batch, channels, samples)`` waveform.

    Measured on the CPU with ``loudness.integrated_loudness``. Excerpts shorter
    than the BS.1770 gating block (400ms) are right-padded with silence so the
    measurement is valid.
    """
    min_samples = int(np.ceil(0.4 * sample_rate))
    if waveform.shape[-1] < min_samples:
        pad_right = min_samples - waveform.shape[-1]
        waveform = np.pad(waveform, ((0, 0), (0, 0), (0, pad_right)))
    audio_transposed = np.transpose(waveform, (0, 2, 1))  # [B, T, C]
    values = [
        loudness.integrated_loudness(np.ascontiguousarray(item), sample_rate)
        for item in audio_transposed
    ]
    return np.array(values, dtype=np.float32)
