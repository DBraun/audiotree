"""Helper functions for transforms with dual numpy/JAX implementations."""

from typing import Tuple

import jax
from jax import numpy as jnp
from jax import random
import librosa
import librosax
import numpy as np

from audiotree import AudioTree
from audiotree.loudness import safe_gain_db, shift_lufs, shift_lufs_windows


# =============================================================================
# Backend-agnostic utilities
# =============================================================================


def _db2linear_jax(decibels):
    return jnp.pow(10.0, decibels / 20.0)


def _db2linear_np(decibels):
    return np.power(10.0, decibels / 20.0)


def _check_uniform_range(
    name: str, min_name: str, max_name: str, min_value: float, max_value: float
) -> None:
    """Reject an inverted sampling range so both backends fail identically.

    The NumPy backends draw with ``rng.uniform(min, max)``, which raises
    numpy's generic ``ValueError: high - low < 0`` for an inverted range --
    on the first element, inside a grain worker. ``jax.random.uniform`` does
    not error at all: it silently clamps every draw to ``minval``, so the
    transform applies a constant parameter instead. Validating here, ahead of
    either draw, keeps the two backends consistent with a single clear error
    (see ``_check_roll_range`` for the integer-draw equivalent).
    """
    if min_value > max_value:
        raise ValueError(
            f"{name} requires {min_name} <= {max_name}, but got "
            f"{min_name}={min_value} > {max_name}={max_value}."
        )


def _check_phase_amount(name: str, amount: float) -> None:
    """Reject a negative phase amount so both backends fail identically.

    The phase transforms draw from ``[-pi * amount, pi * amount]``, which is
    an inverted range when ``amount`` is negative: numpy raises its generic
    ``high - low < 0`` and ``jax.random.uniform`` silently pins every draw to
    ``minval``. A negative amount also has no meaning of its own -- the range
    is symmetric, so ``-a`` could only ever mean ``a``.
    """
    if amount < 0:
        raise ValueError(
            f"{name} requires amount >= 0, but got amount={amount}: the phase "
            f"offsets are drawn from [-pi * amount, pi * amount], which is "
            f"inverted for a negative amount."
        )


def _check_stft_length(name: str, num_samples: int, frame_length: int) -> None:
    """Reject audio shorter than one STFT frame so both backends fail identically.

    On a too-short clip ``librosa.stft`` only warns (``n_fft=... is too large
    for input signal``) and zero-pads, while ``librosax.stft`` raises jax's
    ``ValueError: window is longer than input signal`` -- and whether the JAX
    path survives depends on how much ``_pad_to_hop_multiple`` happens to pad.
    A clip shorter than one frame cannot be analyzed at the requested
    resolution, so both backends refuse it identically instead of one of them
    silently analyzing zero-padding.
    """
    if num_samples < frame_length:
        raise ValueError(
            f"{name} requires audio of at least frame_length={frame_length} "
            f"samples, but got {num_samples} samples. Use a shorter "
            f"frame_length or longer audio."
        )


# =============================================================================
# Volume Norm - JAX and NumPy implementations
# =============================================================================


def _volume_norm_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> AudioTree:
    """JAX implementation of volume normalization."""
    _check_uniform_range("volume_norm", "min_db", "max_db", min_db, max_db)
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    target_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)
    # Silent items (``-inf`` LUFS) get a 0 dB gain and keep their ``-inf``, rather
    # than being scaled by ``+inf`` into an all-NaN waveform.
    gain_db = safe_gain_db(audio_tree.lufs, target_db, xp=jnp)

    waveform = waveform * _db2linear_jax(gain_db)[:, None, None]
    return audio_tree.replace(
        waveform=waveform,
        lufs=shift_lufs(audio_tree.lufs, gain_db, xp=jnp),
        lufs_windows=shift_lufs_windows(audio_tree.lufs_windows, gain_db, xp=jnp),
    )


def _volume_norm_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> AudioTree:
    """NumPy implementation of volume normalization."""
    _check_uniform_range("volume_norm", "min_db", "max_db", min_db, max_db)
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    target_db = rng.uniform(min_db, max_db, size=(B,)).astype(np.float32)
    # See ``_volume_norm_jax``: keep silence silent instead of producing NaN.
    gain_db = safe_gain_db(audio_tree.lufs, target_db, xp=np)

    waveform = waveform * _db2linear_np(gain_db)[:, None, None]
    return audio_tree.replace(
        waveform=waveform,
        lufs=shift_lufs(audio_tree.lufs, gain_db, xp=np),
        lufs_windows=shift_lufs_windows(audio_tree.lufs_windows, gain_db, xp=np),
    )


# =============================================================================
# Volume Change - JAX and NumPy implementations
# =============================================================================


def _volume_change_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> Tuple[AudioTree, jnp.ndarray]:
    """JAX implementation of volume change."""
    _check_uniform_range("volume_change", "min_db", "max_db", min_db, max_db)
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    gain_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)

    waveform = waveform * _db2linear_jax(gain_db)[:, None, None]
    return audio_tree.replace(waveform=waveform), gain_db


def _volume_change_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> Tuple[AudioTree, np.ndarray]:
    """NumPy implementation of volume change."""
    _check_uniform_range("volume_change", "min_db", "max_db", min_db, max_db)
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    gain_db = rng.uniform(min_db, max_db, size=(B,)).astype(np.float32)

    waveform = waveform * _db2linear_np(gain_db)[:, None, None]
    return audio_tree.replace(waveform=waveform), gain_db


# =============================================================================
# Rescale Audio - JAX and NumPy implementations
# =============================================================================


def _rescale_audio_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of audio rescaling."""
    waveform = audio_tree.waveform
    maxes = jnp.max(jnp.absolute(waveform), axis=[-2, -1])
    maxes = jnp.expand_dims(maxes, [-2, -1])
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    waveform = waveform / maxes
    return audio_tree.replace(waveform=waveform)


def _rescale_audio_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of audio rescaling."""
    waveform = audio_tree.waveform
    maxes = np.max(np.absolute(waveform), axis=(-2, -1))
    maxes = np.expand_dims(maxes, (-2, -1))
    maxes = np.maximum(maxes, np.ones_like(maxes))
    waveform = waveform / maxes
    return audio_tree.replace(waveform=waveform)


# =============================================================================
# Invert Phase - JAX and NumPy implementations
# =============================================================================


def _peak_norm_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of peak normalization."""
    waveform = audio_tree.waveform
    peaks = jnp.max(jnp.absolute(waveform), axis=[-2, -1])
    peaks = jnp.expand_dims(peaks, [-2, -1])
    peaks = jnp.maximum(peaks, 1e-8)
    waveform = waveform / peaks
    return audio_tree.replace(waveform=waveform)


def _peak_norm_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of peak normalization."""
    waveform = audio_tree.waveform
    peaks = np.max(np.absolute(waveform), axis=(-2, -1))
    peaks = np.expand_dims(peaks, (-2, -1))
    peaks = np.maximum(peaks, 1e-8)
    waveform = waveform / peaks
    return audio_tree.replace(waveform=waveform)


def _invert_phase_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of phase inversion."""
    waveform = -audio_tree.waveform
    return audio_tree.replace(waveform=waveform)


def _invert_phase_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of phase inversion."""
    waveform = -audio_tree.waveform
    return audio_tree.replace(waveform=waveform)


# =============================================================================
# Swap Stereo - JAX and NumPy implementations
# =============================================================================


def _check_swappable(audio_tree: AudioTree) -> None:
    """Reject audio with more channels than a swap is defined for.

    Exchanging left and right is defined for stereo, and is the identity for
    mono (there is only one channel, so the only permutation of the channels is
    the trivial one). For three or more channels there is no such thing as
    "the" swap: reversing the channel order, which is what this used to do,
    turns 5.1 into nonsense rather than swapping its front pair. The channel
    count is static under ``jax.jit``, so this raises at trace time on either
    backend.
    """
    channels = audio_tree.waveform.shape[1]
    if channels > 2:
        raise ValueError(
            f"swap_stereo is defined for mono (no-op) and stereo audio, but got "
            f"{channels} channels: which channels to exchange is undefined. Use "
            f"`mono()` or `stereo()` first if a swap is what you want."
        )


def _swap_stereo_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of stereo swap."""
    _check_swappable(audio_tree)
    waveform = jnp.flip(audio_tree.waveform, axis=1)
    return audio_tree.replace(waveform=waveform)


def _swap_stereo_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of stereo swap."""
    _check_swappable(audio_tree)
    waveform = np.flip(audio_tree.waveform, axis=1)
    return audio_tree.replace(waveform=waveform)


# =============================================================================
# Corrupt Phase - JAX and NumPy implementations
# =============================================================================


def _pad_to_hop_multiple(waveform: jnp.ndarray, hop_length: int) -> jnp.ndarray:
    """Right-pad a waveform with zeros up to a whole number of hops.

    ``librosax.istft`` reconstructs only ``(n_frames - 1) * hop_length`` samples
    and zero-fills the remainder of whatever ``length=`` asks for, so a signal
    whose length is not a multiple of ``hop_length`` comes back with its last
    ``length % hop_length`` samples silenced -- 68 samples of a 1 s @ 44.1 kHz
    clip at ``hop_length=1024``, which is audible. Padding up to the next
    multiple of the hop buys the one extra analysis frame that covers the tail;
    passing the original ``length`` to ``istft`` then trims the padding back
    off. ``librosa.istft`` reconstructs the tail itself, so the NumPy backend
    needs no such padding.
    """
    pad = -waveform.shape[-1] % hop_length
    if pad == 0:
        return waveform
    return jnp.pad(waveform, ((0, 0), (0, 0), (0, pad)))


def _corrupt_phase_jax(
    audio_tree: AudioTree,
    rng: jax.Array,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """JAX implementation of phase corruption."""
    _check_phase_amount("corrupt_phase", amount)
    waveform = audio_tree.waveform
    B, C, length = waveform.shape
    _check_stft_length("corrupt_phase", length, frame_length)

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        _pad_to_hop_multiple(waveform, hop_length),
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
    )

    # One random phase offset per (batch, channel, frequency), broadcast over frames.
    amt = random.uniform(
        rng, shape=stft_data.shape[:-1], minval=-jnp.pi * amount, maxval=jnp.pi * amount
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=-1)

    waveform = librosax.istft(
        stft_data,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    )

    lufs = audio_tree.lufs if keep_lufs else None
    lufs_windows = audio_tree.lufs_windows if keep_lufs else None
    return audio_tree.replace(waveform=waveform, lufs=lufs, lufs_windows=lufs_windows)


def _corrupt_phase_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """NumPy implementation of phase corruption."""
    _check_phase_amount("corrupt_phase", amount)
    waveform = audio_tree.waveform
    B, C, length = waveform.shape
    _check_stft_length("corrupt_phase", length, frame_length)

    hop_length = int(frame_length * hop_factor)

    # librosa's stft/istft operate on the last axis and broadcast over any leading
    # dims, so the whole (B, C, T) batch runs in a single call.
    stft_data = librosa.stft(
        waveform,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
    )  # (B, C, freq, frames)

    # One random phase offset per (batch, channel, frequency), broadcast over frames.
    amt = rng.uniform(-np.pi * amount, np.pi * amount, size=stft_data.shape[:-1])
    stft_data = stft_data * np.expand_dims(np.exp(1j * amt), axis=-1)

    result = librosa.istft(
        stft_data,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    ).astype(waveform.dtype)

    lufs = audio_tree.lufs if keep_lufs else None
    lufs_windows = audio_tree.lufs_windows if keep_lufs else None
    return audio_tree.replace(waveform=result, lufs=lufs, lufs_windows=lufs_windows)


# =============================================================================
# Shift Phase - JAX and NumPy implementations
# =============================================================================


def _shift_phase_jax(
    audio_tree: AudioTree,
    key: jax.Array,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """JAX implementation of phase shift."""
    _check_phase_amount("shift_phase", amount)
    waveform = audio_tree.waveform
    B, C, length = waveform.shape
    _check_stft_length("shift_phase", length, frame_length)

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        _pad_to_hop_multiple(waveform, hop_length),
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
    )

    # One phase shift per batch item, broadcast over channels/frequencies/frames.
    # Drawing one per channel instead would rotate the two halves of a stereo
    # image by different angles, decorrelating a pair that started identical.
    amts = random.uniform(
        key,
        shape=(B,),
        minval=-jnp.pi * amount,
        maxval=jnp.pi * amount,
    )

    stft_data = stft_data * jnp.exp(1j * amts)[:, None, None, None]

    waveform = librosax.istft(
        stft_data,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    )

    lufs = audio_tree.lufs if keep_lufs else None
    lufs_windows = audio_tree.lufs_windows if keep_lufs else None
    return audio_tree.replace(waveform=waveform, lufs=lufs, lufs_windows=lufs_windows)


def _shift_phase_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """NumPy implementation of phase shift."""
    _check_phase_amount("shift_phase", amount)
    waveform = audio_tree.waveform
    B, C, length = waveform.shape
    _check_stft_length("shift_phase", length, frame_length)

    hop_length = int(frame_length * hop_factor)

    # One phase shift per batch item, broadcast over channels/frequencies/frames.
    amts = rng.uniform(-np.pi * amount, np.pi * amount, size=(B,))

    # librosa's stft/istft operate on the last axis and broadcast over any leading
    # dims, so the whole (B, C, T) batch runs in a single call.
    stft_data = librosa.stft(
        waveform,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
    )  # (B, C, freq, frames)
    stft_data = stft_data * np.exp(1j * amts)[:, None, None, None]

    result = librosa.istft(
        stft_data,
        n_fft=frame_length,
        hop_length=hop_length,
        window=window,
        center=True,
        length=length,
    ).astype(waveform.dtype)

    lufs = audio_tree.lufs if keep_lufs else None
    lufs_windows = audio_tree.lufs_windows if keep_lufs else None
    return audio_tree.replace(waveform=result, lufs=lufs, lufs_windows=lufs_windows)


# =============================================================================
# Roll - JAX and NumPy implementations
# =============================================================================


def _check_roll_range(min_seconds: float, max_seconds: float) -> None:
    """Reject an inverted roll range so both backends fail identically.

    ``_roll_np`` draws with ``rng.integers(min, max + 1)``, which raises
    ``ValueError: low >= high`` for an inverted range, while ``_roll_jax`` draws
    with ``jax.random.randint``, which silently returns ``minval`` for every draw
    when ``minval > maxval`` -- applying a constant roll of ``min_seconds``
    instead of erroring. Validating here, ahead of either draw, keeps the two
    backends consistent with a single clear error. Negative bounds are allowed
    (a negative roll shifts left), so only their ordering is checked.
    """
    if min_seconds > max_seconds:
        raise ValueError(
            f"roll requires min_seconds <= max_seconds, but got "
            f"min_seconds={min_seconds} > max_seconds={max_seconds}."
        )


def _invalidate_offset(metadata: dict) -> dict:
    """Drop a stale source-file ``offset`` after a time shift.

    ``AudioTree.from_file`` records ``metadata["offset"]`` as the source-file
    time (in seconds) of sample 0. Rolling shifts the waveform along the time
    axis, so that recorded offset no longer points at sample 0 and is set to
    ``None``. Metadata without an ``offset`` key is returned unchanged.

    Args:
        metadata: The AudioTree metadata dict to inspect.

    Returns:
        The metadata dict, with ``offset`` invalidated to ``None`` if present.
    """
    if "offset" not in metadata:
        return metadata
    return {**metadata, "offset": None}


def _roll_jax(
    audio_tree: AudioTree,
    rng: jax.Array,
    min_seconds: float,
    max_seconds: float,
    mode: str = "wrap",
) -> AudioTree:
    """JAX implementation of audio roll."""
    _check_roll_range(min_seconds, max_seconds)
    B, C, T = audio_tree.waveform.shape

    min_samples = int(min_seconds * audio_tree.sample_rate)
    max_samples = int(max_seconds * audio_tree.sample_rate)

    roll_amounts = random.randint(
        rng, shape=(B,), minval=min_samples, maxval=max_samples + 1
    )

    @jax.vmap
    def roll_single_item(
        item_audio: jnp.ndarray, roll_amount: jnp.ndarray
    ) -> jnp.ndarray:
        if mode == "wrap":
            return jnp.roll(item_audio, shift=roll_amount, axis=-1)
        elif mode == "constant":
            indices = jnp.arange(T)
            rolled_indices = indices - roll_amount
            valid_mask = (rolled_indices >= 0) & (rolled_indices < T)
            rolled_item = jnp.where(
                valid_mask[None, :],
                item_audio[:, jnp.clip(rolled_indices, 0, T - 1)],
                0.0,
            )
            return rolled_item
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'wrap' or 'constant'.")

    rolled_audio = roll_single_item(audio_tree.waveform, roll_amounts)
    # "wrap" reorders existing samples (integrated loudness preserved); "constant"
    # zeros out part of the signal, changing it. Either way, rolling moves samples
    # across window boundaries, so ``lufs_windows`` is invalidated below.
    lufs = None if mode == "constant" else audio_tree.lufs
    metadata = _invalidate_offset(audio_tree.metadata)
    return audio_tree.replace(
        waveform=rolled_audio, lufs=lufs, lufs_windows=None, metadata=metadata
    )


def _roll_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_seconds: float,
    max_seconds: float,
    mode: str = "wrap",
) -> AudioTree:
    """NumPy implementation of audio roll."""
    _check_roll_range(min_seconds, max_seconds)
    waveform = audio_tree.waveform
    B, C, T = waveform.shape

    min_samples = int(min_seconds * audio_tree.sample_rate)
    max_samples = int(max_seconds * audio_tree.sample_rate)

    roll_amounts = rng.integers(min_samples, max_samples + 1, size=(B,))

    result = np.zeros_like(waveform)
    for b in range(B):
        if mode == "wrap":
            result[b] = np.roll(waveform[b], shift=roll_amounts[b], axis=-1)
        elif mode == "constant":
            roll_amt = roll_amounts[b]
            if roll_amt >= 0:
                if roll_amt < T:
                    result[b, :, roll_amt:] = waveform[b, :, : T - roll_amt]
            else:
                if -roll_amt < T:
                    result[b, :, : T + roll_amt] = waveform[b, :, -roll_amt:]
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'wrap' or 'constant'.")

    # "wrap" reorders existing samples (integrated loudness preserved); "constant"
    # zeros out part of the signal, changing it. Either way, rolling moves samples
    # across window boundaries, so ``lufs_windows`` is invalidated below.
    lufs = None if mode == "constant" else audio_tree.lufs
    metadata = _invalidate_offset(audio_tree.metadata)
    return audio_tree.replace(
        waveform=result, lufs=lufs, lufs_windows=None, metadata=metadata
    )


# =============================================================================
# Trim - JAX and NumPy implementations
# =============================================================================


def _trim_jax(audio_tree: AudioTree, length: float, mode: str = "wrap") -> AudioTree:
    """JAX implementation of trim/pad."""
    waveform = audio_tree.waveform
    T = waveform.shape[-1]
    target_T = int(length * audio_tree.sample_rate)

    if T == target_T:
        return audio_tree
    elif T < target_T:
        waveform = jnp.pad(
            waveform,
            pad_width=((0, 0), (0, 0), (0, target_T - T)),
            mode=mode,
        )
    else:
        waveform = waveform[..., :target_T]

    # Changing the audio length changes its integrated loudness.
    return audio_tree.replace(waveform=waveform, lufs=None, lufs_windows=None)


def _trim_np(audio_tree: AudioTree, length: float, mode: str = "wrap") -> AudioTree:
    """NumPy implementation of trim/pad."""
    waveform = audio_tree.waveform
    T = waveform.shape[-1]
    target_T = int(length * audio_tree.sample_rate)

    if T == target_T:
        return audio_tree
    elif T < target_T:
        waveform = np.pad(
            waveform,
            pad_width=((0, 0), (0, 0), (0, target_T - T)),
            mode=mode,
        )
    else:
        waveform = waveform[..., :target_T]

    # Changing the audio length changes its integrated loudness.
    return audio_tree.replace(waveform=waveform, lufs=None, lufs_windows=None)
