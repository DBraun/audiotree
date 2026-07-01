"""Helper functions for transforms with dual numpy/JAX implementations."""

from typing import Tuple

import jax
from jax import numpy as jnp
from jax import random
import librosa
import librosax
import numpy as np

from audiotree import AudioTree


# =============================================================================
# Backend-agnostic utilities
# =============================================================================


def _db2linear_jax(decibels):
    return jnp.pow(10.0, decibels / 20.0)


def _db2linear_np(decibels):
    return np.power(10.0, decibels / 20.0)


def _shift_lufs_windows(lufs_windows, gain_db):
    """Shift per-window LUFS by a per-item dB gain (a constant gain shifts every
    window equally), or return ``None`` if ``lufs_windows`` is not populated.

    Works for both NumPy and JAX arrays: ``gain_db`` is ``(batch,)`` and
    broadcasts over the window axis.
    """
    if lufs_windows is None:
        return None
    return lufs_windows + gain_db[:, None]


# =============================================================================
# Volume Norm - JAX and NumPy implementations
# =============================================================================


def _volume_norm_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> AudioTree:
    """JAX implementation of volume normalization."""
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    target_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.lufs

    waveform = waveform * _db2linear_jax(gain_db)[:, None, None]
    lufs_windows = _shift_lufs_windows(audio_tree.lufs_windows, gain_db)
    return audio_tree.replace(
        waveform=waveform, lufs=target_db, lufs_windows=lufs_windows
    )


def _volume_norm_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> AudioTree:
    """NumPy implementation of volume normalization."""
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    target_db = rng.uniform(min_db, max_db, size=(B,)).astype(np.float32)
    gain_db = target_db - audio_tree.lufs

    waveform = waveform * _db2linear_np(gain_db)[:, None, None]
    lufs_windows = _shift_lufs_windows(audio_tree.lufs_windows, gain_db)
    return audio_tree.replace(
        waveform=waveform, lufs=target_db, lufs_windows=lufs_windows
    )


# =============================================================================
# Volume Change - JAX and NumPy implementations
# =============================================================================


def _volume_change_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> Tuple[AudioTree, jnp.ndarray]:
    """JAX implementation of volume change."""
    waveform = audio_tree.waveform
    B = waveform.shape[0]

    gain_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)

    waveform = waveform * _db2linear_jax(gain_db)[:, None, None]
    return audio_tree.replace(waveform=waveform), gain_db


def _volume_change_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> Tuple[AudioTree, np.ndarray]:
    """NumPy implementation of volume change."""
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


def _swap_stereo_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of stereo swap."""
    waveform = jnp.flip(audio_tree.waveform, axis=1)
    return audio_tree.replace(waveform=waveform)


def _swap_stereo_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of stereo swap."""
    waveform = np.flip(audio_tree.waveform, axis=1)
    return audio_tree.replace(waveform=waveform)


# =============================================================================
# Corrupt Phase - JAX and NumPy implementations
# =============================================================================


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
    waveform = audio_tree.waveform
    B, C, length = waveform.shape

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        waveform, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

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
    waveform = audio_tree.waveform
    B, C, length = waveform.shape

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
    waveform = audio_tree.waveform
    B, C, length = waveform.shape

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        waveform, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

    amt = random.uniform(
        key,
        shape=stft_data.shape[:-2],
        minval=-jnp.pi * amount,
        maxval=jnp.pi * amount,
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=(-2, -1))

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
    waveform = audio_tree.waveform
    B, C, length = waveform.shape

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
