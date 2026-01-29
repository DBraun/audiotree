"""Helper functions for transforms with dual numpy/JAX implementations."""

from typing import Any, Tuple

import jax
from jax import numpy as jnp
from jax import random
from jax.tree_util import DictKey
import librosa
import librosax
import numpy as np

from audiotree import AudioTree

KeyLeafPairs = list[tuple[list[DictKey], Any]]


# =============================================================================
# Backend-agnostic utilities
# =============================================================================


def _db2linear_jax(decibels):
    return jnp.pow(10.0, decibels / 20.0)


def _db2linear_np(decibels):
    return np.power(10.0, decibels / 20.0)


# =============================================================================
# Volume Norm - JAX and NumPy implementations
# =============================================================================


def _volume_norm_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> AudioTree:
    """JAX implementation of volume normalization."""
    audio_data = audio_tree.audio_data
    B = audio_data.shape[0]

    target_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)
    gain_db = target_db - audio_tree.loudness

    audio_data = audio_data * _db2linear_jax(gain_db)[:, None, None]
    return audio_tree.replace(audio_data=audio_data, loudness=target_db)


def _volume_norm_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> AudioTree:
    """NumPy implementation of volume normalization."""
    audio_data = audio_tree.audio_data
    B = audio_data.shape[0]

    target_db = rng.uniform(min_db, max_db, size=(B,)).astype(np.float32)
    loudness = audio_tree.loudness
    gain_db = target_db - loudness

    audio_data = audio_data * _db2linear_np(gain_db)[:, None, None]
    return audio_tree.replace(audio_data=audio_data, loudness=target_db)


# =============================================================================
# Volume Change - JAX and NumPy implementations
# =============================================================================


def _volume_change_jax(
    audio_tree: AudioTree, key: jax.Array, min_db: float, max_db: float
) -> Tuple[AudioTree, jnp.ndarray]:
    """JAX implementation of volume change."""
    audio_data = audio_tree.audio_data
    B = audio_data.shape[0]

    gain_db = random.uniform(key, shape=(B,), minval=min_db, maxval=max_db)

    audio_data = audio_data * _db2linear_jax(gain_db)[:, None, None]
    return audio_tree.replace(audio_data=audio_data), gain_db


def _volume_change_np(
    audio_tree: AudioTree, rng: np.random.Generator, min_db: float, max_db: float
) -> Tuple[AudioTree, np.ndarray]:
    """NumPy implementation of volume change."""
    audio_data = audio_tree.audio_data
    B = audio_data.shape[0]

    gain_db = rng.uniform(min_db, max_db, size=(B,)).astype(np.float32)

    audio_data = audio_data * _db2linear_np(gain_db)[:, None, None]
    return audio_tree.replace(audio_data=audio_data), gain_db


# =============================================================================
# Rescale Audio - JAX and NumPy implementations
# =============================================================================


def _rescale_audio_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of audio rescaling."""
    audio_data = audio_tree.audio_data
    maxes = jnp.max(jnp.absolute(audio_data), axis=[-2, -1])
    maxes = jnp.expand_dims(maxes, [-2, -1])
    maxes = jnp.maximum(maxes, jnp.ones_like(maxes))
    audio_data = audio_data / maxes
    return audio_tree.replace(audio_data=audio_data)


def _rescale_audio_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of audio rescaling."""
    audio_data = audio_tree.audio_data
    maxes = np.max(np.absolute(audio_data), axis=(-2, -1))
    maxes = np.expand_dims(maxes, (-2, -1))
    maxes = np.maximum(maxes, np.ones_like(maxes))
    audio_data = audio_data / maxes
    return audio_tree.replace(audio_data=audio_data)


# =============================================================================
# Invert Phase - JAX and NumPy implementations
# =============================================================================


def _invert_phase_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of phase inversion."""
    audio_data = -audio_tree.audio_data
    return audio_tree.replace(audio_data=audio_data)


def _invert_phase_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of phase inversion."""
    audio_data = -audio_tree.audio_data
    return audio_tree.replace(audio_data=audio_data)


# =============================================================================
# Swap Stereo - JAX and NumPy implementations
# =============================================================================


def _swap_stereo_jax(audio_tree: AudioTree) -> AudioTree:
    """JAX implementation of stereo swap."""
    audio_data = jnp.flip(audio_tree.audio_data, axis=1)
    return audio_tree.replace(audio_data=audio_data)


def _swap_stereo_np(audio_tree: AudioTree) -> AudioTree:
    """NumPy implementation of stereo swap."""
    audio_data = np.flip(audio_tree.audio_data, axis=1)
    return audio_tree.replace(audio_data=audio_data)


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
) -> AudioTree:
    """JAX implementation of phase corruption."""
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        audio_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

    amt = random.uniform(
        rng, shape=stft_data.shape[:-1], minval=-jnp.pi * amount, maxval=jnp.pi * amount
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=-1)

    audio_data = librosax.istft(
        stft_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True, length=length
    )

    return audio_tree.replace(audio_data=audio_data)


def _corrupt_phase_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
) -> AudioTree:
    """NumPy implementation of phase corruption."""
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    hop_length = int(frame_length * hop_factor)

    # Process each batch/channel separately since librosa doesn't support batched input
    result = np.zeros_like(audio_data)
    for b in range(B):
        for c in range(C):
            stft_data = librosa.stft(
                audio_data[b, c], n_fft=frame_length, hop_length=hop_length, window=window, center=True
            )

            amt = rng.uniform(-np.pi * amount, np.pi * amount, size=stft_data.shape[:-1])
            stft_data = stft_data * np.expand_dims(np.exp(1j * amt), axis=-1)

            result[b, c] = librosa.istft(
                stft_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True, length=length
            )

    return audio_tree.replace(audio_data=result)


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
) -> AudioTree:
    """JAX implementation of phase shift."""
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    hop_length = int(frame_length * hop_factor)

    stft_data = librosax.stft(
        audio_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True
    )

    amt = random.uniform(
        key,
        shape=stft_data.shape[:-2],
        minval=-jnp.pi * amount,
        maxval=jnp.pi * amount,
    )

    stft_data = stft_data * jnp.expand_dims(jnp.exp(1j * amt), axis=(-2, -1))

    audio_data = librosax.istft(
        stft_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True, length=length
    )

    return audio_tree.replace(audio_data=audio_data)


def _shift_phase_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
) -> AudioTree:
    """NumPy implementation of phase shift."""
    audio_data = audio_tree.audio_data
    B, C, length = audio_data.shape

    hop_length = int(frame_length * hop_factor)

    # Generate one phase shift per batch item
    amts = rng.uniform(-np.pi * amount, np.pi * amount, size=(B,))

    result = np.zeros_like(audio_data)
    for b in range(B):
        for c in range(C):
            stft_data = librosa.stft(
                audio_data[b, c], n_fft=frame_length, hop_length=hop_length, window=window, center=True
            )

            stft_data = stft_data * np.exp(1j * amts[b])

            result[b, c] = librosa.istft(
                stft_data, n_fft=frame_length, hop_length=hop_length, window=window, center=True, length=length
            )

    return audio_tree.replace(audio_data=result)


# =============================================================================
# Roll - JAX and NumPy implementations
# =============================================================================


def _roll_jax(
    audio_tree: AudioTree,
    rng: jax.Array,
    min_seconds: float,
    max_seconds: float,
    mode: str = "wrap",
) -> AudioTree:
    """JAX implementation of audio roll."""
    B, C, T = audio_tree.audio_data.shape

    min_samples = int(min_seconds * audio_tree.sample_rate)
    max_samples = int(max_seconds * audio_tree.sample_rate)

    roll_amounts = random.randint(
        rng, shape=(B,), minval=min_samples, maxval=max_samples + 1
    )

    @jax.vmap
    def roll_single_item(item_audio: jnp.ndarray, roll_amount: jnp.ndarray) -> jnp.ndarray:
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

    rolled_audio = roll_single_item(audio_tree.audio_data, roll_amounts)
    return audio_tree.replace(audio_data=rolled_audio)


def _roll_np(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_seconds: float,
    max_seconds: float,
    mode: str = "wrap",
) -> AudioTree:
    """NumPy implementation of audio roll."""
    audio_data = audio_tree.audio_data
    B, C, T = audio_data.shape

    min_samples = int(min_seconds * audio_tree.sample_rate)
    max_samples = int(max_seconds * audio_tree.sample_rate)

    roll_amounts = rng.integers(min_samples, max_samples + 1, size=(B,))

    result = np.zeros_like(audio_data)
    for b in range(B):
        if mode == "wrap":
            result[b] = np.roll(audio_data[b], shift=roll_amounts[b], axis=-1)
        elif mode == "constant":
            roll_amt = roll_amounts[b]
            if roll_amt >= 0:
                if roll_amt < T:
                    result[b, :, roll_amt:] = audio_data[b, :, :T - roll_amt]
            else:
                if -roll_amt < T:
                    result[b, :, :T + roll_amt] = audio_data[b, :, -roll_amt:]
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'wrap' or 'constant'.")

    return audio_tree.replace(audio_data=result)


# =============================================================================
# Trim - JAX and NumPy implementations
# =============================================================================


def _trim_jax(audio_tree: AudioTree, length: float, mode: str = "wrap") -> AudioTree:
    """JAX implementation of trim/pad."""
    audio_data = audio_tree.audio_data
    T = audio_data.shape[-1]
    target_T = int(length * audio_tree.sample_rate)

    if T < target_T:
        audio_data = jnp.pad(
            audio_data,
            pad_width=((0, 0), (0, 0), (0, target_T - T)),
            mode=mode,
        )
    elif T > target_T:
        audio_data = audio_data[..., :target_T]

    return audio_tree.replace(audio_data=audio_data)


def _trim_np(audio_tree: AudioTree, length: float, mode: str = "wrap") -> AudioTree:
    """NumPy implementation of trim/pad."""
    audio_data = audio_tree.audio_data
    T = audio_data.shape[-1]
    target_T = int(length * audio_tree.sample_rate)

    if T < target_T:
        audio_data = np.pad(
            audio_data,
            pad_width=((0, 0), (0, 0), (0, target_T - T)),
            mode=mode,
        )
    elif T > target_T:
        audio_data = audio_data[..., :target_T]

    return audio_tree.replace(audio_data=audio_data)
