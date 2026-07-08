"""JAX-based transforms for GPU/JIT usage.

These transforms use JAX operations and are designed for use inside @jax.jit functions.
They work on batched AudioTree data (leading batch axis).

Example:
    from audiotree.transforms import jax as jax_transforms
    import argbind

    # Bind all transforms for argbind configuration
    transforms_lib = argbind.bind_module(jax_transforms)

    @argbind.bind("train", "val")
    def augment_batch(rng, batch, transforms: list[str] = None):
        for transform_name in transforms or []:
            transform = getattr(transforms_lib, transform_name)()
            if hasattr(transform, "random_map"):
                rng, subkey = jax.random.split(rng)
                batch = transform.random_map(batch, subkey)
            elif hasattr(transform, "map"):
                batch = transform.map(batch)
        return batch
"""

from typing import Callable

from einops import rearrange
import jax
from jax import numpy as jnp

from audiotree import AudioTree
from audiotree.transforms.decorators import random_transform, map_transform
from audiotree.transforms.helpers import (
    _volume_norm_jax,
    _volume_change_jax,
    _rescale_audio_jax,
    _peak_norm_jax,
    _invert_phase_jax,
    _swap_stereo_jax,
    _corrupt_phase_jax,
    _shift_phase_jax,
    _roll_jax,
    _trim_jax,
    _shift_lufs_windows,
)


@random_transform
def volume_norm(
    audio_tree: AudioTree,
    rng: jax.Array,
    min_db: float = 0.0,
    max_db: float = 0.0,
) -> AudioTree:
    """Normalize volume to a randomly selected loudness value specified in LUFS.

    Args:
        audio_tree: Input audio to normalize
        rng: JAX random key
        min_db: Minimum target loudness in LUFS
        max_db: Maximum target loudness in LUFS

    Returns:
        AudioTree with normalized loudness
    """
    audio_tree = audio_tree.replace_lufs()
    return _volume_norm_jax(audio_tree, rng, min_db, max_db)


@random_transform
def volume_change(
    audio_tree: AudioTree,
    rng: jax.Array,
    min_db: float = 0.0,
    max_db: float = 0.0,
) -> AudioTree:
    """Change the volume by a uniformly randomly selected decibel value.

    Args:
        audio_tree: Input audio
        rng: JAX random key
        min_db: Minimum gain change in dB
        max_db: Maximum gain change in dB

    Returns:
        AudioTree with volume changed
    """
    audio_tree, gain_db = _volume_change_jax(audio_tree, rng, min_db, max_db)
    if audio_tree.lufs is not None:
        audio_tree = audio_tree.replace(
            lufs=(audio_tree.lufs + gain_db),
            lufs_windows=_shift_lufs_windows(audio_tree.lufs_windows, gain_db),
        )
    return audio_tree


@random_transform
def invert_phase(audio_tree: AudioTree, rng: jax.Array) -> AudioTree:
    """Invert the phase of all channels of audio.

    Args:
        audio_tree: Input audio
        rng: JAX random key (unused but required for random_transform)

    Returns:
        AudioTree with inverted phase
    """
    return _invert_phase_jax(audio_tree)


@map_transform
def trim(audio_tree: AudioTree, length: float = 1.0, mode: str = "wrap") -> AudioTree:
    """Adjust audio length to a fixed length in seconds.

    Args:
        audio_tree: Input audio
        length: Target length in seconds
        mode: Padding mode - "wrap" (circular) or "constant" (zero padding)

    Returns:
        AudioTree adjusted to target length
    """
    return _trim_jax(audio_tree, length, mode)


@map_transform
def mono(audio_tree: AudioTree) -> AudioTree:
    """Convert audio to mono by averaging channels."""
    return audio_tree.to_mono()


@map_transform
def stereo(audio_tree: AudioTree) -> AudioTree:
    """Convert audio to stereo by duplicating mono channel."""
    return audio_tree.to_stereo()


@map_transform
def resample(audio_tree: AudioTree, sample_rate: int | None = None) -> AudioTree:
    """Resample audio to a new sample rate (JAX/Julius port for JAX waveforms)."""
    if sample_rate is None:
        raise ValueError(
            "resample requires a target sample_rate, e.g., resample(sample_rate=16000)."
        )
    return audio_tree.resample(sample_rate)


@map_transform
def identity(audio_tree: AudioTree) -> AudioTree:
    """Return audio without any modifications."""
    return audio_tree


@map_transform
def rescale_audio(audio_tree: AudioTree) -> AudioTree:
    """Rescale audio so the largest absolute value is 1.0."""
    return _rescale_audio_jax(audio_tree).replace(lufs=None, lufs_windows=None)


@map_transform
def peak_norm(audio_tree: AudioTree) -> AudioTree:
    """Peak-normalize audio so the largest absolute value is 1.0.

    Unlike :func:`rescale_audio`, which only scales down audio that exceeds the
    [-1.0, 1.0] range, this always divides by the peak (clamped to a small
    epsilon) so the result peaks at 1.0. The peak is computed per item in the
    batch, across channels and samples.
    """
    return _peak_norm_jax(audio_tree).replace(lufs=None, lufs_windows=None)


@random_transform
def swap_stereo(audio_tree: AudioTree, rng: jax.Array) -> AudioTree:
    """Swap the channels of stereo audio.

    Args:
        audio_tree: Input audio (must be stereo)
        rng: JAX random key (unused but required for random_transform)

    Returns:
        AudioTree with swapped channels
    """
    return _swap_stereo_jax(audio_tree)


@random_transform
def corrupt_phase(
    audio_tree: AudioTree,
    rng: jax.Array,
    amount: float = 1.0,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """Perform phase corruption on audio.

    Args:
        audio_tree: Input audio
        rng: JAX random key
        amount: Maximum phase shift in multiples of pi (0.0 to 1.0)
        hop_factor: Hop size as fraction of frame_length
        frame_length: STFT frame length in samples
        window: Window function name
        keep_lufs: If True, preserve the cached ``lufs`` and ``lufs_windows``.
            Phase corruption leaves the magnitude spectrum (and thus energy)
            intact, so loudness is approximately unchanged; the cached values
            are invalidated by default to be safe.

    Returns:
        AudioTree with corrupted phase
    """
    return _corrupt_phase_jax(
        audio_tree, rng, amount, hop_factor, frame_length, window, keep_lufs
    )


@random_transform
def shift_phase(
    audio_tree: AudioTree,
    rng: jax.Array,
    amount: float = 1.0,
    keep_lufs: bool = False,
) -> AudioTree:
    """Perform a phase shift on audio.

    Args:
        audio_tree: Input audio
        rng: JAX random key
        amount: Maximum phase shift in multiples of pi
        keep_lufs: If True, preserve the cached ``lufs`` and ``lufs_windows``. A
            phase shift leaves the magnitude spectrum (and thus energy) intact, so
            loudness is approximately unchanged; the cached values are invalidated
            by default to be safe.

    Returns:
        AudioTree with shifted phase
    """
    return _shift_phase_jax(audio_tree, rng, amount, keep_lufs=keep_lufs)


@random_transform
def roll(
    audio_tree: AudioTree,
    rng: jax.Array,
    min_seconds: float = 0.0,
    max_seconds: float = 0.0,
    mode: str = "wrap",
) -> AudioTree:
    """Apply a circular shift (roll) to audio data.

    Args:
        audio_tree: Input audio
        rng: JAX random key
        min_seconds: Minimum roll amount in seconds (negative = left)
        max_seconds: Maximum roll amount in seconds (positive = right)
        mode: Padding mode - "wrap" (circular) or "constant" (zero padding)

    Returns:
        AudioTree with rolled audio
    """
    return _roll_jax(audio_tree, rng, min_seconds, max_seconds, mode)


def encode_with_codec(
    encoder_fn: Callable[[AudioTree], jnp.ndarray],
    num_codebooks: int,
):
    """Create a transform that encodes audio using a neural codec.

    Args:
        encoder_fn: Function that takes AudioTree and returns tokens
        num_codebooks: Number of codebooks in the codec

    Returns:
        Transform function that can be used with .map()
    """

    @map_transform
    def _encode_with_codec_transform(audio_tree: AudioTree) -> AudioTree:
        if audio_tree.codes is None:
            B, C, T = audio_tree.waveform.shape
            codes = encoder_fn(audio_tree)
            codes = rearrange(
                codes,
                "(B C) K S -> B (K C) S",
                B=B,
                C=C,
            )
            audio_tree = audio_tree.replace(codes=codes)
        return audio_tree

    return _encode_with_codec_transform()


def encode_latents(encoder_fn: Callable[[AudioTree], jnp.ndarray]):
    """Create a transform that encodes audio using a neural network.

    Args:
        encoder_fn: Function that takes AudioTree and returns latent sequence

    Returns:
        Transform function that can be used with .map()
    """

    @map_transform
    def _encode_latents_transform(audio_tree: AudioTree) -> AudioTree:
        if audio_tree.latents is None:
            latents = encoder_fn(audio_tree)
            audio_tree = audio_tree.replace(latents=latents)
        return audio_tree

    return _encode_latents_transform()
