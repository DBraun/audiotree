"""NumPy-based transforms for grain data pipelines (CPU).

These transforms use NumPy operations and are designed for use with grain's
.random_map() and .map() methods. For GPU/JIT usage, use audiotree.transforms.jax.

Example:
    from audiotree.transforms import volume_norm, trim

    # Direct usage
    transform = volume_norm(min_db=-20, max_db=-15)
    ds = ds.random_map(transform, seed=42)

    # With argbind
    import argbind
    volume_norm = argbind.bind(volume_norm)

    args = argbind.parse_args()
    with argbind.scope(args):
        transform = volume_norm()
        ds = ds.random_map(transform, seed=42)
"""

from typing import Callable

from einops import rearrange
import grain
from jax import numpy as jnp
import numpy as np

from audiotree import AudioTree
from audiotree.transforms.decorators import random_transform, map_transform
from audiotree.transforms.helpers import (
    _volume_norm_np,
    _volume_change_np,
    _rescale_audio_np,
    _peak_norm_np,
    _invert_phase_np,
    _swap_stereo_np,
    _corrupt_phase_np,
    _shift_phase_np,
    _roll_np,
    _trim_np,
    _shift_lufs_windows,
)


@random_transform
def volume_norm(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_db: float = 0.0,
    max_db: float = 0.0,
) -> AudioTree:
    """Normalize volume to a randomly selected loudness value specified in LUFS.

    Args:
        audio_tree: Input audio to normalize
        rng: numpy random Generator
        min_db: Minimum target loudness in LUFS
        max_db: Maximum target loudness in LUFS

    Returns:
        AudioTree with normalized loudness

    Example:
        transform = volume_norm(min_db=-20, max_db=-15)
        ds = ds.random_map(transform, seed=42)
    """
    audio_tree = audio_tree.replace_lufs()
    return _volume_norm_np(audio_tree, rng, min_db, max_db)


@random_transform
def volume_change(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_db: float = 0.0,
    max_db: float = 0.0,
) -> AudioTree:
    """Change the volume by a uniformly randomly selected decibel value.

    Args:
        audio_tree: Input audio
        rng: numpy random Generator
        min_db: Minimum gain change in dB
        max_db: Maximum gain change in dB

    Returns:
        AudioTree with volume changed

    Example:
        transform = volume_change(min_db=-12, max_db=12, prob=0.9)
        ds = ds.random_map(transform, seed=42)
    """
    audio_tree, gain_db = _volume_change_np(audio_tree, rng, min_db, max_db)
    if audio_tree.lufs is not None:
        audio_tree = audio_tree.replace(
            lufs=(audio_tree.lufs + gain_db),
            lufs_windows=_shift_lufs_windows(audio_tree.lufs_windows, gain_db),
        )
    return audio_tree


@random_transform
def invert_phase(
    audio_tree: AudioTree,
    rng: np.random.Generator,
) -> AudioTree:
    """Invert the phase of all channels of audio.

    For data augmentation, it's common to use prob=0.5 to apply this transform
    probabilistically.

    Args:
        audio_tree: Input audio
        rng: numpy random Generator (unused but required for random_transform)

    Returns:
        AudioTree with inverted phase

    Example:
        transform = invert_phase(prob=0.5)
        ds = ds.random_map(transform, seed=42)
    """
    return _invert_phase_np(audio_tree)


@map_transform
def trim(
    audio_tree: AudioTree,
    length: float = 1.0,
    mode: str = "wrap",
) -> AudioTree:
    """Adjust audio length to a fixed length in seconds.

    If audio is shorter than the target length, it will be padded according to mode.
    If audio is longer, it will be trimmed.

    Args:
        audio_tree: Input audio
        length: Target length in seconds
        mode: Padding mode if audio needs to be lengthened
            - "wrap": Circular shift (default). Audio wraps around.
            - "constant": Zero padding.

    Returns:
        AudioTree adjusted to target length

    Example:
        # Trim to 3 seconds
        transform = trim(length=3.0)
        ds = ds.map(transform)

        # Pad short audio with zeros
        transform = trim(length=5.0, mode="constant")
        ds = ds.map(transform)
    """
    return _trim_np(audio_tree, length, mode)


@map_transform
def mono(audio_tree: AudioTree) -> AudioTree:
    """Convert audio to mono by averaging channels.

    Args:
        audio_tree: Input audio

    Returns:
        AudioTree with mono audio

    Example:
        transform = mono()
        ds = ds.map(transform)
    """
    return audio_tree.to_mono()


@map_transform
def stereo(audio_tree: AudioTree) -> AudioTree:
    """Convert audio to stereo by duplicating mono channel.

    Args:
        audio_tree: Input audio

    Returns:
        AudioTree with stereo audio

    Example:
        transform = stereo()
        ds = ds.map(transform)
    """
    return audio_tree.to_stereo()


@map_transform
def resample(audio_tree: AudioTree, sample_rate: int | None = None) -> AudioTree:
    """Resample audio to a new sample rate.

    Wraps :meth:`~audiotree.core.AudioTree.resample`: NumPy-backed waveforms
    resample on CPU via librosa, JAX-backed waveforms use the JAX/Julius port.

    Args:
        audio_tree: Input audio
        sample_rate: Target sample rate in Hz (e.g. 16000). Required.

    Returns:
        AudioTree resampled to ``sample_rate``

    Example:
        transform = resample(sample_rate=16000)
        ds = ds.map(transform)
    """
    if sample_rate is None:
        raise ValueError(
            "resample requires a target sample_rate, e.g. resample(sample_rate=16000)."
        )
    return audio_tree.resample(sample_rate)


@map_transform
def identity(audio_tree: AudioTree) -> AudioTree:
    """Return audio without any modifications.

    Useful as a placeholder or for testing.

    Args:
        audio_tree: Input audio

    Returns:
        AudioTree unchanged

    Example:
        transform = identity()
        ds = ds.map(transform)
    """
    return audio_tree


@map_transform
def rescale_audio(audio_tree: AudioTree) -> AudioTree:
    """Rescale audio so the largest absolute value is 1.0.

    If all values are already in [-1.0, 1.0], no transformation is applied.
    Useful if transforms have caused the audio to clip.

    Args:
        audio_tree: Input audio

    Returns:
        AudioTree with rescaled audio

    Example:
        transform = rescale_audio()
        ds = ds.map(transform)
    """
    return _rescale_audio_np(audio_tree).replace(lufs=None, lufs_windows=None)


@map_transform
def peak_norm(audio_tree: AudioTree) -> AudioTree:
    """Peak-normalize audio so the largest absolute value is 1.0.

    Unlike :func:`rescale_audio`, which only scales down audio that exceeds the
    [-1.0, 1.0] range, this always divides by the peak so the result peaks at
    1.0. The peak is computed per item in the batch (across channels and
    samples) and clamped to a small epsilon to avoid division by zero on silent
    audio.

    Args:
        audio_tree: Input audio

    Returns:
        AudioTree with peak-normalized audio

    Example:
        transform = peak_norm()
        ds = ds.map(transform)
    """
    return _peak_norm_np(audio_tree).replace(lufs=None, lufs_windows=None)


@random_transform
def swap_stereo(
    audio_tree: AudioTree,
    rng: np.random.Generator,
) -> AudioTree:
    """Swap the channels of stereo audio.

    For data augmentation, it's common to use prob=0.5 to apply this transform
    probabilistically.

    Args:
        audio_tree: Input audio (must be stereo)
        rng: numpy random Generator (unused but required for random_transform)

    Returns:
        AudioTree with swapped channels

    Example:
        transform = swap_stereo(prob=0.5)
        ds = ds.random_map(transform, seed=42)
    """
    return _swap_stereo_np(audio_tree)


@random_transform
def corrupt_phase(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float = 1.0,
    hop_factor: float = 0.5,
    frame_length: int = 2048,
    window: str = "hann",
    keep_lufs: bool = False,
) -> AudioTree:
    """Perform phase corruption on audio.

    The phase shift range is [-pi * amount, pi * amount], independently
    selected for each frequency in the STFT.

    Args:
        audio_tree: Input audio
        rng: numpy random Generator
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

    Example:
        transform = corrupt_phase(amount=0.5, hop_factor=0.5)
        ds = ds.random_map(transform, seed=42)
    """
    return _corrupt_phase_np(
        audio_tree, rng, amount, hop_factor, frame_length, window, keep_lufs
    )


@random_transform
def shift_phase(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    amount: float = 1.0,
    keep_lufs: bool = False,
) -> AudioTree:
    """Perform a phase shift on audio.

    The phase shift range is [-pi * amount, pi * amount].

    Args:
        audio_tree: Input audio
        rng: numpy random Generator
        amount: Maximum phase shift in multiples of pi
        keep_lufs: If True, preserve the cached ``lufs`` and ``lufs_windows``. A
            phase shift leaves the magnitude spectrum (and thus energy) intact, so
            loudness is approximately unchanged; the cached values are invalidated
            by default to be safe.

    Returns:
        AudioTree with shifted phase

    Example:
        transform = shift_phase(amount=0.5)
        ds = ds.random_map(transform, seed=42)
    """
    return _shift_phase_np(audio_tree, rng, amount, keep_lufs=keep_lufs)


@random_transform
def roll(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_seconds: float = 0.0,
    max_seconds: float = 0.0,
    mode: str = "wrap",
) -> AudioTree:
    """Apply a circular shift (roll) to audio data.

    The amount of roll is randomly selected per item in the batch (not per channel).
    Positive values roll the audio to the right, negative values roll to the left.

    Args:
        audio_tree: Input audio
        rng: numpy random Generator
        min_seconds: Minimum roll amount in seconds (negative = left)
        max_seconds: Maximum roll amount in seconds (positive = right)
        mode: Padding mode - "wrap" (circular) or "constant" (zero padding)

    Returns:
        AudioTree with rolled audio

    Example:
        transform = roll(min_seconds=-1.0, max_seconds=1.0, mode="wrap")
        ds = ds.random_map(transform, seed=42)
    """
    return _roll_np(audio_tree, rng, min_seconds, max_seconds, mode)


def encode_with_codec(
    encoder_fn: Callable[[AudioTree], jnp.ndarray],
    num_codebooks: int,
):
    """Create a transform that encodes audio using a neural codec.

    Use a neural audio codec such as DAC or EnCodec to encode audio into tokens.

    Note: This transform uses JAX operations as it's intended for GPU inference.
    For CPU grain pipelines, consider running inference separately.

    Args:
        encoder_fn: Function that takes AudioTree and returns tokens
            shaped ((B*C), K, S) where K is codebooks, S is sequence length
        num_codebooks: Number of codebooks in the codec

    Returns:
        Transform function that can be used with .map()

    Example:
        def my_encoder(audio_tree):
            # Your codec encoder here
            return tokens  # Shape: ((B*C), K, S)

        transform = encode_with_codec(my_encoder, num_codebooks=9)
        ds = ds.map(transform)
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

    Use a neural network to set the latents of the AudioTree.

    Note: This transform uses JAX operations as it's intended for GPU inference.
    For CPU grain pipelines, consider running inference separately.

    Args:
        encoder_fn: Function that takes AudioTree and returns latent sequence

    Returns:
        Transform function that can be used with .map()

    Example:
        def my_encoder(audio_tree):
            # Your encoder here
            return latents  # Shape: (B, D, S)

        transform = encode_latents(my_encoder)
        ds = ds.map(transform)
    """

    @map_transform
    def _encode_latents_transform(audio_tree: AudioTree) -> AudioTree:
        if audio_tree.latents is None:
            latents = encoder_fn(audio_tree)
            audio_tree = audio_tree.replace(latents=latents)
        return audio_tree

    return _encode_latents_transform()


class choose(grain.transforms.RandomMap):
    r"""Choose c transform(s) among transforms with optional probability weights.

    With probability prob, choose c transform(s) from the list of transforms
    and apply them sequentially.

    Args:
        \*transforms: Variable number of transforms to choose from
        c: Number of transforms to choose
        weights: Optional probability weights for each transform
        prob: Probability of applying any transforms at all

    Example::

        transform = choose(
            volume_change(min_db=-6, max_db=6),
            invert_phase(),
            swap_stereo(),
            c=2,
            weights=[0.5, 0.3, 0.2],
            prob=0.9,
        )
        ds = ds.random_map(transform, seed=42)
    """

    def __init__(self, *transforms, c: int = 1, weights=None, prob: float = 1.0):
        if weights is not None:
            assert len(weights) == len(transforms)

        assert c <= len(transforms)

        self.c = c
        self.weights = weights
        assert 0 <= prob <= 1
        self.prob = prob

        self.transforms = transforms

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() >= self.prob:
            return element

        transforms = rng.choice(
            self.transforms, size=(self.c,), replace=False, p=self.weights
        )

        for transform in transforms:
            if isinstance(transform, grain.transforms.Map):
                element = transform.map(element)
            elif isinstance(transform, grain.transforms.RandomMap):
                element = transform.random_map(element, rng)
            elif hasattr(transform, "np_random_map"):
                element = transform.np_random_map(element, rng)
            else:
                element = transform(element, rng)

        return element
