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

import grain
import numpy as np

from audiotree import AudioTree
from audiotree.transforms.decorators import random_transform, map_transform
from audiotree.loudness import shift_lufs, shift_lufs_windows
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
)


@random_transform
def volume_norm(
    audio_tree: AudioTree,
    rng: np.random.Generator,
    min_db: float = 0.0,
    max_db: float = 0.0,
) -> AudioTree:
    """Normalize volume to a randomly selected loudness value specified in LUFS.

    A tree arriving with ``lufs`` already populated (e.g. restored from a
    written manifest) is trusted and not re-measured; loudness is measured
    only when ``lufs`` is unset. The cache is trustworthy because every
    operation that changes the audio clears it.

    Args:
        audio_tree: Input audio to normalize
        rng: numpy random Generator
        min_db: Minimum target loudness in LUFS
        max_db: Maximum target loudness in LUFS

    Returns:
        AudioTree with normalized loudness

    Raises:
        ValueError: If ``min_db > max_db``.

    Example:
        transform = volume_norm(min_db=-20, max_db=-15)
        ds = ds.random_map(transform, seed=42)
    """
    # A cached ``lufs`` (e.g. restored from a written manifest) is trusted, the
    # same way ``normalize_lufs`` trusts it: every operation that changes the
    # audio clears the cache, so a populated value describes this waveform.
    if audio_tree.lufs is None:
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

    Raises:
        ValueError: If ``min_db > max_db``.

    Example:
        transform = volume_change(min_db=-12, max_db=12, prob=0.9)
        ds = ds.random_map(transform, seed=42)
    """
    audio_tree, gain_db = _volume_change_np(audio_tree, rng, min_db, max_db)
    if audio_tree.lufs is not None:
        audio_tree = audio_tree.replace(
            lufs=shift_lufs(audio_tree.lufs, gain_db, xp=np),
            lufs_windows=shift_lufs_windows(audio_tree.lufs_windows, gain_db, xp=np),
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
            "resample requires a target sample_rate, e.g., resample(sample_rate=16000)."
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
    return _rescale_audio_np(audio_tree)


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
    return _peak_norm_np(audio_tree)


@random_transform
def swap_stereo(
    audio_tree: AudioTree,
    rng: np.random.Generator,
) -> AudioTree:
    """Exchange the left and right channels of stereo audio.

    Mono audio passes through unchanged: with one channel the only possible
    permutation is the identity. Audio with three or more channels raises,
    because which pair to exchange is undefined.

    For data augmentation, it's common to use prob=0.5 to apply this transform
    probabilistically.

    Args:
        audio_tree: Input audio, mono or stereo
        rng: numpy random Generator (unused but required for random_transform)

    Returns:
        AudioTree with swapped channels

    Raises:
        ValueError: If the audio has more than two channels.

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
    selected for each channel and frequency of the STFT, and shared across
    frames. Contrast :func:`shift_phase`, which rotates the whole spectrum of
    an item by one angle.

    Args:
        audio_tree: Input audio
        rng: numpy random Generator
        amount: Maximum phase shift in multiples of pi (0.0 to 1.0)
        hop_factor: Hop size as fraction of frame_length, in ``(0, 0.5]``.
            Larger hops leave the analysis windows unable to reconstruct the
            signal, so they are rejected.
        frame_length: STFT frame length in samples
        window: Window function name
        keep_lufs: If True, preserve the cached ``lufs`` and ``lufs_windows``.
            Phase corruption leaves the magnitude spectrum (and thus energy)
            intact, so loudness is approximately unchanged; the cached values
            are invalidated by default to be safe.

    Returns:
        AudioTree with corrupted phase

    Raises:
        ValueError: If ``amount`` is negative, ``hop_factor`` is outside
            ``(0, 0.5]``, or the audio is shorter than ``frame_length``
            samples.

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

    The phase shift range is [-pi * amount, pi * amount]. One angle is drawn
    per item in the batch and applied to every channel and frequency, so a
    stereo image stays coherent. Contrast :func:`corrupt_phase`, which draws an
    angle per channel and frequency.

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

    Raises:
        ValueError: If ``amount`` is negative, or the audio is shorter than
            one STFT frame (2048 samples).

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

    Rolling invalidates the recorded source-file ``offset`` provenance (it no
    longer says where sample 0 came from). With ``prob < 1`` the offset is
    dropped for the *whole batch*, not just the rolled items: a per-item mix
    of "valid" and "invalidated" cannot be represented in one array, and
    keeping stale offsets on the rolled items would be worse.

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


class choose(grain.transforms.RandomMap):
    r"""Choose c transform(s) among transforms with optional probability weights.

    With probability prob, choose c transform(s) from the list of transforms
    and apply them sequentially.

    NumPy backend only. Which transforms run is decided in Python, so this
    cannot be traced by ``jax.jit``; there is deliberately no
    ``audiotree.transforms.jax.choose``. Compose JAX transforms explicitly
    instead.

    This is a hand-written ``grain.transforms.RandomMap``, not a decorated
    transform, so it does **not** take ``split_seed``, ``scope`` or
    ``output_key``; scope the transforms handed to it instead. Its ``prob`` is
    also a single draw for the whole element, not one per batch item as it is
    for a decorated transform.

    Args:
        \*transforms: Transforms to choose from. Each must be a
            ``grain.transforms.Map`` or ``grain.transforms.RandomMap`` — which
            is what every ``audiotree`` transform constructor returns.
        c: Number of transforms to choose
        weights: Optional probability weights for each transform. Must be one
            weight per transform, each non-negative, summing to 1.
        prob: Probability of applying any transforms at all

    Raises:
        TypeError: If a positional argument is not a grain transform.
        ValueError: If ``c``, ``weights`` or ``prob`` are out of range.

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
        for index, transform in enumerate(transforms):
            if not isinstance(
                transform, (grain.transforms.Map, grain.transforms.RandomMap)
            ):
                raise TypeError(
                    f"choose() argument {index} is a {type(transform).__name__}, "
                    f"not a grain Map/RandomMap transform. Pass a constructed "
                    f"transform, e.g. choose(invert_phase(), swap_stereo())."
                )
        if weights is not None:
            if len(weights) != len(transforms):
                raise ValueError(
                    f"choose() got {len(weights)} weights for {len(transforms)} "
                    f"transforms; there must be exactly one weight per transform."
                )
            # Validated here, with a clear message, rather than surfacing as
            # numpy's "probabilities do not sum to 1" on the first element
            # inside a grain worker.
            if any(weight < 0 for weight in weights):
                raise ValueError(
                    f"choose() weights must be non-negative, but got {list(weights)!r}."
                )
            total = float(sum(weights))
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"choose() weights must sum to 1, but got {list(weights)!r} "
                    f"(sum {total})."
                )
            # Renormalize the float32-ish residue away: `rng.choice` checks the
            # sum to ~1.5e-8, tighter than the tolerance above.
            weights = [weight / total for weight in weights]
        if not 0 <= c <= len(transforms):
            raise ValueError(
                f"choose() cannot pick c={c} of {len(transforms)} transforms."
            )
        if not 0 <= prob <= 1:
            raise ValueError(f"choose() got prob={prob}, which is not in [0, 1].")

        self.c = c
        self.weights = weights
        self.prob = prob
        self.transforms = transforms

    def random_map(self, element, rng: np.random.Generator):
        if rng.random() >= self.prob:
            return element

        # Draw indices rather than letting numpy build an object array out of
        # the transforms themselves: the transforms stay exactly the objects
        # that were passed in.
        indices = rng.choice(
            len(self.transforms), size=(self.c,), replace=False, p=self.weights
        )

        for index in indices:
            transform = self.transforms[index]
            if isinstance(transform, grain.transforms.Map):
                element = transform.map(element)
            else:
                element = transform.random_map(element, rng)

        return element
