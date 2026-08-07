from typing import Any, Dict, List

import jax
import platform

if platform.system() == "Darwin":
    jax.config.update("jax_platform_name", "cpu")
from jax import numpy as jnp
import numpy as np
import pytest

import audiotree.transforms
from audiotree import AudioTree
from audiotree.transforms.base import BaseRandomTransform, BaseMapTransform
from audiotree.transforms.decorators import map_transform, random_transform
from audiotree.transforms import (
    identity,
    volume_change,
    volume_norm,
    shift_phase,
    corrupt_phase,
    rescale_audio,
    peak_norm,
    resample,
    invert_phase,
    swap_stereo,
    encode_latents,
    encode_with_codec,
    trim,
    roll,
)
from audiotree.transforms import jax as jax_transforms


class ReturnConfigTransform(BaseRandomTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "minval": 0,
            "maxval": 1,
        }

    @staticmethod
    def _apply_transform(
        element: jnp.ndarray, rng, minval: float = 0, maxval: float = 1
    ):
        return {"minval": minval, "maxval": maxval}


class AddSomethingTransform(BaseMapTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "offset": 1,
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, offset: int = 1):
        waveform = audio_tree.waveform + offset
        audio_tree = audio_tree.replace(waveform=waveform)
        return audio_tree


@pytest.mark.parametrize("split_seed", [False, True])
def test_config_001(split_seed: bool):
    def make_tree(v: float):
        return AudioTree(np.full(shape=(1, 1, 44_100), fill_value=v), 44_100)

    audio_zero = make_tree(0.0)
    element = {
        "a": audio_zero,
        "b": audio_zero,
        "c": [audio_zero, audio_zero],
        "d": {"e": audio_zero, "f": audio_zero},
    }

    config = {
        "b": {"minval": -1, "maxval": 3},
        "c": {"maxval": 4},
        "d":
        # todo: it's ugly how a config parameter `minval` can potentially clash with the data `e` or `f`
        {"minval": -3, "e": {"minval": -2}, "f": {"maxval": 2}},
    }

    transform = ReturnConfigTransform(config=config, split_seed=split_seed)
    seed = 42
    transformed_element = transform.random_map(element, rng=np.random.default_rng(seed))

    expected = {
        "a": {"minval": 0, "maxval": 1},
        "b": {"minval": -1, "maxval": 3},
        "c": [{"minval": 0, "maxval": 4}, {"minval": 0, "maxval": 4}],
        "d": {
            "e": {"minval": -2, "maxval": 1},
            "f": {"minval": -3, "maxval": 2},
        },
    }
    assert expected == transformed_element


def test_scope():
    def make_tree(v: float):
        return AudioTree(np.full(shape=(1, 1, 44_100), fill_value=v), 44_100)

    audio_zero = make_tree(0.0)
    element = {
        "a": audio_zero,
        "b": audio_zero,
        "c": [audio_zero, audio_zero],
        "d": {"e": audio_zero, "f": audio_zero},
        "g": [audio_zero, audio_zero],
    }

    scope = {
        "b": {"scope": True},
        "d": {
            # todo: it's ugly how `scope` can potentially clash with `f`
            "scope": True,
            "f": {"scope": False},
        },
        "g": {"scope": True},
    }

    transform = AddSomethingTransform(scope=scope)
    transformed_element = transform.map(element)

    expected = {
        "a": make_tree(0),
        "b": make_tree(1),
        "c": [make_tree(0), make_tree(0)],
        "d": {
            "e": make_tree(1),
            "f": make_tree(0),
        },
        "g": [make_tree(1), make_tree(1)],
    }
    assert are_equal_pytree(expected, transformed_element)


def are_equal_pytree(pytree1, pytree2):
    """
    Compares the equality of two PyTrees.

    Args:
    pytree1: The first PyTree.
    pytree2: The second PyTree.

    Returns:
    A boolean indicating if the two PyTrees are equal.
    """

    # Define a function to compare individual elements
    def compare_elements(x, y):
        return jnp.array_equal(x, y)

    # Apply the comparison element-wise across the PyTrees
    try:
        comparison_tree = jax.tree_util.tree_map(compare_elements, pytree1, pytree2)
    except Exception:
        return False

    # Aggregate the results into a single boolean value
    are_equal = jax.tree_util.tree_reduce(
        lambda x, y: x & y, comparison_tree, initializer=True
    )

    return are_equal


class Multiply(BaseMapTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {"mult": 1}

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, mult: float = 1) -> AudioTree:
        return audio_tree.replace(waveform=audio_tree.waveform * mult)


def test_output_key_001():
    """
    Test that `output_key` can be a function
    """

    def make_tree(v: float):
        return AudioTree(jnp.full(shape=(1, 1, 44100), fill_value=v), 44100)

    audio_tree = {"src": {"GT": make_tree(1)}, "target": {"GT": make_tree(2)}}

    def output_key(path: List[str]) -> str:
        return path[-1] + "_modified"

    config = {"src": {"mult": 2}, "target": {"GT": {"mult": 3}}}

    transform = Multiply(output_key=output_key, config=config)

    transformed_audio_tree = transform.map(audio_tree)

    expected = {
        "src": {"GT": make_tree(1), "GT_modified": make_tree(2)},
        "target": {"GT": make_tree(2), "GT_modified": make_tree(6)},
    }

    assert are_equal_pytree(expected, transformed_audio_tree)


def test_output_key_002():
    """
    Same as test_output_key_001 but output_key is a string.
    """

    def make_tree(v: float):
        return AudioTree(jnp.full(shape=(1, 1, 44100), fill_value=v), 44100)

    audio_tree = {"src": {"GT": make_tree(1)}, "target": {"GT": make_tree(2)}}

    output_key = "modified"

    config = {"src": {"mult": 2}, "target": {"GT": {"mult": 3}}}

    transform = Multiply(output_key=output_key, config=config)

    transformed_audio_tree = transform.map(audio_tree)

    expected = {
        "src": {"GT": make_tree(1), "modified": make_tree(2)},
        "target": {"GT": make_tree(2), "modified": make_tree(6)},
    }

    assert are_equal_pytree(expected, transformed_audio_tree)


def test_output_key_003():
    """
    Same as test_output_key_002 but the scope is specified
    """

    def make_tree(v: float):
        return AudioTree(jnp.full(shape=(1, 1, 44100), fill_value=v), 44100)

    audio_tree = {"src": {"GT": make_tree(1)}, "target": {"GT": make_tree(2)}}

    output_key = "modified"

    config = {"src": {"mult": 2}, "target": {"GT": {"mult": 3}}}
    scope = {"src": {"scope": True}}

    transform = Multiply(output_key=output_key, config=config, scope=scope)

    transformed_audio_tree = transform.map(audio_tree)

    expected = {
        "src": {"GT": make_tree(1), "modified": make_tree(2)},
        "target": {"GT": make_tree(2)},
    }

    assert are_equal_pytree(expected, transformed_audio_tree)


def test_output_key_004():
    """ """

    def make_tree(v: float):
        return AudioTree(jnp.full(shape=(1, 1, 44100), fill_value=v), 44100)

    audio_tree = {
        "src": [make_tree(1), make_tree(2)],
        "target": [make_tree(3), make_tree(4)],
    }

    output_key = "modified"

    config = {"src": {"mult": 2}, "target": {"mult": 3}}
    scope = {"src": {"scope": True}}

    transform = Multiply(output_key=output_key, config=config, scope=scope)

    transformed_audio_tree = transform.map(audio_tree)

    expected = {
        "src": [make_tree(1), make_tree(2)],
        "modified": [make_tree(2), make_tree(4)],
        "target": [make_tree(3), make_tree(4)],
    }

    assert are_equal_pytree(expected, transformed_audio_tree)


def test_output_key_rejects_a_non_dict_element():
    """``output_key`` renames dict keys, so a bare AudioTree is a usage error.

    A ``raise`` rather than an ``assert``, so it survives ``python -O`` -- see
    ``test_transform_guards_survive_python_O``.
    """
    transform = Multiply(output_key="modified")
    with pytest.raises(TypeError, match="not a dict"):
        transform.map(AudioTree(jnp.zeros((1, 1, 64)), 44100))


@pytest.mark.parametrize("prob", [-0.1, 1.5, float("nan")])
def test_random_transform_rejects_out_of_range_prob(prob):
    """``prob`` outside [0, 1] silently means "always" or "never" if unchecked."""
    with pytest.raises(ValueError, match=r"prob=.*not in \[0, 1\]"):
        volume_change(prob=prob)


def test_transform_guards_survive_python_O():
    """The two guards above are ``raise``, not ``assert``, so ``-O`` keeps them.

    Under ``-O`` an ``assert`` is compiled out entirely, so the failure would be
    a transform that silently always (or never) fires, or one that drops the
    audio it just transformed.
    """
    import subprocess
    import sys

    script = """
import numpy as np
from audiotree import AudioTree
from audiotree.transforms import volume_change

assert_removed = True
try:
    assert False
except AssertionError:
    assert_removed = False
if assert_removed is not True:
    raise SystemExit("-O did not strip asserts; the test proves nothing")

try:
    volume_change(prob=1.5)
except ValueError as exc:
    if "not in [0, 1]" not in str(exc):
        raise SystemExit(f"wrong message: {exc}")
else:
    raise SystemExit("prob=1.5 was accepted under -O")

from audiotree.transforms.base import BaseMapTransform

class Noop(BaseMapTransform):
    @staticmethod
    def get_default_config():
        return {}

    @staticmethod
    def _apply_transform(audio_tree):
        return audio_tree

try:
    Noop(output_key="modified").map(
        AudioTree(np.zeros((1, 1, 64), dtype=np.float32), 44100)
    )
except TypeError as exc:
    if "not a dict" not in str(exc):
        raise SystemExit(f"wrong message: {exc}")
else:
    raise SystemExit("a non-dict element was accepted under -O")
"""
    subprocess.run([sys.executable, "-O", "-c", script], check=True)


def test_volume_change():
    audio_tree = AudioTree(
        waveform=np.ones(shape=(1, 1, 44100), dtype=np.float32), sample_rate=44100
    )

    config = {
        "min_db": 20,
        "max_db": 20,
    }
    transform = volume_change(**config)

    seed = 0
    transformed_audio_tree = transform.random_map(
        audio_tree, rng=np.random.default_rng(seed)
    )

    assert np.allclose(audio_tree.waveform * 10, transformed_audio_tree.waveform)


def test_transforms():
    """Test that all numpy transforms can be instantiated and applied."""
    # Use numpy arrays for the numpy module transforms
    audio_tree = AudioTree(
        waveform=np.ones(shape=(1, 1, 44100), dtype=np.float32), sample_rate=44100
    )
    audio_tree = audio_tree.replace_lufs()  # Required for volume_norm
    rng = np.random.default_rng(0)

    # Test all transforms (with prob=1.0 to avoid edge cases)
    volume_change().random_map(audio_tree, rng=rng)
    volume_norm().random_map(audio_tree, rng=rng)
    shift_phase().random_map(audio_tree, rng=rng)
    corrupt_phase().random_map(audio_tree, rng=rng)
    swap_stereo().random_map(audio_tree, rng=rng)
    invert_phase().random_map(audio_tree, rng=rng)
    rescale_audio().map(audio_tree)
    identity().map(audio_tree)


class _FakeCodec:
    """Minimal AudioCodec-protocol stand-in for tests."""

    def __init__(
        self, num_codebooks: int = 4, frame_rate: int = 100, scale: bool = False
    ):
        self.num_codebooks = num_codebooks
        self.frame_rate = frame_rate
        self.scale = scale

    def encode(self, audio_tree: AudioTree):
        B = audio_tree.waveform.shape[0]
        frames = (
            audio_tree.waveform.shape[-1] * self.frame_rate // audio_tree.sample_rate
        )
        codes = jnp.zeros((B, self.num_codebooks, frames), dtype=jnp.int32)
        scale = jnp.ones((B, 1)) if self.scale else None
        return codes, scale

    def encode_to_latent(self, audio_tree: AudioTree):
        B = audio_tree.waveform.shape[0]
        return jnp.zeros((B, 8))


def test_encode_with_codec_stores_codes_and_scale():
    B = 2
    audio_tree = AudioTree(waveform=jnp.zeros((B, 1, 44100)), sample_rate=44100)

    out = encode_with_codec(_FakeCodec()).map(audio_tree)
    assert out.codes is not None
    assert out.codes.shape == (B, 4, 100)
    assert "codec_scale" not in out.metadata  # codec returned scale=None

    out = encode_with_codec(_FakeCodec(scale=True)).map(audio_tree)
    assert out.metadata["codec_scale"].shape == (B, 1)

    # Idempotent: an AudioTree that already has codes passes through unchanged.
    again = encode_with_codec(_FakeCodec(num_codebooks=9)).map(out)
    assert again.codes.shape == (B, 4, 100)


def test_only_apply_to_audiotree():
    """Only `src` can have its `latents` set because `src` is an AudioTree while `other` is a simple array."""

    B = 2

    waveform = {
        "src": AudioTree(waveform=jnp.zeros(shape=(B, 1, 44100)), sample_rate=44100),
        "other": jnp.zeros((B,)),
    }

    transform = encode_latents(_FakeCodec())
    out = transform.map(waveform)
    # Both src and other get processed, but only src has latents since it's an AudioTree
    assert out["src"].latents is not None
    assert out["src"].latents.shape == (B, 8)


def test_trim_shorten():
    """Test that Trim correctly shortens audio that is longer than the target length."""

    # Create audio with 2 seconds of data at 44100 Hz
    sample_rate = 44_100
    waveform = jnp.ones(shape=(1, 1, sample_rate * 2))  # 2 seconds
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)

    # Test trimming to 0.5 seconds
    transform = trim(length=0.5)
    trimmed_audio = transform.map(audio_tree)

    expected_samples = int(0.5 * sample_rate)  # 22050 samples
    assert trimmed_audio.waveform.shape == (1, 1, expected_samples)
    # Verify the trimmed audio contains the first part of the original
    assert jnp.array_equal(trimmed_audio.waveform, waveform[:, :, :expected_samples])

    # Test trimming to 1 second
    transform = trim(length=1.0)
    trimmed_audio = transform.map(audio_tree)

    expected_samples = sample_rate  # 44100 samples
    assert trimmed_audio.waveform.shape == (1, 1, expected_samples)
    assert jnp.array_equal(trimmed_audio.waveform, waveform[:, :, :expected_samples])


def test_trim_lengthen():
    """Test that Trim correctly lengthens audio that is shorter than the target length."""

    # Create audio with 0.5 seconds of data at 44100 Hz
    sample_rate = 44_100
    original_samples = int(0.5 * sample_rate)  # 22050 samples
    waveform = jnp.arange(original_samples, dtype=jnp.float32).reshape(1, 1, -1)
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)

    # Test lengthening to 1 second with "wrap" mode (default)
    transform = trim(length=1.0, mode="wrap")
    lengthened_audio = transform.map(audio_tree)

    expected_samples = sample_rate  # 44100 samples
    assert lengthened_audio.waveform.shape == (1, 1, expected_samples)

    # With wrap mode, the audio should repeat to fill the target length
    # Check that the first part matches the original
    assert jnp.array_equal(lengthened_audio.waveform[:, :, :original_samples], waveform)
    # Check that the wrapped part starts repeating from the beginning
    wrapped_part = lengthened_audio.waveform[:, :, original_samples:expected_samples]
    expected_wrap = waveform[:, :, : expected_samples - original_samples]
    assert jnp.array_equal(wrapped_part, expected_wrap)

    # Test lengthening to 1.5 seconds with "constant" mode (zero padding)
    transform = trim(length=1.5, mode="constant")
    lengthened_audio = transform.map(audio_tree)

    expected_samples = int(1.5 * sample_rate)  # 66150 samples
    assert lengthened_audio.waveform.shape == (1, 1, expected_samples)

    # With constant mode, the audio should be padded with zeros
    # Check that the first part matches the original
    assert jnp.array_equal(lengthened_audio.waveform[:, :, :original_samples], waveform)
    # Check that the padded part is all zeros
    padded_part = lengthened_audio.waveform[:, :, original_samples:]
    assert jnp.all(padded_part == 0)


def test_roll_wrap_mode():
    """Test roll transform with wrap mode (circular shift)."""
    B, C, T = 2, 2, 100
    waveform = np.arange(B * C * T).reshape(B, C, T).astype(np.float32)
    sample_rate = 10000

    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)
    transform = roll(min_seconds=0.001, max_seconds=0.001, mode="wrap")
    rng = np.random.default_rng(42)
    rolled = transform.random_map(audio_tree, rng)

    expected_start = waveform[0, 0, -10:]
    actual_start = rolled.waveform[0, 0, :10]
    assert np.allclose(expected_start, actual_start)


def test_roll_constant_mode():
    """Test roll transform with constant mode (zero padding)."""
    B, C, T = 1, 2, 100
    waveform = np.ones((B, C, T)).astype(np.float32)
    sample_rate = 10000
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)

    # Roll right by 0.002 seconds (20 samples)
    transform = roll(min_seconds=0.002, max_seconds=0.002, mode="constant")
    rng = np.random.default_rng(42)
    rolled = transform.random_map(audio_tree, rng)

    assert np.all(rolled.waveform[0, :, :20] == 0)
    assert np.all(rolled.waveform[0, :, 20:] == 1)


def test_roll_no_change():
    """Test roll transform with zero roll amount."""
    B, C, T = 1, 2, 100
    waveform = np.arange(B * C * T).reshape(B, C, T).astype(np.float32)
    sample_rate = 10000
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)

    transform = roll(min_seconds=0.0, max_seconds=0.0, mode="wrap")
    rng = np.random.default_rng(42)
    rolled = transform.random_map(audio_tree, rng)

    assert np.array_equal(rolled.waveform, waveform)


def test_peak_norm():
    """peak_norm scales each batch item so its peak is 1.0."""
    # Two items with different peaks (0.5 and 0.25) across channels.
    waveform = np.zeros((2, 2, 4), dtype=np.float32)
    waveform[0, 0, 1] = 0.5
    waveform[0, 1, 2] = -0.25
    waveform[1, 0, 0] = 0.25
    waveform[1, 1, 3] = -0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_lufs()

    result = peak_norm().map(audio_tree)

    # Each item now peaks at exactly 1.0, computed across channels and samples.
    peaks = np.max(np.abs(result.waveform), axis=(-2, -1))
    np.testing.assert_allclose(peaks, [1.0, 1.0], atol=1e-6)
    # Inter-channel balance is preserved (both channels scaled by the same factor).
    np.testing.assert_allclose(result.waveform[0, 1, 2], -0.5, atol=1e-6)
    # Volume changed, so cached loudness is invalidated.
    assert result.lufs is None


def test_peak_norm_silence():
    """peak_norm leaves silence untouched without dividing by zero."""
    audio_tree = AudioTree(
        waveform=np.zeros((1, 2, 8), dtype=np.float32), sample_rate=44100
    )
    result = peak_norm().map(audio_tree)
    assert np.all(np.isfinite(result.waveform))
    assert np.all(result.waveform == 0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_volume_norm_silence(backend):
    """volume_norm leaves silence alone instead of scaling it by +inf into NaN."""
    sr = 44100
    silence = np.zeros((2, 1, sr), dtype=np.float32)
    lib = volume_norm
    if backend == "jax":
        silence = jnp.asarray(silence)
        lib = jax_transforms.volume_norm
    audio_tree = AudioTree.create(silence, sr).replace_lufs()
    assert np.all(np.asarray(audio_tree.lufs) == -np.inf)

    seed = jax.random.key(0) if backend == "jax" else np.random.default_rng(0)
    result = lib(min_db=-20, max_db=-16).random_map(audio_tree, seed)

    assert not np.isnan(np.asarray(result.waveform)).any()
    assert np.all(np.asarray(result.waveform) == 0.0)
    assert np.all(np.asarray(result.lufs) == -np.inf)


def test_resample_transform():
    """The resample transform changes the sample rate via AudioTree.resample."""
    audio_tree = AudioTree(np.zeros((1, 1, 44_100), dtype=np.float32), 44_100)
    result = resample(sample_rate=22_050).map(audio_tree)
    assert isinstance(result.waveform, np.ndarray)  # NumPy backend stays NumPy
    assert result.sample_rate == 22_050
    assert result.waveform.shape == (1, 1, 22_050)


def test_trim_invalidates_loudness():
    """Resizing the waveform invalidates cached loudness; a no-op preserves it."""
    sample_rate = 44100
    waveform = np.random.randn(2, 1, sample_rate * 2).astype(np.float32) * 0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate).replace_lufs()
    assert audio_tree.lufs is not None

    # Shorten -> loudness invalidated.
    assert trim(length=1.0).map(audio_tree).lufs is None
    # Lengthen -> loudness invalidated.
    assert trim(length=3.0).map(audio_tree).lufs is None
    # Same length is a no-op and keeps the cached loudness.
    same = trim(length=2.0).map(audio_tree)
    assert same.lufs is not None


def test_roll_loudness_invalidation():
    """Constant-mode roll invalidates loudness; wrap-mode preserves it."""
    waveform = np.random.randn(1, 2, 10000).astype(np.float32) * 0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=10000).replace_lufs()
    rng = np.random.default_rng(0)

    wrapped = roll(min_seconds=0.1, max_seconds=0.1, mode="wrap").random_map(
        audio_tree, rng
    )
    assert wrapped.lufs is not None

    constant = roll(min_seconds=0.1, max_seconds=0.1, mode="constant").random_map(
        audio_tree, rng
    )
    assert constant.lufs is None


def test_roll_invalidates_offset():
    """Rolling shifts the start, so the recorded source-file offset is stale."""
    waveform = np.arange(10000, dtype=np.float32).reshape(1, 1, 10000)
    audio_tree = AudioTree(
        waveform=waveform, sample_rate=10000, metadata={"offset": np.array([5.0])}
    )
    rng = np.random.default_rng(0)

    for mode in ("wrap", "constant"):
        rolled = roll(min_seconds=0.1, max_seconds=0.1, mode=mode).random_map(
            audio_tree, rng
        )
        assert rolled.metadata["offset"] is None

    # A tree without an offset key is left untouched (no spurious None added).
    no_offset = AudioTree(waveform=waveform, sample_rate=10000)
    rolled = roll(min_seconds=0.1, max_seconds=0.1).random_map(no_offset, rng)
    assert "offset" not in rolled.metadata


def test_to_stereo_invalidates_loudness():
    """Duplicating a mono channel changes loudness, so it is invalidated."""
    mono_tree = AudioTree(
        waveform=np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
        sample_rate=44100,
    ).replace_lufs()
    assert mono_tree.lufs is not None

    stereo_tree = mono_tree.to_stereo()
    assert stereo_tree.num_channels == 2
    assert stereo_tree.lufs is None

    # Already-stereo audio is unchanged, so its loudness is preserved.
    assert stereo_tree.replace_lufs().to_stereo().lufs is not None


def test_phase_transforms_keep_lufs():
    """Phase transforms invalidate loudness by default, kept via keep_lufs."""
    waveform = np.random.randn(2, 1, 44100).astype(np.float32) * 0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_lufs()
    original_loudness = audio_tree.lufs
    assert original_loudness is not None
    rng = np.random.default_rng(0)

    for transform in (shift_phase, corrupt_phase):
        # Default: loudness invalidated.
        assert transform().random_map(audio_tree, rng).lufs is None
        # keep_lufs=True: cached value carried through unchanged.
        kept = transform(keep_lufs=True).random_map(audio_tree, rng)
        np.testing.assert_array_equal(kept.lufs, original_loudness)


# =============================================================================
# prob < 1
# =============================================================================


def _transform_names(namespace, base_class):
    """Names in ``namespace`` whose transform derives from ``base_class``.

    Discovered rather than listed, so a new transform is covered the day it is
    added. The decorators attach the generated class as ``.Transform``, which
    also avoids instantiating non-transform exports (``AudioCodec`` is a
    Protocol and raises).
    """
    return sorted(
        name
        for name in namespace.__all__
        if issubclass(
            getattr(getattr(namespace, name), "Transform", type(None)), base_class
        )
    )


#: Transforms that go through ``random_map`` and therefore honor ``prob``.
_RANDOM_TRANSFORM_NAMES = _transform_names(audiotree.transforms, BaseRandomTransform)
#: Transforms that go through ``map`` and take no ``prob``.
_MAP_TRANSFORM_NAMES = _transform_names(audiotree.transforms, BaseMapTransform)


def _prob_test_tree(backend, batch_size):
    """A tree shaped like one off the data loader: stereo, with lufs and metadata."""
    sr = 16000
    ramp = np.linspace(-0.4, 0.4, sr, dtype=np.float32)
    waveform = np.tile(ramp[None, None, :], (batch_size, 2, 1))
    if backend == "jax":
        waveform = jnp.asarray(waveform)
    filepaths = [f"f{i}.wav" for i in range(batch_size)]
    return AudioTree.create(waveform, sr, filepath=filepaths).replace_lufs()


@pytest.mark.parametrize("name", _RANDOM_TRANSFORM_NAMES)
@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_prob_lt_one_runs(name, backend, batch_size):
    """``prob < 1`` must work for every random transform, on both backends.

    Regression test: masking used to require the transformed and original trees
    to have identical treedefs, so any transform that nulls ``lufs`` (or adds a
    metadata key) crashed on the JAX backend, and the string-encoded
    ``metadata["filepath"]`` that ``from_file`` always sets crashed the rest.
    """
    lib = jax_transforms if backend == "jax" else audiotree.transforms
    tree = _prob_test_tree(backend, batch_size)
    seed = jax.random.key(0) if backend == "jax" else np.random.default_rng(0)

    result = getattr(lib, name)(prob=0.5).random_map(tree, seed)

    assert result.waveform.shape == tree.waveform.shape
    assert not np.isnan(np.asarray(result.waveform)).any()


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_prob_is_drawn_per_batch_item(backend):
    """``prob`` is one draw per item, not one coin flip for the whole batch.

    A whole-batch draw yields batches that are entirely transformed or entirely
    untransformed, which silently removes the augmentation diversity ``prob`` is
    supposed to provide.
    """
    batch_size = 16
    tree = _prob_test_tree(backend, batch_size)
    lib = jax_transforms if backend == "jax" else audiotree.transforms

    def flips(seed_value):
        seed = (
            jax.random.key(seed_value)
            if backend == "jax"
            else np.random.default_rng(seed_value)
        )
        out = lib.invert_phase(prob=0.5).random_map(tree, seed)
        return np.array(
            [
                bool(
                    np.all(np.asarray(out.waveform[i]) == -np.asarray(tree.waveform[i]))
                )
                for i in range(batch_size)
            ]
        )

    # At least one batch is a genuine mixture rather than all-or-nothing.
    assert any(0 < flips(s).sum() < batch_size for s in range(5))

    # And the marginal per-item rate tracks `prob`.
    total = sum(int(flips(s).sum()) for s in range(40))
    np.testing.assert_allclose(total / (40 * batch_size), 0.5, atol=0.08)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_prob_lt_one_keeps_one_structure(backend, batch_size):
    """The output treedef must not depend on how the coin fell.

    A transform that nulls ``lufs`` produces a different structure than the
    input; mixing them per item has to canonicalize, or the results cannot be
    batched, scanned, or jitted together. The NumPy backend used to shortcut
    ``batch_size == 1`` by returning one branch or the other wholesale, which
    made the structure of every grain element a coin flip.
    """
    lib = jax_transforms if backend == "jax" else audiotree.transforms
    tree = _prob_test_tree(backend, batch_size)

    def structure(s):
        seed = jax.random.key(s) if backend == "jax" else np.random.default_rng(s)
        return jax.tree.structure(lib.shift_phase(prob=0.5).random_map(tree, seed))

    assert len({structure(s) for s in range(12)}) == 1


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_prob_lt_one_with_output_key(backend):
    """``prob`` < 1 and ``output_key`` together used to be an unconditional crash.

    Masking now happens before the renamed subtree is merged back in, so the
    new key holds a per-item mixture and the original is left alone.
    """
    lib = jax_transforms if backend == "jax" else audiotree.transforms
    batch_size = 16
    element = {"src": _prob_test_tree(backend, batch_size)}
    seed = jax.random.key(0) if backend == "jax" else np.random.default_rng(0)

    out = lib.volume_change(
        min_db=20, max_db=20, prob=0.5, output_key="modified"
    ).random_map(element, seed)

    assert set(out) == {"src", "modified"}
    original = np.asarray(element["src"].waveform)
    np.testing.assert_allclose(np.asarray(out["src"].waveform), original)
    # Every item of the new leaf is either transformed (+20 dB) or untouched.
    ratio = np.asarray(out["modified"].waveform) / original
    per_item = ratio.reshape(batch_size, -1)[:, 0]
    assert np.all(np.isclose(per_item, 10.0) | np.isclose(per_item, 1.0))
    assert 0 < np.isclose(per_item, 10.0).sum() < batch_size


# =============================================================================
# split_seed
# =============================================================================


def _pair_element(backend, batch_size=1):
    """A ``{"dry": ..., "wet": ...}`` element whose two leaves start identical."""
    waveform = np.ones((batch_size, 1, 8), dtype=np.float32)
    if backend == "jax":
        waveform = jnp.asarray(waveform)
    return {
        "dry": AudioTree(waveform, 16000),
        "wet": AudioTree(waveform, 16000),
    }


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("split_seed", [False, True])
def test_split_seed_false_locks_leaves_together(backend, split_seed):
    """``split_seed=False`` must mean "every leaf draws the same" on both backends.

    The NumPy path shared one *stateful* ``np.random.Generator`` across leaves,
    so each leaf advanced the stream and drew a different gain -- silently
    decorrelating the dry/wet pair the flag exists to keep locked.
    """
    lib = jax_transforms if backend == "jax" else audiotree.transforms
    seed = jax.random.key(0) if backend == "jax" else np.random.default_rng(0)

    out = lib.volume_change(min_db=-12, max_db=12, split_seed=split_seed).random_map(
        _pair_element(backend), seed
    )

    dry = np.asarray(out["dry"].waveform)
    wet = np.asarray(out["wet"].waveform)
    assert np.allclose(dry, wet) == (not split_seed)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_split_seed_false_locks_the_prob_mask(backend):
    """The Bernoulli mask honors ``split_seed`` too, not just the transform draw.

    Otherwise a locked dry/wet pair is re-decorrelated by the mask as soon as
    ``prob`` drops below one.
    """
    lib = jax_transforms if backend == "jax" else audiotree.transforms
    batch_size = 16
    element = _pair_element(backend, batch_size)
    seed = jax.random.key(0) if backend == "jax" else np.random.default_rng(0)

    out = lib.invert_phase(prob=0.5, split_seed=False).random_map(element, seed)

    dry = np.asarray(out["dry"].waveform)
    wet = np.asarray(out["wet"].waveform)
    np.testing.assert_array_equal(dry, wet)
    # And the mask really is a mixture, so the equality above is not vacuous.
    flipped = (dry.reshape(batch_size, -1)[:, 0] < 0).sum()
    assert 0 < flipped < batch_size


# =============================================================================
# transform parameters
# =============================================================================


def test_container_valued_parameters_survive():
    """Dict/list/None parameter values used to be flattened away or crash.

    ``_get_config_val`` matched on the last key of the flattened config path, so
    a dict parameter's *inner* key was compared against the parameter name (and
    leaked into the sibling parameters' namespace), a list parameter crashed on
    ``SequenceKey.key``, and ``None`` vanished entirely because it is not a
    pytree leaf.
    """
    seen = {}

    @map_transform
    def withopts(audio_tree, opts={"a": 1}, bands=[200.0, 4000.0], a=99, opt=7):
        seen.update(opts=opts, bands=bands, a=a, opt=opt)
        return audio_tree

    tree = AudioTree(np.ones((1, 1, 4), dtype=np.float32), 16000)
    withopts(opts={"a": 2}, bands=[100.0, 900.0], opt=None).map(tree)
    assert seen == {"opts": {"a": 2}, "bands": [100.0, 900.0], "a": 99, "opt": None}

    # The inner key of a dict parameter must not leak onto a sibling parameter
    # of the same name, even when the element has a leaf keyed 'a'.
    seen.clear()
    withopts().map({"a": tree})
    assert seen == {"opts": {"a": 1}, "bands": [200.0, 4000.0], "a": 99, "opt": 7}


def test_config_rejects_unknown_leaf():
    """A config entry that is neither a parameter nor a path is a typo."""
    with pytest.raises(ValueError, match="not a parameter of this transform"):
        ReturnConfigTransform(config={"b": {"minvla": -1}})


def test_required_parameters_are_supported():
    """A decorated function may declare parameters with no default.

    They used to be dropped from the allow-list, so passing one raised
    "unexpected parameter" while omitting it deferred a ``TypeError`` to the
    first batch -- both contradicting the signature the decorator publishes.
    """

    @random_transform
    def myfx(audio_tree, rng, amount, gain=1.0):
        return audio_tree.replace(waveform=audio_tree.waveform * amount * gain)

    tree = AudioTree(np.full((1, 1, 4), 2.0, dtype=np.float32), 16000)
    for transform in (myfx(0.5), myfx(amount=0.5)):
        out = transform.random_map(tree, np.random.default_rng(0))
        np.testing.assert_allclose(out.waveform, 1.0)

    with pytest.raises(TypeError, match="missing required parameter.*'amount'"):
        myfx()

    # Keyword-only required parameters work the same way.
    @random_transform
    def kwonly(audio_tree, rng, *, amount):
        return audio_tree.replace(waveform=audio_tree.waveform * amount)

    out = kwonly(amount=0.5).random_map(tree, np.random.default_rng(0))
    np.testing.assert_allclose(out.waveform, 1.0)


def test_variadic_parameters_are_rejected_at_decoration():
    """`**kwargs` can never be configured, so say so where it is written."""
    with pytest.raises(TypeError, match=r"declares \*\*extra"):

        @map_transform
        def variadic(audio_tree, **extra):
            return audio_tree


@pytest.mark.parametrize("name", ["volume_norm", "roll", "mono", "identity"])
def test_reserved_parameters_are_documented_in_args(name):
    """The shared parameters belong in the Args block, not after the Example.

    They used to be appended to the very end of the docstring, so Napoleon read
    them as prose glued to the example and no transform documented them.
    """
    transform = getattr(audiotree.transforms, name)
    doc = transform.__doc__
    expected = ["scope:", "output_key:"]
    if issubclass(transform.Transform, BaseRandomTransform):
        expected = ["prob:", "split_seed:"] + expected

    args_at = doc.index("Args:")
    example_at = doc.index("Example:")
    for entry in expected:
        assert args_at < doc.index(entry) < example_at, entry


def test_prob_lt_one_is_jittable():
    """The masked path must survive `jax.jit` (no Python branch on the mask)."""
    tree = _prob_test_tree("jax", 4)

    @jax.jit
    def run(t, key):
        return jax_transforms.corrupt_phase(prob=0.5).random_map(t, key)

    out = run(tree, jax.random.key(0))
    assert out.waveform.shape == tree.waveform.shape


# =============================================================================
# JAX Module Tests
# =============================================================================


def test_jax_transforms():
    """Test that all JAX transforms can be instantiated and applied."""
    # Use JAX arrays for the JAX module transforms
    audio_tree = AudioTree(waveform=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)
    audio_tree = audio_tree.replace_lufs()  # Required for volume_norm
    rng = jax.random.key(0)

    # Test all transforms (with prob=1.0 to avoid edge cases)
    rng, subkey = jax.random.split(rng)
    jax_transforms.volume_change().random_map(audio_tree, subkey)

    rng, subkey = jax.random.split(rng)
    jax_transforms.volume_norm().random_map(audio_tree, subkey)

    rng, subkey = jax.random.split(rng)
    jax_transforms.shift_phase().random_map(audio_tree, subkey)

    rng, subkey = jax.random.split(rng)
    jax_transforms.corrupt_phase().random_map(audio_tree, subkey)

    rng, subkey = jax.random.split(rng)
    jax_transforms.swap_stereo().random_map(audio_tree, subkey)

    rng, subkey = jax.random.split(rng)
    jax_transforms.invert_phase().random_map(audio_tree, subkey)

    jax_transforms.rescale_audio().map(audio_tree)
    jax_transforms.peak_norm().map(audio_tree)
    jax_transforms.identity().map(audio_tree)


def test_jax_volume_change():
    """Test JAX volume_change transform."""
    audio_tree = AudioTree(waveform=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)

    config = {
        "min_db": 20,
        "max_db": 20,
    }
    transform = jax_transforms.volume_change(**config)

    rng = jax.random.key(0)
    transformed_audio_tree = transform.random_map(audio_tree, rng)

    assert jnp.allclose(audio_tree.waveform * 10, transformed_audio_tree.waveform)


def test_jax_roll_wrap_mode():
    """Test JAX roll transform with wrap mode (circular shift)."""
    B, C, T = 2, 2, 100
    waveform = jnp.arange(B * C * T).reshape(B, C, T).astype(jnp.float32)
    sample_rate = 10000

    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)
    transform = jax_transforms.roll(min_seconds=0.001, max_seconds=0.001, mode="wrap")
    rng = jax.random.key(42)
    rolled = transform.random_map(audio_tree, rng)

    expected_start = waveform[0, 0, -10:]
    actual_start = rolled.waveform[0, 0, :10]
    assert jnp.allclose(expected_start, actual_start)


def test_jax_roll_constant_mode():
    """Test JAX roll transform with constant mode (zero padding)."""
    B, C, T = 1, 2, 100
    waveform = jnp.ones((B, C, T)).astype(jnp.float32)
    sample_rate = 10000
    audio_tree = AudioTree(waveform=waveform, sample_rate=sample_rate)

    # Roll right by 0.002 seconds (20 samples)
    transform = jax_transforms.roll(
        min_seconds=0.002, max_seconds=0.002, mode="constant"
    )
    rng = jax.random.key(42)
    rolled = transform.random_map(audio_tree, rng)

    assert jnp.all(rolled.waveform[0, :, :20] == 0)
    assert jnp.all(rolled.waveform[0, :, 20:] == 1)


def test_jax_peak_norm():
    """Test JAX peak_norm scales each item to a peak of 1.0."""
    waveform = jnp.array([[[0.0, 0.5, -0.25, 0.1]]], dtype=jnp.float32)  # peak 0.5
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_lufs()

    result = jax_transforms.peak_norm().map(audio_tree)

    assert jnp.allclose(jnp.max(jnp.abs(result.waveform)), 1.0)
    assert jnp.allclose(result.waveform[0, 0, 1], 1.0)


def test_jax_resample_transform():
    """The JAX resample transform changes the sample rate and stays a JAX array."""
    audio_tree = AudioTree(jnp.zeros((1, 1, 44_100)), 44_100)
    result = jax_transforms.resample(sample_rate=22_050).map(audio_tree)
    assert isinstance(result.waveform, jax.Array)  # JAX backend stays JAX
    assert result.sample_rate == 22_050
    assert result.waveform.shape == (1, 1, 22_050)
    assert result.lufs is None


def test_transform_discovery_is_not_empty():
    """Guard the discovery itself: an empty list would silently pass everything."""
    assert len(_RANDOM_TRANSFORM_NAMES) >= 7
    assert len(_MAP_TRANSFORM_NAMES) >= 7
    assert not set(_RANDOM_TRANSFORM_NAMES) & set(_MAP_TRANSFORM_NAMES)


@pytest.mark.parametrize("name", _MAP_TRANSFORM_NAMES)
def test_map_transforms_reject_prob(name):
    """`prob` is meaningless for a map transform, so it must not be accepted.

    It used to be swallowed by the decorator's ``**transform_params``, so a
    user who wrote ``trim(length=1.0, prob=0.5)`` got an unconditional trim and
    no indication that half of it was ignored.
    """
    with pytest.raises(TypeError, match="unexpected parameter|prob"):
        getattr(audiotree.transforms, name)(prob=0.5)


def test_unknown_parameter_message_lists_only_accepted_reserved():
    """The error's "plus ..." list must match what the transform kind accepts.

    It used to always append "plus prob, split_seed, scope, output_key", but a
    map transform rejects ``prob``/``split_seed``, so following the message
    walked the user straight into the next TypeError.
    """
    with pytest.raises(TypeError, match=r"plus scope, output_key\.") as excinfo:
        trim(lengthh=1.0)
    assert "prob" not in str(excinfo.value)
    assert "split_seed" not in str(excinfo.value)

    with pytest.raises(TypeError, match=r"plus prob, split_seed, scope, output_key\."):
        volume_norm(min_dB=-20)


def test_repr_includes_non_default_reserved_settings():
    """The repr renders as a constructor call, so it must not drop settings.

    It used to omit ``prob``/``split_seed``/``scope``/``output_key``, so the
    repr of a transform with non-default reserved settings reconstructed a
    different transform.
    """
    transform = volume_norm(min_db=-20.0, prob=0.5, scope=["dry"])
    assert repr(transform) == "volume_norm(min_db=-20.0, prob=0.5, scope=['dry'])"

    # Settings left at their defaults stay out of the repr.
    assert repr(volume_norm(min_db=-20.0)) == "volume_norm(min_db=-20.0)"

    # A map transform renders its own reserved settings (there is no prob).
    assert repr(trim(length=2.0, output_key="out")) == (
        "trim(length=2.0, output_key='out')"
    )
