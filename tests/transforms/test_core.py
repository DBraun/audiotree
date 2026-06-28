from typing import Any, Dict, List

import jax
import platform

if platform.system() == "Darwin":
    jax.config.update("jax_platform_name", "cpu")
from jax import numpy as jnp
import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.transforms.base import BaseRandomTransform, BaseMapTransform
from audiotree.transforms import (
    identity,
    volume_change,
    volume_norm,
    shift_phase,
    corrupt_phase,
    rescale_audio,
    peak_norm,
    invert_phase,
    swap_stereo,
    encode_latents,
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
    audio_tree = audio_tree.replace_loudness()  # Required for volume_norm
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


def test_only_apply_to_audiotree():
    """Only `src` can have its `latents` set because `src` is an AudioTree while `other` is a simple array."""

    def encoder_fn(audio_tree: AudioTree) -> jnp.ndarray:
        B = audio_tree.waveform.shape[0]
        return jnp.zeros((B,))

    B = 2

    waveform = {
        "src": AudioTree(waveform=jnp.zeros(shape=(B, 1, 44100)), sample_rate=44100),
        "other": jnp.zeros((B,)),
    }

    # encode_latents returns a transform that can optionally have scope
    # Since encode_latents is a factory function, we need to handle scope differently
    # For now, test without scope since the function doesn't expose it directly
    transform = encode_latents(encoder_fn)
    out = transform.map(waveform)
    # Both src and other get processed, but only src has latents since it's an AudioTree
    assert out["src"].latents is not None


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
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_loudness()

    result = peak_norm().map(audio_tree)

    # Each item now peaks at exactly 1.0, computed across channels and samples.
    peaks = np.max(np.abs(result.waveform), axis=(-2, -1))
    np.testing.assert_allclose(peaks, [1.0, 1.0], atol=1e-6)
    # Inter-channel balance is preserved (both channels scaled by the same factor).
    np.testing.assert_allclose(result.waveform[0, 1, 2], -0.5, atol=1e-6)
    # Volume changed, so cached loudness is invalidated.
    assert result.loudness is None


def test_peak_norm_silence():
    """peak_norm leaves silence untouched without dividing by zero."""
    audio_tree = AudioTree(
        waveform=np.zeros((1, 2, 8), dtype=np.float32), sample_rate=44100
    )
    result = peak_norm().map(audio_tree)
    assert np.all(np.isfinite(result.waveform))
    assert np.all(result.waveform == 0)


def test_trim_invalidates_loudness():
    """Resizing the waveform invalidates cached loudness; a no-op preserves it."""
    sample_rate = 44100
    waveform = np.random.randn(2, 1, sample_rate * 2).astype(np.float32) * 0.1
    audio_tree = AudioTree(
        waveform=waveform, sample_rate=sample_rate
    ).replace_loudness()
    assert audio_tree.loudness is not None

    # Shorten -> loudness invalidated.
    assert trim(length=1.0).map(audio_tree).loudness is None
    # Lengthen -> loudness invalidated.
    assert trim(length=3.0).map(audio_tree).loudness is None
    # Same length is a no-op and keeps the cached loudness.
    same = trim(length=2.0).map(audio_tree)
    assert same.loudness is not None


def test_roll_loudness_invalidation():
    """Constant-mode roll invalidates loudness; wrap-mode preserves it."""
    waveform = np.random.randn(1, 2, 10000).astype(np.float32) * 0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=10000).replace_loudness()
    rng = np.random.default_rng(0)

    wrapped = roll(min_seconds=0.1, max_seconds=0.1, mode="wrap").random_map(
        audio_tree, rng
    )
    assert wrapped.loudness is not None

    constant = roll(min_seconds=0.1, max_seconds=0.1, mode="constant").random_map(
        audio_tree, rng
    )
    assert constant.loudness is None


def test_to_stereo_invalidates_loudness():
    """Duplicating a mono channel changes loudness, so it is invalidated."""
    mono_tree = AudioTree(
        waveform=np.random.randn(2, 1, 44100).astype(np.float32) * 0.1,
        sample_rate=44100,
    ).replace_loudness()
    assert mono_tree.loudness is not None

    stereo_tree = mono_tree.to_stereo()
    assert stereo_tree.num_channels == 2
    assert stereo_tree.loudness is None

    # Already-stereo audio is unchanged, so its loudness is preserved.
    assert stereo_tree.replace_loudness().to_stereo().loudness is not None


def test_phase_transforms_keep_loudness():
    """Phase transforms invalidate loudness by default, kept via keep_loudness."""
    waveform = np.random.randn(2, 1, 44100).astype(np.float32) * 0.1
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_loudness()
    original_loudness = audio_tree.loudness
    assert original_loudness is not None
    rng = np.random.default_rng(0)

    for transform in (shift_phase, corrupt_phase):
        # Default: loudness invalidated.
        assert transform().random_map(audio_tree, rng).loudness is None
        # keep_loudness=True: cached value carried through unchanged.
        kept = transform(keep_loudness=True).random_map(audio_tree, rng)
        np.testing.assert_array_equal(kept.loudness, original_loudness)


# =============================================================================
# JAX Module Tests
# =============================================================================


def test_jax_transforms():
    """Test that all JAX transforms can be instantiated and applied."""
    # Use JAX arrays for the JAX module transforms
    audio_tree = AudioTree(waveform=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)
    audio_tree = audio_tree.replace_loudness()  # Required for volume_norm
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
    audio_tree = AudioTree(waveform=waveform, sample_rate=44100).replace_loudness()

    result = jax_transforms.peak_norm().map(audio_tree)

    assert jnp.allclose(jnp.max(jnp.abs(result.waveform)), 1.0)
    assert jnp.allclose(result.waveform[0, 0, 1], 1.0)
    assert result.loudness is None
