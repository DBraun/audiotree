from typing import Any, Dict, List

import jax
import platform
if platform.system() == "Darwin":
    jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp
import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.transforms.base import BaseRandomTransform, BaseMapTransform
from audiotree.transforms import (
    Identity,
    VolumeChange,
    VolumeNorm,
    ShiftPhase,
    CorruptPhase,
    RescaleAudio,
    InvertPhase,
    SwapStereo,
    NeuralLatentEncodeTransform,
    Trim,
)


class ReturnConfigTransform(BaseRandomTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "minval": 0,
            "maxval": 1,
        }

    @staticmethod
    def _apply_transform(
        element: jnp.ndarray, rng: jax.Array, minval: float, maxval: float
    ):
        return {"minval": minval, "maxval": maxval}


class AddSomethingTransform(BaseMapTransform):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "offset": 1,
        }

    @staticmethod
    def _apply_transform(audio_tree: AudioTree, offset: int):
        audio_data = audio_tree.audio_data + offset
        audio_tree = audio_tree.replace(audio_data=audio_data)
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
    except Exception as e:
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
    def _apply_transform(audio_tree: AudioTree, mult: float) -> AudioTree:
        return audio_tree.replace(audio_data=audio_tree.audio_data * mult)


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

    audio_tree = AudioTree(audio_data=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)

    config = {
        "min_db": 20,
        "max_db": 20,
    }
    transform = VolumeChange(config=config)

    seed = 0
    transformed_audio_tree = transform.random_map(
        audio_tree, rng=np.random.default_rng(seed)
    )

    assert jnp.allclose(audio_tree.audio_data * 10, transformed_audio_tree.audio_data)


def test_transforms():
    audio_tree = AudioTree(audio_data=jnp.ones(shape=(1, 1, 44100)), sample_rate=44100)
    prob = 0.5
    rng = np.random.default_rng(0)
    VolumeChange(prob=prob).random_map(audio_tree, rng=rng)
    VolumeNorm(prob=prob).random_map(audio_tree, rng=rng)
    ShiftPhase(prob=prob).random_map(audio_tree, rng=rng)
    CorruptPhase(prob=prob).random_map(audio_tree, rng=rng)
    SwapStereo().map(audio_tree)
    RescaleAudio().map(audio_tree)
    InvertPhase().map(audio_tree)
    Identity().map(audio_tree)


def test_only_apply_to_audiotree():
    """Only `src` can have its `latents` set because `src` is an AudioTree while `other` is a simple array."""

    def encoder_fn(audio_tree: AudioTree) -> jnp.ndarray:
        B = audio_tree.audio_data.shape[0]
        return jnp.zeros((B,))

    B = 2

    audio_data = {
        "src": AudioTree(audio_data=jnp.zeros(shape=(B, 1, 44100)), sample_rate=44100),
        "other": jnp.zeros((B,)),
    }

    transform = NeuralLatentEncodeTransform(
        encoder_fn=encoder_fn, scope={"src": {"scope": True}}
    )
    out = transform.map(audio_data)
    assert out["src"].latents is not None

    transform = NeuralLatentEncodeTransform(encoder_fn=encoder_fn)
    out = transform.map(audio_data)
    assert out["src"].latents is not None


def test_trim_shorten():
    """Test that Trim correctly shortens audio that is longer than the target length."""

    # Create audio with 2 seconds of data at 44100 Hz
    sample_rate = 44_100
    audio_data = jnp.ones(shape=(1, 1, sample_rate * 2))  # 2 seconds
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)

    # Test trimming to 0.5 seconds
    transform = Trim(config={"length": 0.5})
    trimmed_audio = transform.map(audio_tree)

    expected_samples = int(0.5 * sample_rate)  # 22050 samples
    assert trimmed_audio.audio_data.shape == (1, 1, expected_samples)
    # Verify the trimmed audio contains the first part of the original
    assert jnp.array_equal(trimmed_audio.audio_data, audio_data[:, :, :expected_samples])

    # Test trimming to 1 second
    transform = Trim(config={"length": 1.0})
    trimmed_audio = transform.map(audio_tree)

    expected_samples = sample_rate  # 44100 samples
    assert trimmed_audio.audio_data.shape == (1, 1, expected_samples)
    assert jnp.array_equal(trimmed_audio.audio_data, audio_data[:, :, :expected_samples])


def test_trim_lengthen():
    """Test that Trim correctly lengthens audio that is shorter than the target length."""

    # Create audio with 0.5 seconds of data at 44100 Hz
    sample_rate = 44_100
    original_samples = int(0.5 * sample_rate)  # 22050 samples
    audio_data = jnp.arange(original_samples, dtype=jnp.float32).reshape(1, 1, -1)
    audio_tree = AudioTree(audio_data=audio_data, sample_rate=sample_rate)

    # Test lengthening to 1 second with "wrap" mode (default)
    transform = Trim(config={"length": 1.0, "mode": "wrap"})
    lengthened_audio = transform.map(audio_tree)

    expected_samples = sample_rate  # 44100 samples
    assert lengthened_audio.audio_data.shape == (1, 1, expected_samples)

    # With wrap mode, the audio should repeat to fill the target length
    # Check that the first part matches the original
    assert jnp.array_equal(lengthened_audio.audio_data[:, :, :original_samples], audio_data)
    # Check that the wrapped part starts repeating from the beginning
    wrapped_part = lengthened_audio.audio_data[:, :, original_samples:expected_samples]
    expected_wrap = audio_data[:, :, :expected_samples - original_samples]
    assert jnp.array_equal(wrapped_part, expected_wrap)

    # Test lengthening to 1.5 seconds with "constant" mode (zero padding)
    transform = Trim(config={"length": 1.5, "mode": "constant"})
    lengthened_audio = transform.map(audio_tree)

    expected_samples = int(1.5 * sample_rate)  # 66150 samples
    assert lengthened_audio.audio_data.shape == (1, 1, expected_samples)

    # With constant mode, the audio should be padded with zeros
    # Check that the first part matches the original
    assert jnp.array_equal(lengthened_audio.audio_data[:, :, :original_samples], audio_data)
    # Check that the padded part is all zeros
    padded_part = lengthened_audio.audio_data[:, :, original_samples:]
    assert jnp.all(padded_part == 0)
