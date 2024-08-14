from typing import Any, Dict, List

import jax
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
    def _apply_transform(element: jnp.ndarray, offset: int):
        return element + offset


@pytest.mark.parametrize("split_seed", [False, True])
def test_config_001(split_seed: bool):

    v = jnp.full((1,), fill_value=1)
    element = {"a": v, "b": v, "c": [v, v], "d": {"e": v, "f": v}}

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
    transformed_element = jax.tree.map(
        lambda x: np.array(x).tolist(), transformed_element
    )

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

    v = jnp.zeros((1,))
    element = {"a": v, "b": v, "c": [v, v], "d": {"e": v, "f": v}, "g": [v, v]}

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
    transformed_element = jax.tree.map(
        lambda x: np.array(x.reshape()).astype(int).tolist(), transformed_element
    )

    expected = {
        "a": 0,
        "b": 1,
        "c": [0, 0],
        "d": {
            "e": 1,
            "f": 0,
        },
        "g": [1, 1],
    }
    assert expected == transformed_element


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
