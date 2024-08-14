from typing import Any, Callable, Dict, List, Union
import warnings

from grain.python import MapTransform, RandomMapTransform
import jax
from jax import random
from jax.tree_util import DictKey, tree_map_with_path
import numpy as np

from audiotree import AudioTree

KeyLeafPairs = list[tuple[list[DictKey], Any]]


def _get_config_val(
    config: KeyLeafPairs,
    lookup_path: list[DictKey],
    lookup_key: str,
    default: Any,
) -> Any:
    """
    Retrieve the configuration value for a given key and path.

    :param config: A list of key-leaf pairs from `tree_util.tree_flatten_with_path`.
    :param lookup_path: Path of the current element.
    :param lookup_key: Configuration key to look up
    :param default: Default value if key is not found
    :return: Configuration value
    """
    longest_len = 0
    matched_value = default
    for config_path, value in config:
        if config_path[-1].key == lookup_key:
            L = len(config_path)
            if config_path[:-1] == lookup_path[: L - 1] and L > longest_len:
                longest_len = L
                matched_value = value
    return matched_value


def _is_in_scope(
    scope: KeyLeafPairs,
    lookup_path: list[DictKey],
) -> bool:
    """
    Retrieve the configuration value for a given key and path.

    :param scope: A list of key-leaf pairs from `tree_util.tree_flatten_with_path`.
    :param lookup_path: Path of the current element.
    :return: Boolean indicating if the path is in scope
    """
    if not scope:
        return True
    matched_value = False
    for config_path, value in scope:
        L = len(config_path)
        if config_path[:-1] == lookup_path[: L - 1]:
            if not value:
                return False
            matched_value = True
    return matched_value


def merge_pytree(tree1, tree2):
    """Order matters!"""

    def is_leaf(leaf: dict):
        # todo: ask JAX experts about this
        if not isinstance(leaf, dict):
            return False
        values = list(leaf.values())
        while isinstance(values, list) and values:
            values = values[0]
        return not values or isinstance(values, AudioTree)

    def _combine(x, y):
        return {**x, **y}

    return jax.tree_util.tree_map(_combine, tree1, tree2, is_leaf=is_leaf)


class BaseTransformMixIn:

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for the transform.

        :return: Default configuration dictionary
        """
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def _pre_transform(element):
        """
        Apply a transform that will occur regardless of ``prob``.
        """
        return element

    @staticmethod
    def _apply_transform(element, rng: jax.Array, **kwargs):
        """
        Apply the transformation to the given element.

        Args:
            element (Any): Element to be transformed.
            rng (jax.Array): jax.random.PRNGKey
            **kwargs: Additional type-annotated keyword arguments for the transformation.

        Returns:
            Any: The transformed element.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _post_process(self, old_tree, new_tree):
        if self.output_key is None:
            return new_tree

        assert isinstance(
            old_tree, dict
        ), "You specified `output_key`, but the transformed element is not a dict."

        def is_leaf(x):
            # todo: ask JAX experts about this
            if not isinstance(x, dict):
                return False
            values = list(x.values())
            while isinstance(values, list):
                values = values[0]
            return isinstance(values, AudioTree)  # or values is None

        # Use output_key to rename the nodes in the tree
        def rename_node(path: list[DictKey], leaf):
            full_path = [k.key for k in path]
            leaf = {
                self.output_key(full_path + [k]): v
                for k, v in leaf.items()
                if _is_in_scope(self.scope, tuple(path) + (DictKey(k),))
            }
            return leaf

        # Rename the deepest keys in the new tree using the `output_key` function.
        new_tree = tree_map_with_path(rename_node, new_tree, is_leaf=is_leaf)

        # Merge the trees. Unfortunately, the order matters.
        new_tree = merge_pytree(new_tree, old_tree)
        return new_tree


class BaseRandomTransform(BaseTransformMixIn, RandomMapTransform):

    def __init__(
        self,
        config: Dict[str, Any] = None,
        split_seed: bool = True,
        prob: float = 1.0,
        scope: Dict[str, Any] = None,
        output_key: Union[str, Callable[[List[str]], str]] = None,
    ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the transform
            split_seed (bool, optional): Whether to split the seed for each leaf. Defaults to True.
            prob (float, optional): Probability of applying the transform. Defaults to 1.0.
            scope (Dict[str, Any], optional): Dictionary indicating which modalities to apply the transform to
            output_key (Union[str, Callable[[List[str]], str]], optional): Key under which to store the transformed
                value. By default, the values will be transformed in-place.
        """
        assert 0 <= prob <= 1
        self.default_config = self.get_default_config()
        self.config = jax.tree_util.tree_flatten_with_path(config or {})[0]
        self.split_seed = split_seed
        self.prob = prob
        self.scope = jax.tree_util.tree_flatten_with_path(scope or {})[0]
        if isinstance(output_key, str):
            # redefine it as a function
            self.output_key = lambda _: output_key
        else:
            self.output_key = output_key
        if output_key is not None and prob < 1.0:
            warnings.warn(
                "You have set a custom `output_key`, but `prob` is less than one. This may result in missing leaves."
            )

    def random_map(
        self, element: Any, rng: Union[np.random.Generator, jax.Array]
    ) -> Any:
        """
        Apply the random mapping to the given element using the provided seed.

        Args:
            element (Any): Input element to transform
            rng (jax.Array): jax.random.PRNGKey or numpy random generator
        Returns:
            Any: transformed element
        """

        if isinstance(rng, jax.Array):
            key = rng
        else:
            # todo: is there a better way to seed jax.random from this numpy random Generator?
            seed = rng.integers(2**63)
            key = random.PRNGKey(seed)

        def pre_transform_map_func(path: list[DictKey], leaf):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._pre_transform(leaf)
                return transformed_leaf
            return leaf

        def map_func(path: list[DictKey], leaf, rng: jax.Array, *config):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._apply_transform(leaf, rng, **config[0])
                return transformed_leaf
            elif self.output_key is not None:
                # The audiotree is not in scope, but output_key has been specified.
                # We drop the leaf and return None instead.
                # Then later when merging the new tree and the old tree, having None in the leaf helps us merge them
                # more easily.
                return None
            return leaf

        def map_use_default_config_val(path: list[DictKey], leaf):
            return {
                key: _get_config_val(self.config, path, key, default)
                for key, default in self.default_config.items()
            }

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree)  # todo: ask JAX experts about this

        element = tree_map_with_path(pre_transform_map_func, element, is_leaf=is_leaf)

        treedef = jax.tree.flatten(element, is_leaf=is_leaf)[1]
        length = treedef.num_leaves
        subkeys = random.split(key, length) if self.split_seed else [key] * length
        subkeys = jax.tree.unflatten(treedef, subkeys)

        config = tree_map_with_path(
            map_use_default_config_val, element, is_leaf=is_leaf
        )

        new_tree = tree_map_with_path(
            map_func, element, subkeys, config, is_leaf=is_leaf
        )
        new_tree = self._post_process(element, new_tree)

        # Determine if we should apply the transform
        key, subkey = random.split(key)
        if self.prob == 1:
            return new_tree
        mask_new = random.uniform(subkey) < self.prob
        selected = jax.tree.map(
            (lambda x, y: jax.numpy.where(mask_new, x, y)), new_tree, element
        )
        return selected


class BaseMapTransform(BaseTransformMixIn, MapTransform):

    def __init__(
        self,
        config: Dict[str, Dict[str, Any]] = None,
        scope: Dict[str, Dict[str, Any]] = None,
        output_key: Union[str, Callable[[List[str]], str]] = None,
    ):
        """
        Initialize the base transform with a configuration, a flag for seed splitting, a probability, a scope, and an
        output key.

        Args:
            config (Dict[str, Dict[str, Any]]): Configuration dictionary for the transform
            scope (Dict[str, Dict[str, Any]]): Dictionary indicating which modalities to apply the transform to
            output_key (Union[str, Callable[[List[str]], str]], optional): Key under which to store the transformed
                value. By default, the values will be transformed in-place.
        """
        self.default_config = self.get_default_config()
        self.config = jax.tree_util.tree_flatten_with_path(config or {})[0]
        self.scope = jax.tree_util.tree_flatten_with_path(scope or {})[0]
        if isinstance(output_key, str):
            # redefine it as a function
            self.output_key = lambda _: output_key
        else:
            self.output_key = output_key

    def map(self, element: Any) -> Any:
        """
        Apply the random mapping to the given element.

        Args:
            element (Any): Input element to transform

        Returns:
            Any: transformed element
        """

        def pre_transform_map_func(path: list[DictKey], leaf):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._pre_transform(leaf)
                return transformed_leaf
            return leaf

        def map_func(path: list[DictKey], leaf, *config):
            if _is_in_scope(self.scope, path):
                transformed_leaf = self._apply_transform(leaf, **config[0])
                return transformed_leaf
            elif self.output_key is not None:
                # The audiotree is not in scope, but output_key has been specified.
                # We drop the leaf and return None instead.
                # Then later when merging the new tree and the old tree, having None in the leaf helps us merge them
                # more easily.
                return None
            return leaf

        def map_use_default_config_val(path: list[DictKey], leaf):
            return {
                key: _get_config_val(self.config, path, key, default)
                for key, default in self.default_config.items()
            }

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree)  # todo: ask JAX experts about this

        element = tree_map_with_path(pre_transform_map_func, element, is_leaf=is_leaf)

        config = tree_map_with_path(
            map_use_default_config_val, element, is_leaf=is_leaf
        )
        new_tree = tree_map_with_path(map_func, element, config, is_leaf=is_leaf)
        return self._post_process(element, new_tree)
