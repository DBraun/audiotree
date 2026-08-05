"""Base classes for transforms."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import warnings

from grain.transforms import Map as MapTransform, RandomMap as RandomMapTransform
import jax
from jax import random
from jax.tree import map_with_path
from jax.tree_util import DictKey
import numpy as np

from audiotree import AudioTree
from audiotree.core import ARRAY_FIELDS

# jax types its pytree key classes (DictKey, SequenceKey, ...) as ``Any``, so they
# cannot appear directly in a type expression; ``KeyPath`` is the path of key
# entries used in the annotations below. ``Sequence`` (covariant) lets the
# tuple-based key paths that jax's ``tree_flatten_with_path`` returns satisfy
# these annotations.
KeyPath = Sequence[Any]
KeyLeafPairs = Sequence[tuple[KeyPath, Any]]


def _get_config_val(
    config: KeyLeafPairs,
    lookup_path: KeyPath,
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


# In the nested-dict scope form, this key marks "select the path I sit under".
# ``_is_in_scope`` compares ``config_path[:-1]``, so a scope entry has to sit one
# level deeper than the leaf it selects; this is the key that fills that level.
_SCOPE_SENTINEL = "scope"


def _as_path(entry) -> tuple:
    """Normalize one entry of the list scope form to a tuple of keys."""
    if isinstance(entry, str):
        return tuple(part for part in entry.split(".") if part)
    if isinstance(entry, (list, tuple)):
        return tuple(entry)
    raise TypeError(
        f"A scope path must be a string like 'wet' or 'input.dry', or a tuple "
        f"of keys; got {entry!r}."
    )


def normalize_scope(scope) -> KeyLeafPairs:
    """Normalize a ``scope`` argument to internal ``(path, include)`` pairs.

    Two spellings are accepted:

    * A **list of paths** — ``["wet"]``, ``["input.dry"]``, ``[("input", "dry")]``.
      This is the clear form: each entry names a subtree to transform.
    * A **nested dict** using the ``"scope"`` sentinel —
      ``{"wet": {"scope": True}}``, optionally with nested exclusions like
      ``{"d": {"scope": True, "f": {"scope": False}}}``.

    The dict form has a sharp edge: because a scope entry is matched one level
    above where it sits, the obvious shorthand ``{"wet": True}`` matches *every*
    leaf rather than only ``"wet"`` — silently transforming the target signal in
    a dry/wet pipeline. That spelling now raises and points at the list form.
    """
    if scope is None:
        return []

    if isinstance(scope, (list, tuple)):
        pairs = []
        for entry in scope:
            path = _as_path(entry)
            if not path:
                raise ValueError(f"Empty scope path in {scope!r}.")
            keys = tuple(DictKey(key) for key in path) + (DictKey(_SCOPE_SENTINEL),)
            pairs.append((keys, True))
        return pairs

    if isinstance(scope, dict):
        flat = jax.tree_util.tree_flatten_with_path(scope)[0]
        for config_path, value in flat:
            key = getattr(config_path[0], "key", None) if config_path else None
            if len(config_path) == 1 and key != _SCOPE_SENTINEL:
                raise ValueError(
                    f"scope={{{key!r}: {value!r}}} selects every leaf, not just "
                    f"{key!r}, because a scope entry is matched one level above "
                    f"where it sits. Write scope=[{key!r}] instead, or the "
                    f"explicit scope={{{key!r}: {{'scope': {value!r}}}}}."
                )
        return flat

    raise TypeError(
        f"scope must be a list of paths (e.g. ['wet']) or a nested dict, got "
        f"{type(scope).__name__}."
    )


def _is_in_scope(
    scope: KeyLeafPairs,
    lookup_path: KeyPath,
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


def _leaf_batch_size(leaf: "AudioTree") -> int:
    """Leading (batch) axis length of an AudioTree leaf."""
    for value in (leaf.waveform, leaf.codes, leaf.latents):
        if value is not None:
            return value.shape[0]
    raise ValueError("Cannot apply `prob` to an AudioTree with no array fields.")


def _select_field(mask, new_value, old_value, xp, field: str):
    """Per-item choice between a transformed field value and the original.

    ``mask`` is a boolean ``(B,)`` array; ``True`` keeps the transformed item.
    A field that only one side populates cannot be mixed item-by-item, so it is
    dropped — the alternative is a tree whose structure depends on a coin flip.
    """
    if new_value is None or old_value is None:
        return None
    if not hasattr(new_value, "shape") or not hasattr(old_value, "shape"):
        # Non-array metadata (e.g. a list of strings) is not per-item indexable
        # here; the transforms do not change it, so keep the original.
        return old_value
    if new_value.shape != old_value.shape:
        raise ValueError(
            f"`prob` < 1 cannot be applied to a transform that changes the shape "
            f"of `{field}` ({old_value.shape} -> {new_value.shape}): whether an "
            f"item is transformed is random, so the output shape would be too. "
            f"Use prob=1.0 and apply the transform to a pre-selected subset."
        )
    broadcast = mask.reshape((mask.shape[0],) + (1,) * (new_value.ndim - 1))
    return xp.where(broadcast, new_value, old_value)


def _select_transformed(new_leaf, old_leaf, mask, xp):
    """Combine a transformed AudioTree with its original, one batch item at a time.

    Fields that one side leaves unpopulated are canonicalized to ``None`` (and
    metadata keys to absent), so the result has one structure regardless of how
    the mask fell — a tree whose treedef depends on the RNG cannot be batched,
    scanned, or jitted.
    """
    if not isinstance(new_leaf, AudioTree) or not isinstance(old_leaf, AudioTree):
        return old_leaf

    updates = {
        name: _select_field(
            mask, getattr(new_leaf, name), getattr(old_leaf, name), xp, name
        )
        for name in ARRAY_FIELDS
    }
    updates["metadata"] = {
        key: _select_field(
            mask, new_leaf.metadata[key], value, xp, f"metadata[{key!r}]"
        )
        for key, value in old_leaf.metadata.items()
        if key in new_leaf.metadata
    }
    return old_leaf.replace(**updates)


def merge_pytree(tree1, tree2):
    """Order matters!"""

    def is_leaf(leaf: dict):
        if not isinstance(leaf, dict):
            return False
        values = list(leaf.values())
        while isinstance(values, list) and values:
            values = values[0]
        return not values or isinstance(values, AudioTree)

    def _combine(x, y):
        return {**x, **y}

    return jax.tree.map(_combine, tree1, tree2, is_leaf=is_leaf)


class BaseTransformMixIn:
    # Set in each concrete transform's ``__init__``; declared here so the shared
    # ``_post_process`` can reference them.
    scope: KeyLeafPairs
    output_key: Optional[Callable[[List[str]], str]]

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
    def _apply_transform(element, rng, **kwargs):
        """
        Apply the transformation to the given element.

        Args:
            element (Any): Element to be transformed.
            rng: Random state (jax.Array or np.random.Generator depending on module)
            **kwargs: Additional keyword arguments for the transformation.

        Returns:
            Any: The transformed element.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _post_process(self, old_tree, new_tree):
        # Capture as a local so it stays narrowed to non-None inside the closure.
        output_key = self.output_key
        if output_key is None:
            return new_tree

        assert isinstance(old_tree, dict), (
            "You specified `output_key`, but the transformed element is not a dict."
        )

        def is_leaf(x):
            if not isinstance(x, dict):
                return False
            values = list(x.values())
            while isinstance(values, list):
                values = values[0]
            return isinstance(values, AudioTree)

        # Use output_key to rename the nodes in the tree
        def rename_node(path: KeyPath, leaf):
            full_path = [k.key for k in path]
            leaf = {
                output_key(full_path + [k]): v
                for k, v in leaf.items()
                if _is_in_scope(self.scope, tuple(path) + (DictKey(k),))
            }
            return leaf

        # Rename the deepest keys in the new tree using the `output_key` function.
        new_tree = map_with_path(rename_node, new_tree, is_leaf=is_leaf)

        # Merge the trees. Unfortunately, the order matters.
        new_tree = merge_pytree(new_tree, old_tree)
        return new_tree


class BaseRandomTransform(BaseTransformMixIn, RandomMapTransform):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        split_seed: bool = True,
        prob: float = 1.0,
        scope: Optional[Dict[str, Any]] = None,
        output_key: Optional[Union[str, Callable[[List[str]], str]]] = None,
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
        self.scope = normalize_scope(scope)
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
            rng: jax.random.PRNGKey (for JAX transforms) or np.random.Generator (for numpy transforms)
        Returns:
            Any: transformed element
        """
        # Detect if we're using numpy or JAX based on rng type
        if isinstance(rng, np.random.Generator):
            return self._random_map_numpy(element, rng)
        else:
            return self._random_map_jax(element, rng)

    def _random_map_jax(self, element: Any, key: jax.Array) -> Any:
        """JAX implementation of random_map."""

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree)

        def pre_transform_map_func(path: KeyPath, leaf):
            if _is_in_scope(self.scope, path):
                return self._pre_transform(leaf)
            return leaf

        def map_func(path: KeyPath, leaf, rng: jax.Array, *config):
            if not is_leaf(leaf):
                return leaf
            if _is_in_scope(self.scope, path):
                return self._apply_transform(leaf, rng, **config[0])
            elif self.output_key is not None:
                return None
            return leaf

        def map_use_default_config_val(path: KeyPath, leaf):
            return {
                k: _get_config_val(self.config, path, k, default)
                for k, default in self.default_config.items()
            }

        element = map_with_path(pre_transform_map_func, element, is_leaf=is_leaf)

        treedef = jax.tree.flatten(element, is_leaf=is_leaf)[1]
        length = treedef.num_leaves
        # Reserve the `prob` key before deriving the per-leaf keys, so it can
        # never collide with one of them (splitting `key` twice would hand the
        # Bernoulli draw the same bits as a leaf's transform key).
        key, prob_key = random.split(key)
        subkeys = random.split(key, length) if self.split_seed else [key] * length
        subkeys = jax.tree.unflatten(treedef, subkeys)

        config = map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)

        new_tree = map_with_path(map_func, element, subkeys, config, is_leaf=is_leaf)
        new_tree = self._post_process(element, new_tree)

        if self.prob == 1:
            return new_tree

        # One Bernoulli draw per batch item, per leaf, so a batch is a mixture of
        # transformed and untransformed items rather than all-or-nothing.
        prob_keys = jax.tree.unflatten(treedef, random.split(prob_key, length))

        def select(new_leaf, old_leaf, leaf_key):
            if not isinstance(old_leaf, AudioTree):
                return old_leaf
            mask = random.bernoulli(
                leaf_key, p=self.prob, shape=(_leaf_batch_size(old_leaf),)
            )
            return _select_transformed(new_leaf, old_leaf, mask, jax.numpy)

        return jax.tree.map(select, new_tree, element, prob_keys, is_leaf=is_leaf)

    def _random_map_numpy(self, element: Any, rng: np.random.Generator) -> Any:
        """NumPy implementation of random_map."""

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree)

        def pre_transform_map_func(path: KeyPath, leaf):
            if _is_in_scope(self.scope, path):
                return self._pre_transform(leaf)
            return leaf

        def map_func(path: KeyPath, leaf, leaf_rng: np.random.Generator, *config):
            if not is_leaf(leaf):
                return leaf
            if _is_in_scope(self.scope, path):
                return self._apply_transform(leaf, leaf_rng, **config[0])
            elif self.output_key is not None:
                return None
            return leaf

        def map_use_default_config_val(path: KeyPath, leaf):
            return {
                k: _get_config_val(self.config, path, k, default)
                for k, default in self.default_config.items()
            }

        element = map_with_path(pre_transform_map_func, element, is_leaf=is_leaf)

        # Create separate RNGs for each leaf if split_seed is True
        treedef = jax.tree.flatten(element, is_leaf=is_leaf)[1]
        length = treedef.num_leaves
        if self.split_seed:
            sub_rngs = [
                np.random.Generator(np.random.PCG64(rng.integers(2**63)))
                for _ in range(length)
            ]
        else:
            sub_rngs = [rng] * length
        sub_rngs = jax.tree.unflatten(treedef, sub_rngs)

        config = map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)

        new_tree = map_with_path(map_func, element, sub_rngs, config, is_leaf=is_leaf)
        new_tree = self._post_process(element, new_tree)

        if self.prob == 1:
            return new_tree

        def select(new_leaf, old_leaf):
            if not isinstance(old_leaf, AudioTree):
                return old_leaf
            batch_size = _leaf_batch_size(old_leaf)
            mask = rng.random(batch_size) < self.prob
            if batch_size == 1:
                # Per-item and per-batch coincide, so select wholesale. This is
                # the grain data-loader case, and unlike the masked path it also
                # works for transforms that change the waveform's length.
                return new_leaf if bool(mask[0]) else old_leaf
            return _select_transformed(new_leaf, old_leaf, mask, np)

        return jax.tree.map(select, new_tree, element, is_leaf=is_leaf)


class BaseMapTransform(BaseTransformMixIn, MapTransform):
    def __init__(
        self,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        scope: Optional[Dict[str, Dict[str, Any]]] = None,
        output_key: Optional[Union[str, Callable[[List[str]], str]]] = None,
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
        self.scope = normalize_scope(scope)
        if isinstance(output_key, str):
            # redefine it as a function
            self.output_key = lambda _: output_key
        else:
            self.output_key = output_key

    def map(self, element: Any) -> Any:
        """
        Apply the mapping to the given element.

        Args:
            element (Any): Input element to transform

        Returns:
            Any: transformed element
        """

        def is_leaf(leaf):
            return isinstance(leaf, AudioTree)

        def pre_transform_map_func(path: KeyPath, leaf):
            if not is_leaf(leaf):
                return leaf
            if _is_in_scope(self.scope, path):
                return self._pre_transform(leaf)
            return leaf

        def map_func(path: KeyPath, leaf, *config):
            if not is_leaf(leaf):
                return leaf
            if _is_in_scope(self.scope, path):
                return self._apply_transform(leaf, **config[0])
            elif self.output_key is not None:
                return None
            return leaf

        def map_use_default_config_val(path: KeyPath, leaf):
            return {
                key: _get_config_val(self.config, path, key, default)
                for key, default in self.default_config.items()
            }

        element = map_with_path(pre_transform_map_func, element, is_leaf=is_leaf)

        config = map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)
        new_tree = map_with_path(map_func, element, config, is_leaf=is_leaf)
        return self._post_process(element, new_tree)
