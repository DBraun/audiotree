"""Base classes for transforms."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from grain.transforms import Map as MapTransform, RandomMap as RandomMapTransform
import jax
from jax import random
from jax.tree import map_with_path
from jax.tree_util import DictKey, SequenceKey
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
# A config entry: where in the element it applies, which parameter it sets, and
# the value. The value is opaque -- see ``flatten_config``.
ConfigEntries = Sequence[tuple[KeyPath, str, Any]]


def flatten_config(config: Dict[str, Any], parameters: Sequence[str]) -> ConfigEntries:
    """Flatten a nested config dict into ``(path, parameter, value)`` entries.

    A config mixes two kinds of keys: the names of the transform's parameters,
    and the keys of the element being transformed (which scope a parameter to a
    subtree). Recursion therefore stops as soon as a key names a parameter --
    the value under it is opaque, so a parameter whose value is a dict, a list
    or ``None`` survives intact instead of being flattened into (or erased
    from) the surrounding namespace.

    :param config: The user's nested configuration dictionary.
    :param parameters: The transform's parameter names.
    :return: One entry per configured parameter.
    """
    entries: List[tuple[KeyPath, str, Any]] = []

    def walk(node: Any, path: tuple) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in parameters:
                    entries.append((path, key, value))
                else:
                    walk(value, path + (DictKey(key),))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, path + (SequenceKey(index),))
        else:
            location = "".join(str(key) for key in path)
            raise ValueError(
                f"config{location} is not a parameter of this transform and not "
                f"a path into the element. Valid parameters: "
                f"{', '.join(sorted(parameters)) or '(none)'}."
            )

    walk(config, ())
    return entries


def _get_config_val(
    config: ConfigEntries,
    lookup_path: KeyPath,
    lookup_key: str,
    default: Any,
) -> Any:
    """
    Retrieve the configuration value for a given key and path.

    The most specific entry wins: an entry applies when its path is a prefix of
    the element's path, and the longest such path is used.

    :param config: Entries from `flatten_config`.
    :param lookup_path: Path of the current element.
    :param lookup_key: Configuration key to look up
    :param default: Default value if key is not found
    :return: Configuration value
    """
    longest_len = -1
    matched_value = default
    for config_path, key, value in config:
        L = len(config_path)
        if key == lookup_key and L > longest_len:
            if tuple(config_path) == tuple(lookup_path[:L]):
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

    Every leaf of the dict form must sit under a ``"scope"`` key. In
    particular, a parameter override inside a scope entry — e.g.
    ``{"dry": {"scope": True, "min_db": -30}}`` — is rejected: per-key
    parameters are not supported, and such keys used to be read as extra scope
    markers rather than as overrides. Use one transform instance per key, each
    with its own parameters and scope, instead.
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
            last_key = getattr(config_path[-1], "key", None)
            if last_key != _SCOPE_SENTINEL:
                location = "".join(str(key) for key in config_path)
                raise ValueError(
                    f"scope{location} = {value!r}: the only value a scope dict "
                    f"may set is the {_SCOPE_SENTINEL!r} marker. Per-key "
                    f"parameter overrides inside `scope` are not supported "
                    f"(and were previously misread as scope markers, not "
                    f"applied as overrides). To use different parameters per "
                    f"key, build one transform instance per key, each scoped "
                    f"to that key."
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


def _mask_transformed(element, new_tree, rngs, draw_mask, xp, is_leaf):
    """Blend ``new_tree`` back into ``element``, one batch item at a time.

    ``element`` is flattened first so that a leaf the transform dropped (``None``,
    which is how an out-of-scope leaf is marked when ``output_key`` is set) is
    carried through untouched rather than exploding as a structure mismatch.
    """

    def select(old_leaf, new_leaf, leaf_rng):
        if not isinstance(old_leaf, AudioTree) or not isinstance(new_leaf, AudioTree):
            return new_leaf
        mask = draw_mask(leaf_rng, _leaf_batch_size(old_leaf))
        return _select_transformed(new_leaf, old_leaf, mask, xp)

    return jax.tree.map(select, element, new_tree, rngs, is_leaf=is_leaf)


def _spawn_numpy_rngs(
    rng: np.random.Generator, length: int, split_seed: bool
) -> List[np.random.Generator]:
    """One child generator per leaf, drawn from ``rng``.

    With ``split_seed=False`` every child is seeded identically, so all leaves
    draw the same numbers. Sharing the parent generator itself would not do
    that: ``np.random.Generator`` is stateful, so each leaf would advance the
    stream and get a *different* draw — the opposite of what the flag means on
    the JAX backend, where an immutable key is shared.
    """
    if split_seed:
        seeds = [rng.integers(2**63) for _ in range(length)]
    else:
        seeds = [rng.integers(2**63)] * length
    return [np.random.Generator(np.random.PCG64(seed)) for seed in seeds]


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

        # A `raise`, not an `assert`: `python -O` strips asserts, and without this
        # check the rename below silently drops the transformed audio instead.
        if not isinstance(old_tree, dict):
            raise TypeError(
                f"You specified `output_key`, but the transformed element is a "
                f"{type(old_tree).__name__}, not a dict."
            )

        def is_leaf(x):
            if not isinstance(x, dict):
                return False
            values = list(x.values())
            while isinstance(values, list) and values:
                values = values[0]
            return not values or isinstance(values, AudioTree)

        # Use output_key to rename the nodes in the tree
        def rename_node(path: KeyPath, leaf):
            full_path = [k.key for k in path]
            renamed = {}
            sources = {}  # output name -> the input key that produced it
            for k, v in leaf.items():
                if not _is_in_scope(self.scope, tuple(path) + (DictKey(k),)):
                    continue
                name = output_key(full_path + [k])
                if name in renamed:
                    # A `raise`, not a silent overwrite: a string `output_key`
                    # meeting several in-scope leaves would keep only one
                    # transformed result and drop the rest.
                    raise ValueError(
                        f"output_key maps both {sources[name]!r} and {k!r} to "
                        f"{name!r}, so all but one transformed value would be "
                        f"silently discarded. With more than one in-scope "
                        f"leaf, pass a callable output_key that returns a "
                        f"distinct name per path, e.g. "
                        f"lambda path: path[-1] + '_augmented'."
                    )
                renamed[name] = v
                sources[name] = k
            return renamed

        # Rename the deepest keys in the new tree using the `output_key` function.
        new_tree = map_with_path(rename_node, new_tree, is_leaf=is_leaf)

        # Merge the trees. Order matters: on a key collision the second argument
        # wins, and the only key that can collide is the one the user explicitly
        # asked `output_key` to write, so the freshly transformed value must win.
        new_tree = merge_pytree(old_tree, new_tree)
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
            split_seed (bool, optional): Whether to give each leaf its own RNG split. When False, every leaf draws
                identically, which keeps a dry/wet (or input/target) pair in lockstep. Defaults to True.
            prob (float, optional): Probability of applying the transform. Defaults to 1.0.
            scope (Dict[str, Any], optional): Dictionary indicating which modalities to apply the transform to
            output_key (Union[str, Callable[[List[str]], str]], optional): Key under which to store the transformed
                value. By default, the values will be transformed in-place. A plain string names exactly one output,
                so it requires a single in-scope leaf; with several, pass a callable that maps each leaf's path to a
                distinct name.
        """
        # A `raise`, not an `assert`: `python -O` strips asserts, and `prob` outside
        # [0, 1] then silently becomes "always" or "never". Matches `choose()`.
        if not 0 <= prob <= 1:
            raise ValueError(
                f"{type(self).__name__} got prob={prob}, which is not in [0, 1]."
            )
        self.default_config = self.get_default_config()
        self.config = flatten_config(config or {}, self.default_config)
        self.split_seed = split_seed
        self.prob = prob
        self.scope = normalize_scope(scope)
        if isinstance(output_key, str):
            # redefine it as a function
            self.output_key = lambda _: output_key
        else:
            self.output_key = output_key

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

        if self.prob < 1:
            # One Bernoulli draw per batch item, per leaf, so a batch is a
            # mixture of transformed and untransformed items rather than
            # all-or-nothing. `split_seed=False` locks the leaves together here
            # too, otherwise a dry/wet pair would be decorrelated by the mask
            # even though both leaves drew the same transform parameters.
            prob_keys = (
                random.split(prob_key, length)
                if self.split_seed
                else [prob_key] * length
            )
            prob_keys = jax.tree.unflatten(treedef, prob_keys)

            def draw_mask(leaf_key, batch_size: int):
                return random.bernoulli(leaf_key, p=self.prob, shape=(batch_size,))

            new_tree = _mask_transformed(
                element, new_tree, prob_keys, draw_mask, jax.numpy, is_leaf
            )

        return self._post_process(element, new_tree)

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

        treedef = jax.tree.flatten(element, is_leaf=is_leaf)[1]
        length = treedef.num_leaves
        sub_rngs = jax.tree.unflatten(
            treedef, _spawn_numpy_rngs(rng, length, self.split_seed)
        )

        config = map_with_path(map_use_default_config_val, element, is_leaf=is_leaf)

        new_tree = map_with_path(map_func, element, sub_rngs, config, is_leaf=is_leaf)

        if self.prob < 1:
            prob_rngs = jax.tree.unflatten(
                treedef, _spawn_numpy_rngs(rng, length, self.split_seed)
            )

            def draw_mask(leaf_rng: np.random.Generator, batch_size: int):
                return leaf_rng.random(batch_size) < self.prob

            new_tree = _mask_transformed(
                element, new_tree, prob_rngs, draw_mask, np, is_leaf
            )

        return self._post_process(element, new_tree)


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
                value. By default, the values will be transformed in-place. A plain string names exactly one output,
                so it requires a single in-scope leaf; with several, pass a callable that maps each leaf's path to a
                distinct name.
        """
        self.default_config = self.get_default_config()
        self.config = flatten_config(config or {}, self.default_config)
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
