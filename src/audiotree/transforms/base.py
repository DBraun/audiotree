"""Base classes for transforms."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from grain.transforms import Map as MapTransform, RandomMap as RandomMapTransform
import jax
from jax import random
from jax.tree import map_with_path
from jax.tree_util import DictKey, GetAttrKey, SequenceKey, tree_flatten_with_path
import numpy as np

from audiotree import AudioTree
from audiotree.core import ARRAY_FIELDS, _require_single_batch

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


def normalize_scope(scope) -> Optional[KeyLeafPairs]:
    """Normalize a ``scope`` argument to internal ``(path, include)`` pairs.

    ``None`` (no scope given) stays ``None``, meaning every leaf is in scope.
    An explicitly **empty** scope (``[]`` or ``{}``) normalizes to an empty
    pair list, meaning *no* leaf is in scope — "transform these subtrees:
    (none)" makes the transform a no-op rather than silently applying it to
    everything.

    Two non-empty spellings are accepted:

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
        return None

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
    scope: Optional[KeyLeafPairs],
    lookup_path: KeyPath,
) -> bool:
    """
    Retrieve the configuration value for a given key and path.

    :param scope: ``None`` (everything in scope), or a list of key-leaf pairs
        from `tree_util.tree_flatten_with_path` — empty meaning nothing is.
    :param lookup_path: Path of the current element.
    :return: Boolean indicating if the path is in scope
    """
    if scope is None:
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
    if isinstance(new_value, dict) and isinstance(old_value, dict):
        return {
            key: _select_field(mask, new_value[key], value, xp, f"{field}[{key!r}]")
            for key, value in old_value.items()
            if key in new_value
        }
    if not hasattr(new_value, "shape") or not hasattr(old_value, "shape"):
        # Non-array extras (e.g. a list of strings) is not per-item indexable
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
    extras keys to absent), so the result has one structure regardless of how
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
    updates["extras"] = _select_field(
        mask, new_leaf.extras, old_leaf.extras, xp, "extras"
    )
    # ``metadata`` values are never *rewritten* by a transform, but a transform
    # may *invalidate* one — ``roll`` sets ``offset`` to ``None`` because a
    # time shift makes the recorded source-file offset wrong. An entry the
    # transform invalidated must stay invalidated for the whole batch: some
    # items were transformed, a per-item mix of "valid" and "invalidated"
    # cannot be represented, and undoing the invalidation would silently
    # attach stale provenance to the transformed items.
    updates["_metadata"] = {
        key: old_value if new_leaf._metadata.get(key, old_value) is not None else None
        for key, old_value in old_leaf._metadata.items()
    }
    return old_leaf.replace(**updates)


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


def _is_audiotree(value):
    return isinstance(value, AudioTree)


def _output_path(path: KeyPath) -> list:
    return [
        key.idx
        if isinstance(key, SequenceKey)
        else key.name
        if isinstance(key, GetAttrKey)
        else getattr(key, "key", str(key))
        for key in path
    ]


class _ConstantOutputKey:
    """A picklable ``output_key`` that always returns the same name.

    A string ``output_key`` used to become ``lambda _: name`` inside
    ``__init__``, which stdlib pickle cannot serialize -- and transforms must
    pickle, because grain's worker processes are spawned, not forked (see
    ``decorators.py``).
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, path: List[str]) -> str:
        return self.name


class BaseTransformMixIn:
    # Set in each concrete transform's ``__init__``; declared here so the shared
    # ``_post_process`` can reference them.
    scope: Optional[KeyLeafPairs]
    output_key: Optional[Callable[[List[str]], str]]
    config: ConfigEntries
    default_config: Dict[str, Any]

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

    def _preprocess(self, element):
        leaves, treedef = tree_flatten_with_path(element, is_leaf=_is_audiotree)
        values = [
            self._pre_transform(leaf)
            if isinstance(leaf, AudioTree) and _is_in_scope(self.scope, path)
            else leaf
            for path, leaf in leaves
        ]
        if all(new is old for (_, old), new in zip(leaves, values)):
            return element
        return treedef.unflatten(values)

    def _config_at(self, path: KeyPath) -> Dict[str, Any]:
        return {
            key: _get_config_val(self.config, path, key, default)
            for key, default in self.default_config.items()
        }

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

        def rename_dict(path, original, transformed):
            updated = dict(original)
            outputs = {}
            sources = {}  # output name -> the input key that produced it
            for key, old_value in original.items():
                value_path = path + (DictKey(key),)
                value, write_output = rename_value(
                    value_path, old_value, transformed[key]
                )
                if not write_output:
                    updated[key] = value
                    continue
                name = output_key(_output_path(value_path))
                if name in outputs:
                    raise ValueError(
                        f"output_key maps both {sources[name]!r} and {key!r} to "
                        f"{name!r}, so all but one transformed value would be "
                        f"silently discarded. With more than one in-scope "
                        f"leaf, pass a callable output_key that returns a "
                        f"distinct name per path, e.g. "
                        f"lambda path: path[-1] + '_augmented'."
                    )
                outputs[name] = value
                sources[name] = key
            return {**updated, **outputs}

        def rename_value(path, original, transformed):
            # Dicts own their output keys. An AudioTree reached through a
            # sequence uses the containing dict key, with the other sequence
            # entries carried over from the original.
            leaves, treedef = tree_flatten_with_path(
                original, is_leaf=lambda value: isinstance(value, (AudioTree, dict))
            )
            values = []
            changed = write_output = False
            for (subpath, old), new in zip(leaves, treedef.flatten_up_to(transformed)):
                full_path = path + subpath
                if isinstance(old, dict):
                    new = rename_dict(full_path, old, new)
                elif isinstance(old, AudioTree) and _is_in_scope(self.scope, full_path):
                    write_output = True
                else:
                    new = old
                changed |= new is not old
                values.append(new)
            return (treedef.unflatten(values) if changed else original), write_output

        return rename_dict((), old_tree, new_tree)


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
            # Redefine it as a function (a picklable one, not a lambda).
            self.output_key = _ConstantOutputKey(output_key)
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
        Raises:
            ValueError: If an in-scope AudioTree has multiple batch axes and
                the transform has a nonzero probability.
                Flatten it first, or apply the transform to rank-3 audio
                inside scan/vmap.
        """
        if self.prob == 0:
            element = self._preprocess(element)
            return self._post_process(element, element)

        # Per-item draws and probability masks require one active batch axis.
        # Validate before preprocessing or consuming RNG state, and only for
        # selected AudioTrees (labels and out-of-scope subtrees pass through).
        for path, leaf in tree_flatten_with_path(element, is_leaf=_is_audiotree)[0]:
            if isinstance(leaf, AudioTree) and _is_in_scope(self.scope, path):
                _require_single_batch(leaf, f"{type(self).__name__}.random_map")

        element = self._preprocess(element)
        leaves, treedef = tree_flatten_with_path(element, is_leaf=_is_audiotree)
        selected = [
            isinstance(leaf, AudioTree) and _is_in_scope(self.scope, path)
            for path, leaf in leaves
        ]
        # Preserve the draw order for each backend, including one RNG slot per
        # leaf. JAX reserves a separate key for probability before splitting.
        use_numpy = isinstance(rng, np.random.Generator)
        if use_numpy:
            leaf_rngs = _spawn_numpy_rngs(rng, len(leaves), self.split_seed)
        else:
            key, prob_key = random.split(rng)
            leaf_rngs = (
                random.split(key, len(leaves))
                if self.split_seed
                else [key] * len(leaves)
            )
        values = [
            self._apply_transform(leaf, leaf_rng, **self._config_at(path))
            if in_scope
            else leaf
            for (path, leaf), leaf_rng, in_scope in zip(leaves, leaf_rngs, selected)
        ]

        if self.prob < 1:
            if use_numpy:
                prob_rngs = _spawn_numpy_rngs(rng, len(leaves), self.split_seed)
            else:
                prob_rngs = (
                    random.split(prob_key, len(leaves))
                    if self.split_seed
                    else [prob_key] * len(leaves)
                )
            for index, ((_, old), new, in_scope, prob_rng) in enumerate(
                zip(leaves, values, selected, prob_rngs)
            ):
                if in_scope and isinstance(new, AudioTree):
                    size = _leaf_batch_size(old)
                    if use_numpy:
                        mask = prob_rng.random(size) < self.prob
                        values[index] = _select_transformed(new, old, mask, np)
                    else:
                        mask = random.bernoulli(prob_rng, p=self.prob, shape=(size,))
                        values[index] = _select_transformed(new, old, mask, jax.numpy)

        return self._post_process(element, treedef.unflatten(values))


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
            # Redefine it as a function (a picklable one, not a lambda).
            self.output_key = _ConstantOutputKey(output_key)
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

        def apply(path, leaf):
            if isinstance(leaf, AudioTree) and _is_in_scope(self.scope, path):
                return self._apply_transform(leaf, **self._config_at(path))
            return leaf

        element = self._preprocess(element)
        new_tree = map_with_path(apply, element, is_leaf=_is_audiotree)
        return self._post_process(element, new_tree)
