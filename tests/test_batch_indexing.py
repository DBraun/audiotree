"""Tests for AudioTree batch indexing/iteration, batch with optional
fields, to_mono strategies, and lazy submodule loading."""

import numpy as np
import pytest

from audiotree import AudioTree


def _tree(batch: int = 3, channels: int = 2, samples: int = 16) -> AudioTree:
    rng = np.random.default_rng(0)
    return AudioTree(
        waveform=rng.standard_normal((batch, channels, samples)).astype(np.float32),
        sample_rate=44_100,
        codes=rng.integers(0, 8, size=(batch, 4, 3), dtype=np.int32),
        extras={"style": rng.integers(0, 5, size=(batch, 4), dtype=np.int32)},
    )


def test_len_and_batch_size():
    tree = _tree(batch=3)
    assert len(tree) == 3
    assert tree.batch_size == 3
    assert tree.num_channels == 2


def test_batch_size_token_only():
    codes = np.zeros((5, 4, 3), dtype=np.int32)
    tree = AudioTree(waveform=None, sample_rate=44_100, codes=codes)
    assert tree.batch_size == 5
    assert len(tree) == 5


def test_getitem_int_keeps_batch_axis():
    tree = _tree(batch=3)
    item = tree[1]
    assert item.waveform.shape == (1,) + tree.waveform.shape[1:]
    assert item.codes.shape == (1,) + tree.codes.shape[1:]
    assert item.extras["style"].shape == (1, 4)
    np.testing.assert_array_equal(item.waveform[0], tree.waveform[1])
    np.testing.assert_array_equal(item.codes[0], tree.codes[1])
    assert item.sample_rate == tree.sample_rate


def test_getitem_slice_and_iteration():
    tree = _tree(batch=3)
    sub = tree[1:3]
    assert sub.batch_size == 2

    items = list(tree)
    assert len(items) == 3
    for i, item in enumerate(items):
        np.testing.assert_array_equal(item.waveform[0], tree.waveform[i])


def test_is_iterable_and_yields_batch_of_one():
    from collections.abc import Iterable, Iterator

    tree = _tree(batch=16)
    # AudioTree advertises itself as a proper Iterable via __iter__.
    assert isinstance(tree, Iterable)
    assert isinstance(iter(tree), Iterator)

    count = 0
    for a_tree_batch_size1 in tree:
        assert a_tree_batch_size1.waveform.shape[0] == 1
        assert isinstance(a_tree_batch_size1, AudioTree)
        count += 1
    assert count == 16


def test_getitem_negative_index():
    tree = _tree(batch=3)
    np.testing.assert_array_equal(tree[-1].waveform[0], tree.waveform[2])


def test_getitem_numpy_scalar_keeps_batch_axis():
    """Any integer scalar indexes the batch, not just the builtin ``int``.

    ``np.argmax`` & friends return ``np.int64``, which used to fall through to
    raw array indexing and scalar-index every leaf -- silently dropping the
    batch axis, so the channel count masqueraded as ``batch_size``.
    """
    import jax.numpy as jnp

    tree = _tree(batch=3, channels=2)
    tree = tree.replace(lufs=np.array([-30.0, -10.0, -50.0], dtype=np.float32))

    for key in (
        np.argmax(tree.lufs),  # np.int64
        np.int32(1),
        np.array(1),  # 0-d array
        jnp.asarray(1),  # 0-d JAX array
        np.where(tree.lufs > -20.0)[0][0],
        1,
    ):
        item = tree[key]
        assert item.batch_size == 1, key
        assert item.waveform.shape == (1, 2, 16), key
        assert item.codes.shape == (1, 4, 3), key
        assert item.lufs.shape == (1,), key
        np.testing.assert_array_equal(item.waveform[0], tree.waveform[1])

    # Negative NumPy scalars index from the end, like the builtin int does.
    np.testing.assert_array_equal(tree[np.int64(-1)].waveform[0], tree.waveform[2])
    with pytest.raises(IndexError):
        tree[np.int64(3)]


def test_getitem_fancy_and_mask_keys_still_work():
    """Lists, index arrays and boolean masks keep selecting sub-batches."""
    tree = _tree(batch=3)
    for key in ([0, 2], np.array([0, 2]), np.array([True, False, True])):
        sub = tree[key]
        assert sub.batch_size == 2
        np.testing.assert_array_equal(sub.waveform[1], tree.waveform[2])
        assert sub.extras["style"].shape == (2, 4)


def test_batch_round_trips_items():
    tree = _tree(batch=3)
    batched = AudioTree.batch([tree[i] for i in range(len(tree))])
    np.testing.assert_array_equal(batched.waveform, tree.waveform)
    np.testing.assert_array_equal(batched.codes, tree.codes)
    np.testing.assert_array_equal(batched.extras["style"], tree.extras["style"])


def test_batch_keeps_jax_arrays_on_device():
    """JAX in, JAX out.

    ``AudioTree.batch`` hardcoded ``np.concatenate``, so the function the README
    tells everyone to pass as ``batch_fn`` turned every device-resident batch
    into NumPy -- a blocking host sync and a silent type change.
    """
    import jax
    import jax.numpy as jnp

    items = [
        AudioTree(
            waveform=jnp.full((1, 2, 16), float(i), dtype=jnp.float32),
            sample_rate=44_100,
            codes=jnp.full((1, 4, 3), i, dtype=jnp.int32),
            extras={"style": jnp.full((1, 4), i, dtype=jnp.int32)},
        )
        for i in range(3)
    ]
    batched = AudioTree.batch(items)

    assert isinstance(batched.waveform, jax.Array)
    assert isinstance(batched.codes, jax.Array)
    assert isinstance(batched.extras["style"], jax.Array)
    assert batched.waveform.shape == (3, 2, 16)
    np.testing.assert_array_equal(np.asarray(batched.waveform[2]), 2.0)

    # A dict of JAX arrays around the trees stays JAX too.
    nested = AudioTree.batch(
        [{"audio": item, "weight": jnp.ones((1,))} for item in items]
    )
    assert isinstance(nested["weight"], jax.Array)
    assert isinstance(nested["audio"].waveform, jax.Array)

    # NumPy items still come back as NumPy.
    numpy_items = [tree for tree in _tree(batch=2)]
    assert isinstance(AudioTree.batch(numpy_items).waveform, np.ndarray)


def test_batch_rejects_an_empty_sequence():
    """An empty batch has no structure to infer; say so instead of IndexError."""
    with pytest.raises(ValueError, match="at least one item"):
        AudioTree.batch([])


def test_batch_token_only_items():
    """Items without a waveform (codes-only training examples) batch fine."""
    items = [
        AudioTree(
            waveform=None,
            sample_rate=48_000,
            codes=np.full((1, 4, 3), i, dtype=np.int32),
        )
        for i in range(3)
    ]
    batched = AudioTree.batch(items)
    assert batched.waveform is None
    assert batched.codes.shape == (3, 4, 3)
    np.testing.assert_array_equal(batched.codes[2], np.full((4, 3), 2))


def test_to_mono_strategies():
    tree = _tree(batch=2, channels=2)
    mono_avg = tree.to_mono()
    np.testing.assert_allclose(
        mono_avg.waveform, tree.waveform.mean(axis=1, keepdims=True), rtol=1e-6
    )
    np.testing.assert_array_equal(
        tree.to_mono("left").waveform, tree.waveform[:, 0:1, :]
    )
    np.testing.assert_array_equal(
        tree.to_mono("right").waveform, tree.waveform[:, 1:2, :]
    )
    with pytest.raises(ValueError, match="strategy"):
        tree.to_mono("center")


def test_to_mono_validates_strategy_on_mono_input():
    """An invalid strategy raises even when there is nothing to mix down.

    The mono short-circuit used to return ``self`` before looking at the
    argument, so a typo passed silently on mono files and only blew up later on
    a stereo one.
    """
    mono = _tree(batch=2, channels=1)
    with pytest.raises(ValueError, match="strategy"):
        mono.to_mono("center")
    # The valid strategies are still no-ops on mono audio.
    for strategy in ("average", "left", "right"):
        assert mono.to_mono(strategy) is mono


def test_lazy_submodules_importable():
    """`audiotree.sources` / `audiotree.transforms` resolve via the lazy
    module __getattr__ (they're no longer imported eagerly, keeping
    `import audiotree` grain-free)."""
    import audiotree

    assert hasattr(audiotree.sources, "TreeDataSource")
    assert hasattr(audiotree.transforms, "identity")
    with pytest.raises(AttributeError):
        audiotree.not_a_module  # noqa: B018
