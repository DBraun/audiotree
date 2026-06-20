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
        metadata={"style": rng.integers(0, 5, size=(batch, 4), dtype=np.int32)},
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
    assert item.metadata["style"].shape == (1, 4)
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


def test_batch_round_trips_items():
    tree = _tree(batch=3)
    batched = AudioTree.batch([tree[i] for i in range(len(tree))])
    np.testing.assert_array_equal(batched.waveform, tree.waveform)
    np.testing.assert_array_equal(batched.codes, tree.codes)
    np.testing.assert_array_equal(batched.metadata["style"], tree.metadata["style"])


def test_batch_token_only_items():
    """Items without a waveform (codes-only training examples) batch fine."""
    items = [
        AudioTree(
            waveform=None, sample_rate=48_000,
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


def test_lazy_submodules_importable():
    """`audiotree.sources` / `audiotree.transforms` resolve via the lazy
    module __getattr__ (they're no longer imported eagerly, keeping
    `import audiotree` grain-free)."""
    import audiotree

    assert hasattr(audiotree.sources, "TreeDataSource")
    assert hasattr(audiotree.transforms, "identity")
    with pytest.raises(AttributeError):
        audiotree.not_a_module  # noqa: B018
