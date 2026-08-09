"""Sweep tests for string ``extras`` leaves across every batch-axis operation.

``extras`` may carry per-item strings (the TreeWriter/TreeDataSource
contract): a bare ``str`` means a batch of 1 and a list of strings holds one
per batch item. Each operation here runs on a tree carrying *both* forms —
including one nested inside a sub-dict, alongside a normal array leaf — and
asserts the strings stay element-aligned with the waveform batch axis. Any new
batch-axis operation belongs in this file.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from audiotree import AudioTree


def _item(i: int, bare: bool) -> AudioTree:
    """A batch-of-1 tree whose string leaves are bare or one-item lists.

    ``bare=True`` is what ``TreeDataSource`` yields per item; both forms mean
    the same thing and must batch/index identically. Every fixture also
    carries encoded provenance, so the ``metadata`` container is exercised
    through the same operations as the string leaves.
    """
    wrap = (lambda s: s) if bare else (lambda s: [s])
    return AudioTree.create(
        waveform=np.full((1, 1, 8), float(i), dtype=np.float32),
        sample_rate=16_000,
        extras={
            "tag": wrap(f"tag{i}"),
            "nested": {"group": wrap(f"group{i}")},
            "style": np.full((1, 4), i, dtype=np.int32),
        },
        filepath=f"file{i}.wav",
        source=f"src{i}",
    )


def _batched(n: int) -> AudioTree:
    """A batch-of-``n`` tree with list-form string leaves and provenance."""
    return AudioTree.create(
        waveform=np.arange(n, dtype=np.float32)[:, None, None]
        * np.ones((n, 1, 8), dtype=np.float32),
        sample_rate=16_000,
        extras={
            "tag": [f"tag{i}" for i in range(n)],
            "nested": {"group": [f"group{i}" for i in range(n)]},
            "style": np.arange(n, dtype=np.int32)[:, None] * np.ones((n, 4), np.int32),
        },
        filepath=[f"file{i}.wav" for i in range(n)],
        source=[f"src{i}" for i in range(n)],
    )


def _assert_aligned(tree: AudioTree, expected: list):
    """Every leaf — arrays, strings and provenance — holds ``expected``'s items in order."""
    ids = [int(v) for v in np.asarray(tree.waveform)[:, 0, 0]]
    assert ids == list(expected)
    assert tree.extras["tag"] == [f"tag{i}" for i in expected]
    assert tree.extras["nested"]["group"] == [f"group{i}" for i in expected]
    np.testing.assert_array_equal(
        np.asarray(tree.extras["style"])[:, 0], np.asarray(expected)
    )
    assert tree.filepath == [f"file{i}.wav" for i in expected]
    assert tree.source == [f"src{i}" for i in expected]


@pytest.mark.parametrize("bare", [True, False], ids=["bare-str", "list-of-str"])
def test_batch_items(bare: bool):
    """``AudioTree.batch`` concatenates string leaves like the batch axis.

    The bare-str case is the docstring's own grain recipe
    (``ds.to_iter_dataset().batch(n, batch_fn=AudioTree.batch)``): it used to
    crash on ``np.concatenate`` of 0-d string arrays.
    """
    batched = AudioTree.batch([_item(i, bare) for i in range(3)])
    _assert_aligned(batched, [0, 1, 2])


def test_batch_mixed_forms():
    """A bare str and a one-item list mean the same thing side by side."""
    batched = AudioTree.batch([_item(0, True), _item(1, False), _item(2, True)])
    _assert_aligned(batched, [0, 1, 2])


def test_batch_of_batches():
    """Batching already-batched trees concatenates their string lists."""
    batched = AudioTree.batch([_batched(2), _batched(3)])
    _assert_aligned(batched, [0, 1, 0, 1, 2])


def test_batch_string_leaves_outside_audiotree():
    """String leaves in a structure *around* the trees concatenate too."""
    batched = AudioTree.batch(
        [{"audio": _item(i, True), "label": f"x{i}"} for i in range(3)]
    )
    _assert_aligned(batched["audio"], [0, 1, 2])
    assert batched["label"] == ["x0", "x1", "x2"]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (1, [1]),
        (-1, [3]),
        (np.int64(2), [2]),
        (slice(1, 3), [1, 2]),
        (slice(None, None, 2), [0, 2]),
        ([0, 2], [0, 2]),
        ([2, 0, -1], [2, 0, 3]),
        (np.array([0, 3]), [0, 3]),
        (jnp.array([1, 2]), [1, 2]),
        (np.array([True, False, False, True]), [0, 3]),
        (jnp.array([False, True, True, False]), [1, 2]),
    ],
    ids=[
        "int",
        "negative-int",
        "np-scalar",
        "slice",
        "strided-slice",
        "list",
        "list-with-negative",
        "np-index-array",
        "jax-index-array",
        "np-bool-mask",
        "jax-bool-mask",
    ],
)
def test_getitem(key, expected):
    """Every documented key form selects string leaves like array rows."""
    _assert_aligned(_batched(4)[key], expected)


def test_getitem_normalizes_a_bare_str():
    """Indexing a batch-of-1 tree returns its bare-str leaves in list form."""
    item = _item(5, bare=True)[0]
    assert item.extras["tag"] == ["tag5"]
    assert item.extras["nested"]["group"] == ["group5"]


def test_getitem_rejects_a_mismatched_mask():
    """A wrong-length boolean mask raises like NumPy, never misaligns."""
    tree = _batched(3)
    with pytest.raises(IndexError):
        tree[np.array([True, False])]


def test_split():
    for part, expected in zip(_batched(4).split(2), ([0, 1], [2, 3])):
        _assert_aligned(part, expected)
    # A batch-of-1 tree with bare-str leaves splits into its list form.
    (only,) = _item(7, bare=True).split(1)
    _assert_aligned(only, [7])


def test_filter_keeps_some():
    kept = _batched(4).filter(lambda item: int(item.waveform[0, 0, 0]) % 2 == 0)
    _assert_aligned(kept, [0, 2])


def test_filter_keeps_none_then_composes():
    """A keep-nothing filter empties the string leaves too, and chains."""
    empty = _batched(3).filter(lambda item: False)
    assert empty.batch_size == 0
    _assert_aligned(empty, [])

    # The empty tree is a valid input to every batch-axis op, not a dead end.
    _assert_aligned(empty.filter(lambda item: True), [])
    _assert_aligned(empty[0:0], [])
    _assert_aligned(AudioTree.batch([empty, _batched(2)]), [0, 1])


@pytest.mark.parametrize("bare", [True, False], ids=["bare-str", "list-of-str"])
def test_mini_batch_round_trip(bare: bool):
    """reshape -> index -> flatten keeps strings aligned at every step."""
    tree = AudioTree.batch([_item(i, bare) for i in range(6)])
    mini = tree.reshape_mini_batches(2)

    # String leaves nest per mini-batch, mirroring the arrays' leading axes.
    assert mini.extras["tag"] == [
        ["tag0", "tag1"],
        ["tag2", "tag3"],
        ["tag4", "tag5"],
    ]
    assert mini.extras["nested"]["group"][1] == ["group2", "group3"]

    # Provenance nests per mini-batch too, matching the documented rank-4
    # behaviour of the .filepath/.source properties.
    assert mini.filepath == [
        ["file0.wav", "file1.wav"],
        ["file2.wav", "file3.wav"],
        ["file4.wav", "file5.wav"],
    ]
    assert mini.source[2] == ["src4", "src5"]

    # Indexing the rank-4 tree selects whole mini-batches, strings included.
    picked = mini[1]
    assert picked.waveform.shape == (1, 2, 1, 8)
    assert picked.extras["tag"] == [["tag2", "tag3"]]
    assert picked.filepath == [["file2.wav", "file3.wav"]]
    _assert_aligned(picked.flatten_mini_batches(), [2, 3])

    for part, expected in zip(mini.split(3), ([0, 1], [2, 3], [4, 5])):
        _assert_aligned(part.flatten_mini_batches(), expected)

    # The full round trip is the identity.
    _assert_aligned(mini.flatten_mini_batches(), [0, 1, 2, 3, 4, 5])


def test_reshape_mini_batches_bare_str():
    """A batch-of-1 tree's bare str nests like a one-item list."""
    mini = _item(4, bare=True).reshape_mini_batches(1)
    assert mini.extras["tag"] == [["tag4"]]
    _assert_aligned(mini.flatten_mini_batches(), [4])


def test_reshape_mini_batches_rejects_a_misaligned_string_leaf():
    """A string list that disagrees with the batch size raises like an array would."""
    tree = _batched(4).replace_extras(tag=["only", "three", "tags"])
    with pytest.raises(ValueError, match="[Ss]tring extras"):
        tree.reshape_mini_batches(2)
