"""Tests for AudioTree core functionality."""

from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from audiotree import transforms
from audiotree.core import AudioTree


def test_audiotree_create_with_filepaths():
    """Test that AudioTree.create accepts and processes filepaths parameter."""
    # Test with 1D audio data and single filepath string
    audio_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree1 = AudioTree.create(audio_1d, 44100, filepath="test1.wav")

    assert tree1.waveform.shape == (
        1,
        1,
        5,
    )  # Should be expanded to (batch, channels, samples)
    assert tree1.filepath == ["test1.wav"]
    assert "filepath" in tree1.metadata
    assert tree1.extras == {}  # provenance never lands in the user dict

    # Test with 2D audio data and single filepath Path
    audio_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
    tree2 = AudioTree.create(audio_2d, 44100, filepath=Path("test2.wav"))

    assert tree2.waveform.shape == (
        1,
        2,
        3,
    )  # Should be expanded to (batch, channels, samples)
    assert tree2.filepath == ["test2.wav"]

    # Test with 3D audio data and a list of one filepath per batch item.
    audio_3d = np.zeros((3, 1, 3))  # (batch=3, channels, samples)
    filepaths = ["file1.wav", "file2.wav", Path("file3.wav")]
    tree3 = AudioTree.create(audio_3d, 44100, filepath=filepaths)

    assert tree3.waveform.shape == (3, 1, 3)  # Should remain unchanged
    assert tree3.filepath == ["file1.wav", "file2.wav", "file3.wav"]

    # Test with no filepaths (should work as before)
    tree4 = AudioTree.create(audio_1d, 44100)

    assert tree4.waveform.shape == (1, 1, 5)
    assert tree4.filepath == []  # Empty list when no filepaths provided
    assert "filepath" not in tree4.metadata


def test_audiotree_create_with_source():
    """AudioTree.create accepts a source group name (single or per-batch-item)."""
    # Single string tags the whole tree; read back via the .source property.
    audio_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree1 = AudioTree.create(audio_1d, 44100, source="music")
    assert tree1.source == ["music"]
    assert "source" in tree1.metadata
    assert tree1.extras == {}

    # A list gives one source name per batch item (unlike from_file).
    audio_batch = np.zeros((3, 1, 4))  # (batch=3, channels, samples)
    tree2 = AudioTree.create(audio_batch, 44100, source=["drums", "vocal", "impulse"])
    assert tree2.source == ["drums", "vocal", "impulse"]

    # No source -> no metadata key, empty list (unchanged behavior).
    tree3 = AudioTree.create(audio_1d, 44100)
    assert tree3.source == []
    assert "source" not in tree3.metadata


def test_filepath_too_long_raises():
    """Filepaths longer than the encoding limit raise instead of truncating."""
    from audiotree.core import _str_max_length

    audio = np.zeros((1, 1, 5))

    # A path exactly at the limit round-trips intact.
    exact = "a" * _str_max_length
    tree = AudioTree.create(audio, 44100, filepath=exact)
    assert tree.filepath == [exact]

    # One character over the limit raises rather than silently truncating.
    too_long = "a" * (_str_max_length + 1)
    with pytest.raises(ValueError, match="exceeds the provenance encoding limit"):
        AudioTree.create(audio, 44100, filepath=too_long)


def test_audiotree_constructor_compatibility():
    """Test that original AudioTree constructor still works for backward compatibility."""
    audio_3d = np.array(
        [[[1.0, 2.0, 3.0]]]
    )  # Use 3D data since constructor won't reshape
    tree = AudioTree(audio_3d, 44100)

    assert tree.waveform.shape == (1, 1, 3)
    assert tree.filepath == []


def test_audiotree_samples_property():
    """Test that the samples property returns the last dimension of the waveform."""
    tree = AudioTree(np.zeros((4, 2, 44100)), 44100)
    assert tree.samples == 44100


def test_write_round_trip(tmp_path):
    """write() saves a batch-of-1 tree and round-trips via from_file."""
    import soundfile

    sr = 16000
    waveform = (np.random.default_rng(0).standard_normal((1, 2, sr)) * 0.1).astype(
        np.float32
    )
    tree = AudioTree(waveform=waveform, sample_rate=sr)

    out = tree.write(tmp_path / "out.wav")
    assert isinstance(out, Path) and out.exists()

    # soundfile stores (samples, channels); default WAV subtype is PCM_16.
    data, sr_read = soundfile.read(str(out))
    assert sr_read == sr
    assert data.shape == (sr, 2)
    assert soundfile.info(str(out)).subtype == "PCM_16"

    # Round-trips through from_file (allowing PCM_16 quantization error).
    reloaded = AudioTree.from_file(out, sample_rate=sr)
    assert reloaded.sample_rate == sr
    np.testing.assert_allclose(reloaded.waveform[0], waveform[0], atol=1e-3)


def test_from_file_pads_duration_without_sample_rate(tmp_path):
    """from_file honors ``duration`` even when ``sample_rate`` is None.

    Regression: target_length was computed only when both duration and
    sample_rate were given, so a short file loaded at its native rate (sr=None)
    was returned unpadded despite the docstring's unconditional promise and
    pad_mode defaulting to "constant". librosa.load with sr=None reports the
    native rate, so the target length is computable from it.
    """
    import soundfile

    sr = 16000
    path = tmp_path / "half_second.wav"
    soundfile.write(str(path), np.zeros(sr // 2, np.float32), sr, subtype="FLOAT")

    # With an explicit rate the short file is padded to 2 s.
    assert AudioTree.from_file(path, sample_rate=sr, duration=2.0).waveform.shape == (
        1,
        1,
        2 * sr,
    )

    # With sr=None (native rate) it is padded to the same length.
    reloaded = AudioTree.from_file(path, duration=2.0)
    assert reloaded.sample_rate == sr
    assert reloaded.waveform.shape == (1, 1, 2 * sr)


def test_write_options_and_batch_size_check(tmp_path):
    """write() forwards subtype/format and requires batch_size == 1."""
    import soundfile

    sr = 8000
    single = AudioTree(
        waveform=np.ones((1, 1, sr), dtype=np.float32) * 0.5, sample_rate=sr
    )

    # subtype is forwarded.
    single.write(tmp_path / "a.wav", subtype="PCM_24")
    assert soundfile.info(str(tmp_path / "a.wav")).subtype == "PCM_24"

    # Format is inferred from the extension (FLAC here).
    single.write(tmp_path / "a.flac")
    assert soundfile.info(str(tmp_path / "a.flac")).format == "FLAC"

    # A multi-item batch must be indexed/iterated first. This is a ValueError,
    # not an assert: an assert would vanish under ``python -O`` and soundfile
    # would silently write only the first item.
    batch = AudioTree(waveform=np.zeros((3, 1, sr), dtype=np.float32), sample_rate=sr)
    with pytest.raises(ValueError, match="batch_size == 1"):
        batch.write(tmp_path / "fail.wav")

    # Iterating yields writable batch-of-1 trees.
    for i, item in enumerate(batch):
        item.write(tmp_path / f"item_{i}.wav")
    assert sorted(p.name for p in tmp_path.glob("item_*.wav")) == [
        "item_0.wav",
        "item_1.wav",
        "item_2.wav",
    ]


def test_methods_work_after_reshape_mini_batches():
    """All AudioTree methods must handle the extra leading axis from reshape_mini_batches."""
    rng = np.random.default_rng(0)
    sample_rate = 16000
    audio = rng.uniform(-0.5, 0.5, (4, 2, 8000)).astype(np.float32)
    tree = AudioTree(audio, sample_rate)
    mini = tree.reshape_mini_batches(2)
    assert mini.waveform.shape == (2, 2, 2, 8000)

    assert mini.samples == 8000
    assert mini.num_channels == 2

    # replace_lufs matches the flat computation, reshaped
    flat_tree = tree.replace_lufs()
    mini_tree = mini.replace_lufs()
    flat_loudness = flat_tree.lufs
    mini_loudness = mini_tree.lufs
    assert mini_loudness.shape == (2, 2)
    np.testing.assert_allclose(mini_loudness, flat_loudness.reshape(2, 2), rtol=1e-5)

    # lufs_windows carries the same leading axes; 8000 samples / 0.4s window at
    # 16 kHz (6400 samples) is a single window.
    assert flat_tree.lufs_windows.shape == (4, 1)
    assert mini_tree.lufs_windows.shape == (2, 2, 1)
    np.testing.assert_allclose(
        mini_tree.lufs_windows, flat_tree.lufs_windows.reshape(2, 2, 1), rtol=1e-5
    )

    # normalize_lufs
    normalized = mini.normalize_lufs(-18.0)
    assert normalized.waveform.shape == mini.waveform.shape
    np.testing.assert_allclose(normalized.replace_lufs().lufs, -18.0, atol=0.5)

    # to_mono / to_stereo
    mono = mini.to_mono()
    assert mono.waveform.shape == (2, 2, 1, 8000)
    left = mini.to_mono("left")
    np.testing.assert_array_equal(left.waveform[..., 0, :], mini.waveform[..., 0, :])
    stereo = mono.to_stereo()
    assert stereo.waveform.shape == (2, 2, 2, 8000)
    np.testing.assert_array_equal(
        stereo.waveform[..., 0, :], stereo.waveform[..., 1, :]
    )

    # resample matches the flat computation, reshaped
    resampled = mini.resample(8000)
    assert resampled.waveform.shape == (2, 2, 2, 4000)
    np.testing.assert_allclose(
        np.asarray(resampled.waveform).reshape(4, 2, 4000),
        np.asarray(tree.resample(8000).waveform),
        rtol=1e-5,
    )


def test_provenance_decodes_at_rank_4():
    """filepath/source survive reshape_mini_batches, nesting to match the axes."""
    audio = np.zeros((4, 1, 16), dtype=np.float32)
    tree = AudioTree.create(
        audio,
        16000,
        filepath=[f"f{i}.wav" for i in range(4)],
        source=["music", "music", "speech", "speech"],
    )
    assert tree.filepath == ["f0.wav", "f1.wav", "f2.wav", "f3.wav"]

    mini = tree.reshape_mini_batches(2)
    assert mini.metadata["filepath"].shape == (2, 2, 1024)
    # Previously a ValueError from NumPy ("truth value of an array ... is
    # ambiguous"): the decoder assumed the leading axis was the batch.
    assert mini.filepath == [["f0.wav", "f1.wav"], ["f2.wav", "f3.wav"]]
    assert mini.source == [["music", "music"], ["speech", "speech"]]

    # Indexing the leading axis keeps the nesting aligned with the arrays.
    assert mini[1].waveform.shape == (1, 2, 1, 16)
    assert mini[1].filepath == [["f2.wav", "f3.wav"]]

    # Flattening restores the flat, one-string-per-item view.
    assert mini.flatten_mini_batches().filepath == tree.filepath


def test_provenance_without_leading_axis_raises():
    """A hand-encoded row with no batch axis names the field instead of crashing."""
    from audiotree.core import _str_max_length

    encoded = AudioTree._encode_filepaths(["only.wav"])[0]  # (1024,), no batch axis
    assert encoded.shape == (_str_max_length,)
    tree = AudioTree(np.zeros((1, 1, 16)), 16000, metadata={"filepath": encoded})
    with pytest.raises(ValueError, match="must have a leading batch axis"):
        tree.filepath


def test_rank_4_rejected_where_it_has_no_meaning():
    """Operations defined per batch item name the rank instead of misbehaving."""
    tree = AudioTree(np.zeros((4, 1, 16), dtype=np.float32), 16000)
    mini = tree.reshape_mini_batches(2)

    # write(): batch_size counts mini-batches at rank 4, so a (1, B, C, T) tree
    # slipped past the batch-of-1 check and reached soundfile, which complained
    # about the *transposed* shape and named neither the rank nor the fix.
    assert mini[0].batch_size == 1
    with pytest.raises(ValueError, match="rank 4"):
        mini[0].write("/dev/null")

    # A rank-5 tree is readable by nothing here.
    with pytest.raises(ValueError, match="rank 4"):
        mini.reshape_mini_batches(1)

    # Per-item filtering would leave the mini-batches ragged.
    with pytest.raises(ValueError, match="rank 4"):
        mini.filter(lambda item: True)

    # create() encodes one provenance row per batch item, and a mini-batched
    # waveform has no single such axis: a broadcast filepath used to come out
    # one row per mini-batch, i.e. fewer strings than items.
    with pytest.raises(ValueError, match="rank-4 waveform"):
        AudioTree.create(np.zeros((2, 2, 1, 16)), 16000, filepath="x.wav")


def test_write_without_waveform_raises():
    """A token-only tree says so rather than raising TypeError on None."""
    tree = AudioTree(waveform=None, sample_rate=16000, codes=np.zeros((1, 4, 8)))
    with pytest.raises(ValueError, match="needs a waveform"):
        tree.write("/dev/null")


def test_batch_axis_ops_work_on_token_only_tree():
    """split/filter/reshape/flatten read the batch axis from codes, not waveform.

    A token-only tree (waveform=None) has its batch axis on ``codes`` /
    ``latents``; these are pure batch-axis operations, so they must derive the
    size from whichever leaf is present instead of dereferencing ``None.shape``.
    """
    tok = AudioTree.create(None, 16000, codes=np.zeros((4, 2, 10), dtype=np.int32))
    assert len(tok) == 4

    parts = tok.split(2)
    assert [p.codes.shape for p in parts] == [(2, 2, 10), (2, 2, 10)]

    assert tok.filter(lambda t: True).codes.shape == (4, 2, 10)

    mini = tok.reshape_mini_batches(2)
    assert mini.codes.shape == (2, 2, 2, 10)
    assert mini.flatten_mini_batches().codes.shape == (4, 2, 10)


def test_create_rejects_provenance_list_of_wrong_length():
    """A per-item filepath/source list must match the batch, or provenance
    silently misaligns (fewer strings than items). A scalar still broadcasts and
    a correct-length list is accepted."""
    waveform = np.zeros((4, 1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="2 filepath for a batch of 4"):
        AudioTree.create(waveform, 16000, filepath=["x.wav", "y.wav"])

    with pytest.raises(ValueError, match="2 source for a batch of 4"):
        AudioTree.create(waveform, 16000, source=["a", "b"])

    # A correct-length list is accepted and round-trips per item.
    ok = AudioTree.create(
        waveform,
        16000,
        filepath=["a", "b", "c", "d"],
        source=["w", "x", "y", "z"],
    )
    assert ok.filepath == ["a", "b", "c", "d"]
    assert ok.source == ["w", "x", "y", "z"]

    # A single value still broadcasts across the whole batch.
    bc = AudioTree.create(waveform, 16000, filepath="one.wav")
    assert bc.metadata["filepath"].shape[0] == 4
    assert bc.filepath == ["one.wav"] * 4


def test_audiotree_create_audio_dimensionality():
    """Test that AudioTree.create handles different audio dimensionalities correctly."""
    sample_rate = 44100

    # Test 1D audio
    audio_1d = np.array([1.0, 2.0, 3.0])
    tree1 = AudioTree.create(audio_1d, sample_rate)
    assert tree1.waveform.shape == (1, 1, 3)

    # Test 2D audio
    audio_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tree2 = AudioTree.create(audio_2d, sample_rate)
    assert tree2.waveform.shape == (1, 2, 3)

    # Test 3D audio (already correct)
    audio_3d = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    tree3 = AudioTree.create(audio_3d, sample_rate)
    assert tree3.waveform.shape == (2, 1, 3)


def test_audiotree_create_extras_handling():
    """Test that AudioTree.create handles extras correctly with filepaths."""
    waveform = np.array([1.0, 2.0, 3.0])
    sample_rate = 44100

    # Test with existing extras and filepaths
    existing_extras = {"custom_key": "custom_value"}
    tree1 = AudioTree.create(
        waveform, sample_rate, extras=existing_extras, filepath="test.wav"
    )

    # Should preserve existing extras; the filepath goes to metadata, so the
    # user's dict is exactly what was passed in.
    assert tree1.extras == {"custom_key": "custom_value"}
    assert "filepath" in tree1.metadata
    assert tree1.filepath == ["test.wav"]

    # Test that original extras dict is not modified
    assert "filepath" not in existing_extras

    # Test with filepaths but no existing extras
    tree2 = AudioTree.create(waveform, sample_rate, filepath="test2.wav")
    assert "filepath" in tree2.metadata
    assert tree2.extras == {}
    assert tree2.filepath == ["test2.wav"]


def test_replace_extras():
    """replace_extras merges kwargs into extras without mutating the original."""
    tree = AudioTree.create(
        np.zeros((2, 1, 100)),
        44100,
        extras={"energy": np.array([0.8, 0.9]), "tag": np.array([1, 2])},
    )

    tagged = tree.replace_extras(onsets=np.array([[0.1], [0.2]]), tag=np.array([3, 4]))

    # New key added, colliding key overwritten, other keys preserved
    np.testing.assert_array_equal(tagged.extras["onsets"], [[0.1], [0.2]])
    np.testing.assert_array_equal(tagged.extras["tag"], [3, 4])
    np.testing.assert_array_equal(tagged.extras["energy"], [0.8, 0.9])

    # Everything else carries over untouched
    assert tagged.sample_rate == tree.sample_rate
    np.testing.assert_array_equal(tagged.waveform, tree.waveform)

    # The original tree and its extras dict are not mutated
    assert set(tree.extras) == {"energy", "tag"}
    np.testing.assert_array_equal(tree.extras["tag"], [1, 2])

    # No kwargs is a no-op copy
    same = tree.replace_extras()
    assert set(same.extras) == {"energy", "tag"}


def test_split_by_batch():
    x = AudioTree(np.zeros((4, 1, 44100)), 44100)
    trees = [x, x, x]
    big_tree = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *trees)
    assert big_tree.waveform.shape == (12, 1, 44100)
    split_trees = big_tree.split(2)
    assert len(split_trees) == 2
    assert split_trees[0].waveform.shape == (6, 1, 44100)


def test_split_preserves_string_list_extras():
    """split() slices list-of-strings extras element-wise, not character-wise.

    Regression: split() used a bare tree_map with no is_leaf, so jax descended
    into the list and sliced each string's characters
    (``['a.wav', ...]`` -> ``['a.', ...]``). It must treat the list as one leaf,
    exactly like __getitem__ does.
    """
    names = ["a.wav", "b.wav", "c.wav", "d.wav"]
    tree = AudioTree.create(
        np.zeros((4, 1, 8), np.float32), 16000, extras={"names": names}
    )

    first, second = tree.split(2)
    assert first.extras["names"] == ["a.wav", "b.wav"]
    assert second.extras["names"] == ["c.wav", "d.wav"]

    # A bare-string leaf is a batch of 1 (the TreeDataSource per-item form);
    # batch ops slice it element-wise and return the list form, never
    # character-slicing it.
    tagged = AudioTree.create(
        np.zeros((1, 1, 8), np.float32), 16000, extras={"tag": "hello"}
    )
    assert tagged.split(1)[0].extras["tag"] == ["hello"]

    # filter() (split + _batch_audiotrees) and AudioTree.batch round-trip a
    # string-list leaf instead of crashing on it.
    assert tree.filter(lambda t: True).extras["names"] == names
    assert AudioTree.batch([first, second]).extras["names"] == names


def test_split_by_mini_batch():
    """Test that split_by_mini_batch correctly reshapes AudioTree with mini-batch dimension."""

    # Create an AudioTree with batch size 12
    sample_rate = 44_100
    batch_size = 12
    channels = 2
    samples = 1000

    # Create distinctive audio data so we can verify correct reshaping
    waveform = np.arange(batch_size * channels * samples, dtype=np.float32).reshape(
        batch_size, channels, samples
    )
    audio_tree = AudioTree(waveform, sample_rate)

    # Test splitting into mini-batches of size 3
    mini_batch_size = 3
    reshaped_tree = audio_tree.reshape_mini_batches(mini_batch_size)

    # Expected shape: (num_mini_batches=4, mini_batch_size=3, channels=2, samples=1000)
    expected_shape = (4, 3, channels, samples)
    assert reshaped_tree.waveform.shape == expected_shape

    # Verify the data is correctly reshaped (not just shape but actual values)
    # The first mini-batch should contain the first 3 samples from the original batch
    first_mini_batch = reshaped_tree.waveform[0]  # Shape: (3, 2, 1000)
    expected_first_mini_batch = waveform[:3]  # First 3 samples from original
    np.testing.assert_array_equal(first_mini_batch, expected_first_mini_batch)

    # The last mini-batch should contain samples 9-11 from the original batch
    last_mini_batch = reshaped_tree.waveform[-1]  # Shape: (3, 2, 1000)
    expected_last_mini_batch = waveform[9:12]  # Last 3 samples from original
    np.testing.assert_array_equal(last_mini_batch, expected_last_mini_batch)

    # Test with different mini-batch size
    mini_batch_size_2 = 4
    reshaped_tree_2 = audio_tree.reshape_mini_batches(mini_batch_size_2)

    # Expected shape: (num_mini_batches=3, mini_batch_size=4, channels=2, samples=1000)
    expected_shape_2 = (3, 4, channels, samples)
    assert reshaped_tree_2.waveform.shape == expected_shape_2

    # Test that sample_rate is preserved
    assert reshaped_tree.sample_rate == sample_rate
    assert reshaped_tree_2.sample_rate == sample_rate

    # Test edge case: mini_batch_size equals batch_size
    reshaped_tree_full = audio_tree.reshape_mini_batches(batch_size)
    assert reshaped_tree_full.waveform.shape == (1, batch_size, channels, samples)
    np.testing.assert_array_equal(reshaped_tree_full.waveform[0], waveform)


def test_unsplit_mini_batch():
    """Test that unsplit_mini_batch correctly flattens mini-batch dimension back to batch dimension."""

    # Create an AudioTree with batch size 12
    sample_rate = 44_100
    batch_size = 12
    channels = 2
    samples = 1000

    # Create distinctive audio data so we can verify correct reshaping
    original_audio_data = np.arange(
        batch_size * channels * samples, dtype=np.float32
    ).reshape(batch_size, channels, samples)
    audio_tree = AudioTree(original_audio_data, sample_rate)

    # Test round-trip: split then unsplit with mini-batch size 3
    mini_batch_size = 3
    batched_tree = audio_tree.reshape_mini_batches(mini_batch_size)
    unbatched_tree = batched_tree.flatten_mini_batches()

    # Verify we get back the original shape
    assert unbatched_tree.waveform.shape == original_audio_data.shape
    # Verify we get back the exact same data
    np.testing.assert_array_equal(unbatched_tree.waveform, original_audio_data)
    # Verify sample_rate is preserved
    assert unbatched_tree.sample_rate == sample_rate

    # Test with different mini-batch size
    mini_batch_size_2 = 4
    batched_tree_2 = audio_tree.reshape_mini_batches(mini_batch_size_2)
    unbatched_tree_2 = batched_tree_2.flatten_mini_batches()

    assert unbatched_tree_2.waveform.shape == original_audio_data.shape
    np.testing.assert_array_equal(unbatched_tree_2.waveform, original_audio_data)

    # Test edge case: mini_batch_size equals batch_size
    batched_tree_full = audio_tree.reshape_mini_batches(batch_size)
    unbatched_tree_full = batched_tree_full.flatten_mini_batches()

    assert unbatched_tree_full.waveform.shape == original_audio_data.shape
    np.testing.assert_array_equal(unbatched_tree_full.waveform, original_audio_data)

    # Test that unsplitting preserves extras if present (one string per item,
    # the tree_writer contract).
    names = [f"item{i}.wav" for i in range(batch_size)]
    audio_tree_with_extras = AudioTree(
        original_audio_data, sample_rate, extras={"test_key": names}
    )
    batched_with_extras = audio_tree_with_extras.reshape_mini_batches(mini_batch_size)
    unbatched_with_extras = batched_with_extras.flatten_mini_batches()

    assert unbatched_with_extras.extras == {"test_key": names}

    # Test direct unsplit on already mini-batched data
    # Create data that's already in mini-batch format
    mini_batched_data = np.arange(4 * 3 * channels * samples, dtype=np.float32).reshape(
        4,
        3,
        channels,
        samples,  # (num_mini_batches, mini_batch_size, channels, samples)
    )
    mini_batched_tree = AudioTree(mini_batched_data, sample_rate)
    flattened_tree = mini_batched_tree.flatten_mini_batches()

    expected_shape = (12, channels, samples)  # 4 * 3 = 12
    assert flattened_tree.waveform.shape == expected_shape


def test_forward_batches_with_scan():
    """Demonstrate memory-efficient inference on large AudioTrees.

    A large batch like [4096, 2, 44100] fits in GPU memory as raw data, but
    processing it through a model creates intermediate activations that don't
    fit. Using reshape_mini_batches + nnx.scan + flatten_mini_batches, we can
    process it in chunks while keeping everything jittable.

    Note: this trick saves memory only for inference (forward pass). During
    training, backprop through scan still materializes all intermediate
    activations, so there is no memory benefit.
    """
    from flax import nnx

    # A toy model that doubles the audio
    class ToyModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.scale = nnx.Param(jax.numpy.array(2.0))

        def __call__(self, x: AudioTree) -> AudioTree:
            return x.replace(waveform=x.waveform * self.scale[...])

    model = ToyModel(rngs=nnx.Rngs(0))

    # Simulate a large batch (small here so the test is fast)
    batch_size = 12
    mini_batch_size = 3
    x = AudioTree(np.ones((batch_size, 2, 1000), dtype=np.float32), 44_100)

    # --- Naive approach: process all at once (would OOM on large batches) ---
    @nnx.jit
    def forward_naive(_model, _x):
        return _model(_x)

    out_naive = forward_naive(model, x)

    # --- Memory-efficient approach: reshape → scan → flatten ---
    #
    # reshape_mini_batches and flatten_mini_batches happen outside jit because
    # they use Python-level shape assertions. The scan body runs inside jit.
    x_mini = x.reshape_mini_batches(mini_batch_size)
    # x_mini.waveform.shape == (4, 3, 2, 1000)

    @nnx.jit
    def forward_batches(_model, _x):
        @nnx.scan(in_axes=(None, 0), out_axes=0)
        def scan_fn(__model, __x):
            return __model(__x)

        return scan_fn(_model, _x)

    out_mini = forward_batches(model, x_mini)
    out_batched = out_mini.flatten_mini_batches()

    # Both approaches produce the same result
    np.testing.assert_allclose(out_naive.waveform, out_batched.waveform, rtol=1e-5)
    assert out_naive.waveform.shape == (batch_size, 2, 1000)
    assert out_batched.waveform.shape == (batch_size, 2, 1000)
    np.testing.assert_allclose(out_batched.waveform, 2.0)


def _tone(sample_rate: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_replace_lufs_windows_shape_and_default_window():
    """replace_lufs fills both ``lufs`` (B,) and ``lufs_windows`` (B, n_windows)."""
    sr = 44100
    tree = AudioTree.create(_tone(sr, 2.0), sr).replace_lufs()  # default 0.4s window
    assert tree.lufs.shape == (1,)
    # 2.0s / 0.4s -> 5 non-overlapping windows.
    assert tree.lufs_windows.shape == (1, 5)
    # A steady tone is uniformly loud window to window (the first window carries a
    # small K-filter start-up transient, hence the loose tolerance), and its gated
    # integrated LUFS sits near the ungated per-window LUFS.
    windows = np.asarray(tree.lufs_windows[0])
    np.testing.assert_allclose(windows, windows.mean(), atol=0.05)
    assert abs(float(tree.lufs[0]) - float(windows.mean())) < 1.0


def test_replace_lufs_windows_custom_duration_and_short_excerpt():
    """The window length is configurable; a sub-window excerpt yields no windows."""
    sr = 44100
    tree = AudioTree.create(_tone(sr, 2.0), sr)
    assert tree.replace_lufs(lufs_window_sec=1.0).lufs_windows.shape == (1, 2)
    # Shorter than one 0.4s window -> empty (but ``lufs`` is still computed).
    short = AudioTree.create(_tone(sr, 0.2), sr).replace_lufs()
    assert short.lufs.shape == (1,)
    assert short.lufs_windows.shape == (1, 0)
    with pytest.raises(ValueError):
        tree.replace_lufs(lufs_window_sec=0.2)


def test_replace_lufs_windows_tracks_per_window_loudness():
    """Per-window LUFS reflects a loud-then-quiet signal window by window."""
    sr = 44100
    loud = _tone(sr, 1.0, amplitude=0.5)
    quiet = _tone(sr, 1.0, amplitude=0.005)
    waveform = np.concatenate([loud, quiet])[None, None, :]
    tree = AudioTree.create(waveform, sr).replace_lufs()  # 2.0s -> 5 windows of 0.4s
    windows = np.asarray(tree.lufs_windows[0])
    # The first ~half of the windows are much louder than the last ~half.
    assert windows[0] > windows[-1] + 20.0


def test_replace_lufs_windows_hop_and_silence():
    """`lufs_hop_sec` overlaps windows; fully silent windows are -inf."""
    sr = 44100
    tone = _tone(sr, 2.0)
    # 0.4s windows stepping 0.2s over 2.0s -> (2.0 - 0.4) / 0.2 + 1 = 9 windows.
    overlapped = AudioTree.create(tone, sr).replace_lufs(
        lufs_window_sec=0.4, lufs_hop_sec=0.2
    )
    assert overlapped.lufs_windows.shape == (1, 9)
    # Non-overlapping (default hop) gives 2.0 / 0.4 = 5 windows.
    assert AudioTree.create(tone, sr).replace_lufs().lufs_windows.shape == (1, 5)
    with pytest.raises(ValueError):
        AudioTree.create(tone, sr).replace_lufs(lufs_hop_sec=0.0)

    # Ungated windows report -inf for digital silence (comparable across windows).
    silent = AudioTree.create(np.zeros(2 * sr, dtype=np.float32), sr).replace_lufs()
    assert np.all(np.isneginf(np.asarray(silent.lufs_windows[0])))


def test_replace_lufs_rejects_a_sub_sample_hop():
    """A positive hop that rounds to 0 samples raises up front.

    ``lufs_hop_sec=1e-5`` passed the positivity check but spans 0 samples at
    44.1 kHz, which used to surface as a bare ZeroDivisionError deep in the
    window count.
    """
    sr = 44100
    tree = AudioTree.create(_tone(sr, 2.0), sr)
    with pytest.raises(ValueError, match="lufs_hop_sec.*at least 1 sample"):
        tree.replace_lufs(lufs_hop_sec=1e-5)


def test_replace_lufs_rejects_more_than_five_channels_on_both_engines():
    """The documented 5-channel limit holds for the NumPy engine too.

    Only the JAX engine used to enforce it; the NumPy engine silently computed
    a value for 6+ channels.
    """
    sr = 44100
    tree = AudioTree(np.zeros((1, 6, sr), dtype=np.float32), sr)
    for engine in ("numpy", "jax"):
        with pytest.raises(ValueError, match="five channels"):
            tree.replace_lufs(engine=engine)


def test_replace_lufs_engine_forces_jax_kernel_but_keeps_array_type():
    """`engine="jax"` runs the FIR kernel yet returns NumPy loudness."""
    import jax.numpy as jnp

    sr = 44100
    tone = _tone(sr, 2.0)
    np_tree = AudioTree.create(tone, sr)
    jx_tree = AudioTree.create(jnp.asarray(tone), sr)

    # Default engine=None -> native NumPy/CPU kernel for a NumPy waveform.
    assert isinstance(np_tree.replace_lufs().lufs, np.ndarray)

    # engine="jax" forces the vmapped jaxloudnorm kernel even for a NumPy
    # waveform, but loudness comes back as NumPy so the tree stays on one
    # device -- no manual device_put/device_get needed.
    forced = np_tree.replace_lufs(device="cpu", engine="jax")
    assert isinstance(forced.lufs, np.ndarray)
    assert isinstance(forced.lufs_windows, np.ndarray)
    # It ran the JAX kernel, so it matches the JAX path (not the exact-IIR NumPy
    # path) to high precision.
    jax_native = jx_tree.replace_lufs()
    np.testing.assert_allclose(forced.lufs, np.asarray(jax_native.lufs), atol=1e-3)
    np.testing.assert_allclose(
        forced.lufs_windows, np.asarray(jax_native.lufs_windows), atol=1e-3
    )
    # ...and the device is optional: engine alone picks the kernel.
    np.testing.assert_allclose(
        np_tree.replace_lufs(engine="jax").lufs, forced.lufs, atol=1e-6
    )

    # normalize_lufs forwards device/engine to replace_lufs and keeps NumPy output.
    normalized = np_tree.normalize_lufs(-18.0, device="cpu", engine="jax")
    assert isinstance(normalized.lufs, np.ndarray)
    np.testing.assert_allclose(float(normalized.lufs[0]), -18.0, atol=1e-4)


def test_replace_lufs_device_and_engine_are_independent():
    """`device=` says where, `engine=` says which kernel -- neither implies the other.

    ``backend="cpu"`` used to conflate the two: it forced the FIR approximation
    even for a NumPy waveform already sitting on the CPU, so "the exact IIR
    meter, on this device" was unaskable.
    """
    import jax
    import jax.numpy as jnp

    sr = 44100
    tone = _tone(sr, 2.0)
    np_tree = AudioTree.create(tone, sr)
    jx_tree = AudioTree.create(jnp.asarray(tone), sr)

    exact = float(np_tree.replace_lufs().lufs[0])  # NumPy engine, exact IIR
    approx = float(np_tree.replace_lufs(engine="jax").lufs[0])  # FIR approximation
    assert exact != approx  # the two kernels really are different

    # device="cpu" alone keeps the waveform's own engine -- it is a placement
    # request, not a kernel request.
    assert float(np_tree.replace_lufs(device="cpu").lufs[0]) == exact
    np.testing.assert_allclose(
        float(jx_tree.replace_lufs(device="cpu").lufs[0]), approx, atol=1e-4
    )

    # The newly askable combination: the exact IIR meter on a JAX waveform.
    iir_on_jax = jx_tree.replace_lufs(engine="numpy")
    assert isinstance(iir_on_jax.lufs, jax.Array)  # output follows the waveform
    np.testing.assert_allclose(float(iir_on_jax.lufs[0]), exact, atol=1e-4)

    # A concrete jax.Device is accepted alongside the platform names.
    np.testing.assert_allclose(
        float(np_tree.replace_lufs(device=jax.devices("cpu")[0], engine="jax").lufs[0]),
        approx,
        atol=1e-6,
    )

    # Bad values name themselves, and the CPU-only engine refuses an accelerator.
    with pytest.raises(ValueError, match="device must be"):
        np_tree.replace_lufs(device="jax")
    with pytest.raises(ValueError, match="engine must be"):
        np_tree.replace_lufs(engine="cpu")
    with pytest.raises(TypeError, match="device must be"):
        np_tree.replace_lufs(device=3)
    from audiotree.core import _resolve_lufs_engine

    with pytest.raises(ValueError, match="CPU"):
        _resolve_lufs_engine("numpy", "gpu", input_is_numpy=True)
    # On a non-CPU device, engine=None picks the only kernel that can run there.
    assert _resolve_lufs_engine(None, "gpu", input_is_numpy=True) == "jax"
    assert _resolve_lufs_engine(None, None, input_is_numpy=True) == "numpy"
    assert _resolve_lufs_engine(None, None, input_is_numpy=False) == "jax"


def test_normalize_lufs_shifts_windows():
    """normalize_lufs retargets ``lufs`` and shifts ``lufs_windows`` by the same gain."""
    sr = 44100
    tree = AudioTree.create(_tone(sr, 2.0), sr).replace_lufs()
    before = np.asarray(tree.lufs_windows[0])
    gain = -18.0 - float(tree.lufs[0])
    normalized = tree.normalize_lufs(-18.0)
    np.testing.assert_allclose(float(normalized.lufs[0]), -18.0, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(normalized.lufs_windows[0]), before + gain, atol=1e-3
    )


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_normalize_lufs_silence_is_not_nan(backend):
    """Silence has no gain to a target; it is passed through, not turned into NaN."""
    import jax.numpy as jnp

    sr = 44100
    silence = np.zeros((1, 1, sr), dtype=np.float32)
    if backend == "jax":
        silence = jnp.asarray(silence)
    tree = AudioTree.create(silence, sr).replace_lufs()
    assert float(tree.lufs[0]) == -np.inf  # below the BS.1770 absolute gate

    out = tree.normalize_lufs(-18.0)
    assert not np.isnan(np.asarray(out.waveform)).any()
    assert np.all(np.asarray(out.waveform) == 0.0)  # left untouched
    assert float(out.lufs[0]) == -np.inf  # still identifiable as silent
    assert not np.isnan(np.asarray(out.lufs_windows)).any()


def test_normalize_lufs_mixed_batch_only_skips_the_silent_item():
    """A silent item must not poison the audible items batched alongside it."""
    sr = 44100
    waveform = np.stack([_tone(sr, 1.0)[None, :], np.zeros((1, sr), dtype=np.float32)])
    tree = AudioTree.create(waveform, sr).replace_lufs()

    out = tree.normalize_lufs(-18.0)
    assert not np.isnan(np.asarray(out.waveform)).any()
    np.testing.assert_allclose(float(out.lufs[0]), -18.0, atol=1e-4)
    assert float(out.lufs[1]) == -np.inf
    assert np.all(np.asarray(out.waveform[1]) == 0.0)


def test_normalize_lufs_max_gain_db_caps_amplification():
    """``max_gain_db`` bounds how far a quiet item is amplified."""
    sr = 44100
    tree = AudioTree.create(_tone(sr, 1.0, amplitude=1e-3), sr).replace_lufs()
    before = float(tree.lufs[0])
    assert np.isfinite(before) and before < -18.0  # quiet, but measurable

    uncapped = tree.normalize_lufs(-18.0)
    np.testing.assert_allclose(float(uncapped.lufs[0]), -18.0, atol=1e-4)

    capped = tree.normalize_lufs(-18.0, max_gain_db=6.0)
    np.testing.assert_allclose(float(capped.lufs[0]), before + 6.0, atol=1e-4)


def test_replace_lufs_windows_numpy_jax_agree():
    """NumPy and JAX backends produce close (not bit-identical) windowed LUFS."""
    import jax.numpy as jnp

    sr = 44100
    waveform = _tone(sr, 2.0)
    np_tree = AudioTree.create(waveform, sr).replace_lufs()
    jax_tree = AudioTree.create(jnp.asarray(waveform), sr).replace_lufs()
    assert np_tree.lufs_windows.shape == jax_tree.lufs_windows.shape == (1, 5)
    np.testing.assert_allclose(
        np.asarray(np_tree.lufs_windows), np.asarray(jax_tree.lufs_windows), atol=0.2
    )


# =============================================================================
# ExcerptConfig.search resolution
# =============================================================================


def test_search_function_resolution():
    """Names, legacy spellings, dotted paths and callables all resolve."""
    from audiotree.core import (
        ExcerptConfig,
        _resolve_search_function,
        search_bias_early,
        search_uniform,
    )

    assert ExcerptConfig().search == "uniform"
    assert _resolve_search_function("uniform") is search_uniform
    assert _resolve_search_function("bias_early") is search_bias_early
    # A callable passes through, and a dotted path is imported.
    assert _resolve_search_function(np.mean) is np.mean
    assert _resolve_search_function("numpy.mean") is np.mean


@pytest.mark.parametrize(
    "spec",
    [
        "uniforn",  # typo
        "no_such_module.f",
        "numpy.no_such_attribute",
        # The whole point: a config string is not evaluated as code.
        "(lambda *a, **k: __import__('sys').exit(1))",
    ],
)
def test_search_function_rejects_bad_specs(spec):
    """An unknown search_function raises when the params object is built."""
    from audiotree.core import ExcerptConfig

    with pytest.raises(ValueError):
        ExcerptConfig(search=spec)


def _write_half_silent_wav(path, sample_rate=16000, seconds=4.0):
    """A file whose first half is silent and second half is a loud tone."""
    import soundfile

    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio[: n // 2] = 0.0
    soundfile.write(str(path), audio, sample_rate)
    return path


def test_loudest_excerpt_finds_the_loud_half(tmp_path):
    """The saliency search returns an excerpt above the cutoff when one exists."""
    from audiotree.core import ExcerptConfig

    path = _write_half_silent_wav(tmp_path / "half.wav")
    params = ExcerptConfig(strategy="loudest", num_tries=32, lufs_cutoff=-40.0)
    tree = AudioTree.loudest_excerpt(
        str(path),
        rng=np.random.default_rng(0),
        excerpt=params,
        duration=0.5,
        sample_rate=16000,
    )
    assert float(tree.lufs[0]) > -40.0


def test_loudest_excerpt_terminates_on_fully_silent_audio(tmp_path):
    """A file that can never pass the cutoff must stop after num_tries, not hang."""
    import soundfile

    from audiotree.core import ExcerptConfig

    path = tmp_path / "silent.wav"
    soundfile.write(str(path), np.zeros(16000 * 2, dtype=np.float32), 16000)

    params = ExcerptConfig(strategy="loudest", num_tries=3, lufs_cutoff=-40.0)
    tree = AudioTree.loudest_excerpt(
        str(path),
        rng=np.random.default_rng(0),
        excerpt=params,
        duration=0.5,
        sample_rate=16000,
    )
    # Returns the best (still silent) excerpt rather than looping forever.
    assert float(tree.lufs[0]) == -np.inf


def test_loudest_excerpt_accepts_bias_early_by_name(tmp_path):
    """``search`` selects the searcher by registered name."""
    from audiotree.core import ExcerptConfig

    path = _write_half_silent_wav(tmp_path / "half2.wav")
    params = ExcerptConfig(
        strategy="loudest", num_tries=8, lufs_cutoff=-40.0, search="bias_early"
    )
    tree = AudioTree.loudest_excerpt(
        str(path),
        rng=np.random.default_rng(1),
        excerpt=params,
        duration=0.5,
        sample_rate=16000,
    )
    assert tree.waveform.shape == (1, 1, 8000)


def _tone_wav(path, sample_rate=16000, seconds=4.0):
    """A file of constant tone, long enough for several distinct excerpts."""
    import soundfile

    t = np.arange(int(sample_rate * seconds)) / sample_rate
    soundfile.write(
        str(path), (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sample_rate
    )
    return path


def test_excerpt_uses_one_searcher_signature(tmp_path):
    """Every shipped searcher is callable through ``excerpt``.

    ``excerpt`` used to call its searchers with four positional arguments while
    every searcher in the module (and everything ``_resolve_search_function``
    returns) also requires ``attempt``/``max_attempts``, so no shipped searcher
    could actually be used: ``search_bias_early`` raised TypeError and the name
    ``"bias_early"`` raised "'str' object is not callable".
    """
    from audiotree.core import ExcerptConfig, search_bias_early

    path = _tone_wav(tmp_path / "tone.wav")
    common = dict(duration=0.5, sample_rate=16000)

    for search in ("uniform", "bias_early", search_bias_early):
        tree = AudioTree.excerpt(
            str(path),
            np.random.default_rng(0),
            excerpt=ExcerptConfig(strategy="loudest", search=search),
            **common,
        )
        assert tree.waveform.shape == (1, 1, 8000)


def test_excerpt_strategies(tmp_path):
    """``start`` is deterministic, ``random`` moves, ``loudest`` measures."""
    from audiotree.core import ExcerptConfig

    path = _tone_wav(tmp_path / "tone2.wav")
    common = dict(duration=0.5, sample_rate=16000)

    def offset_of(tree):
        return float(tree.extras["offset"][0])

    starts = [
        offset_of(
            AudioTree.excerpt(
                str(path),
                np.random.default_rng(seed),
                excerpt=ExcerptConfig(strategy="start"),
                **common,
            )
        )
        for seed in range(3)
    ]
    assert starts == [0.0, 0.0, 0.0]

    randoms = [
        offset_of(AudioTree.excerpt(str(path), np.random.default_rng(seed), **common))
        for seed in range(3)
    ]
    assert len(set(randoms)) == 3
    assert all(0.0 <= o <= 3.5 for o in randoms)

    # ``offset`` is the earliest allowed start, not the chosen one.
    bounded = offset_of(
        AudioTree.excerpt(str(path), np.random.default_rng(0), offset=2.0, **common)
    )
    assert bounded >= 2.0

    # ``loudest`` goes through ``loudest_excerpt``, so it measures loudness.
    loud = AudioTree.excerpt(
        str(path),
        np.random.default_rng(0),
        excerpt=ExcerptConfig(strategy="loudest"),
        **common,
    )
    assert loud.lufs is not None and float(loud.lufs[0]) > -40.0


def test_excerpt_rejects_bad_arguments(tmp_path):
    """A missing duration, or an offset the strategy cannot honor, raises."""
    from audiotree.core import ExcerptConfig

    path = _tone_wav(tmp_path / "tone3.wav")
    # ``duration`` is a required parameter (it used to be an Optional that
    # unconditionally raised when omitted).
    with pytest.raises(TypeError, match="duration"):
        AudioTree.excerpt(str(path), np.random.default_rng(0), sample_rate=16000)
    with pytest.raises(ValueError, match="positive duration"):
        AudioTree.excerpt(
            str(path), np.random.default_rng(0), duration=0.0, sample_rate=16000
        )
    with pytest.raises(ValueError, match="offset"):
        AudioTree.excerpt(
            str(path),
            np.random.default_rng(0),
            offset=1.0,
            duration=0.5,
            sample_rate=16000,
            excerpt=ExcerptConfig(strategy="loudest"),
        )


def test_excerpt_can_skip(tmp_path):
    """``on_failure='skip'`` propagates the ``None`` through ``excerpt``."""
    import soundfile

    from audiotree.core import ExcerptConfig

    path = tmp_path / "silent.wav"
    soundfile.write(str(path), np.zeros(16000 * 2, dtype=np.float32), 16000)

    assert (
        AudioTree.excerpt(
            str(path),
            np.random.default_rng(0),
            duration=0.5,
            sample_rate=16000,
            excerpt=ExcerptConfig(strategy="loudest", num_tries=2, on_failure="skip"),
        )
        is None
    )


# =============================================================================
# Provenance broadcasting and typed ``replace``
# =============================================================================


@pytest.mark.parametrize("key", ["filepath", "source"])
def test_create_broadcasts_a_single_name_over_the_batch(key):
    """A single string tags every item, not just item 0.

    Tagging only item 0 meant ``tree[2].filepath == []``, and a filter that
    dropped item 0 lost the provenance entirely -- which then lands in written
    manifests.
    """
    waveform = np.zeros((4, 1, 8), dtype=np.float32)
    tree = AudioTree.create(waveform, 44100, **{key: "music.wav"})

    assert getattr(tree, key) == ["music.wav"] * 4
    assert tree.metadata[key].shape[0] == 4
    assert getattr(tree[2], key) == ["music.wav"]
    assert getattr(tree[1:], key) == ["music.wav"] * 3

    # A list still means one name per item, and a batch of one is unchanged.
    per_item = AudioTree.create(waveform, 44100, **{key: list("abcd")})
    assert getattr(per_item, key) == list("abcd")
    single = AudioTree.create(np.zeros((1, 1, 8)), 44100, **{key: "one.wav"})
    assert single.metadata[key].shape[0] == 1


def test_create_broadcasts_over_token_only_batches():
    """Token-only trees have a batch axis too, taken from ``codes``."""
    codes = np.zeros((3, 4, 2), dtype=np.int32)
    tree = AudioTree.create(None, 44100, codes=codes, filepath="tokens.wav")
    assert tree.filepath == ["tokens.wav"] * 3


def test_replace_is_typed_and_stays_in_sync():
    """``replace`` is the primary mutation API, so it must be visible to checkers.

    ``flax.struct.dataclass`` installs an untyped ``replace(**updates)``; the
    typed one is restored after the class body, and its ``Unpack``ed TypedDict
    has to keep listing exactly the dataclass fields.
    """
    import dataclasses
    import inspect
    import typing

    from audiotree.core import _AudioTreeFields

    assert tuple(typing.get_type_hints(_AudioTreeFields)) == tuple(
        f.name for f in dataclasses.fields(AudioTree)
    )

    signature = inspect.signature(AudioTree.replace)
    updates = signature.parameters["updates"]
    assert updates.kind is inspect.Parameter.VAR_KEYWORD
    assert "_AudioTreeFields" in str(updates.annotation)

    # ...and it still behaves like ``dataclasses.replace``.
    tree = AudioTree.create(np.zeros((2, 1, 8), dtype=np.float32), 44100)
    quieter = tree.replace(waveform=tree.waveform + 1.0)
    assert quieter.sample_rate == 44100
    np.testing.assert_array_equal(quieter.waveform, tree.waveform + 1.0)


# =============================================================================
# Derived-field invalidation
# =============================================================================


def _encoded_tree(sample_rate: int = 16000, channels: int = 2) -> AudioTree:
    """A tree carrying every derived field: loudness, codec tokens and latents."""
    tone = np.broadcast_to(_tone(sample_rate, 1.0), (2, channels, sample_rate))
    return AudioTree.create(
        np.ascontiguousarray(tone),
        sample_rate,
        codes=np.ones((2, 4, 50), dtype=np.int32),
        latents=np.ones((2, 8, 50), dtype=np.float32),
        extras={"codec_scale": np.ones((2, 1), dtype=np.float32)},
    ).replace_lufs()


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda tree: tree.resample(8000), id="resample"),
        pytest.param(lambda tree: tree.to_mono(), id="to_mono"),
        pytest.param(lambda tree: tree.to_mono("left"), id="to_mono_left"),
        pytest.param(lambda tree: tree.to_mono().to_stereo(), id="to_stereo"),
    ],
)
def test_length_rate_channel_changes_invalidate_every_derived_field(operation):
    """Codec tokens describe one waveform; a changed waveform must drop them.

    ``resample``/``to_mono``/``to_stereo`` used to clear only ``lufs`` and
    ``lufs_windows``, so ``codes``/``latents``/``extras["codec_scale"]``
    survived describing the *old* audio -- and ``encode_with_codec`` is
    idempotent, so a resample-after-encode happily reused them.
    """
    tree = _encoded_tree()
    assert tree.codes is not None and tree.lufs is not None

    out = operation(tree)
    assert out.lufs is None
    assert out.lufs_windows is None
    assert out.codes is None
    assert out.latents is None
    assert "codec_scale" not in out.extras


def test_normalize_lufs_keeps_loudness_but_drops_codec_fields():
    """A gain has a closed form for LUFS and none for codec tokens."""
    tree = _encoded_tree()
    out = tree.normalize_lufs(-18.0)

    # Loudness is shifted, not discarded...
    np.testing.assert_allclose(float(out.lufs[0]), -18.0, atol=1e-4)
    assert out.lufs_windows is not None
    # ...but the tokens described the audio at its old level.
    assert out.codes is None
    assert out.latents is None
    assert "codec_scale" not in out.extras


def test_no_op_conversions_keep_derived_fields():
    """Operations that do not touch the audio are exempt from invalidation."""
    mono = _encoded_tree(channels=1)
    stereo = _encoded_tree(channels=2)
    for unchanged in (
        mono.resample(mono.sample_rate),
        mono.to_mono(),
        mono.to_mono("left"),
        stereo.to_stereo(),
    ):
        assert unchanged.codes is not None
        assert unchanged.lufs is not None
        assert "codec_scale" in unchanged.extras

    # Re-batching the same items is not an audio change either.
    for unchanged in (
        stereo[0],
        stereo[1:],
        stereo.split(2)[0],
        stereo.filter(lambda item: True),
    ):
        assert unchanged.codes is not None
        assert unchanged.lufs is not None
        assert "codec_scale" in unchanged.extras


def test_invalidate_derived_rejects_unknown_keep():
    """``keep`` names derived fields only; a typo must not silently clear one."""
    tree = _encoded_tree()
    with pytest.raises(ValueError, match="keep must name derived fields"):
        tree._invalidate_derived(keep=("waveform",))

    # And it is the single source of truth for what "derived" means.
    from audiotree.core import DERIVED_FIELDS, DERIVED_EXTRAS_KEYS

    assert DERIVED_FIELDS == ("lufs", "lufs_windows", "codes", "latents")
    assert DERIVED_EXTRAS_KEYS == ("codec_scale",)


# =============================================================================
# AudioTree.from_manifest
# =============================================================================


def _write_manifest(directory, **writer_kwargs):
    """Write a small three-item dataset and return its manifest path."""
    from audiotree import AudioWriter

    rng = np.random.default_rng(0)
    tree = AudioTree.create(
        rng.uniform(-0.5, 0.5, (3, 1, 8000)).astype(np.float32),
        sample_rate=8000,
        lufs=np.array([-30.0, -18.0, -15.0], dtype=np.float32),
        filepath=["a.wav", "b.wav", "c.wav"],
        extras={"frame_id": np.array([10, 20, 30], dtype=np.int32)},
    )
    with AudioWriter(directory, **writer_kwargs) as writer:
        writer.write(tree, tags={"split": "train"})
    return Path(directory) / "manifest.npz"


def _retag_manifest(manifest_path, column, values):
    """Rewrite one column of an existing manifest, leaving the rest alone."""
    from audiotree import _manifest

    stored = _manifest.read_columns(manifest_path)
    columns = {name: list(array) for name, array in stored.columns.items()}
    columns[column] = list(values)
    _manifest.write(manifest_path, columns, stored.num_entries)


def test_from_manifest_refuses_filenames_outside_the_audio_dir(tmp_path):
    """A tampered manifest cannot make from_manifest read arbitrary files.

    ``pathlib`` drops the left operand of a join when the right side is
    absolute and never normalizes ``..``, so an unchecked ``audio_dir /
    filename`` hands back the contents of any readable file as ``waveform``.
    """
    dataset = tmp_path / "dataset"
    manifest_path = _write_manifest(dataset)

    secret = tmp_path / "secret.wav"
    AudioTree.create(np.ones((1, 1, 8000), dtype=np.float32), 8000).write(secret)

    for filename in ("../secret.wav", str(secret)):
        _retag_manifest(manifest_path, "filename", [filename] * 3)
        with pytest.raises(ValueError, match="Refusing to open|resolves outside"):
            AudioTree.from_manifest(manifest_path)


def test_from_manifest_and_audio_data_source_agree(tmp_path):
    """One manifest, one parser: both readers must produce the same tree."""
    from audiotree.sources import AudioDataSource

    manifest_path = _write_manifest(tmp_path / "dataset")

    combined = AudioTree.from_manifest(manifest_path)
    source = AudioDataSource(manifest_path)
    per_item = AudioTree.batch([source[i] for i in range(len(source))])

    assert combined.batch_size == per_item.batch_size == 3
    assert combined.sample_rate == per_item.sample_rate
    np.testing.assert_allclose(combined.waveform, per_item.waveform)
    np.testing.assert_allclose(combined.lufs, per_item.lufs)
    assert combined.filepath == per_item.filepath == ["a.wav", "b.wav", "c.wav"]
    # AudioDataSource routes each row through ``from_file``, which additionally
    # records the read ``offset``; every manifest-derived field must match.
    assert per_item.extras.keys() - combined.extras.keys() == {"offset"}
    assert combined.extras.keys() <= per_item.extras.keys()
    np.testing.assert_array_equal(
        combined.extras["frame_id"], per_item.extras["frame_id"]
    )


def test_from_manifest_filter_fn_gets_the_same_entries_as_audio_data_source(tmp_path):
    """``filter_fn`` is handed a manifest-entry dict, like AudioDataSource's."""
    from audiotree import _manifest
    from audiotree.sources import AudioDataSource

    manifest_path = _write_manifest(tmp_path / "dataset")
    expected_keys = [set(entry) for entry in _manifest.read_entries(manifest_path)]

    seen = []

    def predicate(entry):
        seen.append(entry)
        return entry["tags"]["split"] == "train" and entry.get("lufs", -np.inf) > -20

    loud = AudioTree.from_manifest(manifest_path, filter_fn=predicate)

    assert [type(entry) for entry in seen] == [dict] * 3
    assert [set(entry) for entry in seen] == expected_keys
    assert loud.batch_size == 2
    np.testing.assert_allclose(loud.lufs, [-18.0, -15.0])

    # The very same predicate must select the very same items in the other reader.
    source = AudioDataSource(manifest_path, filter_fn=predicate)
    assert [entry["filename"] for entry in source.get_all_entries()] == [
        str(seen[1]["filename"]),
        str(seen[2]["filename"]),
    ]


def test_from_manifest_rejects_a_partly_present_column(tmp_path):
    """A masked-out cell is not filler to load; it is a hole and must be named."""
    from audiotree import _manifest

    manifest_path = _write_manifest(tmp_path / "dataset", write_audio=False)
    _retag_manifest(manifest_path, "lufs", [-30.0, None, -15.0])
    assert "lufs" in _manifest.read_columns(manifest_path).present

    with pytest.raises(ValueError, match="value for 2 of the 3 selected entries"):
        AudioTree.from_manifest(manifest_path)

    # Filtering the hole away leaves a loadable manifest.
    kept = AudioTree.from_manifest(manifest_path, filter_fn=lambda e: "lufs" in e)
    np.testing.assert_allclose(kept.lufs, [-30.0, -15.0])


def test_from_manifest_rejects_a_pickled_manifest(tmp_path):
    """Reading a manifest never unpickles: that would be code execution."""
    manifest_path = _write_manifest(tmp_path / "dataset", write_audio=False)
    with np.load(manifest_path, allow_pickle=False) as npz:
        payload = dict(npz)
    # A valid header over a pickled column: exactly what a pre-1.0 manifest, or
    # a tampered one, looks like.
    payload["filename"] = np.array(list(payload["filename"]), dtype=object)
    np.savez(manifest_path, **payload)

    with pytest.raises(ValueError, match="pickled object arrays"):
        AudioTree.from_manifest(manifest_path)


def test_from_manifest_closes_the_npz_handle(tmp_path):
    """The manifest must not stay open: a leaked handle is a leaked descriptor."""
    manifest_path = _write_manifest(tmp_path / "dataset", write_audio=False)
    tree = AudioTree.from_manifest(manifest_path)
    assert tree.batch_size == 3
    # An open NpzFile keeps the archive mapped; deletion is the portable check.
    manifest_path.unlink()
    assert not manifest_path.exists()


# =============================================================================
# Public guarantees survive ``python -O``
# =============================================================================


def test_shape_guarantees_raise_instead_of_asserting():
    """These checks are exceptions, not asserts, so ``-O`` cannot delete them."""
    tree = AudioTree(np.zeros((5, 1, 32), dtype=np.float32), 44100)

    with pytest.raises(ValueError, match="divisible by the number of splits"):
        tree.split(2)
    with pytest.raises(ValueError, match="divisible by mini_batch_size"):
        tree.reshape_mini_batches(2)
    with pytest.raises(ValueError, match="at least 4 dimensions"):
        tree.flatten_mini_batches()

    # The rank checks are the same kind of public precondition.
    mini = AudioTree(np.zeros((4, 1, 32), dtype=np.float32), 44100)
    mini = mini.reshape_mini_batches(2)
    for call in (
        lambda: mini[0].write("/dev/null"),
        lambda: mini.reshape_mini_batches(1),
        lambda: mini.filter(lambda item: True),
    ):
        with pytest.raises(ValueError, match="rank 4"):
            call()


def test_public_checks_survive_python_O():
    """Run the same guarantees in a ``python -O`` subprocess.

    An ``assert`` is compiled out entirely under ``-O``, so the failure would be
    a silently truncated write or a bogus reshape rather than an exception.
    """
    import subprocess
    import sys

    script = """
import numpy as np
from audiotree import AudioTree

assert_removed = True
try:
    assert False
except AssertionError:
    assert_removed = False
if assert_removed is not True:
    raise SystemExit("-O did not strip asserts; the test proves nothing")

tree = AudioTree(np.zeros((5, 1, 32), dtype=np.float32), 44100)
mini = AudioTree(np.zeros((4, 1, 32), dtype=np.float32), 44100).reshape_mini_batches(2)
for call, needle in (
    (lambda: tree.split(2), "divisible"),
    (lambda: tree.reshape_mini_batches(2), "divisible"),
    (lambda: tree.flatten_mini_batches(), "4 dimensions"),
    (lambda: tree.write("/dev/null"), "batch_size == 1"),
    (lambda: mini[0].write("/dev/null"), "rank 4"),
    (lambda: mini.reshape_mini_batches(1), "rank 4"),
    (lambda: mini.filter(lambda item: True), "rank 4"),
    (lambda: AudioTree.create(np.zeros((2, 2, 1, 8)), 44100), "rank-4 waveform"),
    (lambda: AudioTree(None, 44100).write("/dev/null"), "needs a waveform"),
):
    try:
        call()
    except ValueError as exc:
        if needle not in str(exc):
            raise SystemExit(f"wrong message: {exc}")
    else:
        raise SystemExit(f"no exception for {needle}")
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


class TestBackendAndDevice:
    """`.backend` / `.device` answer "where is this tree?" without leaf-poking."""

    def _tree(self, xp):
        waveform = xp.zeros((2, 1, 800), dtype=xp.float32)
        return AudioTree.create(waveform, 16000).replace_lufs()

    def test_reports_the_array_library(self):
        assert self._tree(np).backend == "numpy"
        assert self._tree(jnp).backend == "jax"

    def test_device_is_none_on_numpy_and_a_device_on_jax(self):
        assert self._tree(np).device is None
        device = self._tree(jnp).device
        assert device is not None and device == jnp.zeros(1).device

    def test_round_trips_through_jax_device_put_and_get(self):
        numpy_tree = self._tree(np)
        moved = jax.device_put(numpy_tree)
        assert moved.backend == "jax" and moved.device is not None
        assert jax.device_get(moved).backend == "numpy"

    def test_mixed_tree_is_reported_not_raised(self):
        """A NumPy-namespace transform on a JAX tree converts what it touches."""
        tree = self._tree(jnp)
        mixed = tree.replace(lufs=np.asarray(tree.lufs))
        assert mixed.backend == "mixed"
        with pytest.raises(ValueError, match="more than one device"):
            _ = mixed.device

    def test_numpy_transform_on_a_jax_tree_is_visible(self):
        """The case the property exists for, end to end through grain."""
        tree = self._tree(jnp)
        assert tree.backend == "jax"
        dataset = (
            grain.MapDataset.source([tree]).seed(0).apply([transforms.trim(length=400)])
        )
        assert dataset[0].backend == "numpy"

    def test_waveformless_tree_does_not_crash(self):
        codes = np.zeros((2, 4, 10), dtype=np.int32)
        assert (
            AudioTree(waveform=None, sample_rate=16000, codes=codes).backend == "numpy"
        )
