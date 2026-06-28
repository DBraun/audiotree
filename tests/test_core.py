"""Tests for AudioTree core functionality."""

from pathlib import Path

import jax
import numpy as np
import pytest

from audiotree.core import AudioTree


def test_audiotree_create_with_filepaths():
    """Test that AudioTree.create accepts and processes filepaths parameter."""
    # Test with 1D audio data and single filepath string
    audio_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree1 = AudioTree.create(audio_1d, 44100, filepaths="test1.wav")

    assert tree1.waveform.shape == (
        1,
        1,
        5,
    )  # Should be expanded to (batch, channels, samples)
    assert tree1.filepath == ["test1.wav"]
    assert "filepath" in tree1.metadata

    # Test with 2D audio data and single filepath Path
    audio_2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2 channels, 3 samples
    tree2 = AudioTree.create(audio_2d, 44100, filepaths=Path("test2.wav"))

    assert tree2.waveform.shape == (
        1,
        2,
        3,
    )  # Should be expanded to (batch, channels, samples)
    assert tree2.filepath == ["test2.wav"]

    # Test with 3D audio data and list of filepaths
    audio_3d = np.array([[[1.0, 2.0, 3.0]]])  # Already correct shape: (1, 1, 3)
    filepaths = ["file1.wav", "file2.wav", Path("file3.wav")]
    tree3 = AudioTree.create(audio_3d, 44100, filepaths=filepaths)

    assert tree3.waveform.shape == (1, 1, 3)  # Should remain unchanged
    assert tree3.filepath == ["file1.wav", "file2.wav", "file3.wav"]

    # Test with no filepaths (should work as before)
    tree4 = AudioTree.create(audio_1d, 44100)

    assert tree4.waveform.shape == (1, 1, 5)
    assert tree4.filepath == []  # Empty list when no filepaths provided
    assert "filepath" not in tree4.metadata


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


def test_write_options_and_batch_assertion(tmp_path):
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

    # A multi-item batch must be indexed/iterated first.
    batch = AudioTree(waveform=np.zeros((3, 1, sr), dtype=np.float32), sample_rate=sr)
    with pytest.raises(AssertionError, match="batch_size == 1"):
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

    # replace_loudness matches the flat computation, reshaped
    flat_loudness = tree.replace_loudness().loudness
    mini_loudness = mini.replace_loudness().loudness
    assert mini_loudness.shape == (2, 2)
    np.testing.assert_allclose(mini_loudness, flat_loudness.reshape(2, 2), rtol=1e-5)

    # normalize_loudness
    normalized = mini.normalize_loudness(-18.0)
    assert normalized.waveform.shape == mini.waveform.shape
    np.testing.assert_allclose(normalized.replace_loudness().loudness, -18.0, atol=0.5)

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


def test_audiotree_create_metadata_handling():
    """Test that AudioTree.create handles metadata correctly with filepaths."""
    waveform = np.array([1.0, 2.0, 3.0])
    sample_rate = 44100

    # Test with existing metadata and filepaths
    existing_metadata = {"custom_key": "custom_value"}
    tree1 = AudioTree.create(
        waveform, sample_rate, metadata=existing_metadata, filepaths="test.wav"
    )

    # Should preserve existing metadata and add filepath
    assert tree1.metadata["custom_key"] == "custom_value"
    assert "filepath" in tree1.metadata
    assert tree1.filepath == ["test.wav"]

    # Test that original metadata dict is not modified
    assert "filepath" not in existing_metadata

    # Test with filepaths but no existing metadata
    tree2 = AudioTree.create(waveform, sample_rate, filepaths="test2.wav")
    assert "filepath" in tree2.metadata
    assert tree2.filepath == ["test2.wav"]


def test_split_by_batch():
    x = AudioTree(np.zeros((4, 1, 44100)), 44100)
    trees = [x, x, x]
    big_tree = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *trees)
    assert big_tree.waveform.shape == (12, 1, 44100)
    split_trees = big_tree.split(2)
    assert len(split_trees) == 2
    assert split_trees[0].waveform.shape == (6, 1, 44100)


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

    # Test that unsplitting preserves metadata if present
    audio_tree_with_metadata = AudioTree(
        original_audio_data, sample_rate, metadata={"test_key": "test_value"}
    )
    batched_with_metadata = audio_tree_with_metadata.reshape_mini_batches(
        mini_batch_size
    )
    unbatched_with_metadata = batched_with_metadata.flatten_mini_batches()

    assert unbatched_with_metadata.metadata == {"test_key": "test_value"}

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
