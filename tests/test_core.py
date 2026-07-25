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


def test_audiotree_create_with_source():
    """AudioTree.create accepts a source group name (single or per-batch-item)."""
    # Single string tags the whole tree; read back via the .source property.
    audio_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree1 = AudioTree.create(audio_1d, 44100, source="music")
    assert tree1.source == ["music"]
    assert "source" in tree1.metadata

    # A list gives one source name per batch item (unlike from_file).
    audio_batch = np.zeros((3, 1, 4))  # (batch=3, channels, samples)
    tree2 = AudioTree.create(audio_batch, 44100, source=["drums", "vocal", "impulse"])
    assert tree2.source == ["drums", "vocal", "impulse"]

    # No source -> no metadata key, empty list (unchanged behavior).
    tree3 = AudioTree.create(audio_1d, 44100)
    assert tree3.source == []
    assert "source" not in tree3.metadata


def test_filepath_too_long_raises():
    """Filepaths longer than the metadata limit raise instead of truncating."""
    from audiotree.core import _str_max_length

    audio = np.zeros((1, 1, 5))

    # A path exactly at the limit round-trips intact.
    exact = "a" * _str_max_length
    tree = AudioTree.create(audio, 44100, filepaths=exact)
    assert tree.filepath == [exact]

    # One character over the limit raises rather than silently truncating.
    too_long = "a" * (_str_max_length + 1)
    with pytest.raises(ValueError, match="exceeds the metadata encoding limit"):
        AudioTree.create(audio, 44100, filepaths=too_long)


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


def test_replace_metadata():
    """replace_metadata merges kwargs into metadata without mutating the original."""
    tree = AudioTree.create(
        np.zeros((2, 1, 100)),
        44100,
        metadata={"energy": np.array([0.8, 0.9]), "tag": np.array([1, 2])},
    )

    tagged = tree.replace_metadata(
        onsets=np.array([[0.1], [0.2]]), tag=np.array([3, 4])
    )

    # New key added, colliding key overwritten, other keys preserved
    np.testing.assert_array_equal(tagged.metadata["onsets"], [[0.1], [0.2]])
    np.testing.assert_array_equal(tagged.metadata["tag"], [3, 4])
    np.testing.assert_array_equal(tagged.metadata["energy"], [0.8, 0.9])

    # Everything else carries over untouched
    assert tagged.sample_rate == tree.sample_rate
    np.testing.assert_array_equal(tagged.waveform, tree.waveform)

    # The original tree and its metadata dict are not mutated
    assert set(tree.metadata) == {"energy", "tag"}
    np.testing.assert_array_equal(tree.metadata["tag"], [1, 2])

    # No kwargs is a no-op copy
    same = tree.replace_metadata()
    assert set(same.metadata) == {"energy", "tag"}


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
    assert tree.replace_lufs(window_duration_sec=1.0).lufs_windows.shape == (1, 2)
    # Shorter than one 0.4s window -> empty (but ``lufs`` is still computed).
    short = AudioTree.create(_tone(sr, 0.2), sr).replace_lufs()
    assert short.lufs.shape == (1,)
    assert short.lufs_windows.shape == (1, 0)
    with pytest.raises(ValueError):
        tree.replace_lufs(window_duration_sec=0.2)


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
    """`hop_duration_sec` overlaps windows; fully silent windows are -inf."""
    sr = 44100
    tone = _tone(sr, 2.0)
    # 0.4s windows stepping 0.2s over 2.0s -> (2.0 - 0.4) / 0.2 + 1 = 9 windows.
    overlapped = AudioTree.create(tone, sr).replace_lufs(
        window_duration_sec=0.4, hop_duration_sec=0.2
    )
    assert overlapped.lufs_windows.shape == (1, 9)
    # Non-overlapping (default hop) gives 2.0 / 0.4 = 5 windows.
    assert AudioTree.create(tone, sr).replace_lufs().lufs_windows.shape == (1, 5)
    with pytest.raises(ValueError):
        AudioTree.create(tone, sr).replace_lufs(hop_duration_sec=0.0)

    # Ungated windows report -inf for digital silence (comparable across windows).
    silent = AudioTree.create(np.zeros(2 * sr, dtype=np.float32), sr).replace_lufs()
    assert np.all(np.isneginf(np.asarray(silent.lufs_windows[0])))


def test_replace_lufs_backend_forces_jax_kernel_but_keeps_array_type():
    """`backend="cpu"` runs the JAX kernel on XLA CPU yet returns NumPy loudness."""
    import jax.numpy as jnp

    sr = 44100
    tone = _tone(sr, 2.0)
    np_tree = AudioTree.create(tone, sr)
    jx_tree = AudioTree.create(jnp.asarray(tone), sr)

    # Default backend=None -> native NumPy/CPU kernel for a NumPy waveform.
    assert isinstance(np_tree.replace_lufs().lufs, np.ndarray)

    # backend="cpu" forces the vmapped jaxloudnorm kernel (on XLA CPU) even for a
    # NumPy waveform, but loudness comes back as NumPy so the tree stays on one
    # device -- no manual device_put/device_get needed.
    forced = np_tree.replace_lufs(backend="cpu")
    assert isinstance(forced.lufs, np.ndarray)
    assert isinstance(forced.lufs_windows, np.ndarray)
    # It ran the JAX kernel, so it matches the JAX path (not the exact-IIR NumPy
    # path) to high precision.
    jax_native = jx_tree.replace_lufs()
    np.testing.assert_allclose(forced.lufs, np.asarray(jax_native.lufs), atol=1e-3)
    np.testing.assert_allclose(
        forced.lufs_windows, np.asarray(jax_native.lufs_windows), atol=1e-3
    )

    with pytest.raises(ValueError):
        np_tree.replace_lufs(backend="jax")

    # normalize_lufs forwards backend to replace_lufs and keeps NumPy output.
    normalized = np_tree.normalize_lufs(-18.0, backend="cpu")
    assert isinstance(normalized.lufs, np.ndarray)
    np.testing.assert_allclose(float(normalized.lufs[0]), -18.0, atol=1e-4)


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
