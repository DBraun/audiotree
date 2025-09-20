"""Tests for AudioWriter class."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile

from audiotree import AudioTree, AudioWriter
from audiotree.datasources import ManifestDataSource


def test_basic_sequential_writing():
    """Test basic sequential writing of AudioTree batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test AudioTree with batch of 3
        audio_data = np.random.randn(3, 2, 44100)  # 3 batch, 2 channels, 1 second
        tree = AudioTree.create(audio_data, sample_rate=44100)

        # Write with AudioWriter
        with AudioWriter(output_dir, pattern="test_{index:03d}.wav", manifest_format=None) as writer:
            paths = writer.write(tree)

        # Check files were created
        assert len(paths) == 3
        for i, path in enumerate(paths):
            assert path.exists()
            assert path.name == f"test_{i:03d}.wav"

            # Verify audio content
            data, sr = soundfile.read(path)
            assert sr == 44100
            assert data.shape == (44100, 2)  # (samples, channels)


def test_multiple_writes():
    """Test writing multiple AudioTree objects sequentially."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir, pattern="audio_{index:04d}.wav", manifest_format=None)

        # Write first batch
        tree1 = AudioTree.create(np.random.randn(2, 1, 22050), sample_rate=22050)
        paths1 = writer.write(tree1)

        # Write second batch
        tree2 = AudioTree.create(np.random.randn(3, 1, 22050), sample_rate=22050)
        paths2 = writer.write(tree2)

        # Check continuous indexing
        assert len(paths1) == 2
        assert len(paths2) == 3
        assert paths1[0].name == "audio_0000.wav"
        assert paths1[1].name == "audio_0001.wav"
        assert paths2[0].name == "audio_0002.wav"
        assert paths2[2].name == "audio_0004.wav"

        # Check stats
        stats = writer.get_stats()
        assert stats['total_files'] == 5
        assert stats['current_index'] == 5


def test_resampling():
    """Test automatic resampling when target sample rate is specified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree at 44100 Hz
        tree = AudioTree.create(np.random.randn(1, 1, 44100), sample_rate=44100)

        # Write with resampling to 16000 Hz
        writer = AudioWriter(output_dir, sample_rate=16000, manifest_format=None)
        paths = writer.write(tree)

        # Check output sample rate
        data, sr = soundfile.read(paths[0])
        assert sr == 16000




def test_context_manager():
    """Test context manager behavior for automatic manifest saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Use context manager
        with AudioWriter(output_dir, manifest_format="npz") as writer:
            tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
            writer.write(tree)

            # Manifest should not exist yet
            manifest_path = output_dir / "manifest.npz"
            assert not manifest_path.exists()

        # After exiting context, manifest should exist
        assert manifest_path.exists()


def test_manual_save_manifest():
    """Test manual manifest saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir, manifest_format="npz")
        tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
        writer.write(tree)

        # Manually save manifest
        manifest_path = writer.save_manifest()
        assert manifest_path.exists()
        assert manifest_path.name == "manifest.npz"


def test_no_manifest():
    """Test writer with manifest disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with AudioWriter(output_dir, manifest_format=None) as writer:
            tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
            writer.write(tree)

        # No manifest should be created
        assert not (output_dir / "manifest.npz").exists()

        # save_manifest should return None
        assert writer.save_manifest() is None


def test_directory_creation():
    """Test automatic directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Specify nested directory that doesn't exist
        output_dir = Path(tmpdir) / "nested" / "output" / "dir"
        assert not output_dir.exists()

        writer = AudioWriter(output_dir, manifest_format=None)
        assert output_dir.exists()  # Should be created

        tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
        paths = writer.write(tree)
        assert paths[0].exists()


def test_mono_audio():
    """Test writing mono audio (single channel)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create mono audio
        tree = AudioTree.create(np.random.randn(1, 1, 16000), sample_rate=16000)

        writer = AudioWriter(output_dir, manifest_format=None)
        paths = writer.write(tree)

        # Check output
        data, sr = soundfile.read(paths[0])
        assert data.shape == (16000,)  # Mono is 1D array in soundfile


def test_stereo_audio():
    """Test writing stereo audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create stereo audio
        tree = AudioTree.create(np.random.randn(1, 2, 16000), sample_rate=16000)

        writer = AudioWriter(output_dir, manifest_format=None)
        paths = writer.write(tree)

        # Check output
        data, sr = soundfile.read(paths[0])
        assert data.shape == (16000, 2)  # Stereo is (samples, 2)


def test_npz_manifest():
    """Test NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        audio_data = np.random.randn(3, 1, 44100)
        tree = AudioTree.create(
            audio_data,
            sample_rate=44100,
            loudness=np.array([-20.0, -15.0, -18.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            note_duration=np.array([1.0, 0.5, 0.75]),
            filepaths=["source1.wav", "source2.wav", "source3.wav"]
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz") as writer:
            writer.write(tree, tags={"dataset": "test", "version": 1})

        # Check manifest file
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify NPZ contents
        data = np.load(manifest_path, allow_pickle=True)

        # Check arrays
        assert len(data['index']) == 3
        assert len(data['filename']) == 3
        assert all(data['sample_rate'] == 44100)
        assert all(data['channels'] == 1)
        assert all(data['samples'] == 44100)

        # Check AudioTree metadata
        assert np.allclose(data['loudness'], [-20.0, -15.0, -18.0])
        assert np.allclose(data['pitch'], [60.0, 62.0, 64.0])
        assert np.allclose(data['velocity'], [64, 80, 100])
        assert np.allclose(data['note_duration'], [1.0, 0.5, 0.75])

        # Check filepaths
        assert list(data['filepath']) == ["source1.wav", "source2.wav", "source3.wav"]

        # Check tags
        assert all(data['tags_dataset'] == 'test')
        assert all(data['tags_version'] == 1)


def test_npz_manifest_compressed():
    """Test compressed NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-18.0, -22.0])
        )

        # Write with compressed NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz", compress_manifest=True) as writer:
            writer.write(tree)

        # Check NPZ file exists
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify
        data = np.load(manifest_path, allow_pickle=True)
        assert len(data['index']) == 2
        assert np.allclose(data['loudness'], [-18.0, -22.0])


def test_npz_manifest_uncompressed():
    """Test uncompressed NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-18.0, -22.0])
        )

        # Write with uncompressed NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz", compress_manifest=False) as writer:
            writer.write(tree)

        # Check NPZ file exists
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify
        data = np.load(manifest_path, allow_pickle=True)
        assert len(data['index']) == 2
        assert np.allclose(data['loudness'], [-18.0, -22.0])


def test_npz_manifest_no_timestamp():
    """Test NPZ manifest without timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050
        )

        # Write without timestamps
        with AudioWriter(output_dir, manifest_format="npz", include_timestamp=False) as writer:
            writer.write(tree)

        # Load manifest
        data = np.load(output_dir / "manifest.npz", allow_pickle=True)

        # Verify timestamp field is not present
        assert 'timestamp' not in data


def test_npz_manifest_with_timestamp():
    """Test NPZ manifest with timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050
        )

        # Write with timestamps
        with AudioWriter(output_dir, manifest_format="npz", include_timestamp=True) as writer:
            writer.write(tree)

        # Load manifest
        data = np.load(output_dir / "manifest.npz", allow_pickle=True)

        # Verify timestamp field is present
        assert 'timestamp' in data
        assert len(data['timestamp']) == 2
        assert all(isinstance(ts, (str, np.str_)) for ts in data['timestamp'])


def test_external_progress_bar():
    """Test AudioWriter with external tqdm progress bar."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock progress bar to track updates
        class MockProgressBar:
            def __init__(self):
                self.total_updates = 0
                self.closed = False

            def update(self, n):
                self.total_updates += n

            def close(self):
                self.closed = True

        # Create mock progress bar
        mock_pbar = MockProgressBar()

        # Create AudioTree with 3 batches
        tree1 = AudioTree.create(np.random.randn(3, 1, 8000), 8000)
        tree2 = AudioTree.create(np.random.randn(2, 1, 8000), 8000)

        # Write with external progress bar
        with AudioWriter(output_dir, pbar=mock_pbar, close_pbar=True) as writer:
            writer.write(tree1)
            writer.write(tree2)

        # Check progress bar was updated correctly
        assert mock_pbar.total_updates == 5  # 3 + 2 files
        assert mock_pbar.closed  # Should be closed when close_pbar=True


def test_internal_progress_bar():
    """Test AudioWriter with internal progress bar creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        tree = AudioTree.create(np.random.randn(3, 1, 8000), 8000)

        # Try to create with internal progress bar
        # This will work if tqdm is installed, otherwise silently continue
        with AudioWriter(
            output_dir,
            show_progress=True,
            progress_desc="Test progress",
            manifest_format=None
        ) as writer:
            writer.write(tree)

        # Should have written files regardless of tqdm availability
        assert len(list(output_dir.glob("*.wav"))) == 3


def test_progress_bar_no_close():
    """Test that progress bar is not closed when close_pbar=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock progress bar
        class MockProgressBar:
            def __init__(self):
                self.total_updates = 0
                self.closed = False

            def update(self, n):
                self.total_updates += n

            def close(self):
                self.closed = True

        mock_pbar = MockProgressBar()
        tree = AudioTree.create(np.random.randn(2, 1, 8000), 8000)

        # Write with close_pbar=False (default)
        with AudioWriter(output_dir, pbar=mock_pbar, close_pbar=False) as writer:
            writer.write(tree)

        # Progress bar should be updated but not closed
        assert mock_pbar.total_updates == 2
        assert not mock_pbar.closed


def test_manifest_datasource_npz():
    """Test reading NPZ manifest with ManifestDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create and write AudioTree with metadata
        audio_data = np.random.randn(3, 2, 16000)
        tree = AudioTree.create(
            audio_data,
            sample_rate=16000,
            loudness=np.array([-20.0, -15.0, -25.0]),
            pitch=np.array([60.0, 62.0, 58.0]),
            velocity=np.array([64, 80, 45]),
            filepaths=["orig1.wav", "orig2.wav", "orig3.wav"]
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz") as writer:
            paths = writer.write(tree, tags={"experiment": "test_npz"})

        # Read back with ManifestDataSource
        source = ManifestDataSource.from_writer_output(output_dir, manifest_format="npz")

        assert len(source) == 3

        # Check first item
        loaded_tree = source[0]
        assert loaded_tree.sample_rate == 16000
        assert loaded_tree.audio_data.shape == (1, 2, 16000)
        assert np.allclose(loaded_tree.loudness, [-20.0])
        assert np.allclose(loaded_tree.pitch, [60.0])
        assert np.allclose(loaded_tree.velocity, [64])
        # Check metadata via get_entry since metadata was simplified for batching
        entry = source.get_entry(0)
        assert entry['filepath'] == "orig1.wav"
        assert entry['tags']['experiment'] == "test_npz"


def test_dtype_preservation():
    """Test that AudioWriter preserves correct dtypes in NPZ manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with specific dtypes
        tree = AudioTree.create(
            np.random.randn(4, 1, 1000).astype(np.float32),
            sample_rate=44100,
            loudness=np.array([-20.0, -18.0, -22.0, -15.0], dtype=np.float32),
            pitch=np.array([60.0, 62.0, 64.0, 66.0], dtype=np.float32),
            velocity=np.array([64, 80, 100, 127], dtype=np.int16),
            note_duration=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz", compress_manifest=False) as writer:
            writer.write(tree)

        # Load the NPZ manifest directly
        manifest_data = np.load(output_dir / "manifest.npz")

        # Check dtypes are preserved correctly
        assert manifest_data['index'].dtype == np.int32, f"index dtype is {manifest_data['index'].dtype}"
        assert manifest_data['sample_rate'].dtype == np.int32, f"sample_rate dtype is {manifest_data['sample_rate'].dtype}"
        assert manifest_data['channels'].dtype == np.int32, f"channels dtype is {manifest_data['channels'].dtype}"
        assert manifest_data['samples'].dtype == np.int32, f"samples dtype is {manifest_data['samples'].dtype}"

        assert manifest_data['loudness'].dtype == np.float32, f"loudness dtype is {manifest_data['loudness'].dtype}"
        assert manifest_data['pitch'].dtype == np.float32, f"pitch dtype is {manifest_data['pitch'].dtype}"
        assert manifest_data['note_duration'].dtype == np.float32, f"note_duration dtype is {manifest_data['note_duration'].dtype}"

        # Most importantly, velocity should be int16
        assert manifest_data['velocity'].dtype == np.int16, f"velocity dtype is {manifest_data['velocity'].dtype}"

        # Verify values are correct
        np.testing.assert_array_equal(manifest_data['velocity'], [64, 80, 100, 127])
        np.testing.assert_array_almost_equal(manifest_data['loudness'], [-20.0, -18.0, -22.0, -15.0])


def test_metadata_array_preservation():
    """Test that AudioWriter preserves metadata arrays in NPZ manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with metadata arrays
        batch_size = 3
        param_dim = 185

        # Create AudioTree with metadata containing arrays
        tree = AudioTree.create(
            np.random.randn(batch_size, 2, 1000).astype(np.float32),
            sample_rate=44100,
        )

        # Add metadata with arrays
        tree = tree.replace(metadata={
            "params": np.random.randn(batch_size, param_dim).astype(np.float32),
            "frame_indices": np.array([10, 20, 30], dtype=np.int32),
            "confidence": np.array([0.9, 0.85, 0.95], dtype=np.float32),
        })

        # Write with NPZ manifest
        with AudioWriter(output_dir, manifest_format="npz") as writer:
            writer.write(tree)

        # Load the NPZ manifest directly
        manifest_data = np.load(output_dir / "manifest.npz")

        # Check metadata fields were saved
        assert 'metadata_params' in manifest_data
        assert 'metadata_frame_indices' in manifest_data
        assert 'metadata_confidence' in manifest_data

        # Check shapes - params should be [batch_size, param_dim]
        assert manifest_data['metadata_params'].shape == (batch_size, param_dim)
        assert manifest_data['metadata_params'].dtype == np.float32

        # Check other metadata arrays
        assert manifest_data['metadata_frame_indices'].shape == (batch_size,)
        assert manifest_data['metadata_frame_indices'].dtype == np.int32
        np.testing.assert_array_equal(manifest_data['metadata_frame_indices'], [10, 20, 30])

        assert manifest_data['metadata_confidence'].shape == (batch_size,)
        assert manifest_data['metadata_confidence'].dtype == np.float32
        np.testing.assert_array_almost_equal(manifest_data['metadata_confidence'], [0.9, 0.85, 0.95])


if __name__ == "__main__":
    # Run tests
    test_basic_sequential_writing()
    test_multiple_writes()
    test_resampling()
    test_context_manager()
    test_manual_save_manifest()
    test_no_manifest()
    test_directory_creation()
    test_mono_audio()
    test_stereo_audio()
    test_npz_manifest()
    test_npz_manifest_compressed()
    test_npz_manifest_uncompressed()
    test_npz_manifest_no_timestamp()
    test_npz_manifest_with_timestamp()
    test_external_progress_bar()
    test_internal_progress_bar()
    test_progress_bar_no_close()
    test_manifest_datasource_npz()
    print("All tests passed!")