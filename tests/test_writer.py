"""Tests for AudioWriter class."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile

from audiotree import AudioTree, AudioWriter
from audiotree.sources import ManifestDataSource


def test_basic_sequential_writing():
    """Test basic sequential writing of AudioTree batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test AudioTree with batch of 3
        audio_data = np.random.randn(3, 2, 44100)  # 3 batch, 2 channels, 1 second
        audio_tree = AudioTree.create(audio_data, sample_rate=44100)

        # Write with AudioWriter
        with AudioWriter(output_dir, pattern="test_{index:03d}.wav") as writer:
            paths = writer.write(audio_tree)

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

        writer = AudioWriter(output_dir, pattern="audio_{index:04d}.wav")

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
        audio_tree = AudioTree.create(np.random.randn(1, 1, 44100), sample_rate=44100)

        # Write with resampling to 16000 Hz
        writer = AudioWriter(output_dir, sample_rate=16000)
        paths = writer.write(audio_tree)

        # Check output sample rate
        data, sr = soundfile.read(paths[0])
        assert sr == 16000


def test_context_manager():
    """Test context manager behavior for automatic manifest saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Use context manager
        with AudioWriter(output_dir) as writer:
            audio_tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
            writer.write(audio_tree)

            # Manifest should not exist yet
            manifest_path = output_dir / "manifest.npz"
            assert not manifest_path.exists()

        # After exiting context, manifest should exist
        assert manifest_path.exists()


def test_manual_save_manifest():
    """Test manual manifest saving."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir)
        audio_tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
        writer.write(audio_tree)

        # Manually save manifest
        manifest_path = writer.save_manifest()
        assert manifest_path.exists()
        assert manifest_path.name == "manifest.npz"


def test_directory_creation():
    """Test automatic directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Specify nested directory that doesn't exist
        output_dir = Path(tmpdir) / "nested" / "output" / "dir"
        assert not output_dir.exists()

        writer = AudioWriter(output_dir)
        assert output_dir.exists()  # Should be created

        audio_tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
        paths = writer.write(audio_tree)
        assert paths[0].exists()


def test_mono_audio():
    """Test writing mono audio (single channel)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create mono audio
        audio_tree = AudioTree.create(np.random.randn(1, 1, 16000), sample_rate=16000)

        writer = AudioWriter(output_dir)
        paths = writer.write(audio_tree)

        # Check output
        data, sr = soundfile.read(paths[0])
        assert data.shape == (16000,)  # Mono is 1D array in soundfile


def test_stereo_audio():
    """Test writing stereo audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create stereo audio
        audio_tree = AudioTree.create(np.random.randn(1, 2, 16000), sample_rate=16000)

        writer = AudioWriter(output_dir)
        paths = writer.write(audio_tree)

        # Check output
        data, sr = soundfile.read(paths[0])
        assert data.shape == (16000, 2)  # Stereo is (samples, 2)


def test_npz_manifest():
    """Test NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        audio_data = np.random.randn(3, 1, 44100)
        audio_tree = AudioTree.create(
            audio_data,
            sample_rate=44100,
            loudness=np.array([-20.0, -15.0, -18.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            note_duration=np.array([1.0, 0.5, 0.75]),
            filepaths=["source1.wav", "source2.wav", "source3.wav"]
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree, tags={"dataset": "test", "version": 1})

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
        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-18.0, -22.0])
        )

        # Write with compressed NPZ manifest
        with AudioWriter(output_dir, compress_manifest=True) as writer:
            writer.write(audio_tree)

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
        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-18.0, -22.0])
        )

        # Write with uncompressed NPZ manifest
        with AudioWriter(output_dir, compress_manifest=False) as writer:
            writer.write(audio_tree)

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

        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050
        )

        # Write without timestamps
        with AudioWriter(output_dir, include_timestamp=False) as writer:
            writer.write(audio_tree)

        # Load manifest
        data = np.load(output_dir / "manifest.npz", allow_pickle=True)

        # Verify timestamp field is not present
        assert 'timestamp' not in data


def test_npz_manifest_with_timestamp():
    """Test NPZ manifest with timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050
        )

        # Write with timestamps
        with AudioWriter(output_dir, include_timestamp=True) as writer:
            writer.write(audio_tree)

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

        audio_tree = AudioTree.create(np.random.randn(3, 1, 8000), 8000)

        # Try to create with internal progress bar
        # This will work if tqdm is installed, otherwise silently continue
        with AudioWriter(
            output_dir,
            show_progress=True,
            progress_desc="Test progress",
        ) as writer:
            writer.write(audio_tree)

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
        audio_tree = AudioTree.create(np.random.randn(2, 1, 8000), 8000)

        # Write with close_pbar=False (default)
        with AudioWriter(output_dir, pbar=mock_pbar, close_pbar=False) as writer:
            writer.write(audio_tree)

        # Progress bar should be updated but not closed
        assert mock_pbar.total_updates == 2
        assert not mock_pbar.closed


def test_manifest_datasource_npz():
    """Test reading NPZ manifest with ManifestDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create and write AudioTree with metadata
        audio_data = np.random.randn(3, 2, 16000)
        audio_tree = AudioTree.create(
            audio_data,
            sample_rate=16000,
            loudness=np.array([-20.0, -15.0, -25.0]),
            pitch=np.array([60.0, 62.0, 58.0]),
            velocity=np.array([64, 80, 45]),
            filepaths=["orig1.wav", "orig2.wav", "orig3.wav"]
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            paths = writer.write(audio_tree, tags={"experiment": "test_npz"})

        # Read back with ManifestDataSource
        source = ManifestDataSource.from_writer_output(output_dir)

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
        audio_tree = AudioTree.create(
            np.random.randn(4, 1, 1000).astype(np.float32),
            sample_rate=44100,
            loudness=np.array([-20.0, -18.0, -22.0, -15.0], dtype=np.float32),
            pitch=np.array([60.0, 62.0, 64.0, 66.0], dtype=np.float32),
            velocity=np.array([64, 80, 100, 127], dtype=np.int16),
            note_duration=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir, compress_manifest=False) as writer:
            writer.write(audio_tree)

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
        audio_tree = AudioTree.create(
            np.random.randn(batch_size, 2, 1000).astype(np.float32),
            sample_rate=44100,
        )

        # Add metadata with arrays
        audio_tree = audio_tree.replace(metadata={
            "params": np.random.randn(batch_size, param_dim).astype(np.float32),
            "frame_indices": np.array([10, 20, 30], dtype=np.int32),
            "confidence": np.array([0.9, 0.85, 0.95], dtype=np.float32),
        })

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

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


def test_manifest_only_generation():
    """Test generating manifest without writing audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        audio_data = np.random.randn(3, 2, 22050)
        audio_tree = AudioTree.create(
            audio_data,
            sample_rate=22050,
            loudness=np.array([-20.0, -18.0, -22.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            filepaths=["original1.wav", "original2.wav", "original3.wav"]
        )

        # Write manifest only (no audio files)
        with AudioWriter(output_dir, write_audio=False) as writer:
            paths = writer.write(audio_tree, tags={"dataset": "test", "version": 1})

        # Check that audio files were NOT created
        for path in paths:
            assert not path.exists(), f"Audio file {path} should not exist when write_audio=False"

        # Check that manifest was created
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify manifest contents
        data = np.load(manifest_path, allow_pickle=True)

        # Check basic metadata
        assert len(data['index']) == 3
        assert len(data['filename']) == 3
        assert all(data['sample_rate'] == 22050)
        assert all(data['channels'] == 2)
        assert all(data['samples'] == 22050)

        # Check AudioTree metadata
        assert np.allclose(data['loudness'], [-20.0, -18.0, -22.0])
        assert np.allclose(data['pitch'], [60.0, 62.0, 64.0])
        assert np.allclose(data['velocity'], [64, 80, 100])

        # Check files_written flag
        assert 'files_written' in data
        assert all(data['files_written'] == False)

        # Check stats
        stats = writer.get_stats()
        assert stats['write_audio'] == False
        assert stats['total_files'] == 0  # No files written to disk


def test_manifest_only_with_write_audio_true():
    """Test that files_written flag is True when write_audio=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_tree = AudioTree.create(np.random.randn(2, 1, 8000), sample_rate=8000)

        # Write with audio files (default behavior)
        with AudioWriter(output_dir, write_audio=True) as writer:
            paths = writer.write(audio_tree)

        # Check that audio files WERE created
        for path in paths:
            assert path.exists(), f"Audio file {path} should exist when write_audio=True"

        # Check files_written flag in manifest
        data = np.load(output_dir / "manifest.npz", allow_pickle=True)
        assert 'files_written' in data
        assert all(data['files_written'] == True)

        # Check stats
        stats = writer.get_stats()
        assert stats['write_audio'] == True
        assert stats['total_files'] == 2  # Files written to disk


def test_manifest_only_multiple_writes():
    """Test manifest-only generation with multiple write calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir, write_audio=False)

        # Write first batch
        tree1 = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-18.0, -20.0])
        )
        paths1 = writer.write(tree1)

        # Write second batch
        tree2 = AudioTree.create(
            np.random.randn(3, 1, 22050),
            sample_rate=22050,
            loudness=np.array([-15.0, -25.0, -19.0])
        )
        paths2 = writer.write(tree2)

        # No files should exist
        for path in paths1 + paths2:
            assert not path.exists()

        # Save manifest and check
        manifest_path = writer.save_manifest()
        assert manifest_path.exists()

        data = np.load(manifest_path, allow_pickle=True)
        assert len(data['index']) == 5
        assert all(data['files_written'] == False)
        assert np.allclose(data['loudness'], [-18.0, -20.0, -15.0, -25.0, -19.0])

        # Check stats
        stats = writer.get_stats()
        assert stats['total_files'] == 0
        assert stats['current_index'] == 5


def test_field_validation():
    """Test that AudioWriter validates field consistency across writes."""
    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First audio_tree has loudness and pitch
        tree1 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-20.0, -18.0]),
            pitch=np.array([60.0, 62.0])
        )

        # Second audio_tree only has loudness (missing pitch)
        tree2 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-15.0, -22.0])
        )

        # Third audio_tree has loudness, pitch, and velocity (extra field)
        tree3 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-19.0, -21.0]),
            pitch=np.array([64.0, 66.0]),
            velocity=np.array([80, 90])
        )

        writer = AudioWriter(output_dir)

        # First write should succeed
        writer.write(tree1)

        # Second write should fail (missing pitch)
        with pytest.raises(ValueError, match="missing fields"):
            writer.write(tree2)

        # Third write should fail (extra velocity)
        with pytest.raises(ValueError, match="extra fields"):
            writer.write(tree3)


def test_metadata_different_batch_sizes():
    """Test that metadata arrays work correctly across writes with different batch sizes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First write: batch size 2 with metadata params [2, 10]
        tree1 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-20.0, -18.0])
        )
        tree1 = tree1.replace(metadata={
            "params": np.random.randn(2, 10).astype(np.float32),
            "frame_id": np.array([100, 200], dtype=np.int32)
        })

        # Second write: batch size 3 with metadata params [3, 10]
        tree2 = AudioTree.create(
            np.random.randn(3, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-15.0, -22.0, -19.0])
        )
        tree2 = tree2.replace(metadata={
            "params": np.random.randn(3, 10).astype(np.float32),
            "frame_id": np.array([300, 400, 500], dtype=np.int32)
        })

        # Third write: batch size 1 with metadata params [1, 10]
        tree3 = AudioTree.create(
            np.random.randn(1, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-17.0])
        )
        tree3 = tree3.replace(metadata={
            "params": np.random.randn(1, 10).astype(np.float32),
            "frame_id": np.array([600], dtype=np.int32)
        })

        # Write all trees
        with AudioWriter(output_dir) as writer:
            writer.write(tree1)
            writer.write(tree2)
            writer.write(tree3)

        # Load manifest and verify
        manifest_data = np.load(output_dir / "manifest.npz", allow_pickle=True)

        # Check that metadata was stacked correctly
        assert 'metadata_params' in manifest_data
        assert 'metadata_frame_id' in manifest_data

        # Should have 2 + 3 + 1 = 6 entries total
        assert manifest_data['metadata_params'].shape == (6, 10)
        assert manifest_data['metadata_frame_id'].shape == (6,)

        # Verify dtypes preserved
        assert manifest_data['metadata_params'].dtype == np.float32
        assert manifest_data['metadata_frame_id'].dtype == np.int32

        # Verify frame_ids are correct
        np.testing.assert_array_equal(
            manifest_data['metadata_frame_id'],
            [100, 200, 300, 400, 500, 600]
        )

        # Test reading back with ManifestDataSource
        source = ManifestDataSource.from_writer_output(output_dir)
        assert len(source) == 6

        # Check all items have correct metadata shapes and values
        expected_frame_ids = [100, 200, 300, 400, 500, 600]

        for idx, expected_frame_id in enumerate(expected_frame_ids):
            loaded_tree = source[idx]

            # Verify metadata exists and has correct shape
            assert 'params' in loaded_tree.metadata
            assert loaded_tree.metadata['params'].shape == (1, 10), f"Item {idx}: params shape mismatch"
            assert loaded_tree.metadata['params'].dtype == np.float32, f"Item {idx}: params dtype mismatch"

            assert 'frame_id' in loaded_tree.metadata
            assert loaded_tree.metadata['frame_id'].shape == (1,), f"Item {idx}: frame_id shape mismatch"
            assert loaded_tree.metadata['frame_id'].dtype == np.int32, f"Item {idx}: frame_id dtype mismatch"

            # Verify frame_id value matches
            assert loaded_tree.metadata['frame_id'][0] == expected_frame_id, f"Item {idx}: frame_id value mismatch"

            # Verify AudioTree field shapes
            assert loaded_tree.audio_data.shape == (1, 1, 8000), f"Item {idx}: audio_data shape mismatch"
            assert loaded_tree.loudness is not None, f"Item {idx}: loudness should not be None"
            assert loaded_tree.loudness.shape == (1,), f"Item {idx}: loudness shape mismatch"


def test_audiotree_from_manifest():
    """Test AudioTree.from_manifest loads all items into a single AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple trees with different batch sizes
        tree1 = AudioTree.create(
            np.random.randn(2, 2, 8000),
            sample_rate=8000,
            loudness=np.array([-20.0, -18.0]),
            pitch=np.array([60.0, 62.0]),
        )
        tree1 = tree1.replace(metadata={
            "params": np.random.randn(2, 10).astype(np.float32),
            "frame_id": np.array([100, 200], dtype=np.int32)
        })

        tree2 = AudioTree.create(
            np.random.randn(3, 2, 8000),
            sample_rate=8000,
            loudness=np.array([-15.0, -22.0, -19.0]),
            pitch=np.array([64.0, 66.0, 68.0]),
        )
        tree2 = tree2.replace(metadata={
            "params": np.random.randn(3, 10).astype(np.float32),
            "frame_id": np.array([300, 400, 500], dtype=np.int32)
        })

        # Write to manifest
        with AudioWriter(output_dir) as writer:
            writer.write(tree1)
            writer.write(tree2)

        # Load all at once into single AudioTree
        combined = AudioTree.from_manifest(output_dir / "manifest.npz")

        # Should have 2 + 3 = 5 items in batch dimension
        assert combined.audio_data.shape == (5, 2, 8000)
        assert combined.sample_rate == 8000

        # AudioTree fields should be concatenated
        assert combined.loudness.shape == (5,)
        assert np.allclose(combined.loudness, [-20.0, -18.0, -15.0, -22.0, -19.0])
        assert combined.pitch.shape == (5,)
        assert np.allclose(combined.pitch, [60.0, 62.0, 64.0, 66.0, 68.0])

        # Metadata arrays should be concatenated
        assert combined.metadata['params'].shape == (5, 10)
        assert combined.metadata['frame_id'].shape == (5,)
        np.testing.assert_array_equal(
            combined.metadata['frame_id'],
            [100, 200, 300, 400, 500]
        )


def test_audiotree_from_manifest_without_audio_files():
    """Test AudioTree.from_manifest with manifest-only (no audio files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        audio_tree = AudioTree.create(
            np.random.randn(3, 2, 16000),
            sample_rate=16000,
            loudness=np.array([-20.0, -15.0, -25.0]),
            velocity=np.array([64, 80, 45], dtype=np.int16),
        )

        # Write manifest only
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(audio_tree)

        # Load from manifest without audio files
        combined = AudioTree.from_manifest(output_dir / "manifest.npz")

        # Should have correct shape with zero audio data
        assert combined.audio_data.shape == (3, 2, 16000)
        assert np.all(combined.audio_data == 0.0)

        # Metadata should be preserved
        assert combined.loudness.shape == (3,)
        assert np.allclose(combined.loudness, [-20.0, -15.0, -25.0])
        assert combined.velocity.shape == (3,)
        np.testing.assert_array_equal(combined.velocity, [64, 80, 45])


def test_audiotree_from_manifest_with_filter():
    """Test AudioTree.from_manifest with filter function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with varying loudness
        audio_tree = AudioTree.create(
            np.random.randn(5, 1, 8000),
            sample_rate=8000,
            loudness=np.array([-30.0, -18.0, -25.0, -15.0, -22.0]),
        )

        # Write to manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load only items louder than -20 LUFS
        filtered = AudioTree.from_manifest(
            output_dir / "manifest.npz",
            filter_fn=lambda entry: entry.get('loudness', -float('inf')) > -20.0
        )

        # Should only have 2 items: -18.0 and -15.0
        assert filtered.audio_data.shape == (2, 1, 8000)
        assert filtered.loudness.shape == (2,)
        assert np.allclose(filtered.loudness, [-18.0, -15.0])


def test_manifest_datasource_without_audio_files():
    """Test that ManifestDataSource works with manifest-only (no audio files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        audio_data = np.random.randn(3, 2, 16000)
        audio_tree = AudioTree.create(
            audio_data,
            sample_rate=16000,
            loudness=np.array([-20.0, -15.0, -25.0]),
            pitch=np.array([60.0, 62.0, 58.0]),
            velocity=np.array([64, 80, 45], dtype=np.int16),
        )
        audio_tree = audio_tree.replace(metadata={
            "params": np.random.randn(3, 10).astype(np.float32),
            "frame_id": np.array([100, 200, 300], dtype=np.int32)
        })

        # Write manifest only (no audio files)
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(audio_tree, tags={"experiment": "no_audio"})

        # Verify no audio files exist
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 0, "No WAV files should exist"

        # Load with ManifestDataSource - should work without audio files
        source = ManifestDataSource.from_writer_output(output_dir)
        assert len(source) == 3

        # Check loaded items
        for idx in range(3):
            loaded_tree = source[idx]

            # Should have correct sample rate
            assert loaded_tree.sample_rate == 16000

            # Audio data should be zeros with correct shape
            assert loaded_tree.audio_data.shape == (1, 2, 16000)
            assert np.all(loaded_tree.audio_data == 0.0)

            # Metadata should be preserved
            assert loaded_tree.loudness is not None
            assert loaded_tree.pitch is not None
            assert loaded_tree.velocity is not None

            # Check metadata arrays
            assert 'params' in loaded_tree.metadata
            assert loaded_tree.metadata['params'].shape == (1, 10)
            assert 'frame_id' in loaded_tree.metadata
            assert loaded_tree.metadata['frame_id'].shape == (1,)

        # Verify specific values for first item
        first_tree = source[0]
        assert np.allclose(first_tree.loudness, [-20.0])
        assert np.allclose(first_tree.pitch, [60.0])
        assert np.allclose(first_tree.velocity, [64])
        assert first_tree.metadata['frame_id'][0] == 100


def test_write_empty_audiotree():
    """Test writing an empty AudioTree (batch_size=0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        empty_tree = AudioTree.create(
            np.zeros((0, 2, 44100)),
            sample_rate=44100,
            loudness=np.array([])
        )

        with AudioWriter(output_dir) as writer:
            paths = writer.write(empty_tree)

        assert len(paths) == 0
        manifest_path = output_dir / "manifest.npz"
        assert not manifest_path.exists()


def test_write_empty_audiotree_first():
    """Test writing an empty AudioTree first, then non-empty trees."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with AudioWriter(output_dir) as writer:
            empty_tree = AudioTree.create(
                np.zeros((0, 1, 8000)),
                sample_rate=8000,
                loudness=np.array([])
            )
            paths1 = writer.write(empty_tree)
            assert len(paths1) == 0

            nonempty_tree = AudioTree.create(
                np.random.randn(2, 1, 8000),
                sample_rate=8000,
                loudness=np.array([-20.0, -18.0])
            )
            paths2 = writer.write(nonempty_tree)
            assert len(paths2) == 2

        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        data = np.load(manifest_path, allow_pickle=True)
        assert len(data['index']) == 2
        assert np.allclose(data['loudness'], [-20.0, -18.0])


def test_write_empty_audiotree_after_nonempty():
    """Test writing non-empty trees first, then an empty audio_tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with AudioWriter(output_dir) as writer:
            nonempty_tree = AudioTree.create(
                np.random.randn(2, 1, 8000),
                sample_rate=8000,
                loudness=np.array([-20.0, -18.0])
            )
            paths1 = writer.write(nonempty_tree)
            assert len(paths1) == 2

            empty_tree = AudioTree.create(
                np.zeros((0, 1, 8000)),
                sample_rate=8000,
                loudness=np.array([])
            )
            paths2 = writer.write(empty_tree)
            assert len(paths2) == 0

        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        data = np.load(manifest_path, allow_pickle=True)
        assert len(data['index']) == 2
        assert np.allclose(data['loudness'], [-20.0, -18.0])


def test_write_filtered_empty_audiotree():
    """Test filtering an AudioTree to empty, then writing the result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_data = np.random.randn(3, 2, 44100)
        audio_tree = AudioTree.create(
            audio_data,
            sample_rate=44100,
            loudness=np.array([-20.0, -15.0, -18.0])
        )

        filtered_tree = audio_tree.filter(lambda x: False)

        assert filtered_tree.audio_data.shape[0] == 0
        assert filtered_tree.loudness.shape[0] == 0

        with AudioWriter(output_dir) as writer:
            paths = writer.write(filtered_tree)

        assert len(paths) == 0
        manifest_path = output_dir / "manifest.npz"
        assert not manifest_path.exists()


def test_write_empty_with_metadata_arrays():
    """Test writing empty AudioTree with metadata arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        empty_tree = AudioTree.create(
            np.zeros((0, 2, 44100)),
            sample_rate=44100,
            loudness=np.array([])
        )
        empty_tree = empty_tree.replace(metadata={
            "params": np.zeros((0, 10), dtype=np.float32),
            "frame_id": np.array([], dtype=np.int32)
        })

        assert empty_tree.audio_data.shape == (0, 2, 44100)
        assert empty_tree.metadata['params'].shape == (0, 10)
        assert empty_tree.metadata['frame_id'].shape == (0,)

        with AudioWriter(output_dir) as writer:
            paths = writer.write(empty_tree)

        assert len(paths) == 0
        manifest_path = output_dir / "manifest.npz"
        assert not manifest_path.exists()


def test_filter_with_partial_match():
    """Test filtering AudioTree where some items match."""
    audio_data = np.random.randn(5, 1, 8000)
    audio_tree = AudioTree.create(
        audio_data,
        sample_rate=8000,
        loudness=np.array([-30.0, -18.0, -25.0, -15.0, -22.0])
    )
    audio_tree = audio_tree.replace(metadata={
        "params": np.random.randn(5, 10).astype(np.float32)
    })

    filtered = audio_tree.filter(lambda x: x.loudness[0] > -20.0)

    assert filtered.audio_data.shape[0] == 2
    assert filtered.loudness.shape[0] == 2
    assert np.allclose(filtered.loudness, [-18.0, -15.0])
    assert filtered.metadata['params'].shape == (2, 10)


def test_filter_with_no_match():
    """Test filtering AudioTree where no items match."""
    audio_data = np.random.randn(3, 1, 8000)
    audio_tree = AudioTree.create(
        audio_data,
        sample_rate=8000,
        loudness=np.array([-20.0, -18.0, -22.0])
    )
    audio_tree = audio_tree.replace(metadata={
        "params": np.random.randn(3, 10).astype(np.float32)
    })

    filtered = audio_tree.filter(lambda x: x.loudness[0] > 0.0)

    assert filtered.audio_data.shape[0] == 0
    assert filtered.loudness.shape[0] == 0
    assert filtered.metadata['params'].shape == (0, 10)


def test_filter_with_all_match():
    """Test filtering AudioTree where all items match."""
    audio_data = np.random.randn(3, 1, 8000)
    audio_tree = AudioTree.create(
        audio_data,
        sample_rate=8000,
        loudness=np.array([-20.0, -18.0, -22.0])
    )

    filtered = audio_tree.filter(lambda x: x.loudness[0] < 0.0)

    assert filtered.audio_data.shape[0] == 3
    assert np.allclose(filtered.loudness, [-20.0, -18.0, -22.0])


if __name__ == "__main__":
    # Run tests
    test_basic_sequential_writing()
    test_multiple_writes()
    test_resampling()
    test_context_manager()
    test_manual_save_manifest()
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
    test_dtype_preservation()
    test_metadata_array_preservation()
    test_manifest_only_generation()
    test_manifest_only_with_write_audio_true()
    test_manifest_only_multiple_writes()
    test_field_validation()
    test_metadata_different_batch_sizes()
    test_audiotree_from_manifest()
    test_audiotree_from_manifest_without_audio_files()
    test_audiotree_from_manifest_with_filter()
    test_manifest_datasource_without_audio_files()
    test_write_empty_audiotree()
    test_write_empty_audiotree_first()
    test_write_empty_audiotree_after_nonempty()
    test_write_filtered_empty_audiotree()
    test_write_empty_with_metadata_arrays()
    test_filter_with_partial_match()
    test_filter_with_no_match()
    test_filter_with_all_match()
    print("All tests passed!")