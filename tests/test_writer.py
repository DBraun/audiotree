"""Tests for AudioWriter class."""

import json
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import soundfile

from audiotree import AudioTree, AudioWriter, _format, _manifest
from audiotree.sources import AudioDataSource


def load_manifest(path):
    """Read an NPZ manifest into a plain dict, closing the archive.

    A bare ``np.load`` on an NPZ returns a lazy ``NpzFile`` that keeps the zip
    open. POSIX lets you unlink an open file, so the leak is invisible there;
    Windows refuses, and ``TemporaryDirectory`` cleanup fails with WinError 32.

    ``allow_pickle=False`` deliberately: no manifest audiotree writes may contain
    an object array, so every test that reads one through this helper also
    asserts that invariant. A regression to ``dtype=object`` columns would fail
    the suite here rather than being read back happily.
    """
    with np.load(path, allow_pickle=False) as npz:
        return dict(npz)


def test_lufs_windows_round_trips_through_manifest():
    """The per-window ``lufs_windows`` field persists to the NPZ manifest and back."""
    sr = 44100
    t = np.arange(2 * sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    waveform = np.stack([tone[None, :], (0.5 * tone)[None, :]], axis=0)  # (2, 1, T)
    batch = AudioTree.create(waveform, sample_rate=sr).replace_lufs()
    assert batch.lufs_windows.shape == (2, 5)  # 2.0s / 0.4s window

    with tempfile.TemporaryDirectory() as tmpdir:
        with AudioWriter(tmpdir) as writer:
            writer.write(batch)
        loaded = AudioTree.from_manifest(f"{tmpdir}/manifest.npz")

    assert loaded.lufs_windows.shape == (2, 5)
    np.testing.assert_allclose(
        np.asarray(loaded.lufs_windows), np.asarray(batch.lufs_windows), atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(loaded.lufs), np.asarray(batch.lufs), atol=1e-4
    )


def test_basic_sequential_writing():
    """Test basic sequential writing of AudioTree batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test AudioTree with batch of 3
        waveform = np.random.randn(3, 2, 44100)  # 3 batch, 2 channels, 1 second
        audio_tree = AudioTree.create(waveform, sample_rate=44100)

        # Write with AudioWriter
        with AudioWriter(output_dir, pattern="test_{index:03d}.wav") as writer:
            paths = writer.write(audio_tree)

        # Check files were created
        assert len(paths) == 3
        for i, path in enumerate(paths):
            assert path.exists()
            assert path.name == f"test_{i:03d}.wav"

            # Verify audio content
            data, sr = soundfile.read(str(path))
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
        assert stats["total_files"] == 5
        assert stats["current_index"] == 5


def test_sample_rate_inferred_and_enforced():
    """Sample rate is taken from the first write; later mismatches raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir)

        # The first write sets the writer's sample rate.
        first = AudioTree.create(np.random.randn(1, 1, 44100), sample_rate=44100)
        paths = writer.write(first)
        assert writer.sample_rate == 44100
        _, sr = soundfile.read(str(paths[0]))
        assert sr == 44100

        # A matching sample rate is accepted.
        writer.write(AudioTree.create(np.random.randn(1, 1, 44100), sample_rate=44100))

        # A different sample rate is rejected.
        mismatched = AudioTree.create(np.random.randn(1, 1, 16000), sample_rate=16000)
        with pytest.raises(ValueError, match="sample_rate"):
            writer.write(mismatched)


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
        data, sr = soundfile.read(str(paths[0]))
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
        data, sr = soundfile.read(str(paths[0]))
        assert data.shape == (16000, 2)  # Stereo is (samples, 2)


def test_npz_manifest():
    """Test NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        waveform = np.random.randn(3, 1, 44100)
        audio_tree = AudioTree.create(
            waveform,
            sample_rate=44100,
            lufs=np.array([-20.0, -15.0, -18.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            note_duration=np.array([1.0, 0.5, 0.75]),
            filepath=["source1.wav", "source2.wav", "source3.wav"],
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree, tags={"dataset": "test", "version": 1})

        # Check manifest file
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify NPZ contents
        data = load_manifest(manifest_path)

        # Check arrays
        assert len(data["index"]) == 3
        assert len(data["filename"]) == 3
        assert all(data["sample_rate"] == 44100)
        assert all(data["channels"] == 1)
        assert all(data["samples"] == 44100)

        # Check AudioTree metadata
        assert np.allclose(data["lufs"], [-20.0, -15.0, -18.0])
        assert np.allclose(data["pitch"], [60.0, 62.0, 64.0])
        assert np.allclose(data["velocity"], [64, 80, 100])
        assert np.allclose(data["note_duration"], [1.0, 0.5, 0.75])

        # Check filepaths
        assert list(data["filepath"]) == ["source1.wav", "source2.wav", "source3.wav"]

        # Check tags
        assert all(data["tags_dataset"] == "test")
        assert all(data["tags_version"] == 1)


def test_npz_manifest_compressed():
    """Test compressed NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            lufs=np.array([-18.0, -22.0]),
        )

        # Write with compressed NPZ manifest
        with AudioWriter(output_dir, compress_manifest=True) as writer:
            writer.write(audio_tree)

        # Check NPZ file exists
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify
        data = load_manifest(manifest_path)
        assert len(data["index"]) == 2
        assert np.allclose(data["lufs"], [-18.0, -22.0])


def test_npz_manifest_uncompressed():
    """Test uncompressed NPZ manifest generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            lufs=np.array([-18.0, -22.0]),
        )

        # Write with uncompressed NPZ manifest
        with AudioWriter(output_dir, compress_manifest=False) as writer:
            writer.write(audio_tree)

        # Check NPZ file exists
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify
        data = load_manifest(manifest_path)
        assert len(data["index"]) == 2
        assert np.allclose(data["lufs"], [-18.0, -22.0])


def test_npz_manifest_no_timestamp():
    """Test NPZ manifest without timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_tree = AudioTree.create(np.random.randn(2, 1, 22050), sample_rate=22050)

        # Write without timestamps
        with AudioWriter(output_dir, include_timestamp=False) as writer:
            writer.write(audio_tree)

        # Load manifest
        data = load_manifest(output_dir / "manifest.npz")

        # Verify timestamp field is not present
        assert "timestamp" not in data


def test_npz_manifest_with_timestamp():
    """Test NPZ manifest with timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_tree = AudioTree.create(np.random.randn(2, 1, 22050), sample_rate=22050)

        # Write with timestamps
        with AudioWriter(output_dir, include_timestamp=True) as writer:
            writer.write(audio_tree)

        # Load manifest
        data = load_manifest(output_dir / "manifest.npz")

        # Verify timestamp field is present
        assert "timestamp" in data
        assert len(data["timestamp"]) == 2
        assert all(isinstance(ts, (str, np.str_)) for ts in data["timestamp"])


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

        # ``show_progress=True`` now raises without tqdm rather than silently
        # continuing, so this exercises the installed-extra path only.
        pytest.importorskip("tqdm", reason="needs audiotree[progress]")

        with AudioWriter(
            output_dir,
            show_progress=True,
            progress_desc="Test progress",
        ) as writer:
            writer.write(audio_tree)

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
    """Test reading NPZ manifest with AudioDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create and write AudioTree with metadata
        waveform = np.random.randn(3, 2, 16000)
        audio_tree = AudioTree.create(
            waveform,
            sample_rate=16000,
            lufs=np.array([-20.0, -15.0, -25.0]),
            pitch=np.array([60.0, 62.0, 58.0]),
            velocity=np.array([64, 80, 45]),
            filepath=["orig1.wav", "orig2.wav", "orig3.wav"],
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree, tags={"experiment": "test_npz"})

        # Read back with AudioDataSource
        source = AudioDataSource.from_writer_output(output_dir)

        assert len(source) == 3

        # Check first item
        loaded_tree = source[0]
        assert loaded_tree.sample_rate == 16000
        assert loaded_tree.waveform.shape == (1, 2, 16000)
        assert np.allclose(loaded_tree.lufs, [-20.0])
        assert np.allclose(loaded_tree.pitch, [60.0])
        assert np.allclose(loaded_tree.velocity, [64])
        # Check metadata via get_entry since metadata was simplified for batching
        entry = source.get_entry(0)
        assert entry["filepath"] == "orig1.wav"
        assert entry["tags"]["experiment"] == "test_npz"


def test_dtype_preservation():
    """Test that AudioWriter preserves correct dtypes in NPZ manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with specific dtypes
        audio_tree = AudioTree.create(
            np.random.randn(4, 1, 1000).astype(np.float32),
            sample_rate=44100,
            lufs=np.array([-20.0, -18.0, -22.0, -15.0], dtype=np.float32),
            pitch=np.array([60.0, 62.0, 64.0, 66.0], dtype=np.float32),
            velocity=np.array([64, 80, 100, 127], dtype=np.int16),
            note_duration=np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir, compress_manifest=False) as writer:
            writer.write(audio_tree)

        # Load the NPZ manifest directly
        manifest_data = load_manifest(output_dir / "manifest.npz")

        # Check dtypes are preserved correctly
        assert manifest_data["index"].dtype == np.int32, (
            f"index dtype is {manifest_data['index'].dtype}"
        )
        assert manifest_data["sample_rate"].dtype == np.int32, (
            f"sample_rate dtype is {manifest_data['sample_rate'].dtype}"
        )
        assert manifest_data["channels"].dtype == np.int32, (
            f"channels dtype is {manifest_data['channels'].dtype}"
        )
        assert manifest_data["samples"].dtype == np.int32, (
            f"samples dtype is {manifest_data['samples'].dtype}"
        )

        assert manifest_data["lufs"].dtype == np.float32, (
            f"loudness dtype is {manifest_data['lufs'].dtype}"
        )
        assert manifest_data["pitch"].dtype == np.float32, (
            f"pitch dtype is {manifest_data['pitch'].dtype}"
        )
        assert manifest_data["note_duration"].dtype == np.float32, (
            f"note_duration dtype is {manifest_data['note_duration'].dtype}"
        )

        # Most importantly, velocity should be int16
        assert manifest_data["velocity"].dtype == np.int16, (
            f"velocity dtype is {manifest_data['velocity'].dtype}"
        )

        # Verify values are correct
        np.testing.assert_array_equal(manifest_data["velocity"], [64, 80, 100, 127])
        np.testing.assert_array_almost_equal(
            manifest_data["lufs"], [-20.0, -18.0, -22.0, -15.0]
        )


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
        audio_tree = audio_tree.replace(
            metadata={
                "params": np.random.randn(batch_size, param_dim).astype(np.float32),
                "frame_indices": np.array([10, 20, 30], dtype=np.int32),
                "confidence": np.array([0.9, 0.85, 0.95], dtype=np.float32),
            }
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load the NPZ manifest directly
        manifest_data = load_manifest(output_dir / "manifest.npz")

        # Check metadata fields were saved
        assert "metadata_params" in manifest_data
        assert "metadata_frame_indices" in manifest_data
        assert "metadata_confidence" in manifest_data

        # Check shapes - params should be [batch_size, param_dim]
        assert manifest_data["metadata_params"].shape == (batch_size, param_dim)
        assert manifest_data["metadata_params"].dtype == np.float32

        # Check other metadata arrays
        assert manifest_data["metadata_frame_indices"].shape == (batch_size,)
        assert manifest_data["metadata_frame_indices"].dtype == np.int32
        np.testing.assert_array_equal(
            manifest_data["metadata_frame_indices"], [10, 20, 30]
        )

        assert manifest_data["metadata_confidence"].shape == (batch_size,)
        assert manifest_data["metadata_confidence"].dtype == np.float32
        np.testing.assert_array_almost_equal(
            manifest_data["metadata_confidence"], [0.9, 0.85, 0.95]
        )


def test_manifest_only_generation():
    """Test generating manifest without writing audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        waveform = np.random.randn(3, 2, 22050)
        audio_tree = AudioTree.create(
            waveform,
            sample_rate=22050,
            lufs=np.array([-20.0, -18.0, -22.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            filepath=["original1.wav", "original2.wav", "original3.wav"],
        )

        # Write manifest only (no audio files)
        with AudioWriter(output_dir, write_audio=False) as writer:
            paths = writer.write(audio_tree, tags={"dataset": "test", "version": 1})

        # Check that audio files were NOT created
        for path in paths:
            assert not path.exists(), (
                f"Audio file {path} should not exist when write_audio=False"
            )

        # Check that manifest was created
        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        # Load and verify manifest contents
        data = load_manifest(manifest_path)

        # Check basic metadata
        assert len(data["index"]) == 3
        assert len(data["filename"]) == 3
        assert all(data["sample_rate"] == 22050)
        assert all(data["channels"] == 2)
        assert all(data["samples"] == 22050)

        # Check AudioTree metadata
        assert np.allclose(data["lufs"], [-20.0, -18.0, -22.0])
        assert np.allclose(data["pitch"], [60.0, 62.0, 64.0])
        assert np.allclose(data["velocity"], [64, 80, 100])

        # Check files_written flag
        assert "files_written" in data
        assert not data["files_written"].any()

        # Check stats
        stats = writer.get_stats()
        assert not stats["write_audio"]
        assert stats["total_files"] == 0  # No files written to disk


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
            assert path.exists(), (
                f"Audio file {path} should exist when write_audio=True"
            )

        # Check files_written flag in manifest
        data = load_manifest(output_dir / "manifest.npz")
        assert "files_written" in data
        assert data["files_written"].all()

        # Check stats
        stats = writer.get_stats()
        assert stats["write_audio"]
        assert stats["total_files"] == 2  # Files written to disk


def test_manifest_only_multiple_writes():
    """Test manifest-only generation with multiple write calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        writer = AudioWriter(output_dir, write_audio=False)

        # Write first batch
        tree1 = AudioTree.create(
            np.random.randn(2, 1, 22050),
            sample_rate=22050,
            lufs=np.array([-18.0, -20.0]),
        )
        paths1 = writer.write(tree1)

        # Write second batch
        tree2 = AudioTree.create(
            np.random.randn(3, 1, 22050),
            sample_rate=22050,
            lufs=np.array([-15.0, -25.0, -19.0]),
        )
        paths2 = writer.write(tree2)

        # No files should exist
        for path in paths1 + paths2:
            assert not path.exists()

        # Save manifest and check
        manifest_path = writer.save_manifest()
        assert manifest_path.exists()

        data = load_manifest(manifest_path)
        assert len(data["index"]) == 5
        assert not data["files_written"].any()
        assert np.allclose(data["lufs"], [-18.0, -20.0, -15.0, -25.0, -19.0])

        # Check stats
        stats = writer.get_stats()
        assert stats["total_files"] == 0
        assert stats["current_index"] == 5


def test_field_validation():
    """Test that AudioWriter validates field consistency across writes."""
    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # First audio_tree has loudness and pitch
        tree1 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            lufs=np.array([-20.0, -18.0]),
            pitch=np.array([60.0, 62.0]),
        )

        # Second audio_tree only has loudness (missing pitch)
        tree2 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            lufs=np.array([-15.0, -22.0]),
        )

        # Third audio_tree has loudness, pitch, and velocity (extra field)
        tree3 = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            lufs=np.array([-19.0, -21.0]),
            pitch=np.array([64.0, 66.0]),
            velocity=np.array([80, 90]),
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
            lufs=np.array([-20.0, -18.0]),
        )
        tree1 = tree1.replace(
            metadata={
                "params": np.random.randn(2, 10).astype(np.float32),
                "frame_id": np.array([100, 200], dtype=np.int32),
            }
        )

        # Second write: batch size 3 with metadata params [3, 10]
        tree2 = AudioTree.create(
            np.random.randn(3, 1, 8000),
            sample_rate=8000,
            lufs=np.array([-15.0, -22.0, -19.0]),
        )
        tree2 = tree2.replace(
            metadata={
                "params": np.random.randn(3, 10).astype(np.float32),
                "frame_id": np.array([300, 400, 500], dtype=np.int32),
            }
        )

        # Third write: batch size 1 with metadata params [1, 10]
        tree3 = AudioTree.create(
            np.random.randn(1, 1, 8000), sample_rate=8000, lufs=np.array([-17.0])
        )
        tree3 = tree3.replace(
            metadata={
                "params": np.random.randn(1, 10).astype(np.float32),
                "frame_id": np.array([600], dtype=np.int32),
            }
        )

        # Write all trees
        with AudioWriter(output_dir) as writer:
            writer.write(tree1)
            writer.write(tree2)
            writer.write(tree3)

        # Load manifest and verify
        manifest_data = load_manifest(output_dir / "manifest.npz")

        # Check that metadata was stacked correctly
        assert "metadata_params" in manifest_data
        assert "metadata_frame_id" in manifest_data

        # Should have 2 + 3 + 1 = 6 entries total
        assert manifest_data["metadata_params"].shape == (6, 10)
        assert manifest_data["metadata_frame_id"].shape == (6,)

        # Verify dtypes preserved
        assert manifest_data["metadata_params"].dtype == np.float32
        assert manifest_data["metadata_frame_id"].dtype == np.int32

        # Verify frame_ids are correct
        np.testing.assert_array_equal(
            manifest_data["metadata_frame_id"], [100, 200, 300, 400, 500, 600]
        )

        # Test reading back with AudioDataSource
        source = AudioDataSource.from_writer_output(output_dir)
        assert len(source) == 6

        # Check all items have correct metadata shapes and values
        expected_frame_ids = [100, 200, 300, 400, 500, 600]

        for idx, expected_frame_id in enumerate(expected_frame_ids):
            loaded_tree = source[idx]

            # Verify metadata exists and has correct shape
            assert "params" in loaded_tree.metadata
            assert loaded_tree.metadata["params"].shape == (1, 10), (
                f"Item {idx}: params shape mismatch"
            )
            assert loaded_tree.metadata["params"].dtype == np.float32, (
                f"Item {idx}: params dtype mismatch"
            )

            assert "frame_id" in loaded_tree.metadata
            assert loaded_tree.metadata["frame_id"].shape == (1,), (
                f"Item {idx}: frame_id shape mismatch"
            )
            assert loaded_tree.metadata["frame_id"].dtype == np.int32, (
                f"Item {idx}: frame_id dtype mismatch"
            )

            # Verify frame_id value matches
            assert loaded_tree.metadata["frame_id"][0] == expected_frame_id, (
                f"Item {idx}: frame_id value mismatch"
            )

            # Verify AudioTree field shapes
            assert loaded_tree.waveform.shape == (1, 1, 8000), (
                f"Item {idx}: waveform shape mismatch"
            )
            assert loaded_tree.lufs is not None, (
                f"Item {idx}: loudness should not be None"
            )
            assert loaded_tree.lufs.shape == (1,), (
                f"Item {idx}: loudness shape mismatch"
            )


def test_audiotree_from_manifest():
    """Test AudioTree.from_manifest loads all items into a single AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple trees with different batch sizes
        tree1 = AudioTree.create(
            np.random.randn(2, 2, 8000),
            sample_rate=8000,
            lufs=np.array([-20.0, -18.0]),
            pitch=np.array([60.0, 62.0]),
        )
        tree1 = tree1.replace(
            metadata={
                "params": np.random.randn(2, 10).astype(np.float32),
                "frame_id": np.array([100, 200], dtype=np.int32),
            }
        )

        tree2 = AudioTree.create(
            np.random.randn(3, 2, 8000),
            sample_rate=8000,
            lufs=np.array([-15.0, -22.0, -19.0]),
            pitch=np.array([64.0, 66.0, 68.0]),
        )
        tree2 = tree2.replace(
            metadata={
                "params": np.random.randn(3, 10).astype(np.float32),
                "frame_id": np.array([300, 400, 500], dtype=np.int32),
            }
        )

        # Write to manifest
        with AudioWriter(output_dir) as writer:
            writer.write(tree1)
            writer.write(tree2)

        # Load all at once into single AudioTree
        combined = AudioTree.from_manifest(output_dir / "manifest.npz")

        # Should have 2 + 3 = 5 items in batch dimension
        assert combined.waveform.shape == (5, 2, 8000)
        assert combined.sample_rate == 8000

        # AudioTree fields should be concatenated
        assert combined.lufs.shape == (5,)
        assert np.allclose(combined.lufs, [-20.0, -18.0, -15.0, -22.0, -19.0])
        assert combined.pitch.shape == (5,)
        assert np.allclose(combined.pitch, [60.0, 62.0, 64.0, 66.0, 68.0])

        # Metadata arrays should be concatenated
        assert combined.metadata["params"].shape == (5, 10)
        assert combined.metadata["frame_id"].shape == (5,)
        np.testing.assert_array_equal(
            combined.metadata["frame_id"], [100, 200, 300, 400, 500]
        )


def test_audiotree_from_manifest_without_audio_files():
    """Test AudioTree.from_manifest with manifest-only (no audio files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree
        audio_tree = AudioTree.create(
            np.random.randn(3, 2, 16000),
            sample_rate=16000,
            lufs=np.array([-20.0, -15.0, -25.0]),
            velocity=np.array([64, 80, 45], dtype=np.int16),
        )

        # Write manifest only
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(audio_tree)

        # Load from manifest without audio files
        combined = AudioTree.from_manifest(output_dir / "manifest.npz")

        # Should have correct shape with zero audio data
        assert combined.waveform.shape == (3, 2, 16000)
        assert np.all(combined.waveform == 0.0)

        # Metadata should be preserved
        assert combined.lufs.shape == (3,)
        assert np.allclose(combined.lufs, [-20.0, -15.0, -25.0])
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
            lufs=np.array([-30.0, -18.0, -25.0, -15.0, -22.0]),
        )

        # Write to manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load only items louder than -20 LUFS
        filtered = AudioTree.from_manifest(
            output_dir / "manifest.npz",
            filter_fn=lambda entry: entry.get("lufs", -float("inf")) > -20.0,
        )

        # Should only have 2 items: -18.0 and -15.0
        assert filtered.waveform.shape == (2, 1, 8000)
        assert filtered.lufs.shape == (2,)
        assert np.allclose(filtered.lufs, [-18.0, -15.0])


def test_audiotree_from_manifest_restores_filepaths():
    """from_manifest restores the source filepaths so .filepath works.

    AudioWriter stores filepaths as a top-level ``filepath`` manifest column
    (not under a ``metadata_`` prefix). from_manifest must round-trip that
    column back into ``metadata['filepath']``; otherwise ``.filepath`` is
    silently empty on the loaded tree.
    """
    paths = ["clip_0.wav", "clip_1.wav", "clip_2.wav", "clip_3.wav"]

    # Manifest-only (no audio files written).
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((4, 1, 8000), dtype=np.float32),
            sample_rate=8000,
            filepath=paths,
            metadata={"label": np.arange(4)},
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        loaded = AudioTree.from_manifest(output_dir / "manifest.npz")
        assert loaded.filepath == paths

        # A filter keeps filepaths aligned with the selected rows.
        subset = AudioTree.from_manifest(
            output_dir / "manifest.npz",
            filter_fn=lambda entry: entry["metadata_label"] % 2 == 0,
        )
        assert subset.filepath == ["clip_0.wav", "clip_2.wav"]

    # With real audio files written, filepaths still round-trip.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((4, 1, 8000), dtype=np.float32),
            sample_rate=8000,
            filepath=paths,
        )
        with AudioWriter(output_dir) as writer:
            writer.write(tree)

        loaded = AudioTree.from_manifest(output_dir / "manifest.npz")
        assert loaded.filepath == paths

    # A tree written without filepaths still loads; .filepath stays empty.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(np.zeros((2, 1, 8000), dtype=np.float32), 8000)
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        loaded = AudioTree.from_manifest(output_dir / "manifest.npz")
        assert loaded.filepath == []


def test_manifest_datasource_without_audio_files():
    """Test that AudioDataSource works with manifest-only (no audio files)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create AudioTree with metadata
        waveform = np.random.randn(3, 2, 16000)
        audio_tree = AudioTree.create(
            waveform,
            sample_rate=16000,
            lufs=np.array([-20.0, -15.0, -25.0]),
            pitch=np.array([60.0, 62.0, 58.0]),
            velocity=np.array([64, 80, 45], dtype=np.int16),
        )
        audio_tree = audio_tree.replace(
            metadata={
                "params": np.random.randn(3, 10).astype(np.float32),
                "frame_id": np.array([100, 200, 300], dtype=np.int32),
            }
        )

        # Write manifest only (no audio files)
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(audio_tree, tags={"experiment": "no_audio"})

        # Verify no audio files exist
        wav_files = list(output_dir.glob("*.wav"))
        assert len(wav_files) == 0, "No WAV files should exist"

        # Load with AudioDataSource - should work without audio files
        source = AudioDataSource.from_writer_output(output_dir)
        assert len(source) == 3

        # Check loaded items
        for idx in range(3):
            loaded_tree = source[idx]

            # Should have correct sample rate
            assert loaded_tree.sample_rate == 16000

            # Audio data should be zeros with correct shape
            assert loaded_tree.waveform.shape == (1, 2, 16000)
            assert np.all(loaded_tree.waveform == 0.0)

            # Metadata should be preserved
            assert loaded_tree.lufs is not None
            assert loaded_tree.pitch is not None
            assert loaded_tree.velocity is not None

            # Check metadata arrays
            assert "params" in loaded_tree.metadata
            assert loaded_tree.metadata["params"].shape == (1, 10)
            assert "frame_id" in loaded_tree.metadata
            assert loaded_tree.metadata["frame_id"].shape == (1,)

        # Verify specific values for first item
        first_tree = source[0]
        assert np.allclose(first_tree.lufs, [-20.0])
        assert np.allclose(first_tree.pitch, [60.0])
        assert np.allclose(first_tree.velocity, [64])
        assert first_tree.metadata["frame_id"][0] == 100


def test_write_empty_audiotree():
    """Test writing an empty AudioTree (batch_size=0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        empty_tree = AudioTree.create(
            np.zeros((0, 2, 44100)), sample_rate=44100, lufs=np.array([])
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
                np.zeros((0, 1, 8000)), sample_rate=8000, lufs=np.array([])
            )
            paths1 = writer.write(empty_tree)
            assert len(paths1) == 0

            nonempty_tree = AudioTree.create(
                np.random.randn(2, 1, 8000),
                sample_rate=8000,
                lufs=np.array([-20.0, -18.0]),
            )
            paths2 = writer.write(nonempty_tree)
            assert len(paths2) == 2

        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        data = load_manifest(manifest_path)
        assert len(data["index"]) == 2
        assert np.allclose(data["lufs"], [-20.0, -18.0])


def test_write_empty_audiotree_after_nonempty():
    """Test writing non-empty trees first, then an empty audio_tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with AudioWriter(output_dir) as writer:
            nonempty_tree = AudioTree.create(
                np.random.randn(2, 1, 8000),
                sample_rate=8000,
                lufs=np.array([-20.0, -18.0]),
            )
            paths1 = writer.write(nonempty_tree)
            assert len(paths1) == 2

            empty_tree = AudioTree.create(
                np.zeros((0, 1, 8000)), sample_rate=8000, lufs=np.array([])
            )
            paths2 = writer.write(empty_tree)
            assert len(paths2) == 0

        manifest_path = output_dir / "manifest.npz"
        assert manifest_path.exists()

        data = load_manifest(manifest_path)
        assert len(data["index"]) == 2
        assert np.allclose(data["lufs"], [-20.0, -18.0])


def test_write_filtered_empty_audiotree():
    """Test filtering an AudioTree to empty, then writing the result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        waveform = np.random.randn(3, 2, 44100)
        audio_tree = AudioTree.create(
            waveform, sample_rate=44100, lufs=np.array([-20.0, -15.0, -18.0])
        )

        filtered_tree = audio_tree.filter(lambda x: False)

        assert filtered_tree.waveform.shape[0] == 0
        assert filtered_tree.lufs.shape[0] == 0

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
            np.zeros((0, 2, 44100)), sample_rate=44100, lufs=np.array([])
        )
        empty_tree = empty_tree.replace(
            metadata={
                "params": np.zeros((0, 10), dtype=np.float32),
                "frame_id": np.array([], dtype=np.int32),
            }
        )

        assert empty_tree.waveform.shape == (0, 2, 44100)
        assert empty_tree.metadata["params"].shape == (0, 10)
        assert empty_tree.metadata["frame_id"].shape == (0,)

        with AudioWriter(output_dir) as writer:
            paths = writer.write(empty_tree)

        assert len(paths) == 0
        manifest_path = output_dir / "manifest.npz"
        assert not manifest_path.exists()


def test_filter_with_partial_match():
    """Test filtering AudioTree where some items match."""
    waveform = np.random.randn(5, 1, 8000)
    audio_tree = AudioTree.create(
        waveform,
        sample_rate=8000,
        lufs=np.array([-30.0, -18.0, -25.0, -15.0, -22.0]),
    )
    audio_tree = audio_tree.replace(
        metadata={"params": np.random.randn(5, 10).astype(np.float32)}
    )

    filtered = audio_tree.filter(lambda x: x.lufs[0] > -20.0)

    assert filtered.waveform.shape[0] == 2
    assert filtered.lufs.shape[0] == 2
    assert np.allclose(filtered.lufs, [-18.0, -15.0])
    assert filtered.metadata["params"].shape == (2, 10)


def test_filter_with_no_match():
    """Test filtering AudioTree where no items match."""
    waveform = np.random.randn(3, 1, 8000)
    audio_tree = AudioTree.create(
        waveform, sample_rate=8000, lufs=np.array([-20.0, -18.0, -22.0])
    )
    audio_tree = audio_tree.replace(
        metadata={"params": np.random.randn(3, 10).astype(np.float32)}
    )

    filtered = audio_tree.filter(lambda x: x.lufs[0] > 0.0)

    assert filtered.waveform.shape[0] == 0
    assert filtered.lufs.shape[0] == 0
    assert filtered.metadata["params"].shape == (0, 10)


def test_filter_with_all_match():
    """Test filtering AudioTree where all items match."""
    waveform = np.random.randn(3, 1, 8000)
    audio_tree = AudioTree.create(
        waveform, sample_rate=8000, lufs=np.array([-20.0, -18.0, -22.0])
    )

    filtered = audio_tree.filter(lambda x: x.lufs[0] < 0.0)

    assert filtered.waveform.shape[0] == 3
    assert np.allclose(filtered.lufs, [-20.0, -18.0, -22.0])


if __name__ == "__main__":
    # Run tests
    test_basic_sequential_writing()
    test_multiple_writes()
    test_sample_rate_inferred_and_enforced()
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
    test_audiotree_from_manifest_restores_filepaths()
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


def test_audio_writer_refuses_to_clobber_an_existing_dataset():
    """A second AudioWriter on a finished directory must raise.

    Both writers truncate, so this previously overwrote part of the audio and
    orphaned the rest, referenced by nothing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = AudioWriter(tmpdir)
        writer.write(AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000))
        writer.close()

        with pytest.raises(FileExistsError, match="already contains a dataset"):
            AudioWriter(tmpdir)

        AudioWriter(tmpdir, exist_ok=True)  # opt in


def test_jax_label_columns_are_written_per_item():
    """A ``jax.Array`` label must be split per item, not repeated whole.

    ``_create_manifest_entry`` dispatched on ``isinstance(value, np.ndarray)``,
    which a ``jax.Array`` fails, so every row stored the *entire batch* and the
    manifest ended up with more label values than files, each bound to the wrong
    audio.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with AudioWriter(output_dir) as writer:
            for lufs, pitch in ((-1.0, -2.0), (-3.0, -4.0)):
                tree = AudioTree.create(jnp.zeros((2, 1, 800)), 16000)
                tree = tree.replace(
                    lufs=jnp.array([lufs, pitch], dtype=jnp.float32),
                    metadata={"frame_id": jnp.arange(2, dtype=jnp.int32)},
                )
                writer.write(tree)

        assert len(list(output_dir.glob("*.wav"))) == 4

        data = load_manifest(output_dir / "manifest.npz")
        assert data["lufs"].shape == (4,)
        np.testing.assert_allclose(data["lufs"], [-1.0, -2.0, -3.0, -4.0])
        assert data["metadata_frame_id"].shape == (4,)
        np.testing.assert_array_equal(data["metadata_frame_id"], [0, 1, 0, 1])

        # And the labels round-trip onto the loaded batch in file order.
        loaded = AudioTree.from_manifest(output_dir / "manifest.npz")
        np.testing.assert_allclose(np.asarray(loaded.lufs), [-1.0, -2.0, -3.0, -4.0])


def test_list_label_column_is_written_per_item():
    """A plain Python list is array-like too, and must not be stored whole."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        tree = AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000)
        tree = tree.replace(metadata={"velocity": [10, 20, 30]})

        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        data = load_manifest(output_dir / "manifest.npz")
        np.testing.assert_array_equal(data["metadata_velocity"], [10, 20, 30])


def test_unstorable_column_raises():
    """A value that is neither scalar nor per-item array-like must raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
        tree = tree.replace(metadata={"weird": [object(), object()]})

        with AudioWriter(tmpdir, write_audio=False) as writer:
            with pytest.raises(ValueError, match="metadata_weird"):
                writer.write(tree)


def test_column_shorter_than_batch_raises():
    """A label array with fewer rows than the batch must raise, not be dropped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tree = AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000)
        tree = tree.replace(metadata={"short": np.array([1.0, 2.0])})

        with AudioWriter(tmpdir, write_audio=False) as writer:
            with pytest.raises(ValueError, match="too few for batch index"):
                writer.write(tree)


def test_ragged_column_raises_at_the_offending_write():
    """A metadata key present on some writes and absent on others is rejected.

    The check is eager: a drifting write fails at its own ``write()`` call --
    before its WAVs land and while the manifest is still consistent -- rather
    than being deferred to ``save_manifest()``/``close()``, which would abort
    with the whole manifest unwritten.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tree1 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
        tree1 = tree1.replace(metadata={"frame_id": np.array([1, 2])})
        tree2 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)

        writer = AudioWriter(tmpdir, write_audio=False, manifest_every=0)
        writer.write(tree1)

        with pytest.raises(ValueError, match="metadata keys.*missing keys.*frame_id"):
            writer.write(tree2)  # no metadata at all

        # The good first write still saves cleanly; the manifest is intact.
        manifest_path = writer.save_manifest()
        data = load_manifest(manifest_path)
        assert len(data["index"]) == 2
        np.testing.assert_array_equal(data["metadata_frame_id"], [1, 2])


def test_manifest_is_written_during_the_run():
    """A killed run must leave a manifest for the files already on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)

        writer = AudioWriter(output_dir, manifest_every=4)
        writer.write(tree)
        assert not (output_dir / "manifest.npz").exists()  # still throttled
        writer.write(tree)

        # No close(), no context manager: this is what SIGKILL would leave.
        source = AudioDataSource.from_writer_output(output_dir)
        assert len(source) == 4


def test_manifest_write_failure_leaves_the_previous_manifest_intact(monkeypatch):
    """The manifest is replaced atomically, so a crash mid-save cannot corrupt it.

    A truncated ``manifest.npz`` reads as a broken zip *and* trips
    ``refuse_to_clobber`` on the retry, which used to leave the directory
    unusable in both directions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)

        writer = AudioWriter(output_dir, write_audio=False, manifest_every=0)
        writer.write(tree)
        writer.save_manifest()

        def explode(fileobj, **kwargs):
            fileobj.write(b"PK\x03\x04truncated")
            raise KeyboardInterrupt("killed mid-savez")

        monkeypatch.setattr(np, "savez_compressed", explode)
        writer.write(tree)
        with pytest.raises(KeyboardInterrupt):
            writer.save_manifest()

        data = load_manifest(output_dir / "manifest.npz")
        assert len(data["index"]) == 2  # the good, complete previous manifest


_DETERMINISM_SCRIPT = textwrap.dedent(
    """
    import hashlib, sys, tempfile
    import numpy as np
    from audiotree import AudioTree, AudioWriter

    with tempfile.TemporaryDirectory() as d:
        tree = AudioTree.create(
            np.zeros((3, 1, 80), dtype=np.float32),
            8000,
            lufs=np.zeros(3, dtype=np.float32),
            pitch=np.zeros(3, dtype=np.float32),
        )
        tree = tree.replace(
            metadata={f"col_{i}": np.arange(3, dtype=np.int32) for i in range(12)}
        )
        with AudioWriter(d, write_audio=False) as w:
            w.write(tree, tags={f"tag_{i}": i for i in range(6)})
        blob = (open(f"{d}/manifest.npz", "rb")).read()
        print(hashlib.sha256(blob).hexdigest())
        with np.load(f"{d}/manifest.npz", allow_pickle=False) as npz:
            print(",".join(npz.files))
    """
)


def test_manifest_bytes_are_deterministic_across_hash_seeds():
    """Two runs must produce byte-identical manifests.

    ``_manifest_to_arrays`` iterated a ``set``, so NPZ key order -- and hence the
    file's bytes -- depended on the interpreter's string hash seed.
    """
    import os

    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=True,
        )
        outputs.append(result.stdout.strip().splitlines())

    digest_a, keys_a = outputs[0]
    digest_b, keys_b = outputs[1]
    assert keys_a == keys_b, json.dumps([keys_a, keys_b], indent=2)
    assert digest_a == digest_b


def test_show_progress_without_tqdm_raises(tmp_path, monkeypatch):
    """``show_progress=True`` must fail loudly when tqdm is missing.

    It used to swallow the ImportError and carry on with no progress bar, so the
    flag was a silent no-op for anyone who had not installed the extra.
    """
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("No module named 'tqdm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)
    with pytest.raises(ImportError, match=r"audiotree\[progress\]"):
        AudioWriter(directory=tmp_path, show_progress=True)


# === Manifest encoding: no pickles, explicit presence masks ===


def _one_second(peak=0.5, sample_rate=16000, **fields):
    waveform = np.full((1, 1, sample_rate), peak, dtype=np.float32)
    return AudioTree.create(waveform, sample_rate=sample_rate, **fields)


def test_manifest_stores_strings_without_pickling(tmp_path):
    """String columns are fixed-width unicode, so the reader never unpickles.

    A manifest travels with the data it describes; loading it with
    ``allow_pickle=True`` is arbitrary code execution in every data worker, and
    the only reason it was ever on is that strings were ``dtype=object``.
    """
    tree = _one_second(filepath=["source one.wav"])
    with AudioWriter(tmp_path) as writer:
        writer.write(tree, tags={"dataset": "unicode ✓"})

    with np.load(tmp_path / "manifest.npz", allow_pickle=False) as npz:
        data = dict(npz)

    assert data["filename"].dtype.kind == "U"
    assert data["filepath"].dtype.kind == "U"
    assert data["tags_dataset"].dtype.kind == "U"
    assert all(array.dtype != object for array in data.values())
    assert list(data["filepath"]) == ["source one.wav"]
    assert list(data["tags_dataset"]) == ["unicode ✓"]


def test_absent_tag_is_masked_rather_than_sentinelled(tmp_path):
    """Only the mask says "missing" -- ``""``, ``-1`` and ``NaN`` are values.

    The tag columns used to store ``None`` in an object array and the reader
    dropped every ``None`` *or* ``""`` cell, so a genuinely empty tag vanished.
    """
    writer = AudioWriter(tmp_path)
    writer.write(_one_second(), tags={"split": "train", "score": -1})
    writer.write(_one_second(), tags={"split": ""})
    writer.close()

    with np.load(tmp_path / "manifest.npz", allow_pickle=False) as npz:
        data = dict(npz)

    np.testing.assert_array_equal(data[f"{_manifest.MASK_PREFIX}tags_score"], [1, 0])
    assert f"{_manifest.MASK_PREFIX}tags_split" not in data  # present in both rows

    entries = _manifest.read_entries(tmp_path / "manifest.npz")
    assert entries[0]["tags"] == {"split": "train", "score": -1}
    assert entries[1]["tags"] == {"split": ""}


def test_manifest_only_run_records_no_subtype(tmp_path):
    """With no audio file there is no encoding, and the column says so."""
    with AudioWriter(tmp_path, write_audio=False) as writer:
        writer.write(_one_second())

    with np.load(tmp_path / "manifest.npz", allow_pickle=False) as npz:
        data = dict(npz)
    np.testing.assert_array_equal(data[f"{_manifest.MASK_PREFIX}subtype"], [0])

    (entry,) = _manifest.read_entries(tmp_path / "manifest.npz")
    assert "subtype" not in entry


def test_pre_1_0_pickled_manifest_is_refused(tmp_path):
    """An old object-array manifest fails with re-render advice, not a pickle."""
    manifest_path = tmp_path / "manifest.npz"
    arrays = {
        "index": np.array([0], dtype=np.int32),
        "filename": np.array(["audio_0000.wav"], dtype=object),
    }
    for key, value in _format.header(_format.MANIFEST).items():
        arrays[f"{_format.NPZ_HEADER_PREFIX}{key}"] = np.array(json.dumps(value))
    np.savez(manifest_path, **arrays)

    with pytest.raises(ValueError, match="re-render the dataset"):
        _manifest.read_entries(manifest_path)


def test_manifest_row_count_is_declared_not_guessed(tmp_path):
    """A column shorter than the declared row count is a corrupt manifest."""
    with AudioWriter(tmp_path, write_audio=False) as writer:
        writer.write(AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000))

    with np.load(tmp_path / "manifest.npz", allow_pickle=False) as npz:
        data = dict(npz)
    data["index"] = data["index"][:2]
    np.savez(tmp_path / "manifest.npz", **data)

    with pytest.raises(ValueError, match=r"column 'index' has 2 rows"):
        _manifest.read_entries(tmp_path / "manifest.npz")


# === Subtype: a default that does not destroy float model output ===


def test_default_subtype_keeps_out_of_range_audio(tmp_path):
    """The default must not hard-clip; PCM_16 would silently discard the peak."""
    with AudioWriter(tmp_path) as writer:
        writer.write(_one_second(peak=2.5))

    info = soundfile.info(str(tmp_path / "audio_0000.wav"))
    assert info.subtype == "FLOAT"
    audio, _ = soundfile.read(str(tmp_path / "audio_0000.wav"))
    np.testing.assert_allclose(np.abs(audio).max(), 2.5, atol=1e-6)


def test_default_subtype_falls_back_to_pcm_24_for_flac(tmp_path):
    """FLAC admits no float subtype, so take its widest fixed-point one."""
    with AudioWriter(tmp_path, pattern="audio_{index:04d}.flac") as writer:
        writer.write(_one_second(peak=0.5))

    assert soundfile.info(str(tmp_path / "audio_0000.flac")).subtype == "PCM_24"


def test_written_subtype_is_recorded_in_the_manifest(tmp_path):
    """Which encoding the audio is in is part of what the dataset describes."""
    with AudioWriter(tmp_path, subtype="PCM_24") as writer:
        writer.write(_one_second())

    (entry,) = _manifest.read_entries(tmp_path / "manifest.npz")
    assert entry["subtype"] == "PCM_24"
    assert soundfile.info(str(tmp_path / "audio_0000.wav")).subtype == "PCM_24"


def test_clipping_subtype_warns_when_audio_exceeds_range(tmp_path):
    """An explicit fixed-point subtype must be loud about what it destroys."""
    with AudioWriter(tmp_path, subtype="PCM_16") as writer:
        with pytest.warns(RuntimeWarning, match=r"clipped 1 of 2 items"):
            writer.write(
                AudioTree.batch([_one_second(peak=0.5), _one_second(peak=2.5)])
            )


def test_in_range_audio_does_not_warn(tmp_path):
    """The warning tracks actual clipping, not the choice of subtype."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with AudioWriter(tmp_path, subtype="PCM_16") as writer:
            writer.write(_one_second(peak=0.5))


def test_mini_batched_tree_is_refused(tmp_path):
    """A rank-4 tree wrote nonsense metadata instead of failing.

    ``reshape_mini_batches`` gives a tree two leading axes, and the writer read
    the first as the batch -- so every axis below shifted by one. A 6-item
    ``(3, 2, 1, 800)`` tree produced *three* manifest rows claiming
    ``channels=2, samples=1``. ``write_audio=True`` was saved only by soundfile
    rejecting the transposed shape; ``write_audio=False`` -- the manifest-only
    mode, where nothing else inspects the audio -- recorded it silently.
    """
    tree = AudioTree.create(np.zeros((6, 1, 800), np.float32), 16000)
    mini_batched = tree.reshape_mini_batches(2)
    assert mini_batched.waveform.shape == (3, 2, 1, 800)

    for write_audio in (True, False):
        with AudioWriter(tmp_path / f"out_{write_audio}", write_audio=write_audio) as w:
            with pytest.raises(ValueError, match="needs a rank-3"):
                w.write(mini_batched)

    # Flattening it first is the documented route, and still works.
    with AudioWriter(tmp_path / "flat", write_audio=False) as w:
        w.write(mini_batched.flatten_mini_batches())
    entries = _manifest.read_entries(tmp_path / "flat" / "manifest.npz")
    assert len(entries) == 6
    assert all(e["channels"] == 1 and e["samples"] == 800 for e in entries)


# === Filename pattern must be unique per item when writing audio ===


def test_pattern_without_index_is_refused_when_writing_audio(tmp_path):
    """A pattern with no ``{index}`` writes every item to one file, losing all
    but the last while the manifest still records a row per lost item."""
    with pytest.raises(ValueError, match=r"no '\{index\}' field"):
        AudioWriter(tmp_path, pattern="out.wav")


def test_pattern_with_index_writes_one_file_per_item(tmp_path):
    """A pattern that does carry ``{index}`` gives each item a unique file."""
    tree = AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000)
    with AudioWriter(tmp_path, pattern="clip_{index}.wav") as writer:
        writer.write(tree)

    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [
        "clip_0.wav",
        "clip_1.wav",
        "clip_2.wav",
    ]


def test_manifest_only_run_allows_pattern_without_index(tmp_path):
    """With no audio on disk the filename is a label, so uniqueness is optional."""
    tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    with AudioWriter(tmp_path, pattern="out.wav", write_audio=False) as writer:
        writer.write(tree)  # must not raise

    data = load_manifest(tmp_path / "manifest.npz")
    assert list(data["filename"]) == ["out.wav", "out.wav"]


# === Metadata / filepath drift is rejected eagerly, not at save ===


def test_metadata_key_drift_raises_at_the_write_not_at_close(tmp_path):
    """A write whose metadata keys differ from the first is rejected immediately,
    before its WAVs land, so a long render never loses its manifest at close()."""
    tree1 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    tree1 = tree1.replace(metadata={"snr": np.array([10.0, 20.0])})
    tree2 = AudioTree.create(
        np.zeros((2, 1, 800), dtype=np.float32), 16000
    )  # no metadata

    writer = AudioWriter(tmp_path, manifest_every=0)
    writer.write(tree1)

    with pytest.raises(ValueError, match="metadata keys.*missing keys.*snr"):
        writer.write(tree2)

    # The drifting write left no audio behind: only the first two items exist.
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [
        "audio_0000.wav",
        "audio_0001.wav",
    ]

    # And close() still writes a manifest consistent with the first write.
    writer.close()
    data = load_manifest(tmp_path / "manifest.npz")
    assert len(data["index"]) == 2
    np.testing.assert_allclose(data["metadata_snr"], [10.0, 20.0])


def test_extra_metadata_key_on_later_write_is_named(tmp_path):
    """A later write introducing a new metadata key is rejected, naming the key."""
    tree1 = AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000)
    tree2 = AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000)
    tree2 = tree2.replace(metadata={"snr": np.array([5.0])})

    writer = AudioWriter(tmp_path, write_audio=False, manifest_every=0)
    writer.write(tree1)
    with pytest.raises(ValueError, match="metadata keys.*extra keys.*snr"):
        writer.write(tree2)


def test_short_filepath_list_raises_at_the_write(tmp_path):
    """A filepath list shorter than the batch is rejected at the write itself,
    not deferred to save where it would abort with the manifest unwritten."""
    tree = AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000)
    # Only two encoded paths for a three-item batch. (``create`` guards against
    # this, so encode directly to reach the writer's own coverage check.)
    tree = tree.replace(
        metadata={"filepath": AudioTree._encode_filepaths(["a.wav", "b.wav"])}
    )
    writer = AudioWriter(tmp_path, write_audio=False, manifest_every=0)
    with pytest.raises(ValueError, match="filepath"):
        writer.write(tree)


def test_filepath_presence_drift_raises_at_the_write(tmp_path):
    """Whether a run carries filepaths is fixed by the first write."""
    with_paths = AudioTree.create(
        np.zeros((1, 1, 800), dtype=np.float32), 16000, filepath=["a.wav"]
    )
    without_paths = AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000)

    writer = AudioWriter(tmp_path, write_audio=False, manifest_every=0)
    writer.write(with_paths)
    with pytest.raises(ValueError, match="'filepath' presence"):
        writer.write(without_paths)


def test_consistent_metadata_sequence_still_writes_manifest(tmp_path):
    """The eager check must not reject a genuinely consistent sequence."""
    with AudioWriter(tmp_path, write_audio=False) as writer:
        for snr in ([1.0, 2.0], [3.0, 4.0, 5.0]):
            tree = AudioTree.create(
                np.zeros((len(snr), 1, 800), dtype=np.float32), 16000
            )
            tree = tree.replace(metadata={"snr": np.array(snr)})
            writer.write(tree)

    data = load_manifest(tmp_path / "manifest.npz")
    assert len(data["index"]) == 5
    np.testing.assert_allclose(data["metadata_snr"], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_column_kind_drift_raises_at_the_write_not_at_close(tmp_path):
    """A write whose values change a column's logical kind (int rows, then a
    str row) is rejected at its own call -- the schema checks pass, since the
    *keys* still match -- rather than at close(), where the encoder would abort
    with the manifest unwritten and every WAV on disk orphaned."""
    tree1 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    tree1 = tree1.replace(metadata={"take": np.array([1, 2])})
    tree2 = AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000)
    tree2 = tree2.replace(metadata={"take": "final"})

    writer = AudioWriter(tmp_path, manifest_every=0)
    writer.write(tree1)

    with pytest.raises(
        ValueError, match=r"'metadata_take' holds int values.*str value"
    ):
        writer.write(tree2)

    # The drifting write left no audio behind: only the first two items exist.
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == [
        "audio_0000.wav",
        "audio_0001.wav",
    ]

    # And close() still writes a manifest consistent with the first write.
    writer.close()
    data = load_manifest(tmp_path / "manifest.npz")
    assert len(data["index"]) == 2
    np.testing.assert_array_equal(data["metadata_take"], [1, 2])


def test_tag_kind_drift_raises_at_the_write(tmp_path):
    """Tag columns are pinned the same way: a tag that changes kind across
    writes fails at the offending write, naming the ``tags_*`` column."""
    tree = AudioTree.create(np.zeros((1, 1, 800), dtype=np.float32), 16000)

    writer = AudioWriter(tmp_path, write_audio=False, manifest_every=0)
    writer.write(tree, tags={"quality": 1})
    with pytest.raises(ValueError, match=r"'tags_quality' holds int values.*str value"):
        writer.write(tree, tags={"quality": "high"})


def test_array_column_kind_drift_raises_at_the_write(tmp_path):
    """Array-valued columns are covered too: float32 embedding rows followed by
    int rows fail at the second write, not at close via the encoder."""
    float_tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    float_tree = float_tree.replace(
        metadata={"emb": np.zeros((2, 3), dtype=np.float32)}
    )
    int_tree = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    int_tree = int_tree.replace(metadata={"emb": np.zeros((2, 3), dtype=np.int32)})

    writer = AudioWriter(tmp_path, write_audio=False, manifest_every=0)
    writer.write(float_tree)
    with pytest.raises(
        ValueError, match=r"'metadata_emb' holds float array values.*int array"
    ):
        writer.write(int_tree)


def test_same_kind_values_across_writes_still_pass_the_kind_check(tmp_path):
    """The kind check must not reject a consistent column: a plain str on one
    write and per-item numpy strings on the next are both 'str'."""
    tree1 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    tree1 = tree1.replace(metadata={"label": "warmup"})
    tree2 = AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000)
    tree2 = tree2.replace(metadata={"label": np.array(["a", "b"])})

    with AudioWriter(tmp_path, write_audio=False) as writer:
        writer.write(tree1)
        writer.write(tree2)

    data = load_manifest(tmp_path / "manifest.npz")
    assert list(data["metadata_label"]) == ["warmup", "warmup", "a", "b"]


def test_timestamp_is_minted_once_per_write_call(tmp_path):
    """Every entry of one write() shares one timestamp, so timestamps record
    when the batch landed and get_stats() counts batches, not items."""
    with AudioWriter(tmp_path, write_audio=False, include_timestamp=True) as writer:
        writer.write(AudioTree.create(np.zeros((3, 1, 800), dtype=np.float32), 16000))
        writer.write(AudioTree.create(np.zeros((2, 1, 800), dtype=np.float32), 16000))
        stats = writer.get_stats()

    assert stats["total_batches"] == 2

    timestamps = list(load_manifest(tmp_path / "manifest.npz")["timestamp"])
    assert len(timestamps) == 5
    assert len(set(timestamps[:3])) == 1  # first batch shares its timestamp
    assert len(set(timestamps[3:])) == 1  # so does the second
    assert timestamps[0] != timestamps[3]  # distinct write() calls differ
