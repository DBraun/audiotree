"""Tests for TreeDataSource."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.tree_writer import TreeWriter
from audiotree.sources.tree import TreeDataSource


# === Round-trip tests ===


def test_round_trip_audiotree():
    """Write and read back a single AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(5, 2, 100).astype(np.float32)
        loudness = np.random.randn(5).astype(np.float32)
        tree = AudioTree(
            audio_data=audio, sample_rate=44100, loudness=loudness
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)
        assert len(source) == 5

        for i in range(5):
            sample = source[i]
            assert isinstance(sample, AudioTree)
            assert sample.sample_rate == 44100
            np.testing.assert_array_almost_equal(
                sample.audio_data[0], audio[i], decimal=5
            )
            np.testing.assert_array_almost_equal(
                sample.loudness[0], loudness[i], decimal=5
            )
            assert sample.pitch is None


def test_round_trip_audiotree_with_metadata():
    """AudioTree with metadata arrays round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(3, 1, 50).astype(np.float32)
        mel = np.random.randn(3, 32).astype(np.float32)
        tree = AudioTree(
            audio_data=audio, sample_rate=44100, metadata={"mel": mel}
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)

        for i in range(3):
            sample = source[i]
            assert isinstance(sample, AudioTree)
            np.testing.assert_array_almost_equal(
                sample.audio_data[0], audio[i], decimal=5
            )
            np.testing.assert_array_almost_equal(
                sample.metadata["mel"][0], mel[i], decimal=5
            )


def test_round_trip_nested_metadata():
    """Nested metadata dicts round-trip correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        mel = np.random.randn(2, 16).astype(np.float32)
        mfcc = np.random.randn(2, 8).astype(np.float32)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={"features": {"mel": mel, "mfcc": mfcc}},
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)
        sample = source[0]

        assert isinstance(sample, AudioTree)
        assert "features" in sample.metadata
        assert "mel" in sample.metadata["features"]
        assert "mfcc" in sample.metadata["features"]
        np.testing.assert_array_almost_equal(
            sample.metadata["features"]["mel"][0], mel[0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            sample.metadata["features"]["mfcc"][0], mfcc[0], decimal=5
        )


def test_round_trip_dict_of_audiotrees():
    """Dict of AudioTrees round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        dry_audio = np.random.randn(3, 2, 100).astype(np.float32)
        wet_audio = np.random.randn(3, 2, 100).astype(np.float32)
        dry = AudioTree(audio_data=dry_audio, sample_rate=44100)
        wet = AudioTree(audio_data=wet_audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"dry": dry, "wet": wet})

        source = TreeDataSource.from_directory(output_dir)
        assert len(source) == 3

        sample = source[0]
        assert isinstance(sample, dict)
        assert "dry" in sample and "wet" in sample
        assert isinstance(sample["dry"], AudioTree)
        assert isinstance(sample["wet"], AudioTree)
        np.testing.assert_array_almost_equal(
            sample["dry"].audio_data[0], dry_audio[0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            sample["wet"].audio_data[0], wet_audio[0], decimal=5
        )


def test_round_trip_plain_dict():
    """Plain dict of arrays round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        x = np.random.randn(5, 10).astype(np.float32)
        y = np.arange(5, dtype=np.int32)

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write({"x": x, "y": y})

        source = TreeDataSource.from_directory(output_dir)
        assert len(source) == 5

        for i in range(5):
            sample = source[i]
            assert isinstance(sample, dict)
            np.testing.assert_array_almost_equal(
                sample["x"][0], x[i], decimal=5
            )
            assert sample["y"][0] == y[i]


def test_round_trip_multiple_writes():
    """Data written in multiple batches is read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio1 = np.ones((3, 1, 10), dtype=np.float32)
        audio2 = np.ones((2, 1, 10), dtype=np.float32) * 2

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(AudioTree(audio_data=audio1, sample_rate=44100))
            w.write(AudioTree(audio_data=audio2, sample_rate=44100))

        source = TreeDataSource.from_directory(output_dir)
        assert len(source) == 5

        # First batch
        for i in range(3):
            sample = source[i]
            np.testing.assert_array_almost_equal(
                sample.audio_data[0], audio1[i], decimal=5
            )

        # Second batch
        for i in range(2):
            sample = source[3 + i]
            np.testing.assert_array_almost_equal(
                sample.audio_data[0], audio2[i], decimal=5
            )


# === Raw mode ===


def test_raw_mode():
    """raw=True returns flat dict with leaf path keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
            loudness=np.zeros(3, dtype=np.float32),
            metadata={"mel": np.zeros((3, 16), dtype=np.float32)},
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir, raw=True)
        sample = source[0]

        assert isinstance(sample, dict)
        assert "audio_data" in sample
        assert "loudness" in sample
        assert "metadata.mel" in sample
        assert sample["audio_data"].shape == (1, 2, 100)


# === from_directory ===


def test_from_directory():
    """from_directory convenience constructor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)
        assert len(source) == 2


def test_from_directory_missing():
    """from_directory raises FileNotFoundError for missing manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            TreeDataSource.from_directory(tmpdir)


# === Error handling ===


def test_index_out_of_range():
    """Out-of-range index raises IndexError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)

        with pytest.raises(IndexError):
            _ = source[10]

        with pytest.raises(IndexError):
            _ = source[-1]


def test_version_check():
    """Reading a non-v2 manifest raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write a v1-style manifest
        import json

        manifest = {"version": "1.0", "num_samples": 5, "fields": {}}
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="Unsupported manifest version"):
            TreeDataSource(output_dir / "manifest.json")


# === Metadata ===


def test_get_metadata():
    """get_metadata returns user metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with TreeWriter(
            output_dir,
            expected_samples=2,
            metadata={"description": "test", "version": 1},
        ) as w:
            w.write({"x": np.zeros((2,), dtype=np.float32)})

        source = TreeDataSource.from_directory(output_dir)
        meta = source.get_metadata()
        assert meta["description"] == "test"
        assert meta["version"] == 1


# === Empty/minimal structures ===


def test_empty_metadata_round_trip():
    """AudioTree with empty metadata round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={},
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)
        sample = source[0]
        assert isinstance(sample, AudioTree)
        assert sample.metadata == {}


def test_none_fields_default():
    """Absent optional fields default to None in reconstructed AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)
        sample = source[0]
        assert sample.pitch is None
        assert sample.velocity is None
        assert sample.loudness is None
        assert sample.codes is None
        assert sample.latents is None


# === Grain integration ===


def test_grain_protocol():
    """TreeDataSource satisfies RandomAccessDataSource protocol."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((5, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        source = TreeDataSource.from_directory(output_dir)

        # len() and __getitem__ are required
        assert len(source) == 5
        sample = source[0]
        assert sample is not None

        # Iteration via indexing
        items = [source[i] for i in range(5)]
        assert len(items) == 5
