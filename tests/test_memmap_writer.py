"""Tests for MemmapWriter class."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree, MemmapWriter, FieldSpec
from audiotree.sources import MemmapDataSource


def test_basic_write():
    """Test basic writing of arrays to memmap files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [
            FieldSpec("audio", np.float32, (2, 1000)),
            FieldSpec("label", np.int32, ()),
        ]

        with MemmapWriter(output_dir, field_specs, expected_samples=10) as writer:
            # Write 5 samples
            audio = np.random.randn(5, 2, 1000).astype(np.float32)
            labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
            writer.write_batch({"audio": audio, "label": labels})

        # Check files were created
        assert (output_dir / "audio.bin").exists()
        assert (output_dir / "label.bin").exists()
        assert (output_dir / "manifest.json").exists()

        # Check manifest
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == 5
        assert manifest["expected_samples"] == 10
        assert "audio" in manifest["fields"]
        assert "label" in manifest["fields"]


def test_write_sample():
    """Test writing single samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [
            FieldSpec("data", np.float32, (10,)),
        ]

        with MemmapWriter(output_dir, field_specs, expected_samples=3) as writer:
            for i in range(3):
                data = np.random.randn(10).astype(np.float32)
                writer.write_sample({"data": data})

        # Check manifest
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == 3


def test_string_fields():
    """Test writing string fields to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [
            FieldSpec("value", np.float32, ()),
        ]

        with MemmapWriter(output_dir, field_specs, expected_samples=3) as writer:
            writer.write_batch(
                {"value": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                strings={"text": ["hello", "world", "test"]},
            )

        # Check string field was saved
        assert (output_dir / "text.json").exists()
        with open(output_dir / "text.json") as f:
            texts = json.load(f)
        assert texts == ["hello", "world", "test"]

        # Check manifest references string field
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert "text" in manifest["string_fields"]


def test_metadata():
    """Test custom metadata in manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("x", np.float32, ())]
        metadata = {"sample_rate": 44100, "duration": 3.0}

        with MemmapWriter(
            output_dir, field_specs, expected_samples=1, metadata=metadata
        ) as writer:
            writer.write_sample({"x": np.array(1.0, dtype=np.float32)})

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["sample_rate"] == 44100
        assert manifest["duration"] == 3.0


def test_shape_validation():
    """Test that shape mismatches are caught."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("data", np.float32, (10,))]

        with MemmapWriter(output_dir, field_specs, expected_samples=5) as writer:
            # Wrong shape should raise error
            wrong_shape = np.random.randn(2, 20).astype(np.float32)
            with pytest.raises(ValueError, match="Shape mismatch"):
                writer.write_batch({"data": wrong_shape})


def test_overflow_detection():
    """Test that writing more than expected_samples raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("data", np.float32, ())]

        with MemmapWriter(output_dir, field_specs, expected_samples=3) as writer:
            # Write 3 samples OK
            writer.write_batch({"data": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

            # Writing more should fail
            with pytest.raises(ValueError, match="exceed expected_samples"):
                writer.write_batch({"data": np.array([4.0], dtype=np.float32)})


def test_unknown_field():
    """Test that unknown fields raise error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("data", np.float32, ())]

        with MemmapWriter(output_dir, field_specs, expected_samples=5) as writer:
            with pytest.raises(ValueError, match="Unknown field"):
                writer.write_batch({"unknown": np.array([1.0], dtype=np.float32)})


def test_round_trip():
    """Test write/read round-trip with MemmapDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [
            FieldSpec("audio", np.float32, (2, 100)),
            FieldSpec("label", np.int32, ()),
        ]

        # Write data
        audio_data = np.random.randn(5, 2, 100).astype(np.float32)
        labels = np.arange(5, dtype=np.int32)

        with MemmapWriter(output_dir, field_specs, expected_samples=5) as writer:
            writer.write_batch({"audio": audio_data, "label": labels})

        # Read back with MemmapDataSource
        source = MemmapDataSource.from_directory(output_dir)
        assert len(source) == 5

        for i in range(5):
            sample = source[i]
            # MemmapDataSource adds batch dimension
            np.testing.assert_array_almost_equal(
                sample["audio"][0], audio_data[i], decimal=5
            )
            assert sample["label"][0] == labels[i]


def test_round_trip_with_strings():
    """Test write/read round-trip including string fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("value", np.float32, ())]

        with MemmapWriter(output_dir, field_specs, expected_samples=3) as writer:
            writer.write_batch(
                {"value": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
                strings={"code": ["a = 1", "b = 2", "c = 3"]},
            )

        source = MemmapDataSource.from_directory(output_dir)
        assert len(source) == 3

        sample = source[1]
        assert sample["value"][0] == 2.0
        assert sample["code"] == "b = 2"


def test_get_stats():
    """Test get_stats method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("x", np.float32, ())]

        with MemmapWriter(output_dir, field_specs, expected_samples=10) as writer:
            writer.write_batch({"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

            stats = writer.get_stats()
            assert stats["samples_written"] == 3
            assert stats["expected_samples"] == 10
            assert stats["is_open"] is True


def test_multiple_dtypes():
    """Test writing arrays with different dtypes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [
            FieldSpec("float32_data", np.float32, (10,)),
            FieldSpec("int32_data", np.int32, (5,)),
            FieldSpec("int8_data", np.int8, (3,)),
        ]

        with MemmapWriter(output_dir, field_specs, expected_samples=2) as writer:
            writer.write_batch({
                "float32_data": np.random.randn(2, 10).astype(np.float32),
                "int32_data": np.random.randint(0, 100, (2, 5), dtype=np.int32),
                "int8_data": np.random.randint(-128, 127, (2, 3), dtype=np.int8),
            })

        # Verify dtypes in manifest
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["fields"]["float32_data"]["dtype"] == "float32"
        assert manifest["fields"]["int32_data"]["dtype"] == "int32"
        assert manifest["fields"]["int8_data"]["dtype"] == "int8"


def test_flush():
    """Test explicit flush."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        field_specs = [FieldSpec("x", np.float32, ())]

        writer = MemmapWriter(output_dir, field_specs, expected_samples=5)
        writer.open()
        writer.write_batch({"x": np.array([1.0, 2.0], dtype=np.float32)})
        writer.flush()

        # Data should be on disk after flush
        mm = np.memmap(output_dir / "x.bin", dtype=np.float32, mode="r", shape=(5,))
        assert mm[0] == 1.0
        assert mm[1] == 2.0

        writer.close()


def test_single_audiotree_write():
    """Test writing a single AudioTree directly (not wrapped in a dict)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_data = np.random.randn(5, 2, 100).astype(np.float32)
        loudness = np.random.randn(5).astype(np.float32)
        tree = AudioTree(
            audio_data=audio_data,
            sample_rate=44100,
            loudness=loudness,
            metadata={"mel": np.random.randn(5, 32).astype(np.float32)},
        )

        with MemmapWriter(output_dir, expected_samples=5) as writer:
            writer.write_batch(tree)

        # Check manifest
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["single_audiotree"] is True
        assert manifest["sample_rate"] == 44100
        assert manifest["num_samples"] == 5
        # Fields should be unprefixed
        assert "audio_data" in manifest["fields"]
        assert "loudness" in manifest["fields"]
        assert "mel" in manifest["fields"]
