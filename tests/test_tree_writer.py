"""Tests for TreeWriter."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.tree_writer import TreeWriter


def test_basic_write_audiotree():
    """Write a simple AudioTree, verify files and manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(5, 2, 100).astype(np.float32)
        tree = AudioTree(audio_data=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        assert (output_dir / "audio_data.bin").exists()
        assert (output_dir / "manifest.json").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["version"] == "2.0"
        assert manifest["num_samples"] == 5
        assert "audio_data" in manifest["leaves"]


def test_manifest_structure_audiotree():
    """Verify structure JSON for an AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 1, 50), dtype=np.float32),
            sample_rate=48000,
            loudness=np.zeros(3, dtype=np.float32),
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        structure = manifest["structure"]
        assert structure["type"] == "AudioTree"
        assert structure["sample_rate"] == 48000
        assert "audio_data" in structure["children"]
        assert "loudness" in structure["children"]
        # pitch etc. should be absent (None)
        assert "pitch" not in structure["children"]


def test_write_audiotree_with_metadata():
    """AudioTree with metadata dict containing arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
            metadata={"mel": np.zeros((3, 32), dtype=np.float32)},
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        assert (output_dir / "metadata.mel.bin").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert "metadata.mel" in manifest["leaves"]


def test_write_audiotree_nested_metadata():
    """AudioTree with nested metadata dicts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 50), dtype=np.float32),
            sample_rate=44100,
            metadata={
                "features": {
                    "mel": np.zeros((2, 16), dtype=np.float32),
                    "mfcc": np.zeros((2, 8), dtype=np.float32),
                }
            },
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        assert (output_dir / "metadata.features.mel.bin").exists()
        assert (output_dir / "metadata.features.mfcc.bin").exists()


def test_write_dict_of_audiotrees():
    """Dict of AudioTrees produces prefixed leaf files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        dry = AudioTree(
            audio_data=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )
        wet = AudioTree(
            audio_data=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"dry": dry, "wet": wet})

        assert (output_dir / "dry.audio_data.bin").exists()
        assert (output_dir / "wet.audio_data.bin").exists()


def test_write_plain_dict():
    """Plain dict of arrays (no AudioTree)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        data = {
            "x": np.zeros((5, 10), dtype=np.float32),
            "y": np.arange(5, dtype=np.int32),
        }

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(data)

        assert (output_dir / "x.bin").exists()
        assert (output_dir / "y.bin").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["structure"]["type"] == "dict"


def test_none_fields_excluded():
    """None optional fields produce no .bin files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 1, 50), dtype=np.float32),
            sample_rate=44100,
            # pitch, velocity, etc. are all None
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        assert not (output_dir / "pitch.bin").exists()
        assert not (output_dir / "velocity.bin").exists()
        assert not (output_dir / "codes.bin").exists()


def test_multiple_writes():
    """Two write() calls accumulate correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree1 = AudioTree(
            audio_data=np.ones((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )
        tree2 = AudioTree(
            audio_data=np.ones((2, 1, 10), dtype=np.float32) * 2,
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree1)
            w.write(tree2)

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == 5


def test_shape_validation():
    """Second write with wrong shape raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree1 = AudioTree(
            audio_data=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )
        tree2 = AudioTree(
            audio_data=np.zeros((2, 2, 200), dtype=np.float32),  # wrong samples dim
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree1)
            with pytest.raises(ValueError, match="Shape mismatch"):
                w.write(tree2)


def test_overflow_detection():
    """Writing more than expected_samples raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            with pytest.raises(ValueError, match="exceed expected_samples"):
                w.write(tree)


def test_flush():
    """Data is on disk after flush()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.array([[[1.0, 2.0]]], dtype=np.float32)  # (1, 1, 2)
        tree = AudioTree(audio_data=audio, sample_rate=44100)

        writer = TreeWriter(output_dir, expected_samples=5)
        writer.open()
        writer.write(tree)
        writer.flush()

        mm = np.memmap(
            output_dir / "audio_data.bin",
            dtype=np.float32,
            mode="r",
            shape=(5, 1, 2),
        )
        np.testing.assert_array_equal(mm[0], audio[0])
        del mm

        writer.close()


def test_context_manager():
    """Context manager opens and closes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with TreeWriter(output_dir, expected_samples=3) as w:
            tree = AudioTree(
                audio_data=np.zeros((3, 1, 10), dtype=np.float32),
                sample_rate=44100,
            )
            w.write(tree)

        assert (output_dir / "manifest.json").exists()


def test_get_stats():
    """get_stats returns correct counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with TreeWriter(output_dir, expected_samples=10) as w:
            tree = AudioTree(
                audio_data=np.zeros((3, 1, 10), dtype=np.float32),
                sample_rate=44100,
            )
            w.write(tree)

            stats = w.get_stats()
            assert stats["samples_written"] == 3
            assert stats["expected_samples"] == 10
            assert stats["is_open"] is True
            assert "audio_data" in stats["leaves"]


def test_different_dtypes():
    """Arrays with different dtypes are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        data = {
            "float32": np.zeros((2, 3), dtype=np.float32),
            "float16": np.zeros((2, 3), dtype=np.float16),
            "int32": np.zeros((2,), dtype=np.int32),
        }

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(data)

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["leaves"]["float32"]["dtype"] == "float32"
        assert manifest["leaves"]["float16"]["dtype"] == "float16"
        assert manifest["leaves"]["int32"]["dtype"] == "int32"


def test_empty_metadata():
    """AudioTree with empty metadata dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={},
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        structure = manifest["structure"]
        assert "metadata" in structure["children"]
        assert structure["children"]["metadata"]["type"] == "dict"
        assert structure["children"]["metadata"]["children"] == {}


def test_user_metadata_in_manifest():
    """User metadata is stored in manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with TreeWriter(
            output_dir,
            expected_samples=2,
            metadata={"description": "test dataset", "version": 1},
        ) as w:
            w.write({"x": np.zeros((2,), dtype=np.float32)})

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["metadata"]["description"] == "test dataset"
        assert manifest["metadata"]["version"] == 1


def test_writer_not_open_error():
    """Writing without open() raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(Path(tmpdir), expected_samples=5)

        with pytest.raises(RuntimeError, match="not open"):
            writer.write({"x": np.zeros((2,), dtype=np.float32)})
