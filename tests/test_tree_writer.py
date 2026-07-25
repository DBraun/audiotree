"""Tests for TreeWriter."""

import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.tree_writer import TreeWriter

requires_bagz = pytest.mark.skipif(
    importlib.util.find_spec("bagz") is None,
    reason="bagz not installed (Linux-only wheels)",
)


def test_basic_write_audiotree():
    """Write a simple AudioTree, verify files and manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(5, 2, 100).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        assert (output_dir / "waveform.bin").exists()
        assert (output_dir / "manifest.json").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["version"] == "2.0"
        assert manifest["num_samples"] == 5
        assert "waveform" in manifest["leaves"]


def test_manifest_structure_audiotree():
    """Verify structure JSON for an AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((3, 1, 50), dtype=np.float32),
            sample_rate=48000,
            lufs=np.zeros(3, dtype=np.float32),
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        structure = manifest["structure"]
        assert structure["type"] == "AudioTree"
        assert structure["sample_rate"] == 48000
        assert "waveform" in structure["children"]
        assert "lufs" in structure["children"]
        # pitch etc. should be absent (None)
        assert "pitch" not in structure["children"]


def test_write_audiotree_with_metadata():
    """AudioTree with metadata dict containing arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((3, 2, 100), dtype=np.float32),
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
            waveform=np.zeros((2, 1, 50), dtype=np.float32),
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
            waveform=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )
        wet = AudioTree(
            waveform=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"dry": dry, "wet": wet})

        assert (output_dir / "dry.waveform.bin").exists()
        assert (output_dir / "wet.waveform.bin").exists()


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
            waveform=np.zeros((3, 1, 50), dtype=np.float32),
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
            waveform=np.ones((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )
        tree2 = AudioTree(
            waveform=np.ones((2, 1, 10), dtype=np.float32) * 2,
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
            waveform=np.zeros((3, 2, 100), dtype=np.float32),
            sample_rate=44100,
        )
        tree2 = AudioTree(
            waveform=np.zeros((2, 2, 200), dtype=np.float32),  # wrong samples dim
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree1)
            with pytest.raises(ValueError, match="Shape mismatch"):
                w.write(tree2)


def test_overflow_trimming():
    """Batch exceeding expected_samples is trimmed to fit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.arange(30, dtype=np.float32).reshape(3, 1, 10)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=2) as w:
            n = w.write(tree)
            assert n == 2  # trimmed from 3 to 2

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == 2

        # Memmap should contain only the first 2 samples
        mm = np.memmap(
            output_dir / "waveform.bin", dtype=np.float32, mode="r", shape=(2, 1, 10)
        )
        np.testing.assert_array_equal(mm[0], audio[0])
        np.testing.assert_array_equal(mm[1], audio[1])
        del mm


def test_overflow_returns_zero_when_full():
    """Writing after expected_samples is reached returns 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)
            n = w.write(tree)  # already full
            assert n == 0


def test_undershoot_truncates_memmaps():
    """When fewer samples are written than expected, memmaps are truncated on close."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.arange(20, dtype=np.float32).reshape(2, 1, 10)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=100) as w:
            w.write(tree)  # write only 2 of 100

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["num_samples"] == 2
        assert manifest["expected_samples"] == 100

        # Memmap file should be truncated to actual size (2 samples)
        filepath = output_dir / "waveform.bin"
        expected_bytes = 2 * 1 * 10 * 4  # 2 samples * shape * float32
        assert filepath.stat().st_size == expected_bytes

        mm = np.memmap(filepath, dtype=np.float32, mode="r", shape=(2, 1, 10))
        np.testing.assert_array_equal(mm[0], audio[0])
        np.testing.assert_array_equal(mm[1], audio[1])
        del mm


def test_flush():
    """Data is on disk after flush()."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.array([[[1.0, 2.0]]], dtype=np.float32)  # (1, 1, 2)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        writer = TreeWriter(output_dir, expected_samples=5)
        writer.open()
        writer.write(tree)
        writer.flush()

        mm = np.memmap(
            output_dir / "waveform.bin",
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
                waveform=np.zeros((3, 1, 10), dtype=np.float32),
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
                waveform=np.zeros((3, 1, 10), dtype=np.float32),
                sample_rate=44100,
            )
            w.write(tree)

            stats = w.get_stats()
            assert stats["samples_written"] == 3
            assert stats["expected_samples"] == 10
            assert stats["is_open"] is True
            assert "waveform" in stats["leaves"]


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
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
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


# === String leaf tests ===


@requires_bagz
def test_write_string_list_with_audiotree():
    """Write a dict with List[str] and AudioTree leaves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        pytree = {
            "labels": ["cat", "dog", "bird"],
            "audio": AudioTree(
                waveform=np.zeros((3, 1, 100), dtype=np.float32),
                sample_rate=44100,
            ),
        }

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(pytree)

        assert (output_dir / "labels.bagz").exists()
        assert (output_dir / "audio.waveform.bin").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert "labels" in manifest["string_leaves"]
        assert manifest["string_leaves"]["labels"]["file"] == "labels.bagz"
        assert manifest["num_samples"] == 3


@requires_bagz
def test_write_multiple_string_leaves():
    """Multiple string leaves each get their own bagz file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        pytree = {
            "labels": ["cat", "dog"],
            "source": ["train", "val"],
            "x": np.zeros((2, 3), dtype=np.float32),
        }

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(pytree)

        assert (output_dir / "labels.bagz").exists()
        assert (output_dir / "source.bagz").exists()

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert "labels" in manifest["string_leaves"]
        assert "source" in manifest["string_leaves"]


@requires_bagz
def test_write_string_batch_size_mismatch():
    """String leaf with wrong batch size raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        pytree = {
            "labels": ["cat", "dog"],  # batch 2
            "x": np.zeros((3, 5), dtype=np.float32),  # batch 3
        }

        with TreeWriter(output_dir, expected_samples=3) as w:
            with pytest.raises(ValueError, match="Inconsistent batch sizes"):
                w.write(pytree)


@requires_bagz
def test_get_stats_with_strings():
    """get_stats includes string leaf names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(
                {
                    "labels": ["a", "b", "c"],
                    "x": np.zeros((3,), dtype=np.float32),
                }
            )
            stats = w.get_stats()
            assert "labels" in stats["string_leaves"]
            assert "x" in stats["leaves"]


def _f32(value, shape=(2, 3)):
    return np.full(shape, value, dtype=np.float32)


def test_write_rejects_renamed_leaf():
    """A renamed leaf must raise, not write into the previous leaf's file.

    Leaf extraction is positional and only the leaf *count* was checked, so
    writing {"a", "c"} after {"a", "b"} silently stored c's data in b.bin.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=4) as writer:
            writer.write({"a": _f32(1.0), "b": _f32(2.0)})
            with pytest.raises(ValueError, match="must match the first write"):
                writer.write({"a": _f32(3.0), "c": _f32(4.0)})


@pytest.mark.parametrize(
    "second,match",
    [
        ({"a": _f32(1.0), "b": _f32(2.0)}, "unexpected leaves"),
        ({}, "missing leaves"),
    ],
)
def test_write_rejects_added_or_missing_leaves(second, match):
    """Adding or dropping a leaf mid-dataset is rejected with a named diff."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=4) as writer:
            writer.write({"a": _f32(1.0)})
            with pytest.raises(ValueError, match=match):
                writer.write(second)


def test_write_rejects_changed_sample_rate():
    """A later AudioTree at a different rate must raise rather than be mislabelled.

    ``sample_rate`` is captured from the first write and recorded once in the
    manifest, so a second rate would silently claim to be the first.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=4) as writer:
            writer.write(AudioTree.create(np.zeros((1, 1, 8), np.float32), 16000))
            with pytest.raises(ValueError, match="must match the first write"):
                writer.write(AudioTree.create(np.zeros((1, 1, 8), np.float32), 44100))


def test_write_accepts_identical_structure():
    """The validation must not reject a legitimately unchanged structure."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=4) as writer:
            writer.write({"a": _f32(1.0)})
            writer.write({"a": _f32(9.0)})
        source = TreeDataSource(tmpdir)
        assert len(source) == 4
        np.testing.assert_array_equal(source[2]["a"].ravel(), [9.0, 9.0, 9.0])
