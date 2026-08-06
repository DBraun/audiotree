"""Tests for TreeWriter."""

import importlib.util
import json
import os
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["format"] == "audiotree-tree"
        assert manifest["format_version"] == [1, 0]
        assert manifest["producer"].startswith("audiotree ")
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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
    """on_overflow='trim' trims a batch to fit -- and says so."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.arange(30, dtype=np.float32).reshape(3, 1, 10)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=2, on_overflow="trim") as w:
            with pytest.warns(UserWarning, match="dropping 1 of 3 offered samples"):
                n = w.write(tree)
            assert n == 2  # trimmed from 3 to 2

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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
    """Writing to a full 'trim' writer returns 0 -- with a warning, not silence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2, on_overflow="trim") as w:
            w.write(tree)
            with pytest.warns(UserWarning, match="dropping 2 of 2 offered samples"):
                n = w.write(tree)  # already full
            assert n == 0


def test_trim_warning_names_the_counts():
    """The trim warning must carry written/allocated/offered, not just 'overflow'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=5, on_overflow="trim") as w:
            w.write({"x": _f32(1.0, (4, 3))})
            with pytest.warns(UserWarning) as record:
                w.write({"x": _f32(2.0, (4, 3))})
        message = str(record[0].message)
        assert "4 written" in message
        assert "5 allocated" in message
        assert "expected_samples=5" in message
        assert "4 offered" in message


def test_overflow_error_raises_with_counts():
    """on_overflow='error' refuses the batch and names every count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=5, on_overflow="error") as w:
            w.write({"x": _f32(1.0, (4, 3))})
            with pytest.raises(ValueError) as excinfo:
                w.write({"x": _f32(2.0, (4, 3))})
        message = str(excinfo.value)
        assert "4 written" in message
        assert "5 allocated" in message
        assert "expected_samples=5" in message
        assert "4 offered" in message


def test_overflow_error_leaves_the_written_prefix_readable():
    """A refused batch must not damage what was already written."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(tmpdir, expected_samples=3, on_overflow="error")
        writer.open()
        writer.write({"x": _f32(1.0, (2, 3))})
        with pytest.raises(ValueError):
            writer.write({"x": _f32(2.0, (2, 3))})
        writer.close()

        source = TreeDataSource(tmpdir)
        assert len(source) == 2
        np.testing.assert_array_equal(source[1]["x"].ravel(), [1.0, 1.0, 1.0])


def test_overflow_grows_by_default():
    """The default policy keeps every offered sample, reallocating as needed.

    ``expected_samples`` is documented as an allocation hint, so an underestimate
    must not cost data: it used to be trimmed away without a word.
    """
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as w:  # wildly under-allocated
            for i in range(3):
                assert w.write({"x": _f32(float(i), (4, 3))}) == 4

        manifest = json.loads(
            (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_samples"] == 12
        assert manifest["expected_samples"] == 2  # the hint, recorded as given

        # close() truncates the growth slack away: the file is exactly 12 samples.
        assert (Path(tmpdir) / "x.bin").stat().st_size == 12 * 3 * 4

        source = TreeDataSource(tmpdir)
        assert len(source) == 12
        for i in range(3):
            for j in range(4):
                np.testing.assert_array_equal(
                    source[i * 4 + j]["x"].ravel(), [float(i)] * 3
                )


def test_grow_preserves_data_across_several_leaves():
    """Reallocating must not shuffle, zero, or drop any leaf's existing bytes."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=1) as w:
            for i in range(4):
                w.write(
                    {
                        "a": np.full((3, 2), i, dtype=np.int32),
                        "b": AudioTree(
                            waveform=np.full((3, 1, 5), float(i), dtype=np.float32),
                            sample_rate=16000,
                        ),
                    }
                )

        source = TreeDataSource(tmpdir)
        assert len(source) == 12
        for i in range(4):
            sample = source[i * 3]
            np.testing.assert_array_equal(sample["a"].ravel(), [i, i])
            np.testing.assert_array_equal(sample["b"].waveform.ravel(), [float(i)] * 5)
        assert source[0]["b"].sample_rate == 16000


def test_grow_from_a_zero_sample_allocation():
    """expected_samples=0 is a legal hint when the writer may grow."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=0) as w:
            assert w.write({"x": _f32(7.0, (2, 3))}) == 2

        source = TreeDataSource(tmpdir)
        assert len(source) == 2
        np.testing.assert_array_equal(source[1]["x"].ravel(), [7.0] * 3)


@requires_bagz
def test_grow_keeps_string_leaves_aligned():
    """Growing the memmaps must not desynchronize the append-only bagz files."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as w:
            w.write({"s": ["a", "b", "c"], "x": _f32(1.0, (3, 2))})
            w.write({"s": ["d", "e", "f"], "x": _f32(2.0, (3, 2))})

        source = TreeDataSource(tmpdir)
        assert len(source) == 6
        assert [source[i]["s"] for i in range(6)] == list("abcdef")
        np.testing.assert_array_equal(source[5]["x"].ravel(), [2.0, 2.0])


def test_grown_dataset_is_readable_before_close():
    """A kill between a grow and close() must still leave a readable prefix."""
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(tmpdir, expected_samples=2, manifest_interval=1e-9)
        writer.open()
        for i in range(3):
            writer.write({"x": _f32(float(i), (4, 3))})
        # No flush(), no close() -- exactly what a SIGKILL would leave behind.
        assert len(TreeDataSource(tmpdir)) == 12
        np.testing.assert_array_equal(
            TreeDataSource(tmpdir)[11]["x"].ravel(), [2.0] * 3
        )
        writer.close()


def test_failed_grow_keeps_the_prefix_and_refuses_further_writes(monkeypatch):
    """A grow that dies half-way must not silently write partial batches after.

    Once a leaf's mapping has been released the writer can no longer store that
    leaf, so continuing would drop it from every subsequent batch.
    """
    from audiotree.sources import TreeDataSource
    from audiotree import tree_writer as tw

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(tmpdir, expected_samples=2)
        writer.open()
        writer.write({"x": _f32(1.0, (2, 3)), "y": _f32(2.0, (2, 3))})

        real_truncate = os.truncate
        calls = {"n": 0}

        def flaky_truncate(path, length):
            calls["n"] += 1
            if calls["n"] == 2:  # fail while growing the second leaf
                raise OSError("No space left on device")
            return real_truncate(path, length)

        monkeypatch.setattr(tw.os, "truncate", flaky_truncate)
        with pytest.raises(OSError, match="No space left"):
            writer.write({"x": _f32(3.0, (2, 3)), "y": _f32(4.0, (2, 3))})
        with pytest.raises(RuntimeError, match="cannot accept further writes"):
            writer.write({"x": _f32(5.0, (2, 3)), "y": _f32(6.0, (2, 3))})

        monkeypatch.setattr(tw.os, "truncate", real_truncate)
        writer.close()

        source = TreeDataSource(tmpdir)
        assert len(source) == 2
        np.testing.assert_array_equal(source[1]["x"].ravel(), [1.0] * 3)
        np.testing.assert_array_equal(source[1]["y"].ravel(), [2.0] * 3)


def test_get_stats_reports_the_grown_allocation():
    """get_stats separates the user's hint from the live allocation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as w:
            w.write({"x": _f32(1.0, (5, 3))})
            stats = w.get_stats()
        assert stats["samples_written"] == 5
        assert stats["expected_samples"] == 2
        assert stats["allocated_samples"] >= 5


def test_unknown_on_overflow_is_rejected():
    """A typo'd policy must fail at construction, not silently trim."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="on_overflow must be"):
            TreeWriter(tmpdir, expected_samples=2, on_overflow="truncate")


def test_undershoot_truncates_memmaps():
    """When fewer samples are written than expected, memmaps are truncated on close."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.arange(20, dtype=np.float32).reshape(2, 1, 10)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=100) as w:
            w.write(tree)  # write only 2 of 100

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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

        with open(output_dir / "manifest.json", encoding="utf-8") as f:
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


def test_manifest_is_readable_before_close():
    """A crash mid-render must leave a readable prefix, not an orphaned directory.

    manifest.json used to be written only at the bottom of close(), so a
    multi-hour pre-render killed at 99% produced valid .bin files that
    TreeDataSource refused to open at all.
    """
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(tmpdir, expected_samples=100)
        writer.open()
        for i in range(5):
            writer.write({"x": np.full((10, 2), float(i), dtype=np.float32)})
        writer.flush()

        # Simulate a reader arriving while the writer is still alive.
        manifest = json.loads(
            (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_samples"] == 50
        source = TreeDataSource(tmpdir)
        assert len(source) == 50
        np.testing.assert_array_equal(source[45]["x"].ravel(), [4.0, 4.0])

        writer.close()
        assert len(TreeDataSource(tmpdir)) == 50


def test_writer_refuses_to_clobber_an_existing_dataset():
    """Pointing a writer at a populated directory must raise, not destroy it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as writer:
            writer.write({"x": np.ones((2, 3), dtype=np.float32)})

        with pytest.raises(FileExistsError, match="already contains a dataset"):
            TreeWriter(tmpdir, expected_samples=2).open()

        # Opting in is allowed.
        TreeWriter(tmpdir, expected_samples=2, exist_ok=True).open().close()


def test_non_native_endian_leaf_round_trips():
    """A big-endian leaf is normalized on write, not recorded as an unreadable dtype.

    ``str(np.dtype('>f4'))`` is ``'>f4'``, which TreeDataSource's dtype allowlist
    refuses -- so the dataset was writable but unreadable. The bytes are
    byte-swapped to native order and the manifest names the native dtype.
    """
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        values = np.arange(6, dtype=">f4").reshape(3, 2)
        with TreeWriter(tmpdir, expected_samples=3) as writer:
            writer.write({"x": values})

        manifest = json.loads(
            (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["leaves"]["x"]["dtype"] == "float32"

        source = TreeDataSource(tmpdir)
        np.testing.assert_array_equal(source[1]["x"].ravel(), [2.0, 3.0])


def test_unreadable_dtype_fails_on_the_first_write():
    """A dtype the reader rejects must fail immediately, not after the render."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as writer:
            with pytest.raises(ValueError, match="cannot read back"):
                writer.write({"x": np.array(["ab", "cd"], dtype="<U2")})

        # Nothing half-written was left behind for that leaf.
        assert not (Path(tmpdir) / "x.bin").exists()


def test_colliding_leaf_filenames_raise():
    """Two leaves must never share one .bin file.

    Filenames are dot-joined leaf paths, so ``{"a.b": ...}`` and
    ``{"a": {"b": ...}}`` both mapped to ``a.b.bin``: both read back as
    whichever was written last, and the other leaf vanished.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ones = np.ones((2, 3), dtype=np.float32)
        with TreeWriter(tmpdir, expected_samples=2) as writer:
            with pytest.raises(ValueError, match="both map to the file 'a.b.bin'"):
                writer.write({"a.b": ones * 1.0, "a": {"b": ones * 2.0}})


def test_leaf_name_cannot_escape_the_dataset_directory():
    """A leaf key with '..' must not write outside the dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "dataset").mkdir()
        with TreeWriter(root / "dataset", expected_samples=2) as writer:
            with pytest.raises(ValueError, match=r"\.\."):
                writer.write({"../escaped": np.ones((2, 3), dtype=np.float32)})

        assert not (root / "escaped.bin").exists()


def test_leaf_name_cannot_name_a_subdirectory():
    """A leaf key with a path separator fails clearly rather than at memmap time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with TreeWriter(tmpdir, expected_samples=2) as writer:
            with pytest.raises(ValueError, match="path separators"):
                writer.write({"sub/x": np.ones((2, 3), dtype=np.float32)})


def test_manifest_sample_count_is_refreshed_while_writing():
    """A hard kill must leave a truthful num_samples, without any flush().

    The manifest was stamped once with num_samples=0 and refreshed only by
    flush()/close(), so a SIGKILLed pre-render read back as an empty dataset.
    """
    from audiotree.sources import TreeDataSource

    with tempfile.TemporaryDirectory() as tmpdir:
        # manifest_interval=0.0 disables the timer; a tiny one refreshes always.
        writer = TreeWriter(tmpdir, expected_samples=100, manifest_interval=1e-9)
        writer.open()
        for i in range(5):
            writer.write({"x": np.full((10, 2), float(i), dtype=np.float32)})

        # No flush(), no close() -- exactly what a SIGKILL would leave behind.
        manifest = json.loads(
            (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_samples"] == 50
        assert len(TreeDataSource(tmpdir)) == 50
        np.testing.assert_array_equal(
            TreeDataSource(tmpdir)[45]["x"].ravel(), [4.0] * 2
        )

        writer.close()


def test_close_is_terminal():
    """Reopening a closed writer must raise, not accept writes it will drop.

    After close() the memmaps are gone and the files truncated, so a second
    open() accepted write() calls, dropped the data, and grew the file back with
    zeros that the manifest presented as real samples.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = TreeWriter(tmpdir, expected_samples=4)
        writer.open()
        writer.write({"x": np.ones((2, 3), dtype=np.float32)})
        writer.close()

        with pytest.raises(RuntimeError, match="cannot be reopened"):
            writer.open()
        with pytest.raises(RuntimeError, match="closed"):
            writer.write({"x": np.ones((2, 3), dtype=np.float32)})

        manifest = json.loads(
            (Path(tmpdir) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_samples"] == 2
        assert (Path(tmpdir) / "x.bin").stat().st_size == 2 * 3 * 4
