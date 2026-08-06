"""Tests for TreeDataSource."""

import importlib.util
import json
import locale
import multiprocessing
import os
import pickle
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree
from audiotree.sources.tree import TreeDataSource
from audiotree.tree_writer import TreeWriter

# === Round-trip tests ===


requires_bagz = pytest.mark.skipif(
    importlib.util.find_spec("bagz") is None,
    reason="bagz not installed (Linux-only wheels)",
)


def test_round_trip_audiotree():
    """Write and read back a single AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(5, 2, 100).astype(np.float32)
        loudness = np.random.randn(5).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100, lufs=loudness)

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)
        assert len(source) == 5

        for i in range(5):
            sample = source[i]
            assert isinstance(sample, AudioTree)
            assert sample.sample_rate == 44100
            np.testing.assert_array_almost_equal(
                sample.waveform[0], audio[i], decimal=5
            )
            np.testing.assert_array_almost_equal(sample.lufs[0], loudness[i], decimal=5)
            assert sample.pitch is None


def test_round_trip_audiotree_with_metadata():
    """AudioTree with metadata arrays round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(3, 1, 50).astype(np.float32)
        mel = np.random.randn(3, 32).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100, metadata={"mel": mel})

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)

        for i in range(3):
            sample = source[i]
            assert isinstance(sample, AudioTree)
            np.testing.assert_array_almost_equal(
                sample.waveform[0], audio[i], decimal=5
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
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={"features": {"mel": mel, "mfcc": mfcc}},
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)
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
        dry = AudioTree(waveform=dry_audio, sample_rate=44100)
        wet = AudioTree(waveform=wet_audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"dry": dry, "wet": wet})

        source = TreeDataSource(output_dir)
        assert len(source) == 3

        sample = source[0]
        assert isinstance(sample, dict)
        assert "dry" in sample and "wet" in sample
        assert isinstance(sample["dry"], AudioTree)
        assert isinstance(sample["wet"], AudioTree)
        np.testing.assert_array_almost_equal(
            sample["dry"].waveform[0], dry_audio[0], decimal=5
        )
        np.testing.assert_array_almost_equal(
            sample["wet"].waveform[0], wet_audio[0], decimal=5
        )


def test_round_trip_plain_dict():
    """Plain dict of arrays round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        x = np.random.randn(5, 10).astype(np.float32)
        y = np.arange(5, dtype=np.int32)

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write({"x": x, "y": y})

        source = TreeDataSource(output_dir)
        assert len(source) == 5

        for i in range(5):
            sample = source[i]
            assert isinstance(sample, dict)
            np.testing.assert_array_almost_equal(sample["x"][0], x[i], decimal=5)
            assert sample["y"][0] == y[i]


def test_round_trip_multiple_writes():
    """Data written in multiple batches is read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio1 = np.ones((3, 1, 10), dtype=np.float32)
        audio2 = np.ones((2, 1, 10), dtype=np.float32) * 2

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(AudioTree(waveform=audio1, sample_rate=44100))
            w.write(AudioTree(waveform=audio2, sample_rate=44100))

        source = TreeDataSource(output_dir)
        assert len(source) == 5

        # First batch
        for i in range(3):
            sample = source[i]
            np.testing.assert_array_almost_equal(
                sample.waveform[0], audio1[i], decimal=5
            )

        # Second batch
        for i in range(2):
            sample = source[3 + i]
            np.testing.assert_array_almost_equal(
                sample.waveform[0], audio2[i], decimal=5
            )


# === Missing manifest ===


def test_missing_manifest():
    """Constructor raises FileNotFoundError for missing manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            TreeDataSource(tmpdir)


# === Error handling ===


def test_index_out_of_range():
    """Out-of-range index raises IndexError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)

        with pytest.raises(IndexError):
            _ = source[10]

        with pytest.raises(IndexError):
            _ = source[-1]


def test_rejects_headerless_manifest():
    """A pre-1.0 manifest (no format header) is refused with actionable advice."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        manifest = {"version": "2.0", "num_samples": 5, "leaves": {}}
        with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="re-render the dataset"):
            TreeDataSource(output_dir)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"format": "audiotree-manifest"}, "expected a TreeWriter dataset"),
        ({"format_version": [2, 0]}, "this audiotree reads 1.x"),
        ({"min_reader_version": [1, 7]}, "requires a reader of at least 1.7"),
        ({"format_version": "1.0"}, r"must be a \[major, minor\] pair"),
    ],
)
def test_format_header_dispatch(override, match):
    """Wrong format, future major, and future min-reader are each refused."""
    import json

    import numpy as np

    from audiotree import AudioTree, TreeWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(str(output_dir), expected_samples=1) as writer:
            writer.write(AudioTree.create(np.zeros((1, 1, 8), np.float32), 16000))

        path = output_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(override)
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=match):
            TreeDataSource(output_dir)


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

        source = TreeDataSource(output_dir)
        meta = source.get_metadata()
        assert meta["description"] == "test"
        assert meta["version"] == 1


# === Empty/minimal structures ===


def test_empty_metadata_round_trip():
    """AudioTree with empty metadata round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={},
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert isinstance(sample, AudioTree)
        assert sample.metadata == {}


def test_none_fields_default():
    """Absent optional fields default to None in reconstructed AudioTree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert sample.pitch is None
        assert sample.velocity is None
        assert sample.lufs is None
        assert sample.codes is None
        assert sample.latents is None


# === String leaf round-trip tests ===


@requires_bagz
def test_round_trip_string_list_with_audiotree():
    """List[str] + AudioTree round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        strings = ["hello", "world", "foo"]
        audio = np.random.randn(3, 1, 100).astype(np.float32)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(
                {
                    "strings": strings,
                    "wet": AudioTree(waveform=audio, sample_rate=44100),
                }
            )

        source = TreeDataSource(output_dir)
        assert len(source) == 3

        for i in range(3):
            sample = source[i]
            assert isinstance(sample, dict)
            assert sample["strings"] == strings[i]
            assert isinstance(sample["wet"], AudioTree)
            np.testing.assert_array_almost_equal(
                sample["wet"].waveform[0], audio[i], decimal=5
            )


@requires_bagz
def test_round_trip_multiple_string_leaves():
    """Multiple string leaves round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(
                {
                    "labels": ["cat", "dog"],
                    "sources": ["train", "val"],
                    "x": np.zeros((2, 3), dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir)
        s0 = source[0]
        assert s0["labels"] == "cat"
        assert s0["sources"] == "train"

        s1 = source[1]
        assert s1["labels"] == "dog"
        assert s1["sources"] == "val"


@requires_bagz
def test_round_trip_single_string():
    """A single str leaf (batch size 1) round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=1) as w:
            w.write(
                {
                    "label": "cat",
                    "x": np.zeros((1, 3), dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert sample["label"] == "cat"


@requires_bagz
def test_round_trip_long_strings():
    """Strings longer than 256 chars survive round-trip (no truncation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        long_str = "x" * 1000
        with TreeWriter(output_dir, expected_samples=1) as w:
            w.write({"s": [long_str], "x": np.zeros((1,), dtype=np.float32)})

        source = TreeDataSource(output_dir)
        assert source[0]["s"] == long_str


@requires_bagz
def test_round_trip_empty_string():
    """Empty strings survive round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(
                {
                    "s": ["", "hello"],
                    "x": np.zeros((2,), dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir)
        assert source[0]["s"] == ""
        assert source[1]["s"] == "hello"


@requires_bagz
def test_round_trip_unicode_strings():
    """Unicode strings survive round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        strings = ["cafe\u0301", "\u4f60\u597d", "\U0001f3b5"]
        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"s": strings, "x": np.zeros((3,), dtype=np.float32)})

        source = TreeDataSource(output_dir)
        for i, expected in enumerate(strings):
            assert source[i]["s"] == expected


@requires_bagz
def test_round_trip_strings_multiple_writes():
    """String leaves written across multiple batches read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(
                {
                    "s": ["a", "b", "c"],
                    "x": np.zeros((3,), dtype=np.float32),
                }
            )
            w.write(
                {
                    "s": ["d", "e"],
                    "x": np.ones((2,), dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir)
        assert len(source) == 5
        assert source[0]["s"] == "a"
        assert source[2]["s"] == "c"
        assert source[3]["s"] == "d"
        assert source[4]["s"] == "e"


@requires_bagz
def test_batch_with_strings():
    """AudioTree.batch correctly batches string leaves into List[str]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=4) as w:
            w.write(
                {
                    "strings": ["a", "b", "c", "d"],
                    "wet": AudioTree(
                        waveform=np.arange(4 * 100, dtype=np.float32).reshape(
                            4, 1, 100
                        ),
                        sample_rate=44100,
                    ),
                }
            )

        source = TreeDataSource(output_dir)
        items = [source[i] for i in range(4)]
        batched = AudioTree.batch(items)

        assert batched["strings"] == ["a", "b", "c", "d"]
        assert isinstance(batched["wet"], AudioTree)
        assert batched["wet"].waveform.shape == (4, 1, 100)


def test_backward_compat_no_string_leaves():
    """Old manifests without string_leaves key still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        # Simulate old manifest by removing string_leaves key
        import json

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.pop("string_leaves", None)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert isinstance(sample, AudioTree)


# === Grain integration ===


def test_grain_protocol():
    """TreeDataSource satisfies RandomAccessDataSource protocol."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            waveform=np.zeros((5, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)

        # len() and __getitem__ are required
        assert len(source) == 5
        sample = source[0]
        assert sample is not None

        # Iteration via indexing
        items = [source[i] for i in range(5)]
        assert len(items) == 5


# === exclude_prefixes ===


def test_exclude_audio_data():
    """Excluding wet.waveform gives None audio but keeps dry and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        dry_audio = np.random.randn(3, 2, 100).astype(np.float32)
        wet_audio = np.random.randn(3, 2, 100).astype(np.float32)
        mel = np.random.randn(3, 16).astype(np.float32)
        dry = AudioTree(waveform=dry_audio, sample_rate=44100)
        wet = AudioTree(waveform=wet_audio, sample_rate=44100, metadata={"mel": mel})

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({"dry": dry, "wet": wet})

        source = TreeDataSource(output_dir, exclude_prefixes=["wet.waveform"])
        sample = source[0]

        # wet.waveform excluded -> None
        assert sample["wet"].waveform is None
        # wet metadata still present
        np.testing.assert_array_almost_equal(
            sample["wet"].metadata["mel"][0], mel[0], decimal=5
        )
        # dry.waveform NOT excluded
        np.testing.assert_array_almost_equal(
            sample["dry"].waveform[0], dry_audio[0], decimal=5
        )


def test_exclude_metadata_field():
    """Excluding a metadata field removes its key from the metadata dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        mel = np.random.randn(3, 16).astype(np.float32)
        mfcc = np.random.randn(3, 8).astype(np.float32)
        tree = AudioTree(
            waveform=np.zeros((3, 1, 10), dtype=np.float32),
            sample_rate=44100,
            metadata={"mel": mel, "mfcc": mfcc},
        )

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(output_dir, exclude_prefixes=["metadata.mel"])
        sample = source[0]

        assert "mel" not in sample.metadata
        assert "mfcc" in sample.metadata
        np.testing.assert_array_almost_equal(
            sample.metadata["mfcc"][0], mfcc[0], decimal=5
        )


@requires_bagz
def test_exclude_string_leaf():
    """Excluding a string leaf removes it from the output dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(
                {
                    "label": ["cat", "dog"],
                    "x": np.zeros((2, 3), dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir, exclude_prefixes=["label"])
        sample = source[0]

        assert "label" not in sample
        assert "x" in sample


def test_exclude_prefix_with_subtree():
    """Prefix 'dry' excludes all dry.* leaves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        dry = AudioTree(
            waveform=np.random.randn(2, 1, 50).astype(np.float32),
            sample_rate=44100,
            metadata={"mel": np.random.randn(2, 8).astype(np.float32)},
        )
        wet = AudioTree(
            waveform=np.random.randn(2, 1, 50).astype(np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write({"dry": dry, "wet": wet})

        source = TreeDataSource(output_dir, exclude_prefixes=["dry"])
        sample = source[0]

        # dry AudioTree is reconstructed but all its leaves are excluded
        assert sample["dry"].waveform is None
        assert sample["dry"].metadata == {}
        # wet is untouched
        assert sample["wet"].waveform is not None


def test_exclude_pickle_roundtrip():
    """exclude_prefixes survives pickle/unpickle (grain multiprocessing)."""
    import pickle

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(3, 1, 20).astype(np.float32)
        mel = np.random.randn(3, 8).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100, metadata={"mel": mel})

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(output_dir, exclude_prefixes=["waveform"])

        # Force data files open, then pickle/unpickle
        _ = source[0]
        restored = pickle.loads(pickle.dumps(source))

        sample = restored[0]
        assert sample.waveform is None
        assert "mel" in sample.metadata
        np.testing.assert_array_almost_equal(
            sample.metadata["mel"][0], mel[0], decimal=5
        )


def test_exclude_empty_default():
    """Empty exclude_prefixes behaves identically to no argument."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(2, 1, 10).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source_default = TreeDataSource(output_dir)
        source_empty = TreeDataSource(output_dir, exclude_prefixes=[])

        for i in range(2):
            s1 = source_default[i]
            s2 = source_empty[i]
            np.testing.assert_array_equal(s1.waveform, s2.waveform)


# === load_into_memory ===


def test_load_into_memory_matches_lazy():
    """load_into_memory=True returns identical results to lazy memmap access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(5, 2, 100).astype(np.float32)
        mel = np.random.randn(5, 16).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100, metadata={"mel": mel})

        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write(tree)

        lazy = TreeDataSource(output_dir)
        eager = TreeDataSource(output_dir, load_into_memory=True)

        for i in range(5):
            s_lazy = lazy[i]
            s_eager = eager[i]
            np.testing.assert_array_equal(s_lazy.waveform, s_eager.waveform)
            np.testing.assert_array_equal(
                s_lazy.metadata["mel"], s_eager.metadata["mel"]
            )


def test_load_into_memory_with_exclude():
    """load_into_memory combined with exclude_prefixes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(3, 1, 50).astype(np.float32)
        mel = np.random.randn(3, 8).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100, metadata={"mel": mel})

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(
            output_dir,
            exclude_prefixes=["waveform"],
            load_into_memory=True,
        )

        sample = source[0]
        assert sample.waveform is None
        np.testing.assert_array_almost_equal(
            sample.metadata["mel"][0], mel[0], decimal=5
        )


@requires_bagz
def test_load_into_memory_with_strings():
    """load_into_memory loads string leaves into a list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(
                {
                    "label": ["cat", "dog", "bird"],
                    "x": np.arange(3, dtype=np.float32),
                }
            )

        source = TreeDataSource(output_dir, load_into_memory=True)
        assert source[0]["label"] == "cat"
        assert source[1]["label"] == "dog"
        assert source[2]["label"] == "bird"


def test_load_into_memory_pickle_roundtrip():
    """In-memory data survives pickle for fork-based workers."""
    import pickle

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        audio = np.random.randn(3, 1, 20).astype(np.float32)
        tree = AudioTree(waveform=audio, sample_rate=44100)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write(tree)

        source = TreeDataSource(output_dir, load_into_memory=True)
        restored = pickle.loads(pickle.dumps(source))

        sample = restored[0]
        np.testing.assert_array_almost_equal(sample.waveform[0], audio[0], decimal=5)


def test_excluding_string_leaves_works_without_bagz(tmp_path, monkeypatch):
    """Excluding every string leaf must make a dataset readable without bagz.

    bagz publishes manylinux x86-64 wheels only, and `require_bagz` used to be
    called before the exclusion filter — so `exclude_prefixes=["caption"]` still
    raised ImportError and a Linux-written dataset could not be opened at all
    elsewhere, not even to read its waveforms.
    """
    import builtins

    import numpy as np

    from audiotree import AudioTree, TreeWriter
    from audiotree.sources import TreeDataSource

    pytest.importorskip("bagz", reason="need bagz to write the fixture")

    data_dir = tmp_path / "ds"
    with TreeWriter(str(data_dir), expected_samples=2) as writer:
        writer.write(
            {
                "audio": AudioTree.create(
                    np.zeros((2, 1, 16), dtype=np.float32), 16000
                ),
                "caption": ["a", "b"],
            }
        )

    # Simulate a platform where bagz cannot be imported.
    real_import = builtins.__import__

    def no_bagz(name, *args, **kwargs):
        if name == "bagz":
            raise ImportError("no bagz on this platform")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_bagz)

    # Without the exclusion, the string leaf still needs bagz.
    with pytest.raises(ImportError, match="audiotree\\[bagz\\]"):
        TreeDataSource(data_dir)[0]

    # Excluding it makes the rest of the dataset readable.
    source = TreeDataSource(data_dir, exclude_prefixes=["caption"])
    sample = source[0]
    assert sample["audio"].waveform.shape == (1, 1, 16)
    assert "caption" not in sample


def test_manifest_json_is_utf8_regardless_of_locale(tmp_path, monkeypatch):
    """``manifest.json`` is UTF-8 on every platform, not whatever the locale says.

    ``write_json_atomic`` and the two manifest readers used the default encoding,
    which is UTF-8 on Linux and macOS but cp1252 on Windows. A manifest carrying
    a non-ASCII character -- an accented leaf name, a CJK metadata key -- was
    therefore written on one platform and unreadable on another, and a
    Windows-written manifest was not valid UTF-8 JSON for anyone else. The
    Windows CI leg caught this on its first run.
    """
    from audiotree._fs import write_json_atomic

    payload = {"note": "Frédéric — 練習曲", "leaves": ["café.bin"]}
    path = tmp_path / "manifest.json"
    write_json_atomic(path, payload)

    # Bytes on disk are UTF-8 whatever the interpreter's locale encoding is.
    assert json.loads(path.read_bytes().decode("utf-8")) == payload

    # And the read side does not consult the locale either. cp1252 cannot decode
    # the UTF-8 encoding of "é", so a locale-dependent reader raises here.
    monkeypatch.setattr(
        locale, "getpreferredencoding", lambda do_setlocale=True: "cp1252"
    )
    with open(path, encoding="utf-8") as f:
        assert json.load(f) == payload


def _tiny_dataset(directory, num_samples=64):
    with TreeWriter(directory=directory, expected_samples=num_samples) as writer:
        for start in range(0, num_samples, 8):
            writer.write(
                {
                    "waveform": np.arange(8 * 1 * 16, dtype=np.float32).reshape(
                        8, 1, 16
                    )
                    + start,
                    "lufs": np.arange(start, start + 8, dtype=np.float32),
                }
            )
    return directory


def test_leaf_memmaps_are_held_open_and_reused(tmp_path):
    """One memmap per leaf per process, not one per access.

    Rebuilding the mapping on every ``__getitem__`` cost an open+mmap pair per
    leaf per item -- roughly 15x on a random-order read.
    """
    source = TreeDataSource(directory=_tiny_dataset(tmp_path / "ds"))
    source[0]
    first = {name: id(mm) for name, mm in source._leaf_memmaps.items()}
    assert set(first) == {"waveform", "lufs"}

    for i in range(1, 64):
        source[i]
    assert {name: id(mm) for name, mm in source._leaf_memmaps.items()} == first


def test_held_memmap_is_not_aliased_by_returned_samples(tmp_path):
    """Samples are copies, so nothing hands out a view into the shared mapping."""
    source = TreeDataSource(directory=_tiny_dataset(tmp_path / "ds"))
    tree = source[3]
    assert not np.shares_memory(tree["waveform"], source._leaf_memmaps["waveform"])

    tree["waveform"][:] = -1.0
    assert source[3]["waveform"].max() > 0  # the dataset is untouched


def test_concurrent_first_reads_are_consistent(tmp_path):
    """Grain's prefetch pool opens from many threads at once.

    The lazy open used to be unsynchronized. The window was narrower than it
    looks -- ``_leaf_names = []`` rebinds rather than clears, so an in-flight
    iteration keeps the old complete list, and the "opened" flag was already
    assigned last -- and attempts to provoke a truncated sample (40 leaves, 16
    threads, a 1us switch interval, 60 trials) never produced one. The open is
    now built into locals and published under a lock regardless, which makes
    the remaining window structurally impossible rather than merely unlikely.
    This test guards that; it is not a reproduction of a demonstrated failure.
    """
    import concurrent.futures

    source = TreeDataSource(directory=_tiny_dataset(tmp_path / "ds"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        trees = list(pool.map(source.__getitem__, range(64)))

    for i, tree in enumerate(trees):
        assert set(tree) == {"waveform", "lufs"}, f"sample {i} lost a leaf"
        assert tree["lufs"][0] == float(i)


def test_source_survives_a_pickle_round_trip(tmp_path):
    """Grain spawns workers, so the source is pickled with its handles live.

    A `threading.Lock` cannot be pickled at all and a memmap belongs to the
    process that made it; both are dropped on the way out and rebuilt on first
    use in the worker.
    """
    import pickle

    source = TreeDataSource(directory=_tiny_dataset(tmp_path / "ds"))
    expected = source[5]["waveform"]  # open the handles before pickling

    revived = pickle.loads(pickle.dumps(source))
    assert revived._leaf_memmaps == {}
    assert not revived._data_files_opened
    np.testing.assert_array_equal(revived[5]["waveform"], expected)
    assert revived._leaf_memmaps  # rebuilt on demand


# === Manifest size validation ===


def _rewrite_manifest(directory, mutate):
    """Apply *mutate* to the on-disk manifest dict and write it back."""
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _inflate_shape(manifest):
    manifest["leaves"]["waveform"]["shape_per_sample"] = [1, 200]


def _inflate_num_samples(manifest):
    manifest["num_samples"] = 40


@pytest.mark.parametrize(
    "mutate_manifest,truncate_to",
    [
        # An over-large shape_per_sample. np.memmap does refuse this, but with a
        # bare "mmap length is greater than file size" that names neither the
        # manifest nor the leaf -- and only at the first read.
        pytest.param(_inflate_shape, None, id="over-large-shape"),
        # An over-large num_samples: len() lies and reads run off the end.
        pytest.param(_inflate_num_samples, None, id="over-large-num-samples"),
        # A manifest that was honest when written, against a .bin truncated
        # afterwards (an interrupted copy, a full disk).
        pytest.param(None, 400, id="truncated-bin"),
    ],
)
def test_leaf_file_shorter_than_the_manifest_claims_is_refused(
    tmp_path, mutate_manifest, truncate_to
):
    """_validate_manifest never stat'd a leaf file, so nothing measured the data.

    The failure surfaced later and anonymously -- or, for a shortfall np.memmap
    tolerates, not at all. Now the shortfall is named at construction.
    """
    data_dir = tmp_path / "ds"
    with TreeWriter(data_dir, expected_samples=4) as w:
        w.write(
            AudioTree(
                waveform=np.random.randn(4, 1, 100).astype(np.float32),
                sample_rate=44100,
            )
        )

    if mutate_manifest is not None:
        _rewrite_manifest(data_dir, mutate_manifest)
    if truncate_to is not None:
        os.truncate(data_dir / "waveform.bin", truncate_to)

    with pytest.raises(ValueError, match=r"'waveform.bin' is only \d+ bytes"):
        TreeDataSource(data_dir)


def test_missing_leaf_file_is_refused_by_name(tmp_path):
    """A manifest naming a .bin that is not there fails with both names."""
    data_dir = tmp_path / "ds"
    with TreeWriter(data_dir, expected_samples=2) as w:
        w.write({"x": np.zeros((2, 3), dtype=np.float32)})

    (data_dir / "x.bin").unlink()

    with pytest.raises(ValueError, match=r"leaf 'x' names 'x.bin', which cannot"):
        TreeDataSource(data_dir)


def test_in_progress_dataset_is_readable_after_flush(tmp_path):
    """The size check is `>=`, so a mid-write dataset still opens.

    ``flush()`` is public and the writer commits a manifest right after
    preallocation, so between the first write and ``close()`` every ``.bin`` is
    legitimately *larger* than ``num_samples`` implies. A strict equality check
    would reject this dataset, which is valid and completely readable.
    """
    data_dir = tmp_path / "ds"
    writer = TreeWriter(data_dir, expected_samples=64).open()
    writer.write({"x": np.arange(8, dtype=np.float32)[:, None] * np.ones((1, 4))})
    writer.flush()

    declared = 8 * 4 * np.dtype(np.float32).itemsize
    assert (data_dir / "x.bin").stat().st_size > declared  # preallocated for 64

    try:
        source = TreeDataSource(data_dir)
        assert len(source) == 8
        np.testing.assert_array_equal(source[7]["x"], np.full((1, 4), 7.0, np.float32))
    finally:
        writer.close()

    assert len(TreeDataSource(data_dir)) == 8


# === load_into_memory ===


def test_in_memory_samples_do_not_alias_the_shared_store(tmp_path):
    """Samples are copies, as on the memmap path.

    The in-memory arrays are shared by every sample the source will ever hand
    out, so returning ``arr[idx][np.newaxis]`` -- a writable view -- let one
    caller's in-place write rewrite the dataset for every reader after it.
    """
    source = TreeDataSource(
        directory=_tiny_dataset(tmp_path / "ds"), load_into_memory=True
    )
    tree = source[3]
    assert not np.shares_memory(tree["waveform"], source._in_memory_arrays["waveform"])

    tree["waveform"][:] = -1.0
    assert source[3]["waveform"].max() > 0  # the store is untouched


class _StubBagzWriter:
    """Length-prefixed records; enough for TreeWriter's use of bagz."""

    def __init__(self, path):
        self._file = open(path, "wb")

    def write(self, record: bytes):
        self._file.write(len(record).to_bytes(8, "little") + record)

    def close(self):
        self._file.close()


class _StubBagzReader:
    def __init__(self, path):
        data = Path(path).read_bytes()
        self._records = []
        offset = 0
        while offset < len(data):
            size = int.from_bytes(data[offset : offset + 8], "little")
            offset += 8
            self._records.append(data[offset : offset + size])
            offset += size

    def __getitem__(self, index: int) -> bytes:
        return self._records[index]


@pytest.fixture
def stub_bagz(monkeypatch):
    """Stand in for bagz so string-leaf tests run everywhere.

    bagz publishes manylinux x86-64 wheels only. What these tests exercise is
    which branch ``__getitem__`` takes, not the container format, so a stub
    keeps the coverage on macOS and aarch64 instead of skipping it there.
    """
    module = types.ModuleType("bagz")
    module.Writer = _StubBagzWriter
    module.Reader = _StubBagzReader
    monkeypatch.setitem(sys.modules, "bagz", module)
    return module


def _string_only_source(tmp_path, **kwargs):
    """A dataset whose only *included* leaf is a string leaf."""
    data_dir = tmp_path / "ds"
    with TreeWriter(data_dir, expected_samples=2) as w:
        w.write({"label": ["cat", "dog"], "x": np.zeros((2, 3), dtype=np.float32)})
    return TreeDataSource(data_dir, exclude_prefixes=["x"], **kwargs)


def test_in_memory_string_leaves_survive_the_trip_to_a_worker(tmp_path, stub_bagz):
    """A worker used to get a completely empty sample here.

    ``__getitem__`` branched on whether ``_in_memory_arrays`` held anything
    rather than on the mode, so a source that excludes every array leaf took the
    *file* path -- while ``load_into_memory=True`` had already left
    ``_data_files_opened`` True with no handles behind it on the far side of a
    pickle. ``_ensure_open`` then had nothing to do and both dicts were empty.
    The parent's read below is load-bearing: it is what set the flag.
    """
    source = _string_only_source(tmp_path, load_into_memory=True)
    assert source._in_memory_arrays == {}  # every array leaf excluded
    assert source[0] == {"label": "cat"}  # the parent read that set the flag

    revived = pickle.loads(pickle.dumps(source))  # what spawn does to the source
    assert revived[0] == {"label": "cat"}
    assert revived[1] == {"label": "dog"}


def test_lazy_string_leaves_survive_the_trip_to_a_worker(tmp_path, stub_bagz):
    """The same source in the default lazy mode reopens its reader in the worker."""
    source = _string_only_source(tmp_path)
    assert source[0] == {"label": "cat"}

    revived = pickle.loads(pickle.dumps(source))
    assert revived[1] == {"label": "dog"}


def _read_sample_in_child(source, index, queue):
    queue.put(source[index])


def test_in_memory_source_reads_in_a_real_spawned_worker(tmp_path, stub_bagz):
    """End-to-end version of the above: grain spawns workers on macOS/Windows.

    The child never imports bagz -- ``_in_memory_strings`` is plain ``str`` by
    then -- which is the whole point of loading string leaves up front.
    """
    source = _string_only_source(tmp_path, load_into_memory=True)
    assert source[0] == {"label": "cat"}

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_read_sample_in_child, args=(source, 1, queue))
    process.start()
    try:
        assert queue.get(timeout=120) == {"label": "dog"}
    finally:
        process.join(timeout=120)
