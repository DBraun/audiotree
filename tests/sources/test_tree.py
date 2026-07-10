"""Tests for TreeDataSource."""

import importlib.util
import tempfile
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
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest.pop("string_leaves", None)
        with open(manifest_path, "w") as f:
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
