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

        source = TreeDataSource(output_dir)
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

        source = TreeDataSource(output_dir)

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
        dry = AudioTree(audio_data=dry_audio, sample_rate=44100)
        wet = AudioTree(audio_data=wet_audio, sample_rate=44100)

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

        source = TreeDataSource(output_dir)
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

        source = TreeDataSource(output_dir)
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

        source = TreeDataSource(output_dir, raw=True)
        sample = source[0]

        assert isinstance(sample, dict)
        assert "audio_data" in sample
        assert "loudness" in sample
        assert "metadata.mel" in sample
        assert sample["audio_data"].shape == (1, 2, 100)


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
            audio_data=np.zeros((3, 1, 10), dtype=np.float32),
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
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
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
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
            sample_rate=44100,
        )

        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write(tree)

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert sample.pitch is None
        assert sample.velocity is None
        assert sample.loudness is None
        assert sample.codes is None
        assert sample.latents is None


# === String leaf round-trip tests ===


def test_round_trip_string_list_with_audiotree():
    """List[str] + AudioTree round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        strings = ["hello", "world", "foo"]
        audio = np.random.randn(3, 1, 100).astype(np.float32)

        with TreeWriter(output_dir, expected_samples=3) as w:
            w.write({
                "strings": strings,
                "wet": AudioTree(audio_data=audio, sample_rate=44100),
            })

        source = TreeDataSource(output_dir)
        assert len(source) == 3

        for i in range(3):
            sample = source[i]
            assert isinstance(sample, dict)
            assert sample["strings"] == strings[i]
            assert isinstance(sample["wet"], AudioTree)
            np.testing.assert_array_almost_equal(
                sample["wet"].audio_data[0], audio[i], decimal=5
            )


def test_round_trip_multiple_string_leaves():
    """Multiple string leaves round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write({
                "labels": ["cat", "dog"],
                "sources": ["train", "val"],
                "x": np.zeros((2, 3), dtype=np.float32),
            })

        source = TreeDataSource(output_dir)
        s0 = source[0]
        assert s0["labels"] == "cat"
        assert s0["sources"] == "train"

        s1 = source[1]
        assert s1["labels"] == "dog"
        assert s1["sources"] == "val"


def test_round_trip_single_string():
    """A single str leaf (batch size 1) round-trips."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=1) as w:
            w.write({
                "label": "cat",
                "x": np.zeros((1, 3), dtype=np.float32),
            })

        source = TreeDataSource(output_dir)
        sample = source[0]
        assert sample["label"] == "cat"


def test_round_trip_long_strings():
    """Strings longer than 256 chars survive round-trip (no truncation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        long_str = "x" * 1000
        with TreeWriter(output_dir, expected_samples=1) as w:
            w.write({"s": [long_str], "x": np.zeros((1,), dtype=np.float32)})

        source = TreeDataSource(output_dir)
        assert source[0]["s"] == long_str


def test_round_trip_empty_string():
    """Empty strings survive round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write({
                "s": ["", "hello"],
                "x": np.zeros((2,), dtype=np.float32),
            })

        source = TreeDataSource(output_dir)
        assert source[0]["s"] == ""
        assert source[1]["s"] == "hello"


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


def test_round_trip_strings_multiple_writes():
    """String leaves written across multiple batches read correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=5) as w:
            w.write({
                "s": ["a", "b", "c"],
                "x": np.zeros((3,), dtype=np.float32),
            })
            w.write({
                "s": ["d", "e"],
                "x": np.ones((2,), dtype=np.float32),
            })

        source = TreeDataSource(output_dir)
        assert len(source) == 5
        assert source[0]["s"] == "a"
        assert source[2]["s"] == "c"
        assert source[3]["s"] == "d"
        assert source[4]["s"] == "e"


def test_raw_mode_with_strings():
    """raw=True includes decoded string values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=2) as w:
            w.write({
                "label": ["cat", "dog"],
                "x": np.zeros((2, 3), dtype=np.float32),
            })

        source = TreeDataSource(output_dir, raw=True)
        sample = source[0]
        assert isinstance(sample, dict)
        assert sample["label"] == "cat"
        assert isinstance(sample["x"], np.ndarray)


def test_batch_fn_with_strings():
    """AudioTree.batch_fn correctly batches string leaves into List[str]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        with TreeWriter(output_dir, expected_samples=4) as w:
            w.write({
                "strings": ["a", "b", "c", "d"],
                "wet": AudioTree(
                    audio_data=np.arange(4 * 100, dtype=np.float32).reshape(
                        4, 1, 100
                    ),
                    sample_rate=44100,
                ),
            })

        source = TreeDataSource(output_dir)
        items = [source[i] for i in range(4)]
        batched = AudioTree.batch_fn(items)

        assert batched["strings"] == ["a", "b", "c", "d"]
        assert isinstance(batched["wet"], AudioTree)
        assert batched["wet"].audio_data.shape == (4, 1, 100)


def test_backward_compat_no_string_leaves():
    """Old manifests without string_leaves key still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree(
            audio_data=np.zeros((2, 1, 10), dtype=np.float32),
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
            audio_data=np.zeros((5, 1, 10), dtype=np.float32),
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
