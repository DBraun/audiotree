"""Tests for AudioDataSource."""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree, AudioWriter
from audiotree.sources import AudioDataSource
from audiotree.sources.core import READ_ERROR_KEY, AudioReadError


def test_round_trip_npz_manifest():
    """Test writing with AudioWriter and reading back with AudioDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with metadata
        waveform = np.random.randn(3, 2, 22050)  # 3 batch, 2 channels
        audio_tree = AudioTree.create(
            waveform,
            sample_rate=22050,
            lufs=np.array([-20.0, -18.0, -22.0]),
            pitch=np.array([60.0, 62.0, 64.0]),
            velocity=np.array([64, 80, 100]),
            filepath=["original1.wav", "original2.wav", "original3.wav"],
        )

        # Write with AudioWriter
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree, tags={"dataset": "test", "version": 1})

        # Read back with AudioDataSource
        source = AudioDataSource(output_dir / "manifest.npz")

        # Check length
        assert len(source) == 3

        # Check each item
        for i in range(3):
            loaded_tree = source[i]

            # Check audio shape (should be single item, not batch)
            assert loaded_tree.waveform.shape == (1, 2, 22050)

            # Check restored metadata
            assert loaded_tree.lufs[0] == audio_tree.lufs[i]
            assert loaded_tree.pitch[0] == audio_tree.pitch[i]
            assert loaded_tree.velocity[0] == audio_tree.velocity[i]

            # Check manifest metadata via get_entry (metadata removed from audio_tree for batch compatibility)
            entry = source.get_entry(i)
            assert entry["filepath"] == f"original{i + 1}.wav"
            assert entry["tags"]["dataset"] == "test"
            assert entry["tags"]["version"] == 1


def test_filter_function():
    """Test filtering manifest entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create varied data
        trees = [
            AudioTree.create(
                np.random.randn(1, 1, 8000),
                sample_rate=8000,
                lufs=np.array([-10.0]),
            ),
            AudioTree.create(
                np.random.randn(1, 1, 8000),
                sample_rate=8000,
                lufs=np.array([-25.0]),
            ),
            AudioTree.create(
                np.random.randn(1, 1, 8000),
                sample_rate=8000,
                lufs=np.array([-18.0]),
            ),
        ]

        with AudioWriter(output_dir) as writer:
            for audio_tree in trees:
                writer.write(audio_tree)

        # Filter for loud samples only
        source = AudioDataSource(
            output_dir / "manifest.npz",
            filter_fn=lambda entry: entry.get("lufs", -float("inf")) > -20,
        )

        # Should only have 2 entries (-10 and -18)
        assert len(source) == 2
        assert source[0].lufs[0] == -10.0
        assert source[1].lufs[0] == -18.0


def test_filter_by_tag():
    """Test filtering by tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write data with different tags
        writer = AudioWriter(output_dir)

        for i, category in enumerate(["A", "B", "A", "C"]):
            audio_tree = AudioTree.create(np.random.randn(1, 1, 8000), sample_rate=8000)
            writer.write(audio_tree, tags={"category": category, "index": i})

        writer.save_manifest()

        # Filter by category
        source = AudioDataSource(output_dir / "manifest.npz")
        source_a = source.filter_by_tag("category", "A")

        assert len(source_a) == 2  # Two items with category "A"

        # Check indices
        entries = source_a.get_all_entries()
        assert entries[0]["tags"]["index"] == 0
        assert entries[1]["tags"]["index"] == 2


def test_filter_by_lufs():
    """Test filtering by loudness range."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create data with varying loudness
        loudness_values = [-30.0, -20.0, -15.0, -10.0, -5.0]
        writer = AudioWriter(output_dir)

        for lufs in loudness_values:
            audio_tree = AudioTree.create(
                np.random.randn(1, 1, 8000), sample_rate=8000, lufs=np.array([lufs])
            )
            writer.write(audio_tree)

        writer.save_manifest()

        # Filter by loudness range
        source = AudioDataSource(output_dir / "manifest.npz")
        filtered = source.filter_by_lufs(min_lufs=-20, max_lufs=-10)

        assert len(filtered) == 3  # -20, -15, -10
        loudness_values = [filtered[i].lufs[0] for i in range(len(filtered))]
        assert set(loudness_values) == {-20.0, -15.0, -10.0}


def test_from_writer_output():
    """Test convenience constructor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write some data
        audio_tree = AudioTree.create(np.random.randn(2, 1, 8000), sample_rate=8000)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Use convenience constructor
        source = AudioDataSource.from_writer_output(output_dir)
        assert len(source) == 2
        assert source[0].waveform.shape == (1, 1, 8000)


def test_restores_source_filepath():
    """AudioDataSource exposes the original source path via .filepath.

    The source path is stored as a top-level ``filepath`` manifest column,
    distinct from the on-disk output filename. It is restored for both
    manifest-only and real-audio manifests, matching AudioTree.from_manifest.
    """
    paths = ["src_0.wav", "src_1.wav", "src_2.wav"]

    # Manifest-only (no audio files written).
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((3, 1, 8000), dtype=np.float32), 8000, filepath=paths
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        source = AudioDataSource.from_writer_output(output_dir)
        assert [source[i].filepath[0] for i in range(len(source))] == paths

    # Real audio written: .filepath is the source path, not the output WAV.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            (0.1 * np.random.randn(3, 1, 8000)).astype(np.float32),
            8000,
            filepath=paths,
        )
        with AudioWriter(output_dir) as writer:
            writer.write(tree)

        source = AudioDataSource.from_writer_output(output_dir)
        assert [source[i].filepath[0] for i in range(len(source))] == paths

    # No source paths: real-audio items fall back to the output path.
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            (0.1 * np.random.randn(2, 1, 8000)).astype(np.float32), 8000
        )
        with AudioWriter(output_dir) as writer:
            writer.write(tree)

        source = AudioDataSource.from_writer_output(output_dir)
        assert source[0].filepath[0].endswith("audio_0000.wav")


def test_num_records_limit():
    """Test limiting number of records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write 5 items
        audio_tree = AudioTree.create(np.random.randn(5, 1, 8000), sample_rate=8000)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load only first 3
        source = AudioDataSource(output_dir / "manifest.npz", num_records=3)
        assert len(source) == 3


def test_resampling():
    """Test resampling on load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write at 44100 Hz
        audio_tree = AudioTree.create(np.random.randn(1, 1, 44100), sample_rate=44100)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Read and resample to 16000 Hz
        source = AudioDataSource(output_dir / "manifest.npz", sample_rate=16000)

        loaded = source[0]
        assert loaded.sample_rate == 16000
        assert loaded.waveform.shape[2] == 16000  # 1 second at 16kHz


def test_mono_conversion():
    """Test mono conversion on load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write stereo audio
        audio_tree = AudioTree.create(np.random.randn(1, 2, 8000), sample_rate=8000)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Read as mono
        source = AudioDataSource(output_dir / "manifest.npz", mono=True)

        loaded = source[0]
        assert loaded.waveform.shape[1] == 1  # Mono


def test_get_entry():
    """Test getting raw manifest entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        audio_tree = AudioTree.create(
            np.random.randn(2, 1, 8000),
            sample_rate=8000,
            lufs=np.array([-20.0, -18.0]),
        )
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree, tags={"test": True})

        source = AudioDataSource(output_dir / "manifest.npz")

        # Get raw entry
        entry = source.get_entry(0)
        assert entry["lufs"] == -20.0
        assert entry["tags"]["test"] is True
        assert "filename" in entry


def test_grain_integration():
    """Test that AudioDataSource works as a grain RandomAccessDataSource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write test data
        audio_tree = AudioTree.create(np.random.randn(8, 1, 8000), sample_rate=8000)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Use as grain RandomAccessDataSource
        source = AudioDataSource(output_dir / "manifest.npz")

        # Test it works with basic grain functionality
        assert len(source) == 8

        # Test random access
        item3 = source[3]
        assert isinstance(item3, AudioTree)
        assert item3.waveform.shape == (1, 1, 8000)

        # Test iteration
        items = [source[i] for i in range(min(3, len(source)))]
        assert len(items) == 3
        assert all(isinstance(item, AudioTree) for item in items)


def test_grain_dataloader_with_batch_transform():
    """AudioDataSource items collate through AudioTree.batch."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data
        np.random.seed(42)
        waveform = np.random.randn(8, 2, 1000).astype(np.float32)
        audio_tree = AudioTree(waveform, 44100)

        # Add properties that will be preserved
        audio_tree = audio_tree.replace(
            lufs=np.linspace(-30.0, -10.0, 8),
            pitch=np.linspace(60.0, 72.0, 8),
            velocity=np.arange(32, 40, dtype=np.int16),
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load with AudioDataSource
        source = AudioDataSource.from_writer_output(output_dir)

        # Load all items from AudioDataSource
        items = [source[i] for i in range(len(source))]

        # Test that items from AudioDataSource can be batched
        batch1 = AudioTree.batch(items[0:4])
        batch2 = AudioTree.batch(items[4:8])

        # Check first batch
        assert isinstance(batch1, AudioTree)
        assert batch1.waveform.shape == (4, 2, 1000)
        assert batch1.lufs.shape == (4,)
        assert batch1.pitch.shape == (4,)
        assert batch1.velocity.shape == (4,)

        # Check second batch
        assert batch2.waveform.shape == (4, 2, 1000)
        assert batch2.lufs.shape == (4,)

        # Verify data integrity
        expected_loudness_batch1 = np.linspace(-30.0, -10.0, 8)[:4]
        expected_loudness_batch2 = np.linspace(-30.0, -10.0, 8)[4:]

        assert np.allclose(batch1.lufs, expected_loudness_batch1, atol=0.01)
        assert np.allclose(batch2.lufs, expected_loudness_batch2, atol=0.01)

        # The key result: AudioDataSource items can be successfully batched
        print("✓ AudioDataSource items are compatible with AudioTree.batch")


def test_manifest_metadata_with_batch_transform():
    """Metadata arrays from a manifest survive AudioTree.batch."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with metadata arrays
        np.random.seed(42)
        batch_size = 8
        param_dim = 185

        waveform = np.random.randn(batch_size, 2, 1000).astype(np.float32)
        audio_tree = AudioTree(waveform, 44100)

        # Add metadata arrays that should be preserved through write/read
        audio_tree = audio_tree.replace(
            lufs=np.linspace(-30.0, -10.0, batch_size),
            metadata={
                "params": np.random.randn(batch_size, param_dim).astype(np.float32),
                "confidence": np.linspace(0.5, 1.0, batch_size).astype(np.float32),
            },
        )

        # Write with NPZ manifest (metadata will be saved)
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load data back with metadata
        source = AudioDataSource.from_writer_output(output_dir)

        # Load items and check metadata is present
        items = []
        for i in range(len(source)):
            item = source[i]
            # Check metadata was loaded
            assert "params" in item.metadata
            assert "confidence" in item.metadata
            # Check shapes (should have batch dimension)
            assert item.metadata["params"].shape == (1, param_dim)
            assert item.metadata["confidence"].shape == (1,)
            items.append(item)

        # Create batches
        batch1 = AudioTree.batch(items[0:4])
        AudioTree.batch(items[4:8])

        # Check batched metadata
        assert "params" in batch1.metadata
        assert batch1.metadata["params"].shape == (4, param_dim)
        assert "confidence" in batch1.metadata
        assert batch1.metadata["confidence"].shape == (4,)

        # Verify values are correct
        expected_confidence_batch1 = np.linspace(0.5, 1.0, 8)[:4]
        assert np.allclose(
            batch1.metadata["confidence"].squeeze(), expected_confidence_batch1
        )


def test_manifest_with_batch_transform():
    """AudioDataSource items batch correctly when collated by hand."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create test data with 8 audio samples
        np.random.seed(42)
        waveform = np.random.randn(8, 2, 1000).astype(np.float32)
        audio_tree = AudioTree(waveform, 44100)

        # Add some metadata
        audio_tree = audio_tree.replace(
            lufs=np.linspace(-30.0, -10.0, 8),
            pitch=np.linspace(60.0, 72.0, 8),
            velocity=np.arange(32, 40, dtype=np.int16),
        )

        # Write with NPZ manifest
        with AudioWriter(output_dir) as writer:
            writer.write(audio_tree)

        # Load data back
        source = AudioDataSource.from_writer_output(output_dir)

        # Manually load items and batch them
        # (avoiding grain.DataLoader which seems to have issues in test environment)
        items = []
        for i in range(len(source)):
            item = source[i]
            # Remove metadata to avoid batching issues
            item = item.replace(metadata={})
            items.append(item)

        # Create first batch
        batch1_items = items[0:4]
        batch1 = AudioTree.batch(batch1_items)

        # Create second batch
        batch2_items = items[4:8]
        batch2 = AudioTree.batch(batch2_items)

        # Check first batch
        assert isinstance(batch1, AudioTree)
        assert batch1.waveform.shape == (4, 2, 1000)  # 4 samples batched
        assert batch1.lufs.shape == (4,)
        assert batch1.pitch.shape == (4,)
        assert batch1.velocity.shape == (4,)

        # Check second batch
        assert batch2.waveform.shape == (4, 2, 1000)
        assert batch2.lufs.shape == (4,)

        # Verify data integrity - loudness values should be sequential
        expected_loudness_batch1 = np.linspace(-30.0, -10.0, 8)[:4]
        expected_loudness_batch2 = np.linspace(-30.0, -10.0, 8)[4:]

        assert np.allclose(batch1.lufs, expected_loudness_batch1, atol=0.01)
        assert np.allclose(batch2.lufs, expected_loudness_batch2, atol=0.01)

        # Verify velocity values
        expected_velocity_batch1 = np.arange(32, 36, dtype=np.int16)
        expected_velocity_batch2 = np.arange(36, 40, dtype=np.int16)

        assert np.array_equal(batch1.velocity, expected_velocity_batch1)
        assert np.array_equal(batch2.velocity, expected_velocity_batch2)


if __name__ == "__main__":
    test_round_trip_npz_manifest()
    test_filter_function()
    test_filter_by_tag()
    test_filter_by_lufs()
    test_from_writer_output()
    test_restores_source_filepath()
    test_num_records_limit()
    test_resampling()
    test_mono_conversion()
    test_get_entry()
    test_grain_integration()
    test_grain_dataloader_with_batch_transform()
    test_manifest_metadata_with_batch_transform()
    test_manifest_with_batch_transform()
    print("All tests passed!")


def test_array_fields_keep_their_batch_axis_and_dtype():
    """Array-valued AudioTree fields must survive a manifest round trip intact.

    ``AudioTree.from_file`` only adds a batch axis to a *scalar*, so
    ``lufs_windows``/``codes``/``latents`` used to come back one axis short and
    were then concatenated along the wrong axis by ``AudioTree.batch`` --
    silently interleaving one item's tokens into the next. Scalar fields were
    additionally re-cast to a hard-coded dtype.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        writer = AudioWriter(str(output_dir))
        for i in range(3):
            tree = AudioTree.create(
                np.full((1, 1, 8000), 0.1 * (i + 1), dtype=np.float32),
                sample_rate=16000,
                pitch=np.array([60 + i], dtype=np.int32),
                codes=np.arange(2 * 5, dtype=np.int32).reshape(1, 2, 5) + i,
            )
            writer.write(tree.replace_lufs())
        writer.close()

        source = AudioDataSource.from_writer_output(str(output_dir))
        item = source[0]

        # Every field carries the leading batch axis.
        assert item.lufs.shape == (1,)
        assert item.lufs_windows.ndim == 2 and item.lufs_windows.shape[0] == 1
        assert item.codes.shape == (1, 2, 5)
        # Stored dtypes are preserved rather than re-cast.
        assert item.pitch.dtype == np.int32
        np.testing.assert_array_equal(item.pitch, [60])

        # Batching stacks along the batch axis instead of corrupting the data.
        batched = AudioTree.batch([source[i] for i in range(3)])
        assert batched.codes.shape == (3, 2, 5)
        assert batched.lufs_windows.shape[0] == 3
        np.testing.assert_array_equal(batched.pitch.ravel(), [60, 61, 62])
        for i in range(3):
            np.testing.assert_array_equal(
                batched.codes[i], np.arange(2 * 5, dtype=np.int32).reshape(2, 5) + i
            )


def _write_filter_corpus(output_dir, lufs_values, genres):
    """Write one item per (lufs, genre) pair, each tagged with its genre."""
    writer = AudioWriter(output_dir)
    for lufs, genre in zip(lufs_values, genres):
        writer.write(
            AudioTree.create(
                np.zeros((1, 1, 8000), dtype=np.float32),
                sample_rate=8000,
                lufs=np.array([lufs], dtype=np.float32),
            ),
            tags={"genre": genre},
        )
    writer.save_manifest()


def test_filters_compose_instead_of_re_reading_the_manifest():
    """Each ``filter_*`` narrows the receiver, in either order.

    The helpers used to rebuild an AudioDataSource from the manifest with only
    their own predicate, so a chained call silently returned a *wider* dataset
    containing exactly the entries the caller had already excluded.
    """
    lufs_values = [-30.0, -18.0, -10.0, -25.0]
    genres = ["rock", "jazz", "rock", "jazz"]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_filter_corpus(output_dir, lufs_values, genres)
        source = AudioDataSource.from_writer_output(output_dir)
        assert len(source) == 4

        # tag then lufs: only the loud rock item survives.
        rock = source.filter_by_tag("genre", "rock")
        assert len(rock) == 2
        loud_rock = rock.filter_by_lufs(min_lufs=-20.0)
        assert [e["tags"]["genre"] for e in loud_rock.get_all_entries()] == ["rock"]
        assert [e["lufs"] for e in loud_rock.get_all_entries()] == [-10.0]

        # lufs then tag: same result.
        loud_rock2 = source.filter_by_lufs(min_lufs=-20.0).filter_by_tag(
            "genre", "rock"
        )
        assert [e["lufs"] for e in loud_rock2.get_all_entries()] == [-10.0]

        # The receiver is untouched by the narrowing.
        assert len(source) == 4
        assert len(rock) == 2


def test_filters_compose_with_constructor_filter_and_num_records():
    """A constructor ``filter_fn`` and a ``num_records`` cap survive filtering."""
    lufs_values = [-30.0, -18.0, -10.0, -25.0]
    genres = ["rock", "jazz", "rock", "jazz"]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_filter_corpus(output_dir, lufs_values, genres)

        # Constructor filter is not dropped by a subsequent helper call.
        quiet = AudioDataSource.from_writer_output(
            output_dir, filter_fn=lambda entry: entry["lufs"] < -20.0
        )
        assert len(quiet) == 2
        assert len(quiet.filter_by_lufs(min_lufs=-100.0)) == 2

        # num_records caps the window that filtering sees.
        capped = AudioDataSource.from_writer_output(output_dir, num_records=2)
        assert len(capped) == 2
        rock = capped.filter_by_tag("genre", "rock")
        assert [e["lufs"] for e in rock.get_all_entries()] == [-30.0]

        # A generic predicate composes the same way.
        assert len(capped.filter(lambda entry: entry["lufs"] < 0.0)) == 2


def test_filter_matching_nothing_raises():
    """An empty result is an error, matching the constructor's behavior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        _write_filter_corpus(output_dir, [-30.0, -10.0], ["rock", "rock"])
        source = AudioDataSource.from_writer_output(output_dir)

        with pytest.raises(ValueError, match="No entries left"):
            source.filter_by_tag("genre", "polka")


def test_non_scalar_tag_value_reports_the_tag():
    """A container-valued tag cell fails with an actionable error, not a numpy one.

    ``AudioWriter`` accepts any tag value, and the reader's ``value != ""``
    check used to raise "truth value of an array is ambiguous" -- naming neither
    the tag nor the manifest, so the file looked permanently unreadable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        writer = AudioWriter(output_dir)
        writer.write(
            AudioTree.create(
                np.zeros((1, 1, 8000), dtype=np.float32), sample_rate=8000
            ),
            tags={"embedding": np.array([1.0, 2.0, 3.0])},
        )
        writer.save_manifest()

        with pytest.raises(ValueError, match="non-scalar value for tag 'embedding'"):
            AudioDataSource.from_writer_output(output_dir)


def test_sentinel_values_are_not_dropped():
    """A stored -1 / NaN / "" is user data, not a "missing" marker.

    The reader used to treat those three values as absent and omit the field, so
    an item with ``velocity=-1`` came back with a different pytree structure
    than its siblings and broke batching for the whole dataset.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        writer = AudioWriter(str(output_dir))
        for velocity in (100, -1, 64):
            writer.write(
                AudioTree.create(
                    np.zeros((1, 1, 4000), dtype=np.float32),
                    sample_rate=16000,
                    velocity=np.array([velocity], dtype=np.int32),
                )
            )
        writer.close()

        source = AudioDataSource.from_writer_output(str(output_dir))
        assert source[1].velocity is not None
        np.testing.assert_array_equal(source[1].velocity, [-1])

        batched = AudioTree.batch([source[i] for i in range(3)])
        np.testing.assert_array_equal(batched.velocity.ravel(), [100, -1, 64])


def test_manifest_is_read_without_unpickling(tmp_path, monkeypatch):
    """Reading a manifest must never unpickle it.

    A manifest travels with the data it describes, so a mirrored or downloaded
    dataset would otherwise run whatever it contains, in every data worker.
    """
    with AudioWriter(tmp_path) as writer:
        writer.write(
            AudioTree.create(np.zeros((2, 1, 8000), dtype=np.float32), 8000),
            tags={"dataset": "test"},
        )

    real_load = np.load

    def refuse_pickle(*args, **kwargs):
        assert not kwargs.get("allow_pickle", False), "manifest read with pickling on"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", refuse_pickle)
    source = AudioDataSource.from_writer_output(tmp_path)
    assert len(source) == 2
    assert source.get_entry(0)["tags"] == {"dataset": "test"}


def test_empty_tag_value_survives_the_round_trip(tmp_path):
    """``""`` is a tag value, not a missing marker; absence comes from the mask.

    The reader used to drop every ``None`` *or* ``""`` tag cell, which made an
    empty tag indistinguishable from an unset one.
    """
    writer = AudioWriter(tmp_path)
    writer.write(
        AudioTree.create(np.zeros((1, 1, 8000), dtype=np.float32), 8000),
        tags={"split": "", "note": "kept"},
    )
    writer.write(
        AudioTree.create(np.zeros((1, 1, 8000), dtype=np.float32), 8000),
        tags={"note": "kept"},
    )
    writer.close()

    source = AudioDataSource.from_writer_output(tmp_path)
    assert source.get_entry(0)["tags"] == {"split": "", "note": "kept"}
    assert source.get_entry(1)["tags"] == {"note": "kept"}
    assert len(source.filter_by_tag("split", "")) == 1


def test_entries_report_the_subtype_the_audio_was_written_in(tmp_path):
    """The manifest records the encoding, so a reader need not open every file."""
    with AudioWriter(tmp_path) as writer:
        writer.write(AudioTree.create(np.zeros((1, 1, 8000), dtype=np.float32), 8000))

    source = AudioDataSource.from_writer_output(tmp_path)
    assert source.get_entry(0)["subtype"] == "FLOAT"


# ---------------------------------------------------------------------------
# on_read_error: a manifest can outlive the audio it names.
# ---------------------------------------------------------------------------


def _writer_output_with_one_bad_file(tmp_path, breakage):
    """Write three items, then break the middle one's audio file.

    ``breakage`` is called with the path to ``audio_0001.wav``.
    """
    tree = AudioTree.create(
        np.random.randn(3, 2, 8000).astype(np.float32),
        sample_rate=8000,
        lufs=np.array([-20.0, -18.0, -22.0], dtype=np.float32),
    )
    with AudioWriter(tmp_path) as writer:
        writer.write(tree)
    breakage(Path(tmp_path) / "audio_0001.wav")
    return tmp_path


def _delete(path):
    path.unlink()


def _truncate_header(path):
    path.write_bytes(path.read_bytes()[:20])


def _empty(path):
    path.write_bytes(b"")


BREAKAGES = {"missing": _delete, "truncated_header": _truncate_header, "empty": _empty}


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_getitem_raises_naming_the_path_by_default(tmp_path, breakage):
    """The default policy is unchanged, and the message names the file.

    Every read failure -- a deleted file, an unparseable one -- comes back as
    one catchable type carrying the path, instead of a bare ``EOFError`` whose
    ``str()`` is empty.
    """
    output_dir = _writer_output_with_one_bad_file(tmp_path, BREAKAGES[breakage])
    source = AudioDataSource.from_writer_output(output_dir)

    assert source[0].waveform.shape == (1, 2, 8000)
    with pytest.raises(AudioReadError) as excinfo:
        source[1]
    assert "audio_0001.wav" in str(excinfo.value)
    assert excinfo.value.file_path.endswith("audio_0001.wav")


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_getitem_substitutes_marked_silence(tmp_path, breakage, policy):
    """A non-raising policy hands back a shaped, marked, obviously-empty item."""
    output_dir = _writer_output_with_one_bad_file(tmp_path, BREAKAGES[breakage])
    source = AudioDataSource.from_writer_output(output_dir, on_read_error=policy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        item = source[1]

    # The manifest still knows the shape, so the substitute matches its peers.
    assert item.waveform.shape == (1, 2, 8000)
    assert item.sample_rate == 8000
    assert not np.any(np.asarray(item.waveform))
    assert bool(item.metadata[READ_ERROR_KEY][0]) is True
    assert item.filepath[0].endswith("audio_0001.wav")
    # Manifest-side labels survive: only the audio was lost.
    assert float(item.lufs[0]) == pytest.approx(-18.0)


def test_getitem_warn_names_the_file_and_skip_stays_quiet(tmp_path):
    """The two non-raising policies differ only in whether they warn."""
    output_dir = _writer_output_with_one_bad_file(tmp_path, _empty)

    warning_source = AudioDataSource.from_writer_output(
        output_dir, on_read_error="warn"
    )
    with pytest.warns(UserWarning, match="audio_0001.wav"):
        warning_source[1]

    quiet_source = AudioDataSource.from_writer_output(output_dir, on_read_error="skip")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_source[1]
    # librosa warns on its own when it falls back to audioread; what must be
    # absent is *our* substitution notice.
    assert not [w for w in caught if "Substituting silence" in str(w.message)]


def test_substituted_and_real_items_batch_together(tmp_path):
    """Every item carries the marker, so the batch collates and self-reports."""
    output_dir = _writer_output_with_one_bad_file(tmp_path, _empty)
    source = AudioDataSource.from_writer_output(output_dir, on_read_error="skip")

    batch = AudioTree.batch([source[i] for i in range(len(source))])
    assert batch.waveform.shape == (3, 2, 8000)
    assert np.asarray(batch.metadata[READ_ERROR_KEY]).tolist() == [False, True, False]


def test_getitem_marker_is_absent_under_the_default_policy(tmp_path):
    """The default policy leaves the metadata as it was before the knob existed."""
    with AudioWriter(tmp_path) as writer:
        writer.write(AudioTree.create(np.zeros((2, 1, 8000), dtype=np.float32), 8000))

    source = AudioDataSource.from_writer_output(tmp_path)
    assert READ_ERROR_KEY not in source[0].metadata


def test_unknown_on_read_error_is_rejected_at_construction(tmp_path):
    """A typo'd policy fails before any item is read."""
    with AudioWriter(tmp_path) as writer:
        writer.write(AudioTree.create(np.zeros((1, 1, 8000), dtype=np.float32), 8000))

    with pytest.raises(ValueError, match="on_read_error must be one of"):
        AudioDataSource.from_writer_output(tmp_path, on_read_error="ignore")


# ---------------------------------------------------------------------------
# Synthetic zero waveforms must carry the *target* geometry, not the stored
# original one, so they collate with real resampled/mono items.
# ---------------------------------------------------------------------------


def test_silence_substitute_collates_under_resampling(tmp_path):
    """A resampled substitute matches its resampled peers, so the batch collates.

    The manifest records ``samples`` at the written rate; under on-the-fly
    resampling the substitute has to be sized to the *target* rate the same way
    the real items are, or ``AudioTree.batch`` cannot concatenate them.
    """
    output_dir = _writer_output_with_one_bad_file(
        tmp_path, _empty
    )  # 3x (2, 8000) @ 8 kHz
    source = AudioDataSource.from_writer_output(
        output_dir, sample_rate=16000, on_read_error="skip"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        items = [source[i] for i in range(len(source))]

    real, substitute = items[0], items[1]
    assert bool(real.metadata[READ_ERROR_KEY][0]) is False
    assert bool(substitute.metadata[READ_ERROR_KEY][0]) is True
    # The substitute carries the target geometry, matching its resampled peers.
    assert substitute.waveform.shape == real.waveform.shape == (1, 2, 16000)
    assert substitute.sample_rate == 16000

    batch = AudioTree.batch(items)
    assert batch.waveform.shape == (3, 2, 16000)


def test_silence_substitute_matches_real_length_for_noninteger_ratio(tmp_path):
    """The substitute mirrors librosa's resample length exactly, not a rounded one.

    For a ratio like 44100->48000 the resampled length overshoots
    ``round(duration * sr)`` by a sample (librosa sizes it ``ceil`` of a float
    ratio). Sizing the substitute with ``round`` would leave it one sample short
    of its real peers and break batching; it must reuse the real-item length.
    """
    tree = AudioTree.create(
        np.random.randn(3, 1, 44100).astype(np.float32), sample_rate=44100
    )
    with AudioWriter(tmp_path) as writer:
        writer.write(tree)
    (Path(tmp_path) / "audio_0001.wav").write_bytes(b"")

    source = AudioDataSource.from_writer_output(
        tmp_path, sample_rate=48000, on_read_error="skip"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        items = [source[i] for i in range(len(source))]

    # A real 44100->48000 load lands at 48001 (ceil of the ratio), not 48000.
    assert items[0].waveform.shape[-1] == 48001
    assert items[1].waveform.shape == items[0].waveform.shape
    assert AudioTree.batch(items).waveform.shape == (3, 1, 48001)


def test_token_only_manifest_honors_mono_and_resampling(tmp_path):
    """A manifest with no audio on disk still yields the target geometry.

    ``files_written=False`` builds a synthetic zero waveform from the manifest;
    it must honor ``mono`` and ``sample_rate`` (and ``duration``) instead of
    replaying the stored stereo original-rate shape, or it cannot batch with
    real resampled/mono items.
    """
    tree = AudioTree.create(
        np.zeros((2, 2, 44100), dtype=np.float32), sample_rate=44100
    )
    with AudioWriter(tmp_path, write_audio=False) as writer:
        writer.write(tree)

    source = AudioDataSource.from_writer_output(tmp_path, sample_rate=16000, mono=True)
    item = source[0]
    assert item.waveform.shape == (1, 1, 16000)  # mono, resampled from stereo 44.1 kHz
    assert item.sample_rate == 16000
    batch = AudioTree.batch([source[i] for i in range(len(source))])
    assert batch.waveform.shape == (2, 1, 16000)

    # Duration composes on top of mono + resampling.
    dur_source = AudioDataSource.from_writer_output(
        tmp_path, sample_rate=16000, mono=True, duration=0.5
    )
    assert dur_source[0].waveform.shape == (1, 1, 8000)
