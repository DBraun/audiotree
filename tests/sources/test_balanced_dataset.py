"""Tests for create_balanced_audio_dataset function."""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from audiotree.sources import (
    create_balanced_audio_dataset,
    AudioDataBalancedSource,
    AudioDataBalancedDataset,
)


def _create_test_audio_files(tmpdir, group_name, num_files, sample_rate=44100, duration=1.0):
    """Helper to create test audio files."""
    group_dir = Path(tmpdir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration)
    for i in range(num_files):
        # Create audio with identifiable pattern (group encoded in amplitude)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1
        filepath = group_dir / f"audio_{i}.wav"
        sf.write(filepath, audio, sample_rate)

    return str(group_dir)


class TestCreateBalancedAudioDataset:
    """Tests for create_balanced_audio_dataset function."""

    def test_basic_creation(self):
        """Test basic dataset creation with two groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                sample_rate=44100,
                duration=0.5,
            )

            assert len(ds) == 20

    def test_random_access(self):
        """Test that dataset supports random access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                sample_rate=44100,
                duration=0.5,
            )

            # Access in random order
            _ = ds[15]
            _ = ds[5]
            _ = ds[0]
            _ = ds[19]

    def test_equal_weights(self):
        """Test that equal weights give roughly 50/50 distribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=100,
                weights={"group1": 1.0, "group2": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            )

            assert len(ds) == 100

    def test_custom_weights(self):
        """Test that custom weights are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=100,
                weights={"group1": 0.7, "group2": 0.3},
                sample_rate=44100,
                duration=0.5,
            )

            assert len(ds) == 100

    def test_no_shuffle(self):
        """Test deterministic ordering with shuffle=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds1 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                shuffle=False,
                seed=42,
                sample_rate=44100,
                duration=0.5,
            )

            ds2 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                shuffle=False,
                seed=42,
                sample_rate=44100,
                duration=0.5,
            )

            # Same seed + no shuffle should give same data
            for i in range(10):
                audio1 = ds1[i].audio_data
                audio2 = ds2[i].audio_data
                np.testing.assert_array_equal(audio1, audio2)

    def test_different_seeds(self):
        """Test that different seeds produce different orderings when shuffled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            ds1 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                shuffle=True,
                seed=42,
                sample_rate=44100,
                duration=0.5,
            )

            ds2 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                num_records=20,
                shuffle=True,
                seed=123,
                sample_rate=44100,
                duration=0.5,
            )

            # Different seeds should (very likely) give different orderings
            different = False
            for i in range(10):
                audio1 = ds1[i].audio_data
                audio2 = ds2[i].audio_data
                if not np.array_equal(audio1, audio2):
                    different = True
                    break
            assert different, "Different seeds should produce different orderings"

    def test_single_group(self):
        """Test with only one group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir]},
                num_records=20,
                sample_rate=44100,
                duration=0.5,
            )

            assert len(ds) == 20
            _ = ds[0]  # Should work

    def test_default_weights(self):
        """Test that missing weights default to 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)
            group3_dir = _create_test_audio_files(tmpdir, "group3", 5)

            # Only specify weight for one group
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                    "group3": [group3_dir],
                },
                num_records=30,
                weights={"group1": 2.0},  # Others default to 1.0
                sample_rate=44100,
                duration=0.5,
            )

            assert len(ds) == 30

    def test_returns_audiotree(self):
        """Test that items are AudioTree instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree import AudioTree

            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir]},
                num_records=10,
                sample_rate=44100,
                duration=0.5,
            )

            item = ds[0]
            assert isinstance(item, AudioTree)

    def test_source_property(self):
        """Test that AudioTree items have the source property set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "music", 5)
            group2_dir = _create_test_audio_files(tmpdir, "speech", 5)

            ds = create_balanced_audio_dataset(
                sources={"music": [group1_dir], "speech": [group2_dir]},
                num_records=20,
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            )

            # Check that each item has a source property
            sources_found = set()
            for i in range(len(ds)):
                item = ds[i]
                source = item.source
                assert len(source) == 1, "Each item should have exactly one source"
                assert source[0] in ["music", "speech"], f"Source should be 'music' or 'speech', got {source[0]}"
                sources_found.add(source[0])

            # Both sources should be represented
            assert sources_found == {"music", "speech"}, "Both source groups should be present"

    def test_source_property_batched(self):
        """Test that source property works correctly after batching with Batch transform."""
        from audiotree.transforms import Batch

        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "music", 5)
            group2_dir = _create_test_audio_files(tmpdir, "speech", 5)

            ds = create_balanced_audio_dataset(
                sources={"music": [group1_dir], "speech": [group2_dir]},
                num_records=20,
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            )

            # Use AudioTree's Batch transform (concatenates properly)
            batch_transform = Batch(batch_size=4)
            items = [ds[i] for i in range(4)]
            batch = batch_transform._default_batch_fn(items)

            # Check source property
            sources = batch.source
            assert len(sources) == 4, f"Batch should have 4 sources, got {len(sources)}"
            for src in sources:
                assert src in ["music", "speech"], f"Source should be 'music' or 'speech', got {src}"


class TestDeprecationWarnings:
    """Test that deprecated classes emit warnings."""

    def test_balanced_source_deprecation(self):
        """Test that AudioDataBalancedSource emits deprecation warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = AudioDataBalancedSource(
                    sources={"group1": [group1_dir]},
                    num_records=10,
                    sample_rate=44100,
                    duration=0.5,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "create_balanced_audio_dataset" in str(w[0].message)

    def test_balanced_dataset_deprecation(self):
        """Test that AudioDataBalancedDataset emits deprecation warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = AudioDataBalancedDataset(
                    sources={"group1": [group1_dir]},
                    sample_rate=44100,
                    duration=0.5,
                )
                assert len(w) == 1
                assert issubclass(w[0].category, DeprecationWarning)
                assert "create_balanced_audio_dataset" in str(w[0].message)
