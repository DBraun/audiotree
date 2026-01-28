"""Tests for multiprocessing/multithreading with audio datasets."""

import tempfile
from pathlib import Path

import grain
import numpy as np
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset


def _create_test_audio_files(tmpdir, group_name, num_files, sample_rate=44100, duration=0.5):
    """Helper to create test audio files."""
    group_dir = Path(tmpdir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration)
    for i in range(num_files):
        # Create sine tone audio with frequency based on file index
        t = np.linspace(0, duration, num_samples)
        freq = 440 + i * 10  # Different frequency for each file
        audio = (np.sin(2 * np.pi * freq * t) * 0.1).astype(np.float32)
        filepath = group_dir / f"audio_{i:03d}.wav"
        sf.write(str(filepath), audio, sample_rate)

    return str(group_dir)


class TestMultithreading:
    """Test multithreading with ReadOptions."""

    def test_create_audio_dataset_with_multithreading(self):
        """Test create_audio_dataset works with ReadOptions for multithreading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 20)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=0.5,
            )

            # Convert to IterDataset with multithreading
            read_options = grain.ReadOptions(
                num_threads=2,
                prefetch_buffer_size=4,
            )
            iter_ds = ds.to_iter_dataset(read_options=read_options)

            # Verify we can iterate and get valid AudioTree objects
            count = 0
            for item in iter_ds:
                assert isinstance(item, AudioTree)
                assert item.audio_data.shape[0] == 1  # batch size 1
                assert item.audio_data.shape[1] == 1  # mono
                assert item.sample_rate == 44100
                count += 1

            assert count == 20

    def test_create_balanced_audio_dataset_with_multithreading(self):
        """Test create_balanced_audio_dataset works with ReadOptions for multithreading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            # Create balanced dataset
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                },
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 40))

            # Convert to IterDataset with multithreading
            read_options = grain.ReadOptions(
                num_threads=2,
                prefetch_buffer_size=4,
            )
            iter_ds = ds.to_iter_dataset(read_options=read_options)

            # Verify we can iterate and get valid AudioTree objects
            count = 0
            sources_found = set()
            for item in iter_ds:
                assert isinstance(item, AudioTree)
                assert item.audio_data.shape[0] == 1  # batch size 1
                assert item.sample_rate == 44100
                sources_found.add(item.source[0])
                count += 1

            assert count == 40
            assert sources_found == {"group1", "group2"}


class TestMultiprocessing:
    """Test multiprocessing with mp_prefetch."""

    def test_create_audio_dataset_with_multiprocessing(self):
        """Test create_audio_dataset works with mp_prefetch for multiprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 20)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=0.5,
            )

            # Convert to IterDataset and add multiprocessing
            mp_options = grain.MultiprocessingOptions(
                num_workers=2,
                per_worker_buffer_size=2,
            )
            iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

            # Verify we can iterate and get valid AudioTree objects
            count = 0
            for item in iter_ds:
                assert isinstance(item, AudioTree)
                assert item.audio_data.shape[0] == 1  # batch size 1
                assert item.audio_data.shape[1] == 1  # mono
                assert item.sample_rate == 44100
                count += 1

            assert count == 20

    def test_create_balanced_audio_dataset_with_multiprocessing(self):
        """Test create_balanced_audio_dataset works with mp_prefetch for multiprocessing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            # Create balanced dataset
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                },
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 40))

            # Convert to IterDataset and add multiprocessing
            mp_options = grain.MultiprocessingOptions(
                num_workers=2,
                per_worker_buffer_size=2,
            )
            iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

            # Verify we can iterate and get valid AudioTree objects
            count = 0
            sources_found = set()
            for item in iter_ds:
                assert isinstance(item, AudioTree)
                assert item.audio_data.shape[0] == 1  # batch size 1
                assert item.sample_rate == 44100
                sources_found.add(item.source[0])
                count += 1

            assert count == 40
            assert sources_found == {"group1", "group2"}

    def test_multiprocessing_with_balanced_dataset_different_weights(self):
        """Test that multiprocessing preserves weight-based balancing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 20)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 20)

            # Create balanced dataset with custom weights
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                },
                weights={"group1": 0.7, "group2": 0.3},
                shuffle=True,
                shuffle_seed=42,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 1000))

            # Convert to IterDataset and add multiprocessing
            mp_options = grain.MultiprocessingOptions(
                num_workers=2,
                per_worker_buffer_size=4,
            )
            iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

            # Count occurrences by source
            source_counts = {"group1": 0, "group2": 0}
            for item in iter_ds:
                source = item.source[0]
                source_counts[source] += 1

            # Verify proportions (±5% tolerance due to randomness)
            total = sum(source_counts.values())
            assert total == 1000

            group1_proportion = source_counts["group1"] / total
            group2_proportion = source_counts["group2"] / total

            assert abs(group1_proportion - 0.7) < 0.05
            assert abs(group2_proportion - 0.3) < 0.05


class TestCombinedMultithreadingMultiprocessing:
    """Test combining multithreading and multiprocessing."""

    def test_combined_multithreading_multiprocessing(self):
        """Test using both ReadOptions (multithreading) and mp_prefetch (multiprocessing) together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 15)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 15)

            # Create balanced dataset
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                },
                shuffle=True,
                shuffle_seed=42,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 60))

            # Convert to IterDataset with multithreading
            read_options = grain.ReadOptions(
                num_threads=2,
                prefetch_buffer_size=2,
            )
            iter_ds = ds.to_iter_dataset(read_options=read_options)

            # Add multiprocessing
            # Note: This creates num_workers * num_threads total threads
            mp_options = grain.MultiprocessingOptions(
                num_workers=2,
                per_worker_buffer_size=2,
            )
            iter_ds = iter_ds.mp_prefetch(options=mp_options)

            # Verify iteration works correctly
            count = 0
            sources_found = set()
            for item in iter_ds:
                assert isinstance(item, AudioTree)
                sources_found.add(item.source[0])
                count += 1

            assert count == 60
            assert sources_found == {"group1", "group2"}
