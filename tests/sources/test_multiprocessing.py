"""Tests for multiprocessing/multithreading with audio datasets."""

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import grain
import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset


def _create_test_audio_files(
    tmpdir,
    group_name,
    num_files,
    sample_rate=44100,
    duration: Union[float, Sequence[float]] = 0.5,
):
    """Helper to create test audio files.

    Args:
        tmpdir: Directory to create the group directory under.
        group_name: Name of the group directory.
        num_files: How many files to write.
        sample_rate: Sample rate to write them at.
        duration: Length in seconds of every file, or a sequence of lengths
            cycled over the files. Unequal lengths make a random excerpt offset
            depend on *which* file it was drawn from, not just on the index --
            which is what the single-process parity tests below compare on.

    Returns:
        The group directory as a string.
    """
    group_dir = Path(tmpdir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    durations = (
        [float(duration)] * num_files
        if isinstance(duration, (int, float))
        else [float(duration[i % len(duration)]) for i in range(num_files)]
    )
    for i, file_duration in enumerate(durations):
        # Create sine tone audio with frequency based on file index
        num_samples = int(sample_rate * file_duration)
        t = np.linspace(0, file_duration, num_samples)
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
                num_epochs=1,
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
                assert item.waveform.shape[0] == 1  # batch size 1
                assert item.waveform.shape[1] == 1  # mono
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
                assert item.waveform.shape[0] == 1  # batch size 1
                assert item.sample_rate == 44100
                sources_found.add(item.source[0])
                count += 1

            assert count == 40
            assert sources_found == {"group1", "group2"}


#: Grain's worker processes hand results to the parent through named shared
#: memory. On Windows a named mapping is destroyed as soon as its last handle
#: closes, so by the time the parent attaches the worker has already exited and
#: `SharedMemory(name=..., create=False)` raises
#: `FileNotFoundError: [WinError 2] ... 'wnsm_<id>'`. Grain supports Linux and
#: macOS only -- Windows support is google/grain#793, still open -- so this is
#: upstream, not something audiotree can work around. Single-process use is
#: fine on Windows and stays covered by the rest of the suite.
requires_grain_multiprocessing = pytest.mark.skipif(
    sys.platform == "win32",
    reason="grain multiprocessing is unsupported on Windows (google/grain#793)",
)


@requires_grain_multiprocessing
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
                num_epochs=1,
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
                assert item.waveform.shape[0] == 1  # batch size 1
                assert item.waveform.shape[1] == 1  # mono
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
                assert item.waveform.shape[0] == 1  # batch size 1
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


@requires_grain_multiprocessing
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


#: One item's identity: its group, the file it came from, and where in that
#: file the excerpt started. Object identity says nothing across a process
#: boundary (every item is unpickled fresh) and the waveform is a poor key
#: (two excerpts of one file can be byte-identical), so this triple is what
#: distinguishes a correct parallel stream from a corrupted one.
Item = Tuple[Tuple[str, ...], str, float]


def _item_identity(item: AudioTree) -> Item:
    """Reduce one dataset item to the extras that identifies it."""
    return (
        tuple(item.source),
        os.path.basename(item.filepath[0]),
        round(float(item.offset[0]), 6),
    )


def _drain(dataset, num_workers=None) -> List[Item]:
    """Iterate ``dataset`` to exhaustion, returning each item's identity.

    Args:
        dataset: A finite :class:`grain.MapDataset`.
        num_workers: ``None`` to iterate in this process, or a worker count to
            iterate through ``mp_prefetch``.
    """
    iter_dataset = dataset.to_iter_dataset()
    if num_workers is not None:
        iter_dataset = iter_dataset.mp_prefetch(
            grain.MultiprocessingOptions(
                num_workers=num_workers, per_worker_buffer_size=2
            )
        )
    return [_item_identity(item) for item in iter_dataset]


def _assert_same_stream(
    parallel: List[Item],
    reference: List[Item],
    num_workers: int,
    reference_name: str = "single-process",
):
    """Assert a worker-parallel stream carries exactly the reference items.

    Compares as a multiset, since sharding across workers is allowed to reorder
    the stream but never to drop, duplicate, or alter an item. The message
    names the three failure modes separately because they have different
    causes: missing items mean a dropped tail or shard, duplicates mean two
    workers were handed the same shard, and altered items (an item that is both
    missing and extra, differing only in its offset) mean the workers' excerpt
    RNGs disagree with the reference run's -- a per-worker seed collision.

    Args:
        parallel: Identities collected through ``mp_prefetch``.
        reference: Identities the stream is supposed to carry.
        num_workers: Worker count, for the message.
        reference_name: How to describe ``reference`` in the message.
    """
    parallel_counts = Counter(parallel)
    reference_counts = Counter(reference)
    if parallel_counts == reference_counts:
        return

    missing = reference_counts - parallel_counts
    extra = parallel_counts - reference_counts
    # An item present in both directions with a different offset was not
    # dropped -- it was loaded differently, i.e. from a different RNG.
    missing_files = {(source, name) for source, name, _ in missing}
    altered = sorted(
        (source, name) for source, name, _ in extra if (source, name) in missing_files
    )

    pytest.fail(
        f"mp_prefetch(num_workers={num_workers}) did not yield the "
        f"{reference_name} stream.\n"
        f"  items: {len(parallel)} parallel vs {len(reference)} {reference_name}\n"
        f"  missing (in {reference_name}, never produced): {sorted(missing.items())}\n"
        f"  extra (produced but not by {reference_name}): {sorted(extra.items())}\n"
        f"  same file, different excerpt offset (worker RNG divergence): "
        f"{altered}\n"
        "A worker count must not change the data: it shards the same indices "
        "across processes, so the multiset of items is invariant.",
        pytrace=False,
    )


@pytest.fixture(scope="module")
def parity_corpus(tmp_path_factory):
    """A corpus with unequal file lengths, shared by the parity tests.

    Fourteen files: not a multiple of four, so the last shard is short at
    ``num_workers=4`` and a dropped or duplicated tail cannot hide behind an
    even division.
    """
    tmpdir = tmp_path_factory.mktemp("parity")
    return _create_test_audio_files(
        tmpdir,
        "audio",
        14,
        sample_rate=16000,
        duration=(1.0, 1.5, 2.0),
    )


@pytest.fixture(scope="module")
def parity_groups(tmp_path_factory):
    """Two groups with unequal file lengths, for the balanced parity test."""
    tmpdir = tmp_path_factory.mktemp("parity_groups")
    return {
        "group1": _create_test_audio_files(
            tmpdir, "group1", 7, sample_rate=16000, duration=(1.0, 1.5, 2.0)
        ),
        "group2": _create_test_audio_files(
            tmpdir, "group2", 7, sample_rate=16000, duration=(1.25, 1.75)
        ),
    }


@requires_grain_multiprocessing
class TestMultiprocessingMatchesSingleProcess:
    """The worker count must not change the data, only who loads it.

    The tests above assert that ``mp_prefetch`` yields *something* AudioTree-
    shaped in the expected quantity, which duplicated shards, dropped tail
    batches, and per-worker seed collisions all survive: they keep the count
    and the type intact while changing which audio a run trains on. These
    compare the parallel stream against the single-process one item by item.
    """

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_simple_dataset_stream_matches_single_process(
        self, parity_corpus, num_workers
    ):
        """``create_audio_dataset`` yields the same items at any worker count."""
        dataset = create_audio_dataset(
            sources=parity_corpus,
            shuffle=True,
            shuffle_seed=17,
            num_epochs=1,
            sample_rate=16000,
            duration=0.5,
        )

        single = _drain(dataset)
        assert len(single) == 14
        # Every item distinct, or the multiset comparison below would not be
        # able to see a duplicated shard.
        assert len(set(single)) == len(single)

        _assert_same_stream(_drain(dataset, num_workers), single, num_workers)

    @pytest.mark.parametrize("num_workers", [2, 4])
    def test_balanced_dataset_stream_matches_single_process(
        self, parity_groups, num_workers
    ):
        """The balanced mixture is worker-count invariant too.

        A balanced dataset draws each group's excerpt RNG from a seed derived
        from the group's name, so a worker that re-derived seeds from its own
        index would show up here as the same files at different offsets.
        """
        # 30 is not a multiple of 4, so the last shard is short.
        dataset = create_balanced_audio_dataset(
            sources=parity_groups,
            shuffle=True,
            shuffle_seed=17,
            sample_rate=16000,
            duration=0.5,
        ).slice(slice(0, 30))

        single = _drain(dataset)
        assert len(single) == 30
        assert {source for source, _, _ in single} == {("group1",), ("group2",)}

        _assert_same_stream(_drain(dataset, num_workers), single, num_workers)

    def test_worker_counts_agree_with_each_other(self, parity_corpus):
        """Two different worker counts produce the same stream.

        Guards the case where the single-process path and the worker path are
        both wrong in the same way -- comparing two worker counts against each
        other would still catch a sharding bug that depends on the count.
        """
        dataset = create_audio_dataset(
            sources=parity_corpus,
            shuffle=True,
            shuffle_seed=23,
            num_epochs=1,
            sample_rate=16000,
            duration=0.5,
        )
        two = _drain(dataset, num_workers=2)
        four = _drain(dataset, num_workers=4)
        _assert_same_stream(four, two, num_workers=4, reference_name="num_workers=2")
