"""Tests for create_balanced_audio_dataset function."""

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import (
    WindowParams,
    create_balanced_audio_dataset,
    find_audio_files,
)
from audiotree.sources.core import READ_ERROR_KEY, AudioReadError, _derive_group_seed


def _corrupt_audio_file(directory, name="bad.wav"):
    """Write a zero-byte ``name`` into ``directory`` and return its path."""
    path = Path(directory) / name
    path.write_bytes(b"")
    return path


def _create_test_audio_files(
    tmpdir, group_name, num_files, sample_rate=44100, duration=1.0
):
    """Helper to create test audio files."""
    group_dir = Path(tmpdir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration)
    for i in range(num_files):
        # Create audio with identifiable pattern (group encoded in amplitude)
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1
        filepath = group_dir / f"audio_{i}.wav"
        sf.write(str(filepath), audio, sample_rate)

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
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            assert len(ds) == 20

    def test_random_access(self):
        """Test that dataset supports random access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

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
                weights={"group1": 1.0, "group2": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            ).slice(slice(0, 100))

            assert len(ds) == 100

    def test_custom_weights(self):
        """Test that custom weights are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                weights={"group1": 0.7, "group2": 0.3},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 100))

            assert len(ds) == 100

    def test_no_shuffle(self):
        """Test deterministic ordering with shuffle=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds1 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                shuffle=False,
                shuffle_seed=42,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            ds2 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                shuffle=False,
                shuffle_seed=42,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            # Same seed + no shuffle should give same data
            for i in range(10):
                audio1 = ds1[i].waveform
                audio2 = ds2[i].waveform
                np.testing.assert_array_equal(audio1, audio2)

    def test_different_seeds(self):
        """Test that different seeds produce different orderings when shuffled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10)

            ds1 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                shuffle=True,
                shuffle_seed=42,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            ds2 = create_balanced_audio_dataset(
                sources={"group1": [group1_dir], "group2": [group2_dir]},
                shuffle=True,
                shuffle_seed=123,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            # Different seeds should (very likely) give different orderings
            different = False
            for i in range(10):
                audio1 = ds1[i].waveform
                audio2 = ds2[i].waveform
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
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

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
                weights={"group1": 2.0},  # Others default to 1.0
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 30))

            assert len(ds) == 30

    def test_returns_audiotree(self):
        """Test that items are AudioTree instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree import AudioTree

            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)

            ds = create_balanced_audio_dataset(
                sources={"group1": [group1_dir]},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 10))

            item = ds[0]
            assert isinstance(item, AudioTree)

    def test_source_property(self):
        """Test that AudioTree items have the source property set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "music", 5)
            group2_dir = _create_test_audio_files(tmpdir, "speech", 5)

            ds = create_balanced_audio_dataset(
                sources={"music": [group1_dir], "speech": [group2_dir]},
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            # Check that each item has a source property
            sources_found = set()
            for i in range(len(ds)):
                item = ds[i]
                source = item.source
                assert len(source) == 1, "Each item should have exactly one source"
                assert source[0] in ["music", "speech"], (
                    f"Source should be 'music' or 'speech', got {source[0]}"
                )
                sources_found.add(source[0])

            # Both sources should be represented
            assert sources_found == {"music", "speech"}, (
                "Both source groups should be present"
            )

    def test_source_property_batched(self):
        """The source property survives collation through AudioTree.batch."""

        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "music", 5)
            group2_dir = _create_test_audio_files(tmpdir, "speech", 5)

            ds = create_balanced_audio_dataset(
                sources={"music": [group1_dir], "speech": [group2_dir]},
                shuffle=False,
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            items = [ds[i] for i in range(4)]
            batch = AudioTree.batch(items)

            # Check source property
            sources = batch.source
            assert len(sources) == 4, f"Batch should have 4 sources, got {len(sources)}"
            for src in sources:
                assert src in ["music", "speech"], (
                    f"Source should be 'music' or 'speech', got {src}"
                )

    def test_mix_with_preconstructed_datasets(self):
        """Test mixing file sources with pre-constructed datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree.sources import create_audio_dataset

            # Create file-based source
            group1_dir = _create_test_audio_files(tmpdir, "speech", 10)

            # Create pre-constructed dataset
            group2_dir = _create_test_audio_files(tmpdir, "music", 10)
            preprocessed_ds = create_audio_dataset(
                sources=group2_dir,
                shuffle=True,
                num_epochs=None,
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=999,
            )

            # Mix them together
            ds = create_balanced_audio_dataset(
                sources={"speech": [group1_dir]},
                datasets={"music": preprocessed_ds},
                weights={"speech": 0.6, "music": 0.4},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            assert len(ds) == 20

            # Verify items load
            item = ds[0]
            assert isinstance(item, AudioTree)

    def test_mixed_file_and_preconstructed_groups_batch_together(self):
        """A file group and a pre-built dataset group must share a metadata schema.

        The docstring's own example mixes ``sources=`` with ``datasets=``. File
        groups are stamped with ``source=group_name``; a pre-built dataset built
        with ``create_audio_dataset``'s default ``source=None`` used to carry no
        such key, so any batch spanning both groups failed
        ``AudioTree.batch``'s pytree check on the missing ``source`` key --
        intermittently, since single-group batches collate fine.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree.sources import create_audio_dataset

            speech_dir = _create_test_audio_files(tmpdir, "speech", 5)
            music_dir = _create_test_audio_files(tmpdir, "music", 5)

            # Pre-built exactly as the docstring shows: default source=None.
            preprocessed_ds = create_audio_dataset(
                sources=music_dir,
                num_epochs=None,
                sample_rate=44100,
                duration=0.5,
            )

            ds = create_balanced_audio_dataset(
                sources={"speech": [speech_dir]},
                datasets={"preprocessed": preprocessed_ds},
                weights={"speech": 0.7, "preprocessed": 0.3},
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            ).slice(slice(0, 40))

            items = [ds[i] for i in range(len(ds))]

            # Both groups expose an identical metadata schema, each carrying a
            # `source` set to its group name.
            keys_by_source: Dict[str, set] = {}
            for item in items:
                assert len(item.source) == 1
                keys_by_source.setdefault(item.source[0], set()).update(
                    item.metadata.keys()
                )
            assert set(keys_by_source) == {"speech", "preprocessed"}
            assert keys_by_source["speech"] == keys_by_source["preprocessed"]
            assert "source" in keys_by_source["preprocessed"]

            # A cross-group slice actually collates -- what used to crash.
            sources_seen = {item.source[0] for item in items[:12]}
            assert sources_seen == {"speech", "preprocessed"}, (
                "test needs a slice spanning both groups"
            )
            batch = AudioTree.batch(items[:12])
            assert batch.waveform.shape == (12, 1, 22050)
            for src in batch.source:
                assert src in ("speech", "preprocessed")

    def test_preconstructed_source_is_overwritten_by_group_name(self):
        """The group name the caller chose wins over a dataset's own `source`.

        A pre-built dataset may already carry a `source` (it was built with
        `create_audio_dataset(source=...)`); the balanced-group name the caller
        assigns here is the authoritative one, matching how file groups override
        whatever `create_audio_dataset` was told.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree.sources import create_audio_dataset

            music_dir = _create_test_audio_files(tmpdir, "music", 5)
            preprocessed_ds = create_audio_dataset(
                sources=music_dir,
                num_epochs=None,
                sample_rate=44100,
                duration=0.5,
                source="its_own_name",
            )

            ds = create_balanced_audio_dataset(
                datasets={"preprocessed": preprocessed_ds},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 10))

            for i in range(len(ds)):
                assert ds[i].source == ["preprocessed"]

    def test_datasets_only(self):
        """Test using only pre-constructed datasets without file sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree.sources import create_audio_dataset

            # Create two pre-constructed datasets
            group1_dir = _create_test_audio_files(tmpdir, "group1", 5)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 5)

            ds1 = create_audio_dataset(
                sources=group1_dir, num_epochs=None, sample_rate=44100, duration=0.5
            )
            ds2 = create_audio_dataset(
                sources=group2_dir, num_epochs=None, sample_rate=44100, duration=0.5
            )

            # Mix only datasets, no file sources
            mixed = create_balanced_audio_dataset(
                datasets={"ds1": ds1, "ds2": ds2},
                weights={"ds1": 2.0, "ds2": 1.0},
            ).slice(slice(0, 20))

            assert len(mixed) == 20

    def test_no_sources_or_datasets_error(self):
        """Test that providing neither sources nor datasets raises an error."""
        try:
            create_balanced_audio_dataset()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "At least one of 'sources' or 'datasets' must be provided" in str(e)

    def test_different_sized_file_sources(self):
        """Test that file sources with different sizes are handled correctly.

        File-based sources are automatically repeated before mixing, so
        different-sized source directories work correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create very different sized file sources
            small_dir = _create_test_audio_files(tmpdir, "small", 3)  # 3 files
            large_dir = _create_test_audio_files(tmpdir, "large", 50)  # 50 files

            ds = create_balanced_audio_dataset(
                sources={"small": [small_dir], "large": [large_dir]},
                weights={"small": 0.5, "large": 0.5},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 200))

            # Should successfully create 200 items despite small group having only 3 files
            assert len(ds) == 200

            # Count occurrences
            source_counts = {"small": 0, "large": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Both sources should be represented roughly equally
            total = sum(source_counts.values())
            for group_name in ["small", "large"]:
                proportion = source_counts[group_name] / total
                assert abs(proportion - 0.5) < 0.1, (
                    f"{group_name} proportion {proportion:.3f} should be ~0.5"
                )


class TestBalancedDatasetValidation:
    """Configuration mistakes must fail loudly instead of silently changing data."""

    def test_per_group_seeds_follow_the_group_name_not_its_position(self):
        """Reordering or extending `sources` must not disturb a group's stream.

        Seeds used to be drawn positionally from one RNG, so inserting a group
        (or writing the mapping in a different order) silently handed every
        later group a different stream -- a different corpus for the same config.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = _create_test_audio_files(tmpdir, "a", 8)
            b_dir = _create_test_audio_files(tmpdir, "b", 8)
            c_dir = _create_test_audio_files(tmpdir, "c", 8)

            def a_filepaths(sources):
                ds = create_balanced_audio_dataset(
                    sources=sources,
                    sample_rate=44100,
                    duration=0.5,
                    shuffle_seed=42,
                ).slice(slice(0, 60))
                return [
                    ds[i].filepath[0] for i in range(len(ds)) if ds[i].source[0] == "a"
                ]

            baseline = a_filepaths({"a": [a_dir], "b": [b_dir]})
            assert baseline  # the group is actually represented

            # Same groups, written in the other order.
            reordered = a_filepaths({"b": [b_dir], "a": [a_dir]})
            assert reordered == baseline

            # A third group appears. "a" is drawn less often, but the items it
            # does yield are still the start of the same stream.
            extended = a_filepaths({"a": [a_dir], "c": [c_dir], "b": [b_dir]})
            assert extended
            assert extended == baseline[: len(extended)]

    def test_shuffle_and_excerpt_streams_are_independent(self):
        """A group's shuffle and excerpt seeds differ even from one base seed."""
        assert _derive_group_seed(0, "music", "shuffle") != _derive_group_seed(
            0, "music", "excerpt"
        )
        assert _derive_group_seed(0, "music", "shuffle") != _derive_group_seed(
            0, "speech", "shuffle"
        )
        # Deterministic across processes/runs.
        assert _derive_group_seed(7, "music", "shuffle") == _derive_group_seed(
            7, "music", "shuffle"
        )

    def test_num_epochs_replaces_the_repeat_flag(self):
        """`num_epochs` counts passes per group; None (the default) is infinite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            small_dir = _create_test_audio_files(tmpdir, "small", 3)
            large_dir = _create_test_audio_files(tmpdir, "large", 7)
            sources = {"small": [small_dir], "large": [large_dir]}

            # The default repeats forever, so the mixture is unbounded.
            infinite = create_balanced_audio_dataset(
                sources=sources, sample_rate=44100, duration=0.5
            )
            assert len(infinite) == sys.maxsize
            explicit = create_balanced_audio_dataset(
                sources=sources, num_epochs=None, sample_rate=44100, duration=0.5
            )
            assert len(explicit) == len(infinite)

            # A finite count bounds the mixture by the smallest group, as
            # documented for grain.MapDataset.mix.
            one = create_balanced_audio_dataset(
                sources=sources, num_epochs=1, sample_rate=44100, duration=0.5
            )
            three = create_balanced_audio_dataset(
                sources=sources, num_epochs=3, sample_rate=44100, duration=0.5
            )
            assert len(one) < sys.maxsize
            assert len(three) == 3 * len(one)
            assert isinstance(three[0], AudioTree)

    def test_num_epochs_validation(self):
        """Zero, negative, and the old boolean spelling must all fail loudly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group_dir = _create_test_audio_files(tmpdir, "group1", 2)
            sources = {"group1": [group_dir]}

            with pytest.raises(ValueError, match="num_epochs must be >= 1"):
                create_balanced_audio_dataset(sources=sources, num_epochs=0)
            with pytest.raises(ValueError, match="num_epochs must be >= 1"):
                create_balanced_audio_dataset(sources=sources, num_epochs=-2)
            with pytest.raises(TypeError, match="num_epochs must be an int or None"):
                create_balanced_audio_dataset(sources=sources, num_epochs=True)
            with pytest.raises(TypeError, match="repeat"):
                create_balanced_audio_dataset(sources=sources, repeat=True)

    def test_duration_with_window_params_raises(self):
        """`duration` is meaningless under windowed sampling, so it must not be ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group_dir = _create_test_audio_files(tmpdir, "group1", 2)

            with pytest.raises(ValueError, match="window_params.*duration"):
                create_balanced_audio_dataset(
                    sources={"group1": [group_dir]},
                    duration=0.25,
                    window_params=WindowParams(duration=1.0),
                )

    def test_unknown_weight_key_raises(self):
        """A typo'd group name in `weights` silently dropped that group's weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 2)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 2)

            with pytest.raises(ValueError, match="grpu1"):
                create_balanced_audio_dataset(
                    sources={"group1": [group1_dir], "group2": [group2_dir]},
                    weights={"grpu1": 2.0, "group2": 1.0},
                    sample_rate=44100,
                    duration=0.5,
                )

    def test_group_name_in_both_sources_and_datasets_raises(self):
        """A name in both `sources` and `datasets` silently doubled its share.

        Each build loop appends a dataset and looks up the same
        ``weights.get(name)``, so a colliding name produced two parents at one
        weight -- doubling its effective proportion. The unknown-weight-key
        check can't catch it (the key *is* known), so it needs its own guard.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            from audiotree.sources import create_audio_dataset

            music_dir = _create_test_audio_files(tmpdir, "music", 4)
            prebuilt = create_audio_dataset(
                sources=music_dir,
                num_epochs=None,
                sample_rate=44100,
                duration=0.5,
            )

            with pytest.raises(ValueError, match="music"):
                create_balanced_audio_dataset(
                    sources={"music": [music_dir]},
                    datasets={"music": prebuilt},
                    weights={"music": 0.3},
                    sample_rate=44100,
                    duration=0.5,
                )

    def test_empty_sources_raises(self):
        """An empty mapping used to die inside grain with `min() iterable argument is empty`."""
        with pytest.raises(ValueError, match="No groups to mix"):
            create_balanced_audio_dataset(sources={})

        with pytest.raises(ValueError, match="No groups to mix"):
            create_balanced_audio_dataset(sources={}, datasets={})

    def test_path_sources_are_accepted(self):
        """`pathlib.Path` works wherever a `str` path does."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 3)

            ds = create_balanced_audio_dataset(
                sources={"group1": [Path(group1_dir)]},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 4))
            assert isinstance(ds[0], AudioTree)

            # A bare Path (not wrapped in a list) is a source too.
            ds = create_balanced_audio_dataset(
                sources={"group1": Path(group1_dir)},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 4))
            assert isinstance(ds[0], AudioTree)

            assert find_audio_files(Path(group1_dir)) == find_audio_files(group1_dir)

    def test_source_matching_nothing_warns(self):
        """One typo'd or unmounted path used to shrink the corpus silently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 2)
            missing = str(Path(tmpdir) / "not_mounted")

            with pytest.warns(UserWarning, match="not_mounted"):
                found = find_audio_files([group1_dir, missing])
            assert len(found) == 2

            with pytest.warns(UserWarning, match="not_mounted"):
                create_balanced_audio_dataset(
                    sources={"group1": [group1_dir, missing]},
                    sample_rate=44100,
                    duration=0.5,
                )

    def test_mixed_channel_corpus_names_the_offending_file(self):
        """A stray stereo file used to blow up at batch time with no filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group_dir = Path(tmpdir) / "group1"
            group_dir.mkdir()
            sf.write(
                str(group_dir / "audio_0.wav"),
                np.zeros((44100,), dtype=np.float32),
                44100,
            )
            sf.write(
                str(group_dir / "audio_1.wav"),
                np.zeros((44100, 2), dtype=np.float32),
                44100,
            )

            ds = create_balanced_audio_dataset(
                sources={"group1": [str(group_dir)]},
                sample_rate=44100,
                duration=0.5,
                mono=False,
                shuffle=False,
            ).slice(slice(0, 4))

            with pytest.raises(ValueError, match=r"audio_1\.wav has 2 channels"):
                for i in range(len(ds)):
                    _ = ds[i]

            # mono=True mixes everything down, so the corpus loads fine.
            mono_ds = create_balanced_audio_dataset(
                sources={"group1": [str(group_dir)]},
                sample_rate=44100,
                duration=0.5,
                mono=True,
                shuffle=False,
            ).slice(slice(0, 4))
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert all(mono_ds[i].num_channels == 1 for i in range(len(mono_ds)))

    def test_a_corrupt_file_ends_the_mix_by_default(self):
        """The default policy is unchanged: the run stops, naming the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = _create_test_audio_files(tmpdir, "a", 2)
            b_dir = _create_test_audio_files(tmpdir, "b", 2)
            _corrupt_audio_file(b_dir)

            ds = create_balanced_audio_dataset(
                sources={"a": [a_dir], "b": [b_dir]},
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            ).slice(slice(0, 12))

            with pytest.raises(AudioReadError, match=r"bad\.wav"):
                for i in range(len(ds)):
                    _ = ds[i]

    def test_on_read_error_reaches_every_file_based_group(self):
        """One corrupt file per group is substituted, and the mix still batches.

        The marker lands on *every* item, from every group, so the groups
        collate with each other -- a substitute in one group and a real load in
        another must agree on their metadata keys.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = _create_test_audio_files(tmpdir, "a", 2)
            b_dir = _create_test_audio_files(tmpdir, "b", 2)
            _corrupt_audio_file(a_dir)
            _corrupt_audio_file(b_dir)

            ds = create_balanced_audio_dataset(
                sources={"a": [a_dir], "b": [b_dir]},
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
                on_read_error="skip",
            ).slice(slice(0, 12))

            items = [ds[i] for i in range(len(ds))]
            batch = AudioTree.batch(items)
            assert batch.waveform.shape == (12, 1, 22050)

            flags = np.asarray(batch.metadata[READ_ERROR_KEY])
            # Each group cycles over 3 files, one of which is corrupt, so a
            # third of the mix is substituted -- and both groups contribute.
            assert flags.sum() == 4
            substituted_groups = {
                source for source, flag in zip(batch.source, flags) if flag
            }
            assert substituted_groups == {"a", "b"}
            for waveform, flag in zip(batch.waveform, flags):
                assert bool(np.any(np.asarray(waveform))) is not bool(flag)

    def test_on_read_error_is_rejected_with_window_params(self):
        """Windowed sampling has its own loader, so the knob must not look wired."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = _create_test_audio_files(tmpdir, "a", 2)

            with pytest.raises(ValueError, match="does not support `on_read_error`"):
                create_balanced_audio_dataset(
                    sources={"a": [a_dir]},
                    window_params=WindowParams(duration=0.5),
                    on_read_error="skip",
                )


def _generate_sine_tone(
    frequency: float, duration: float, sample_rate: int = 44100
) -> np.ndarray:
    """Generate a simple sine tone for testing.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Float32 audio array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    return (np.sin(2 * np.pi * frequency * t) * 0.1).astype(np.float32)


def _create_hierarchical_test_data(
    tmpdir: str,
    structure: Dict[str, Dict[str, int]],
    sample_rate: int = 44100,
    duration: float = 1.0,
) -> Dict[str, str]:
    """Create hierarchical audio test data.

    Args:
        tmpdir: Temporary directory
        structure: Dict like {
            "group1": {"sub1": 10, "sub2": 15},  # group1 has 2 subdirs with 10 and 15 files
            "group2": {"subA": 5, "subB": 5, "subC": 10},  # group2 has 3 subdirs
        }
        sample_rate: Sample rate for generated audio
        duration: Duration in seconds for each file

    Returns:
        Dict mapping group names to group directory paths
    """
    base_path = Path(tmpdir)
    group_paths = {}

    for group_name, subdirs in structure.items():
        group_dir = base_path / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        for subdir_name, num_files in subdirs.items():
            subdir_path = group_dir / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)

            # Generate sine tones with slightly different frequencies for variety
            frequency = 440.0 + hash(f"{group_name}_{subdir_name}") % 100
            audio = _generate_sine_tone(frequency, duration, sample_rate)

            for i in range(num_files):
                filepath = subdir_path / f"audio_{i:03d}.wav"
                sf.write(str(filepath), audio, sample_rate)

        group_paths[group_name] = str(group_dir)

    return group_paths


class TestBalancedDatasetHierarchical:
    """Comprehensive tests for hierarchical directory structures and balancing."""

    def test_hierarchical_file_discovery(self):
        """Test 1: Verify rglob correctly discovers all files in nested subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {
                    "subgroup1": 10,
                    "subgroup2": 15,
                    "subgroup3": 8,
                    "subgroup4": 12,
                },
                "group2": {"subgroup1": 7, "subgroup2": 5, "subgroup3": 8},
                "group3": {"subgroup1": 3, "subgroup2": 2},
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            # Create dataset with small num records to test file discovery
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            ).slice(slice(0, 100))

            # Verify dataset was created successfully
            assert len(ds) == 100

            # Load all items and verify we can access files from all subdirectories
            sources_found = set()
            for i in range(len(ds)):
                item = ds[i]
                assert isinstance(item, AudioTree)
                sources_found.add(item.source[0])

            # All three groups should be represented
            assert sources_found == {"group1", "group2", "group3"}

    def test_equal_weights_with_unbalanced_groups(self):
        """Test 2: Verify equal weights produce equal representation despite different group sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {"sub1": 20, "sub2": 25},  # 45 total files
                "group2": {"sub1": 10, "sub2": 10},  # 20 total files
                "group3": {"sub1": 3, "sub2": 2},  # 5 total files
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                weights={"group1": 1.0, "group2": 1.0, "group3": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 900))

            # Count occurrences by source
            source_counts = {"group1": 0, "group2": 0, "group3": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # With equal weights, each group should appear ~300 times (±5%)
            total = sum(source_counts.values())
            for group_name in ["group1", "group2", "group3"]:
                proportion = source_counts[group_name] / total
                expected_proportion = 1.0 / 3.0
                assert abs(proportion - expected_proportion) < 0.05, (
                    f"{group_name} proportion {proportion:.3f} differs from expected "
                    f"{expected_proportion:.3f} by more than 5%"
                )

    def test_custom_weights_with_unbalanced_groups(self):
        """Test 3: Verify custom weights work correctly regardless of underlying file counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {"sub1": 20, "sub2": 25},  # 45 total files
                "group2": {"sub1": 10, "sub2": 10},  # 20 total files
                "group3": {"sub1": 3, "sub2": 2},  # 5 total files
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                weights={"group1": 0.5, "group2": 0.3, "group3": 0.2},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 1000))

            # Count occurrences by source
            source_counts = {"group1": 0, "group2": 0, "group3": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Verify proportions match weights (±5% tolerance)
            total = sum(source_counts.values())
            expected_weights = {"group1": 0.5, "group2": 0.3, "group3": 0.2}
            for group_name, expected_weight in expected_weights.items():
                actual_proportion = source_counts[group_name] / total
                assert abs(actual_proportion - expected_weight) < 0.05, (
                    f"{group_name} proportion {actual_proportion:.3f} differs from expected "
                    f"{expected_weight:.3f} by more than 5%"
                )

    def test_multiple_subdirectories_aggregation(self):
        """Test 4: Verify multiple subdirectories within a group are correctly aggregated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {
                    "subA": 5,
                    "subB": 5,
                    "subC": 5,
                    "subD": 5,
                },  # 4 subdirs, 20 total
                "group2": {"subA": 20},  # 1 subdir, 20 total
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                },
                weights={"group1": 1.0, "group2": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 200))

            # Collect filepaths for group1
            group1_filepaths = []
            source_counts = {"group1": 0, "group2": 0}

            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

                if source == "group1":
                    filepath = item.filepath[0]
                    group1_filepaths.append(filepath)

            # Verify both groups appear roughly equally
            total = sum(source_counts.values())
            for group_name in ["group1", "group2"]:
                proportion = source_counts[group_name] / total
                assert abs(proportion - 0.5) < 0.1

            # Verify filepaths from group1 span all subdirectories
            subdirs_found = set()
            for filepath in group1_filepaths:
                parts = Path(filepath).parts
                # Find the subdirectory name (should be between group1 and filename)
                for i, part in enumerate(parts):
                    if part == "group1" and i + 1 < len(parts):
                        subdirs_found.add(parts[i + 1])
                        break

            # All 4 subdirectories should be represented
            assert subdirs_found == {"subA", "subB", "subC", "subD"}

    def test_extreme_imbalance(self):
        """Test 5: Test balancing works with extreme size ratios (50:1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "large_group": {"sub1": 50, "sub2": 50},  # 100 files
                "small_group": {"sub1": 2},  # 2 files
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "large_group": [group_paths["large_group"]],
                    "small_group": [group_paths["small_group"]],
                },
                weights={"large_group": 1.0, "small_group": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 400))

            # Count occurrences
            source_counts = {"large_group": 0, "small_group": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Verify 50/50 split despite 50:1 file ratio
            total = sum(source_counts.values())
            for group_name in ["large_group", "small_group"]:
                proportion = source_counts[group_name] / total
                assert abs(proportion - 0.5) < 0.1, (
                    f"{group_name} proportion {proportion:.3f} should be ~0.5"
                )

    def test_statistical_distribution_accuracy(self):
        """Test 6: Verify actual distribution matches requested weights within statistical tolerance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {"sub1": 30},
                "group2": {"sub1": 20},
                "group3": {"sub1": 10},
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                weights={"group1": 0.5, "group2": 0.3, "group3": 0.2},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 10000))

            # Count all occurrences
            source_counts = {"group1": 0, "group2": 0, "group3": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Verify with tight tolerance (±1%)
            total = sum(source_counts.values())
            expected_weights = {"group1": 0.5, "group2": 0.3, "group3": 0.2}
            for group_name, expected_weight in expected_weights.items():
                actual_proportion = source_counts[group_name] / total
                assert abs(actual_proportion - expected_weight) < 0.01, (
                    f"{group_name}: expected {expected_weight:.3f}, got {actual_proportion:.3f} "
                    f"(count: {source_counts[group_name]})"
                )

    def test_subdirectory_count_variation(self):
        """Test 7: Verify balancing is based on total files per group, not subdirectory count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {
                    "sub1": 2,
                    "sub2": 2,
                    "sub3": 2,
                    "sub4": 2,
                    "sub5": 2,
                },  # 5 subdirs × 2 = 10
                "group2": {"sub1": 10},  # 1 subdir × 10 = 10
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                },
                weights={"group1": 1.0, "group2": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 200))

            # Count occurrences
            source_counts = {"group1": 0, "group2": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Both groups should appear equally despite different subdirectory counts
            total = sum(source_counts.values())
            for group_name in ["group1", "group2"]:
                proportion = source_counts[group_name] / total
                assert abs(proportion - 0.5) < 0.1, (
                    f"{group_name} proportion {proportion:.3f} should be ~0.5"
                )

    def test_deep_nesting(self):
        """Test 8: Test that arbitrarily deep directory nesting works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            # Create deeply nested structure for group1
            deep_path = base_path / "group1" / "a" / "b" / "c" / "d" / "e"
            deep_path.mkdir(parents=True, exist_ok=True)
            audio = _generate_sine_tone(440.0, 1.0)
            for i in range(5):
                sf.write(str(deep_path / f"deep_{i}.wav"), audio, 44100)

            # Create flat structure for group2
            flat_path = base_path / "group2" / "flat"
            flat_path.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                sf.write(str(flat_path / f"flat_{i}.wav"), audio, 44100)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [str(base_path / "group1")],
                    "group2": [str(base_path / "group2")],
                },
                weights={"group1": 1.0, "group2": 1.0},
                sample_rate=44100,
                duration=0.5,
            ).slice(slice(0, 20))

            # Verify both groups are discovered and accessible
            source_counts = {"group1": 0, "group2": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # Both groups should be represented
            assert source_counts["group1"] > 0
            assert source_counts["group2"] > 0

    def test_mixed_depth_hierarchy(self):
        """Test 9: Verify groups can have different nesting depths without affecting balance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            audio = _generate_sine_tone(440.0, 1.0)

            # group1: flat structure (10 files in root)
            group1_path = base_path / "group1"
            group1_path.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                sf.write(str(group1_path / f"audio_{i}.wav"), audio, 44100)

            # group2: 1-level nesting (2 subdirs × 5 files = 10 files)
            group2_path = base_path / "group2"
            for subdir in ["sub1", "sub2"]:
                subdir_path = group2_path / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                for i in range(5):
                    sf.write(str(subdir_path / f"audio_{i}.wav"), audio, 44100)

            # group3: 2-level nesting (2 dirs × 2 subdirs × 2-3 files = 10 files)
            group3_path = base_path / "group3"
            for dir1 in ["dir1", "dir2"]:
                for dir2 in ["subA", "subB"]:
                    nested_path = group3_path / dir1 / dir2
                    nested_path.mkdir(parents=True, exist_ok=True)
                    num_files = 3 if dir1 == "dir1" and dir2 == "subA" else 2
                    for i in range(num_files):
                        sf.write(str(nested_path / f"audio_{i}.wav"), audio, 44100)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [str(group1_path)],
                    "group2": [str(group2_path)],
                    "group3": [str(group3_path)],
                },
                weights={"group1": 1.0, "group2": 1.0, "group3": 1.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 300))

            # Count occurrences
            source_counts = {"group1": 0, "group2": 0, "group3": 0}
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                source_counts[source] += 1

            # All groups should appear equally despite different depths
            total = sum(source_counts.values())
            for group_name in ["group1", "group2", "group3"]:
                proportion = source_counts[group_name] / total
                expected = 1.0 / 3.0
                assert abs(proportion - expected) < 0.1

    def test_source_tracking_in_nested_structures(self):
        """Test 10: Verify AudioTree.source always reflects the group name, not subdirectory names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            audio = _generate_sine_tone(440.0, 1.0)

            # Create nested structure
            nested_path = base_path / "group1" / "subA" / "subB"
            nested_path.mkdir(parents=True, exist_ok=True)
            sf.write(str(nested_path / "audio.wav"), audio, 44100)

            # Create flat structure
            flat_path = base_path / "group2"
            flat_path.mkdir(parents=True, exist_ok=True)
            sf.write(str(flat_path / "audio.wav"), audio, 44100)

            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [str(base_path / "group1")],
                    "group2": [str(base_path / "group2")],
                },
                sample_rate=44100,
                duration=0.5,
                shuffle=False,
            ).slice(slice(0, 20))

            # Verify source property
            for i in range(len(ds)):
                item = ds[i]
                source = item.source[0]
                assert source in ["group1", "group2"]
                # Source should be group name, not subdirectory name
                assert source != "subA"
                assert source != "subB"

    def test_deterministic_balancing(self):
        """Test 11: Verify same seed produces identical sequence even with complex balancing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {"sub1": 10, "sub2": 10},
                "group2": {"sub1": 5, "sub2": 5},
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            ds1 = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                },
                weights={"group1": 0.7, "group2": 0.3},
                sample_rate=44100,
                duration=0.5,
                shuffle=True,
                shuffle_seed=42,
            ).slice(slice(0, 100))

            ds2 = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                },
                weights={"group1": 0.7, "group2": 0.3},
                sample_rate=44100,
                duration=0.5,
                shuffle=True,
                shuffle_seed=42,
            ).slice(slice(0, 100))

            # Verify identical sequences
            for i in range(len(ds1)):
                item1 = ds1[i]
                item2 = ds2[i]
                assert item1.filepath[0] == item2.filepath[0]
                assert item1.source[0] == item2.source[0]
                np.testing.assert_array_equal(item1.waveform, item2.waveform)

    def test_weight_normalization(self):
        """Test 12: Verify weights are normalized correctly (don't need to sum to 1.0)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = {
                "group1": {"sub1": 10},
                "group2": {"sub1": 10},
                "group3": {"sub1": 10},
            }
            group_paths = _create_hierarchical_test_data(tmpdir, structure)

            # Create dataset with unnormalized weights
            ds_unnormalized = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                weights={"group1": 2.0, "group2": 3.0, "group3": 5.0},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 1000))

            # Create dataset with normalized weights (equivalent)
            ds_normalized = create_balanced_audio_dataset(
                sources={
                    "group1": [group_paths["group1"]],
                    "group2": [group_paths["group2"]],
                    "group3": [group_paths["group3"]],
                },
                weights={"group1": 0.2, "group2": 0.3, "group3": 0.5},
                sample_rate=44100,
                duration=0.5,
                shuffle_seed=42,
            ).slice(slice(0, 1000))

            # Count occurrences for both datasets
            counts_unnorm = {"group1": 0, "group2": 0, "group3": 0}
            counts_norm = {"group1": 0, "group2": 0, "group3": 0}

            for i in range(1000):
                counts_unnorm[ds_unnormalized[i].source[0]] += 1
                counts_norm[ds_normalized[i].source[0]] += 1

            # Both should produce the same proportions
            total_unnorm = sum(counts_unnorm.values())
            total_norm = sum(counts_norm.values())

            for group_name in ["group1", "group2", "group3"]:
                prop_unnorm = counts_unnorm[group_name] / total_unnorm
                prop_norm = counts_norm[group_name] / total_norm
                assert abs(prop_unnorm - prop_norm) < 0.01, (
                    f"{group_name}: unnormalized {prop_unnorm:.3f} != normalized {prop_norm:.3f}"
                )
