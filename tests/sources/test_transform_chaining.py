"""Tests for transform chaining with datasets as documented."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset
from audiotree.transforms import volume_norm, volume_change, trim, invert_phase, mono


def _create_test_audio_files(tmpdir, group_name, num_files, sample_rate=44100, duration=5.0):
    """Helper to create test audio files."""
    group_dir = Path(tmpdir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration)
    for i in range(num_files):
        # Create sine tone audio
        t = np.linspace(0, duration, num_samples)
        freq = 440 + i * 10
        audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
        filepath = group_dir / f"audio_{i:03d}.wav"
        sf.write(str(filepath), audio, sample_rate)

    return str(group_dir)


class TestBasicChaining:
    """Test basic transform chaining with datasets."""

    def test_single_random_map_transform(self):
        """Test chaining a single random transform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 10)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            )

            # Chain volume_norm transform
            transform = volume_norm(min_db=-20, max_db=-15)
            ds = ds.random_map(transform, seed=42)

            # Load item and verify transform was applied
            item = ds[0]
            assert item.loudness is not None
            assert item.loudness[0] >= -20 - 1  # Allow 1dB tolerance
            assert item.loudness[0] <= -15 + 1

    def test_single_map_transform(self):
        """Test chaining a single deterministic transform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 10, duration=5.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            )

            # Chain trim transform
            transform = trim(length=3.0)
            ds = ds.map(transform)

            # Load item and verify transform was applied
            item = ds[0]
            expected_length = int(3.0 * 44100)
            assert item.waveform.shape[-1] == expected_length

    def test_multiple_transform_chain(self):
        """Test chaining multiple transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 10, duration=5.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            )

            # Chain multiple transforms
            ds = ds.random_map(
                volume_norm(min_db=-20, max_db=-15),
                seed=42,
            )
            ds = ds.random_map(
                volume_change(min_db=-3, max_db=3, prob=1.0),
                seed=43,
            )
            ds = ds.map(trim(length=3.0))

            # Load item and verify all transforms were applied
            item = ds[0]

            # trim runs last and changes the audio length, so it invalidates
            # the loudness set by the earlier volume transforms.
            assert item.loudness is None

            # Check Trim was applied
            expected_length = int(3.0 * 44100)
            assert item.waveform.shape[-1] == expected_length


class TestBalancedDatasetChaining:
    """Test transform chaining with balanced datasets."""

    def test_chaining_with_balanced_dataset(self):
        """Test that transform chaining works with balanced datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            group1_dir = _create_test_audio_files(tmpdir, "group1", 10, duration=5.0)
            group2_dir = _create_test_audio_files(tmpdir, "group2", 10, duration=5.0)

            # Create balanced dataset
            ds = create_balanced_audio_dataset(
                sources={
                    "group1": [group1_dir],
                    "group2": [group2_dir],
                },
                shuffle=False,
                sample_rate=44100,
                duration=5.0,
            ).slice(slice(0, 40))

            # Chain transforms
            ds = ds.random_map(
                volume_norm(min_db=-20, max_db=-15),
                seed=42,
            )
            ds = ds.map(trim(length=3.0))

            # Verify transforms applied and source tracking preserved. trim runs
            # last and changes the length, invalidating volume_norm's loudness.
            item = ds[0]
            assert item.loudness is None
            assert item.waveform.shape[-1] == int(3.0 * 44100)
            assert item.source[0] in ["group1", "group2"]

    def test_transform_order_matters(self):
        """Test that transform order affects the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 5, duration=5.0)

            # Create base dataset
            base_ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=5.0,
            )

            # Option 1: Trim then normalize
            ds1 = base_ds.map(trim(length=3.0))
            ds1 = ds1.random_map(
                volume_norm(min_db=-20, max_db=-15),
                seed=42,
            )

            # Option 2: Normalize then trim
            ds2 = base_ds.random_map(
                volume_norm(min_db=-20, max_db=-15),
                seed=42,
            )
            ds2 = ds2.map(trim(length=3.0))

            # Both produce valid results
            item1 = ds1[0]
            item2 = ds2[0]

            assert item1.waveform.shape[-1] == int(3.0 * 44100)
            assert item2.waveform.shape[-1] == int(3.0 * 44100)
            # Order matters for loudness: option 1 normalizes last so loudness is
            # set, while option 2 trims last, which invalidates it.
            assert item1.loudness is not None
            assert item2.loudness is None


class TestProbabilisticTransforms:
    """Test chaining probabilistic transforms."""

    def test_probabilistic_transform_chaining(self):
        """Test chaining transforms with probability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 20, duration=3.0)

            # Create dataset
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=3.0,
            )

            # Chain probabilistic transforms
            ds = ds.random_map(invert_phase(prob=0.5), seed=42)
            ds = ds.random_map(
                volume_change(min_db=-6, max_db=6, prob=0.8),
                seed=43,
            )

            # Load multiple items - some should be affected, some not
            items = [ds[i] for i in range(20)]

            # All items should load successfully
            assert len(items) == 20
            for item in items:
                assert item.waveform is not None


class TestLazyEvaluation:
    """Test that transforms are applied lazily."""

    def test_transforms_are_lazy(self):
        """Test that transforms are not applied until items are accessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 10)

            # Create dataset and chain transforms
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=3.0,
            )

            ds = ds.random_map(
                volume_norm(min_db=-20, max_db=-15),
                seed=42,
            )

            # At this point, no transforms have been applied
            # (we can't easily test this directly, but accessing items triggers it)

            # Access first item - transform applied now
            item = ds[0]
            assert item.loudness is not None

            # Access another item - transform applied independently
            item2 = ds[1]
            assert item2.loudness is not None


class TestStereoMonoChaining:
    """Test chaining with mono/stereo conversions."""

    def test_mono_conversion_in_chain(self):
        """Test adding Mono transform to chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = _create_test_audio_files(tmpdir, "audio", 5, duration=3.0)

            # Create dataset (mono files)
            ds = create_audio_dataset(
                sources=audio_dir,
                shuffle=False,
                repeat=False,
                sample_rate=44100,
                duration=3.0,
                mono=False,  # Keep original channels
            )

            # Chain mono transform
            ds = ds.map(mono())

            # Load item and verify it's mono
            item = ds[0]
            assert item.waveform.shape[1] == 1  # 1 channel
