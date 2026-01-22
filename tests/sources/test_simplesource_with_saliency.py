"""Test load_audio_with_saliency and create_balanced_audio_dataset with SaliencyParams."""

import tempfile
from pathlib import Path

import numpy as np
import grain

from audiotree import AudioTree
from audiotree.core import SaliencyParams
from audiotree.sources import create_balanced_audio_dataset
from audiotree.sources.core import _load_audio_with_saliency
from audiotree.transforms import Batch

# Paths to test audio files
TEST_AUDIO_MONO = Path(__file__).parent.parent / "assets" / "VCTK" / "p225_006_mic1.flac"  # 7.18s, mono, 48kHz
TEST_AUDIO_STEREO = Path(__file__).parent.parent / "assets" / "musdb18hq" / "train" / "A Classic Education - NightOwl" / "mixture.wav"  # 20s, stereo, 44.1kHz


def test_load_audio_with_saliency_basic():
    """Test load_audio_with_saliency function with basic functionality."""
    sample_rate = 44_100

    # Test without saliency
    rng = np.random.default_rng(42)
    result = _load_audio_with_saliency(
        str(TEST_AUDIO_MONO),
        rng,
        sample_rate=sample_rate,
        duration=1.0,
        mono=True,
        saliency_params=None,
    )

    assert isinstance(result, AudioTree)
    assert result.audio_data.shape == (1, 1, sample_rate)
    assert result.sample_rate == sample_rate

    # Test with saliency enabled but no loudness cutoff
    rng = np.random.default_rng(42)
    saliency_params = SaliencyParams(enabled=True, loudness_cutoff=None)
    result = _load_audio_with_saliency(
        str(TEST_AUDIO_MONO),
        rng,
        sample_rate=sample_rate,
        duration=1.0,
        mono=True,
        saliency_params=saliency_params,
    )

    assert isinstance(result, AudioTree)
    assert result.audio_data.shape == (1, 1, sample_rate)


def test_saliency_variety_with_repetition():
    """Test that repeated files get different random excerpts - the key bug fix."""
    sample_rate = 16_000

    # Create dataset with same file repeated many times
    # TEST_AUDIO_MONO is 7.18 seconds long with natural speech variation
    ds = (
        grain.MapDataset.source([str(TEST_AUDIO_MONO)] * 100)
        .random_map(
            lambda path, rng: _load_audio_with_saliency(
                path,
                rng,
                sample_rate=sample_rate,
                duration=1.0,
                mono=True,
                saliency_params=SaliencyParams(enabled=True, loudness_cutoff=None),
            ),
            seed=42
        )
    )

    # Load first 50 excerpts
    excerpts = [ds[i] for i in range(50)]

    # Compute a simple hash of each excerpt to check for uniqueness
    def audio_hash(audio_tree):
        # Use mean and std as a simple signature
        return (float(np.mean(audio_tree.audio_data)), float(np.std(audio_tree.audio_data)))

    hashes = [audio_hash(e) for e in excerpts]
    unique_hashes = len(set(hashes))

    # We should have significant variety (not just 1-2 unique excerpts)
    # With proper RNG seeding, we expect most excerpts to be unique
    assert unique_hashes > 20, f"Expected diverse excerpts, got only {unique_hashes} unique out of 50"
    print(f"Got {unique_hashes} unique excerpts out of 50 - good variety!")


def test_saliency_determinism():
    """Test that the same seed produces identical results."""
    sample_rate = 16_000

    # Create two datasets with same seed
    def make_dataset(seed):
        return (
            grain.MapDataset.source([str(TEST_AUDIO_MONO)] * 10)
            .random_map(
                lambda path, rng: _load_audio_with_saliency(
                    path,
                    rng,
                    sample_rate=sample_rate,
                    duration=1.0,
                    mono=True,
                    saliency_params=SaliencyParams(enabled=True, loudness_cutoff=None),
                ),
                seed=seed
            )
        )

    ds1 = make_dataset(42)
    ds2 = make_dataset(42)

    # Check that results are identical
    for i in range(10):
        audio1 = ds1[i].audio_data
        audio2 = ds2[i].audio_data
        assert np.allclose(audio1, audio2), f"Excerpt {i} differs between runs"

    print("Determinism test passed!")


def test_create_balanced_audio_dataset_with_saliency():
    """Test create_balanced_audio_dataset with SaliencyParams and grain.DataLoader."""
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple copies of test file to simulate a dataset
        num_files = 8
        for i in range(num_files):
            target = output_dir / f"audio_{i:04d}.flac"
            shutil.copy(TEST_AUDIO_MONO, target)

        sample_rate = 44_100
        saliency_params = SaliencyParams(enabled=True, loudness_cutoff=None)
        ds = create_balanced_audio_dataset(
            sources={"test": [str(output_dir)]},
            num_records=num_files,
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            saliency_params=saliency_params,
            seed=42,
        )

        # Load items
        items = []
        for i in range(num_files):
            item = ds[i]
            # Remove metadata to avoid batching issues with variable-length arrays
            item = item.replace(metadata={})
            items.append(item)

        assert len(items) == num_files
        for item in items:
            assert isinstance(item, AudioTree)
            assert item.audio_data.shape == (1, 1, sample_rate)

        # Use grain.DataLoader with Batch transform
        batch_size = 4
        dataloader = grain.DataLoader(
            data_source=items,
            sampler=grain.samplers.SequentialSampler(len(items)),
            operations=[Batch(batch_size=batch_size)],
        )

        batch_count = 0
        total_items = 0

        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, AudioTree)
            assert batch.audio_data.ndim == 3
            current_batch_size = batch.audio_data.shape[0]

            if batch_count < (num_files // batch_size):
                assert current_batch_size == batch_size
            else:
                assert current_batch_size <= batch_size

            assert batch.sample_rate == sample_rate
            assert batch.audio_data.shape[1] == 1
            assert batch.audio_data.shape[2] == sample_rate

            total_items += current_batch_size
            print(f"Batch {batch_count}: {current_batch_size} items, shape={batch.audio_data.shape}")

        assert total_items == num_files
        assert batch_count == (num_files + batch_size - 1) // batch_size

        print(f"Successfully processed {total_items} items in {batch_count} batches")


def test_repeated_dataset_variety():
    """Test that when a dataset is repeated, we get different excerpts each time."""
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create 4 copies of test file
        num_files = 4
        for i in range(num_files):
            target = output_dir / f"audio_{i:04d}.flac"
            shutil.copy(TEST_AUDIO_MONO, target)

        # Create dataset with many more records than files, forcing repetition
        sample_rate = 16_000
        saliency_params = SaliencyParams(enabled=True, loudness_cutoff=None)
        ds = create_balanced_audio_dataset(
            sources={"test": [str(output_dir)]},
            num_records=100,  # Much more than num_files
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            saliency_params=saliency_params,
            seed=42,
        )

        # Load all excerpts
        excerpts = [ds[i] for i in range(100)]

        # Simple hash for checking uniqueness
        def audio_hash(audio_tree):
            return (float(np.mean(audio_tree.audio_data)), float(np.std(audio_tree.audio_data)))

        hashes = [audio_hash(e) for e in excerpts]
        unique_hashes = len(set(hashes))

        # With only 4 files repeated 25 times each, if the bug existed we'd see
        # only 4 unique excerpts. With the fix, we should see many more.
        assert unique_hashes > 30, f"Expected diverse excerpts despite repetition, got only {unique_hashes} unique out of 100"
        print(f"Got {unique_hashes} unique excerpts out of 100 with only 4 source files - excellent variety!")


if __name__ == "__main__":
    test_load_audio_with_saliency_basic()
    print("Basic test passed!")

    test_saliency_variety_with_repetition()
    print("Variety test passed!")

    test_saliency_determinism()
    print("Determinism test passed!")

    test_create_balanced_audio_dataset_with_saliency()
    print("Balanced dataset test passed!")

    test_repeated_dataset_variety()
    print("Repeated dataset variety test passed!")

    print("\nAll tests passed!")
