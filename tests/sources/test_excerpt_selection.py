"""Test load_audio_with_saliency and create_balanced_audio_dataset with ExcerptConfig."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import grain

from audiotree import AudioTree
from audiotree import ExcerptConfig
from audiotree.sources import create_balanced_audio_dataset
from audiotree.sources.core import _load_excerpt

# Paths to test audio files
TEST_AUDIO_MONO = (
    Path(__file__).parent.parent / "assets" / "VCTK" / "p225_006_mic1.flac"
)  # 7.18s, mono, 48kHz
TEST_AUDIO_STEREO = (
    Path(__file__).parent.parent
    / "assets"
    / "musdb18hq"
    / "train"
    / "A Classic Education - NightOwl"
    / "mixture.wav"
)  # 20s, stereo, 44.1kHz


def test_load_excerpt_basic():
    """Test load_audio_with_saliency function with basic functionality."""
    sample_rate = 44_100

    # Test without saliency
    rng = np.random.default_rng(42)
    result = _load_excerpt(
        str(TEST_AUDIO_MONO),
        rng,
        sample_rate=sample_rate,
        duration=1.0,
        mono=True,
        excerpt=ExcerptConfig(strategy="start"),
    )

    assert isinstance(result, AudioTree)
    assert result.waveform.shape == (1, 1, sample_rate)
    assert result.sample_rate == sample_rate

    # Test with saliency enabled but no loudness cutoff
    rng = np.random.default_rng(42)
    excerpt = ExcerptConfig(strategy="random")
    result = _load_excerpt(
        str(TEST_AUDIO_MONO),
        rng,
        sample_rate=sample_rate,
        duration=1.0,
        mono=True,
        excerpt=excerpt,
    )

    assert isinstance(result, AudioTree)
    assert result.waveform.shape == (1, 1, sample_rate)


def test_saliency_variety_with_repetition():
    """Test that repeated files get different random excerpts - the key bug fix."""
    sample_rate = 16_000

    # Create dataset with same file repeated many times
    # TEST_AUDIO_MONO is 7.18 seconds long with natural speech variation
    ds = grain.MapDataset.source([str(TEST_AUDIO_MONO)] * 100).random_map(
        lambda path, rng: _load_excerpt(
            path,
            rng,
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            excerpt=ExcerptConfig(strategy="random"),
        ),
        seed=42,
    )

    # Load first 50 excerpts
    excerpts = [ds[i] for i in range(50)]

    # Compute a simple hash of each excerpt to check for uniqueness
    def audio_hash(audio_tree):
        # Use mean and std as a simple signature
        return (float(np.mean(audio_tree.waveform)), float(np.std(audio_tree.waveform)))

    hashes = [audio_hash(e) for e in excerpts]
    unique_hashes = len(set(hashes))

    # We should have significant variety (not just 1-2 unique excerpts)
    # With proper RNG seeding, we expect most excerpts to be unique
    assert unique_hashes > 20, (
        f"Expected diverse excerpts, got only {unique_hashes} unique out of 50"
    )
    print(f"Got {unique_hashes} unique excerpts out of 50 - good variety!")


def test_saliency_determinism():
    """Test that the same seed produces identical results."""
    sample_rate = 16_000

    # Create two datasets with same seed
    def make_dataset(seed):
        return grain.MapDataset.source([str(TEST_AUDIO_MONO)] * 10).random_map(
            lambda path, rng: _load_excerpt(
                path,
                rng,
                sample_rate=sample_rate,
                duration=1.0,
                mono=True,
                excerpt=ExcerptConfig(strategy="random"),
            ),
            seed=seed,
        )

    ds1 = make_dataset(42)
    ds2 = make_dataset(42)

    # Check that results are identical
    for i in range(10):
        audio1 = ds1[i].waveform
        audio2 = ds2[i].waveform
        assert np.allclose(audio1, audio2), f"Excerpt {i} differs between runs"

    print("Determinism test passed!")


def test_create_balanced_audio_dataset_with_saliency():
    """Test create_balanced_audio_dataset with ExcerptConfig and grain.DataLoader."""
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple copies of test file to simulate a dataset
        num_files = 8
        for i in range(num_files):
            target = output_dir / f"audio_{i:04d}.flac"
            shutil.copy(TEST_AUDIO_MONO, target)

        sample_rate = 44_100
        excerpt = ExcerptConfig(strategy="random")
        ds = create_balanced_audio_dataset(
            sources={"test": [str(output_dir)]},
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            excerpt=excerpt,
            shuffle_seed=42,
        ).slice(slice(0, num_files))

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
            assert item.waveform.shape == (1, 1, sample_rate)

        # Batch with the supported path: AudioTree.batch as grain's batch_fn.
        batch_size = 4
        dataloader = (
            grain.MapDataset.source(items)
            .to_iter_dataset()
            .batch(batch_size, batch_fn=AudioTree.batch)
        )

        batch_count = 0
        total_items = 0

        for batch in dataloader:
            batch_count += 1
            assert isinstance(batch, AudioTree)
            assert batch.waveform.ndim == 3
            current_batch_size = batch.waveform.shape[0]

            if batch_count < (num_files // batch_size):
                assert current_batch_size == batch_size
            else:
                assert current_batch_size <= batch_size

            assert batch.sample_rate == sample_rate
            assert batch.waveform.shape[1] == 1
            assert batch.waveform.shape[2] == sample_rate

            total_items += current_batch_size
            print(
                f"Batch {batch_count}: {current_batch_size} items, shape={batch.waveform.shape}"
            )

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
        excerpt = ExcerptConfig(strategy="random")
        ds = create_balanced_audio_dataset(
            sources={"test": [str(output_dir)]},
            sample_rate=sample_rate,
            duration=1.0,
            mono=True,
            excerpt=excerpt,
            shuffle_seed=42,
        ).slice(slice(0, 100))

        # Load all excerpts
        excerpts = [ds[i] for i in range(100)]

        # Simple hash for checking uniqueness
        def audio_hash(audio_tree):
            return (
                float(np.mean(audio_tree.waveform)),
                float(np.std(audio_tree.waveform)),
            )

        hashes = [audio_hash(e) for e in excerpts]
        unique_hashes = len(set(hashes))

        # With only 4 files repeated 25 times each, if the bug existed we'd see
        # only 4 unique excerpts. With the fix, we should see many more.
        assert unique_hashes > 30, (
            f"Expected diverse excerpts despite repetition, got only {unique_hashes} unique out of 100"
        )
        print(
            f"Got {unique_hashes} unique excerpts out of 100 with only 4 source files - excellent variety!"
        )


if __name__ == "__main__":
    test_load_excerpt_basic()
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


# === ExcerptConfig ===


def test_default_excerpt_varies_the_offset(tmp_path):
    """The default must not read the head of every file forever.

    `saliency_params=None` used to mean "offset 0", so out of the box every
    epoch saw the same leading excerpt and `excerpt_seed` controlled nothing.
    """
    import soundfile

    from audiotree.sources import create_audio_dataset

    sr = 16000
    t = np.arange(sr * 8) / sr
    soundfile.write(
        str(tmp_path / "a.wav"),
        (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
        sr,
    )

    def offsets(**kwargs):
        ds = create_audio_dataset(
            sources=str(tmp_path), duration=1.0, sample_rate=sr, repeat=True, **kwargs
        )
        return [float(ds[i].metadata["offset"][0]) for i in range(6)]

    assert len(set(offsets())) > 1
    # ...and the deterministic behavior is still reachable, by name.
    assert offsets(excerpt=ExcerptConfig(strategy="start")) == [0.0] * 6
    # excerpt_seed now actually selects different excerpts.
    assert offsets(excerpt_seed=1) != offsets(excerpt_seed=2)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(strategy="nope"), "strategy must be one of"),
        (dict(on_failure="nope"), "on_failure must be"),
        (dict(strategy="loudest", num_tries=0), "num_tries must be >= 1"),
        (dict(lufs_cutoff=-30.0), "only appl"),
        (dict(on_failure="skip"), "only appl"),
        (dict(search="bias_early"), "only appl"),
        (dict(strategy="loudest", search="nope"), "Unknown search"),
    ],
)
def test_excerpt_config_rejects_bad_combinations(kwargs, match):
    """Knobs that do nothing under the chosen strategy raise rather than idle.

    Setting `lufs_cutoff` with no loudness search silently did nothing before,
    which is how "I configured it and nothing happened" goes unnoticed.
    """
    with pytest.raises((ValueError, TypeError), match=match):
        ExcerptConfig(**kwargs)


def _half_silent(path, sr=16000, seconds=4.0):
    import soundfile

    n = int(sr * seconds)
    t = np.arange(n) / sr
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    audio[: n // 2] = 0.0
    soundfile.write(str(path), audio, sr)
    return path


@pytest.mark.parametrize(
    "on_failure,check",
    [
        ("keep", lambda out: out is not None and float(out.lufs[0]) == -np.inf),
        ("skip", lambda out: out is None),
    ],
)
def test_on_failure_keep_and_skip(tmp_path, on_failure, check):
    """A file that never clears the cutoff is no longer silently "successful"."""
    import soundfile

    path = tmp_path / "silent.wav"
    soundfile.write(str(path), np.zeros(16000 * 2, dtype=np.float32), 16000)

    out = AudioTree.loudest_excerpt(
        str(path),
        rng=np.random.default_rng(0),
        excerpt=ExcerptConfig(strategy="loudest", num_tries=3, on_failure=on_failure),
        duration=0.5,
        sample_rate=16000,
    )
    assert check(out)


def test_on_failure_raise_names_the_file(tmp_path):
    import soundfile

    path = tmp_path / "silent.wav"
    soundfile.write(str(path), np.zeros(16000 * 2, dtype=np.float32), 16000)

    with pytest.raises(RuntimeError, match="silent.wav"):
        AudioTree.loudest_excerpt(
            str(path),
            rng=np.random.default_rng(0),
            excerpt=ExcerptConfig(strategy="loudest", num_tries=2, on_failure="raise"),
            duration=0.5,
            sample_rate=16000,
        )


def test_loudest_excerpt_stops_early_on_success(tmp_path):
    """A file with loud content returns an excerpt above the cutoff."""
    path = _half_silent(tmp_path / "half.wav")
    out = AudioTree.loudest_excerpt(
        str(path),
        rng=np.random.default_rng(0),
        excerpt=ExcerptConfig(strategy="loudest", num_tries=32, lufs_cutoff=-40.0),
        duration=0.5,
        sample_rate=16000,
    )
    assert float(out.lufs[0]) > -40.0


def test_excerpt_config_is_not_a_pytree():
    """It is CPU-side config; flattening it into leaves was never intended."""
    import jax

    config = ExcerptConfig()
    assert jax.tree.leaves(config) == [config]
