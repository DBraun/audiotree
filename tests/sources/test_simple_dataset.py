"""Tests for create_audio_dataset function."""

import sys
import tempfile
from pathlib import Path

import grain
import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.core import ExcerptConfig
from audiotree.sources import create_audio_dataset, find_audio_files
from audiotree.sources.core import _derive_seed_pair


def _create_test_audio_files(tmpdir, num_files, sample_rate=44100, duration=1.0):
    """Helper to create test audio files."""
    output_dir = Path(tmpdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(sample_rate * duration)
    for i in range(num_files):
        audio = np.random.randn(num_samples).astype(np.float32) * 0.1
        filepath = output_dir / f"audio_{i}.wav"
        sf.write(str(filepath), audio, sample_rate)

    return str(output_dir)


def test_basic_creation():
    """Test basic dataset creation from a single directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 10)

        ds = create_audio_dataset(
            sources=audio_dir,
            sample_rate=44100,
            duration=0.5,
        )

        assert len(ds) == 10

        # Verify loading works
        item = ds[0]
        assert isinstance(item, AudioTree)
        assert item.sample_rate == 44100


def test_multiple_directories():
    """Test loading from multiple directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = _create_test_audio_files(Path(tmpdir) / "dir1", 5)
        dir2 = _create_test_audio_files(Path(tmpdir) / "dir2", 5)

        ds = create_audio_dataset(
            sources=[dir1, dir2],
            sample_rate=44100,
            duration=0.5,
        )

        assert len(ds) == 10


def test_no_shuffle():
    """Test deterministic ordering with shuffle=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 10)

        ds1 = create_audio_dataset(
            sources=audio_dir,
            shuffle=False,
            shuffle_seed=42,
            sample_rate=44100,
            duration=0.5,
        )

        ds2 = create_audio_dataset(
            sources=audio_dir,
            shuffle=False,
            shuffle_seed=42,
            sample_rate=44100,
            duration=0.5,
        )

        # Same seed + no shuffle should give same data
        for i in range(5):
            audio1 = ds1[i].waveform
            audio2 = ds2[i].waveform
            np.testing.assert_array_equal(audio1, audio2)


def test_shuffle():
    """Test that shuffle produces different orderings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 10)

        ds1 = create_audio_dataset(
            sources=audio_dir,
            shuffle=True,
            shuffle_seed=42,
            sample_rate=44100,
            duration=0.5,
        )

        ds2 = create_audio_dataset(
            sources=audio_dir,
            shuffle=True,
            shuffle_seed=99,
            sample_rate=44100,
            duration=0.5,
        )

        # Different seeds should (very likely) give different orderings
        different = False
        for i in range(10):
            audio1 = ds1[i].waveform
            audio2 = ds2[i].waveform
            if not np.array_equal(audio1, audio2):
                different = True
                break
        assert different, "Different seeds should produce different orderings"


def test_num_epochs_none_repeats_forever():
    """`num_epochs=None` is genuinely infinite, not a large finite count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 5)

        ds = create_audio_dataset(
            sources=audio_dir,
            num_epochs=None,
            sample_rate=44100,
            duration=0.5,
        )
        # grain spells "infinite" as sys.maxsize.
        assert len(ds) == sys.maxsize

        # Indices far past the corpus still resolve.
        for i in (0, 19, 10_000_000):
            assert isinstance(ds[i], AudioTree)


def test_num_epochs_one_uses_all_files_once():
    """A single pass covers every file exactly once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 8)

        ds = create_audio_dataset(
            sources=audio_dir,
            num_epochs=1,
            sample_rate=44100,
            duration=0.5,
        )

        assert len(ds) == 8
        # The default is a single pass.
        assert len(create_audio_dataset(sources=audio_dir, duration=0.5)) == 8


def test_num_epochs_finite_count():
    """`num_epochs=n` yields exactly n passes -- what `repeat: bool` could not say."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 4)

        ds = create_audio_dataset(
            sources=audio_dir,
            num_epochs=3,
            shuffle=False,
            sample_rate=44100,
            duration=0.5,
        )

        assert len(ds) == 12
        # Every file is visited three times.
        counts: dict[str, int] = {}
        for i in range(len(ds)):
            path = ds[i].filepath[0]
            counts[path] = counts.get(path, 0) + 1
        assert sorted(counts.values()) == [3, 3, 3, 3]


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_num_epochs_rejects_non_positive_counts(bad):
    """Zero and negatives are mistakes, not silent no-ops."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 2)

        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            create_audio_dataset(sources=audio_dir, num_epochs=bad, duration=0.5)


@pytest.mark.parametrize("bad", [True, False, 1.0, "3"])
def test_num_epochs_rejects_non_integers(bad):
    """The old boolean spelling (and other junk) must fail loudly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 2)

        with pytest.raises(TypeError, match="num_epochs must be an int or None"):
            create_audio_dataset(sources=audio_dir, num_epochs=bad, duration=0.5)


def test_repeat_keyword_is_gone():
    """`repeat=` was deleted outright; it must not be silently accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 2)

        with pytest.raises(TypeError, match="repeat"):
            create_audio_dataset(sources=audio_dir, repeat=True, duration=0.5)


def test_shuffle_and_excerpt_seeds_are_derived_independently():
    """One `shuffle_seed` must not drive both the file order and the excerpts.

    `excerpt_seed` falls back to `shuffle_seed`, and both integers used to reach
    grain verbatim, so the shuffle stream and the excerpt stream were built from
    the same number. They are now two draws of one `SeedSequence`.
    """
    shuffle_stream, excerpt_stream = _derive_seed_pair(42)
    assert shuffle_stream != excerpt_stream
    # A pure function of the caller's seed, so runs stay reproducible.
    assert _derive_seed_pair(42) == (shuffle_stream, excerpt_stream)
    assert _derive_seed_pair(43) != (shuffle_stream, excerpt_stream)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 10)
        files = find_audio_files(audio_dir)

        ds = create_audio_dataset(
            sources=audio_dir,
            shuffle=True,
            shuffle_seed=42,
            sample_rate=44100,
            duration=0.5,
        )
        order = [ds[i].filepath[0] for i in range(len(ds))]

        # grain shuffles with the derived seed, not the caller's raw integer.
        derived = list(grain.MapDataset.source(files).seed(shuffle_stream).shuffle())
        raw = list(grain.MapDataset.source(files).seed(42).shuffle())
        assert order == derived
        assert order != raw

        # An explicit excerpt_seed equal to shuffle_seed reproduces the default,
        # so the fallback is not a second, different code path.
        pinned = create_audio_dataset(
            sources=audio_dir,
            shuffle=True,
            shuffle_seed=42,
            excerpt_seed=42,
            sample_rate=44100,
            duration=0.5,
        )
        for i in range(len(ds)):
            np.testing.assert_array_equal(ds[i].waveform, pinned[i].waveform)


def test_with_saliency():
    """Test dataset creation with saliency parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 5, duration=3.0)

        excerpt = ExcerptConfig(strategy="random")
        ds = create_audio_dataset(
            sources=audio_dir,
            sample_rate=44100,
            duration=1.0,
            excerpt=excerpt,
        )

        assert len(ds) == 5

        # Should load excerpts successfully
        item = ds[0]
        assert isinstance(item, AudioTree)
        assert item.waveform.shape[2] == 44100  # 1 second at 44.1kHz


def test_mono_conversion():
    """Test mono conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create stereo file
        audio = np.random.randn(44100, 2).astype(np.float32) * 0.1
        filepath = output_dir / "stereo.wav"
        sf.write(str(filepath), audio, 44100)

        ds = create_audio_dataset(
            sources=str(output_dir),
            sample_rate=44100,
            duration=1.0,
            mono=True,
        )

        item = ds[0]
        assert item.waveform.shape[1] == 1  # Should be mono


def test_empty_directory():
    """Test that empty directory raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            create_audio_dataset(
                sources=tmpdir,
                sample_rate=44100,
                duration=0.5,
            )
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "No audio files found" in str(e)


def test_filepaths_custom_split():
    """create_audio_dataset accepts an explicit filepaths list for custom splits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 10)
        all_files = find_audio_files(audio_dir)
        assert len(all_files) == 10

        # Split a single directory into train/val without reorganizing on disk.
        train_ds = create_audio_dataset(
            filepaths=all_files[:6], sample_rate=44100, duration=0.5
        )
        val_ds = create_audio_dataset(
            filepaths=all_files[6:], sample_rate=44100, duration=0.5
        )

        assert len(train_ds) == 6
        assert len(val_ds) == 4
        assert isinstance(train_ds[0], AudioTree)


def test_sources_xor_filepaths():
    """Exactly one of sources / filepaths must be provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 2)
        files = find_audio_files(audio_dir)

        # Neither provided.
        with pytest.raises(ValueError, match="exactly one"):
            create_audio_dataset(sample_rate=44100, duration=0.5)

        # Both provided.
        with pytest.raises(ValueError, match="exactly one"):
            create_audio_dataset(
                sources=audio_dir, filepaths=files, sample_rate=44100, duration=0.5
            )


def test_find_audio_files_sorted_recursive_and_filtered():
    """find_audio_files returns sorted paths, recurses, and skips hidden/non-audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sr = 16000
        wav = np.zeros(sr, dtype=np.float32)

        # Audio at the top level and in a subdirectory.
        sf.write(str(root / "b.wav"), wav, sr)
        sf.write(str(root / "a.flac"), wav, sr)
        (root / "sub").mkdir()
        sf.write(str(root / "sub" / "c.wav"), wav, sr)

        # These should all be ignored:
        (root / ".hidden").mkdir()
        sf.write(str(root / ".hidden" / "d.wav"), wav, sr)  # hidden directory
        sf.write(str(root / ".e.wav"), wav, sr)  # hidden file
        (root / "notes.txt").write_text(
            "not audio", encoding="utf-8"
        )  # wrong extension

        found = find_audio_files(str(root))

        # Sorted (deterministic), recursive, hidden + non-audio excluded.
        assert found == sorted(found)
        assert [Path(p).name for p in found] == ["a.flac", "b.wav", "c.wav"]

        # A str and a single-element list are equivalent.
        assert find_audio_files([str(root)]) == found

        # The extension filter is honored.
        only_flac = find_audio_files(str(root), extensions=[".flac"])
        assert [Path(p).name for p in only_flac] == ["a.flac"]


def test_find_audio_files_glob_patterns():
    """find_audio_files expands glob patterns for files and directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sr = 16000
        wav = np.zeros(sr, dtype=np.float32)

        # A musdb18hq-style layout: one directory per track, each with stems.
        for track in ("song_a", "song_b"):
            (root / "train" / track).mkdir(parents=True)
            sf.write(str(root / "train" / track / "mixture.wav"), wav, sr)
            sf.write(str(root / "train" / track / "vocals.wav"), wav, sr)

        # A glob that names a specific file within each track directory.
        mixtures = find_audio_files(str(root / "train" / "*" / "mixture.wav"))
        assert [Path(p).name for p in mixtures] == ["mixture.wav", "mixture.wav"]
        assert [Path(p).parent.name for p in mixtures] == ["song_a", "song_b"]

        # A recursive "**" glob finds every stem under train/.
        all_stems = find_audio_files(str(root / "train" / "**" / "*.wav"))
        assert [Path(p).name for p in all_stems] == [
            "mixture.wav",
            "vocals.wav",
            "mixture.wav",
            "vocals.wav",
        ]

        # A glob that matches directories recurses into each match.
        via_dirs = find_audio_files(str(root / "train" / "*"))
        assert via_dirs == all_stems

        # The extension filter still applies to glob matches.
        assert (
            find_audio_files(str(root / "train" / "*" / "*"), extensions=[".flac"])
            == []
        )

        # Mixing a glob and a plain directory de-duplicates overlapping matches.
        combined = find_audio_files(
            [str(root / "train" / "*" / "mixture.wav"), str(root / "train")]
        )
        assert combined == all_stems


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
