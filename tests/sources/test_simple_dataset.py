"""Tests for create_audio_dataset function."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.core import SaliencyParams
from audiotree.sources import create_audio_dataset, find_audio_files


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


def test_repeat_mode():
    """Test repeat mode for training datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 5)

        # With repeat=True, should repeat files; use slice to limit to 20
        ds = create_audio_dataset(
            sources=audio_dir,
            repeat=True,
            sample_rate=44100,
            duration=0.5,
        ).slice(slice(0, 20))

        assert len(ds) == 20

        # All items should load successfully
        for i in range(20):
            item = ds[i]
            assert isinstance(item, AudioTree)


def test_no_repeat_uses_all_files():
    """Test that without repeat, uses all available files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 8)

        # Without num_records and repeat=False, should use all files
        ds = create_audio_dataset(
            sources=audio_dir,
            repeat=False,
            sample_rate=44100,
            duration=0.5,
        )

        assert len(ds) == 8


def test_with_saliency():
    """Test dataset creation with saliency parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 5, duration=3.0)

        saliency_params = SaliencyParams(enabled=True, loudness_cutoff=None)
        ds = create_audio_dataset(
            sources=audio_dir,
            sample_rate=44100,
            duration=1.0,
            saliency_params=saliency_params,
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
        (root / "notes.txt").write_text("not audio")  # wrong extension

        found = find_audio_files(str(root))

        # Sorted (deterministic), recursive, hidden + non-audio excluded.
        assert found == sorted(found)
        assert [Path(p).name for p in found] == ["a.flac", "b.wav", "c.wav"]

        # A str and a single-element list are equivalent.
        assert find_audio_files([str(root)]) == found

        # The extension filter is honored.
        only_flac = find_audio_files(str(root), extensions=[".flac"])
        assert [Path(p).name for p in only_flac] == ["a.flac"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
