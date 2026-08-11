"""Tests for create_audio_dataset function."""

import gc
import os
import stat
import sys
import tempfile
import warnings
from pathlib import Path

import grain
import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.core import ExcerptConfig
from audiotree.sources import create_audio_dataset, find_audio_files
from audiotree.sources.core import (
    AudioReadError,
    _derive_seed_pair,
    _load_excerpt,
)


def _write_zero_byte(path: Path, valid_bytes: bytes) -> None:
    path.write_bytes(b"")


def _write_truncated_header(path: Path, valid_bytes: bytes) -> None:
    # Cut inside the "fmt " chunk, so the container is unparseable.
    path.write_bytes(valid_bytes[:20])


def _write_garbage(path: Path, valid_bytes: bytes) -> None:
    # No backend recognizes this, which is what raises audioread's
    # NoBackendError (whose str() is empty) on librosa's fallback path.
    path.write_bytes(np.random.default_rng(0).bytes(4096))


def _write_directory(path: Path, valid_bytes: bytes) -> None:
    path.mkdir()


def _write_unreadable(path: Path, valid_bytes: bytes) -> None:
    path.write_bytes(valid_bytes)
    path.chmod(0)


#: The ways an audio file is broken in practice, each mapped to a factory that
#: writes one.
#:
#: A *truncated data chunk* is deliberately absent: a WAV whose header is intact
#: but whose samples are short decodes fine (libsndfile returns what is there),
#: so it is not a read error at all -- see
#: ``test_truncated_data_is_not_a_read_error``.
CORRUPT_FILE_KINDS = {
    "zero_byte": _write_zero_byte,
    "truncated_header": _write_truncated_header,
    "garbage": _write_garbage,
    "directory": _write_directory,
    "unreadable": _write_unreadable,
}


def _make_corrupt_file(directory, kind, name="corrupt.wav", sample_rate=44100):
    """Write one broken ``name`` into ``directory`` and return its path."""
    path = Path(directory) / name
    template = Path(directory) / "_template.wav"
    sf.write(str(template), np.zeros(sample_rate, dtype=np.float32), sample_rate)
    valid_bytes = template.read_bytes()
    template.unlink()
    CORRUPT_FILE_KINDS[kind](path, valid_bytes)
    return path


@pytest.fixture
def restore_permissions():
    """Chmod appended paths back, so tempdir cleanup can remove them."""
    paths = []
    yield paths
    for path in paths:
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass


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


# ---------------------------------------------------------------------------
# on_read_error: one corrupt file must not be able to end a multi-hour run.
# ---------------------------------------------------------------------------

ALL_STRATEGIES = ("start", "random", "loudest")


@pytest.mark.parametrize("kind", sorted(CORRUPT_FILE_KINDS))
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_on_read_error_raise_always_names_the_path(kind, strategy, restore_permissions):
    """The default policy raises AudioReadError, and the path is in the message.

    This is the regression: with ``strategy="start"`` a zero-byte or otherwise
    unparseable file reaches librosa's audioread fallback, which raises an
    ``EOFError`` or an ``audioread.exceptions.NoBackendError`` whose ``str()``
    is *empty* -- a blank line at the end of a long run, naming nothing. (The
    ``soundfile`` errors that ``"random"``/``"loudest"`` hit do name the path;
    they are checked here so every strategy is known to behave the same.)
    """
    if kind == "unreadable":
        if sys.platform == "win32":
            pytest.skip("chmod(0) does not deny the owner a read on Windows")
        if os.geteuid() == 0:
            pytest.skip("root can read a mode-000 file")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_corrupt_file(tmpdir, kind)
        restore_permissions.append(path)

        with pytest.raises(AudioReadError) as excinfo:
            _load_excerpt(
                str(path),
                np.random.default_rng(0),
                sample_rate=44100,
                duration=0.5,
                excerpt=ExcerptConfig(strategy=strategy),
            )

        error = excinfo.value
        message = str(error)
        file_path = error.file_path
        # The original exception is preserved, not swallowed, and its type is
        # named even when its message is empty.
        cause_name = type(error.__cause__).__name__ if error.__cause__ else None

        # Then let the exception go before the tmpdir is removed. Its traceback
        # holds the frames of librosa's audioread fallback, which holds an open
        # handle on the file, and Windows refuses to delete a file that is still
        # open. Measured on a zero-byte file: 1 handle while the exception is
        # alive, 0 after. Exception and traceback reference each other, so the
        # collect is doing real work rather than being superstitious.
        del error, excinfo
        gc.collect()

        assert str(path) in message
        assert file_path == str(path)
        assert cause_name is not None
        assert cause_name in message


def test_on_read_error_raise_is_the_default():
    """A corrupt file still ends the run unless a policy says otherwise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 2)
        _make_corrupt_file(audio_dir, "zero_byte", name="bad.wav")

        ds = create_audio_dataset(
            sources=audio_dir, shuffle=False, sample_rate=44100, duration=0.5
        )
        # find_audio_files sorts, so "audio_0", "audio_1", "bad".
        with pytest.raises(AudioReadError, match="bad.wav"):
            [ds[i] for i in range(len(ds))]


@pytest.mark.parametrize("kind", sorted(CORRUPT_FILE_KINDS))
@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_on_read_error_drops_the_item(kind, policy, restore_permissions):
    """A non-raising policy returns None, grain's filtered-element sentinel."""
    if kind == "unreadable":
        if sys.platform == "win32":
            pytest.skip("chmod(0) does not deny the owner a read on Windows")
        if os.geteuid() == 0:
            pytest.skip("root can read a mode-000 file")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_corrupt_file(tmpdir, kind)
        restore_permissions.append(path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = _load_excerpt(
                str(path),
                np.random.default_rng(0),
                sample_rate=44100,
                duration=0.5,
                on_read_error=policy,
            )

        assert tree is None


def test_on_read_error_warn_names_the_file_and_skip_stays_quiet():
    """The two non-raising policies differ only in whether they warn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_corrupt_file(tmpdir, "zero_byte")

        with pytest.warns(UserWarning, match="corrupt.wav"):
            _load_excerpt(
                str(path),
                np.random.default_rng(0),
                sample_rate=44100,
                duration=0.5,
                on_read_error="warn",
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _load_excerpt(
                str(path),
                np.random.default_rng(0),
                sample_rate=44100,
                duration=0.5,
                on_read_error="skip",
            )
        # librosa may warn on its own when it falls back to audioread; what
        # must be absent is *our* drop notice.
        assert not [w for w in caught if "Skipping unreadable" in str(w.message)]


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_on_read_error_survives_a_whole_epoch_and_batches(policy):
    """A corpus with junk in it iterates to the end and still collates.

    The unreadable files come back as ``None`` under random access, grain's
    filtered-element convention, and ``to_iter_dataset()`` skips them -- so
    iteration yields only real audio and batching needs no special handling.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 3)
        for i, kind in enumerate(["zero_byte", "garbage"]):
            _make_corrupt_file(audio_dir, kind, name=f"bad_{i}.wav")

        ds = create_audio_dataset(
            sources=audio_dir,
            shuffle=False,
            sample_rate=44100,
            duration=0.5,
            on_read_error=policy,
        )
        assert len(ds) == 5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            items = [ds[i] for i in range(len(ds))]
            iterated = list(ds.to_iter_dataset())

        # find_audio_files sorts, so the two bad files come last.
        assert [item is None for item in items] == [False, False, False, True, True]

        real = [item for item in items if item is not None]
        batch = AudioTree.batch(real)
        assert batch.waveform.shape == (3, 1, 22050)

        # The iteration path drops the Nones by itself.
        assert len(iterated) == 3


@pytest.mark.parametrize("policy", ["skip", "warn"])
def test_on_read_error_drops_under_loudest_excerpt(policy):
    """The loudness-search strategy drops an unreadable file like any other."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 1)
        _make_corrupt_file(Path(audio_dir), "zero_byte", name="z_bad.wav")

        ds = create_audio_dataset(
            sources=audio_dir,
            shuffle=False,
            sample_rate=44100,
            duration=0.5,
            excerpt=ExcerptConfig(strategy="loudest"),
            on_read_error=policy,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # find_audio_files sorts, so the good file comes first.
            good, dropped = ds[0], ds[1]

        assert dropped is None
        # The real item is a normal loudest-excerpt load.
        assert good.lufs is not None
        assert good.lufs_windows is not None
        assert AudioTree.batch([good]).waveform.shape == (1, 1, 22050)


def test_truncated_data_is_not_a_read_error():
    """A valid header over a short data chunk decodes; it is not a failure.

    Worth pinning: it is the one "corrupt file" of the set that must *not* take
    the error path, under either policy.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        good = Path(tmpdir) / "good.wav"
        sf.write(str(good), np.ones(44100, dtype=np.float32) * 0.5, 44100)
        truncated = Path(tmpdir) / "truncated.wav"
        truncated.write_bytes(good.read_bytes()[:2000])

        tree = _load_excerpt(
            str(truncated),
            np.random.default_rng(0),
            sample_rate=44100,
            duration=0.5,
            excerpt=ExcerptConfig(strategy="start"),
        )
        assert tree.waveform.shape == (1, 1, 22050)
        # The samples that survived are real audio, zero-padded up to duration.
        assert np.any(np.asarray(tree.waveform))

        kept = _load_excerpt(
            str(truncated),
            np.random.default_rng(0),
            sample_rate=44100,
            duration=0.5,
            excerpt=ExcerptConfig(strategy="start"),
            on_read_error="skip",
        )
        assert kept is not None


def test_on_read_error_rejects_an_unknown_policy():
    """A typo'd policy fails at construction, not per item."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = _create_test_audio_files(tmpdir, 1)
        with pytest.raises(ValueError, match="on_read_error must be one of"):
            create_audio_dataset(sources=audio_dir, on_read_error="ignore")


def test_multichannel_with_unreadable_probe_file_still_loads():
    """An unreadable first file no longer blocks a mono=False dataset.

    The channel probe reads the first file's header; when that file is the
    corrupt one the probe returns None and the check is simply disabled --
    there is no substitute whose shape would have needed it. The corrupt file
    is dropped and the real files load with their own channel count.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dir = Path(tmpdir)
        # Sorts first, so it is the file the channel probe reads.
        _make_corrupt_file(audio_dir, "zero_byte", name="a_bad.wav")
        sf.write(
            str(audio_dir / "b_good.wav"), np.zeros((44100, 2), dtype=np.float32), 44100
        )

        ds = create_audio_dataset(
            sources=str(audio_dir),
            mono=False,
            shuffle=False,
            sample_rate=44100,
            duration=0.5,
            on_read_error="skip",
        )
        assert ds[0] is None
        assert ds[1].waveform.shape == (1, 2, 22050)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
