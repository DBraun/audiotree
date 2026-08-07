"""Tests for length-aware windowed audio sampling (audiotree.sources.windowed)."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import grain
import numpy as np
import pytest
import soundfile as sf

from audiotree import AudioTree
from audiotree.sources import (
    WindowLufsCache,
    WindowParams,
    build_window_lufs_cache,
    create_balanced_audio_dataset,
    create_windowed_audio_dataset,
    load_window_lufs,
    precompute_window_lufs,
    save_window_lufs,
    scan_durations,
)
from audiotree.sources.core import _derive_seed_pair
from audiotree.sources.windowed import _build_slot_index

requires_bagz = pytest.mark.skipif(
    importlib.util.find_spec("bagz") is None,
    reason="bagz not installed (Linux-only wheels)",
)


def _write_file(path, duration_sec, sample_rate=8000, channels=1, level=0.1, seed=0):
    """Write a noise file of a given duration; returns its path as a string."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    audio = (rng.standard_normal((n, channels)) * level).astype(np.float32)
    sf.write(str(path), audio, sample_rate)
    return str(path)


def _mixed_corpus(tmpdir, specs, sample_rate=8000):
    """Create files with given {name: duration_sec}; returns ordered filepaths."""
    d = Path(tmpdir)
    d.mkdir(parents=True, exist_ok=True)
    return [
        _write_file(d / f"{name}.wav", dur, sample_rate=sample_rate, seed=i)
        for i, (name, dur) in enumerate(specs.items())
    ]


# --------------------------------------------------------------------------- #
# Duration scan + slot index
# --------------------------------------------------------------------------- #


def test_scan_durations():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 2.0, "b": 5.0})
        durations = scan_durations(fps)
        assert set(durations) == set(fps)
        assert durations[fps[0]] == pytest.approx(2.0, abs=0.01)
        assert durations[fps[1]] == pytest.approx(5.0, abs=0.01)


def _index_counts(fps, durations, *, duration, hop, alpha):
    file_idx, *_ = _build_slot_index(
        filepaths=fps,
        durations=durations,
        duration=duration,
        hop=hop,
        alpha=alpha,
        lufs_per_file=None,
        lufs_window_sec=1.0,
        lufs_cutoff=-40.0,
    )
    return np.bincount(file_idx, minlength=len(fps))


def test_alpha_zero_is_uniform_per_file():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"short": 4.0, "long": 40.0})
        counts = _index_counts(
            fps, scan_durations(fps), duration=1.0, hop=1.0, alpha=0.0
        )
        assert list(counts) == [1, 1]


def test_alpha_one_is_proportional_to_length():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"short": 4.0, "long": 40.0})  # 10x ratio
        counts = _index_counts(
            fps, scan_durations(fps), duration=1.0, hop=1.0, alpha=1.0
        )
        # natural windows = floor((dur - 1)/1) + 1  ->  4 and 40
        assert list(counts) == [4, 40]


def test_alpha_half_is_sqrt_length():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"short": 4.0, "long": 40.0})
        counts = _index_counts(
            fps, scan_durations(fps), duration=1.0, hop=1.0, alpha=0.5
        )
        # round(4**0.5)=2, round(40**0.5)=6
        assert list(counts) == [2, 6]


def test_slot_offsets_stay_inside_file():
    """Every nominal offset + jitter span must keep the window inside the file."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 10.0})
        durations = scan_durations(fps)
        _, offset, stride, max_offset = _build_slot_index(
            filepaths=fps,
            durations=durations,
            duration=2.0,
            hop=2.0,
            alpha=1.0,
            lufs_per_file=None,
            lufs_window_sec=1.0,
            lufs_cutoff=-40.0,
        )
        # nominal + full jitter (== stride) never exceeds max_offset (== dur - duration).
        assert np.all(offset + stride <= max_offset + 1e-4)
        assert np.all(max_offset == pytest.approx(10.0 - 2.0))


def test_short_file_collapses_to_one_padded_slot():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"tiny": 0.5})  # shorter than duration
        counts = _index_counts(
            fps, scan_durations(fps), duration=2.0, hop=2.0, alpha=1.0
        )
        assert list(counts) == [1]


# --------------------------------------------------------------------------- #
# Dataset construction + sampling
# --------------------------------------------------------------------------- #


def test_dataset_loads_fixed_shape():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0, "b": 8.0})
        ds = create_windowed_audio_dataset(
            filepaths=fps,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            sample_rate=8000,
            mono=True,
            shuffle=False,
            num_epochs=1,
        )
        item = ds[0]
        assert isinstance(item, AudioTree)
        assert item.waveform.shape == (1, 1, 8000)


def test_one_epoch_covers_every_slot_once():
    """With one epoch the dataset length equals the slot count (exact coverage)."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0, "b": 12.0})
        durations = scan_durations(fps)
        counts = _index_counts(fps, durations, duration=1.0, hop=1.0, alpha=1.0)
        ds = create_windowed_audio_dataset(
            filepaths=fps,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            sample_rate=8000,
            shuffle=True,
            num_epochs=1,
        )
        assert len(ds) == int(counts.sum())


def test_global_shuffle_interleaves_files():
    """A long file's slots are scattered, not emitted in one contiguous run."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 20.0, "b": 20.0, "c": 20.0})
        durations = scan_durations(fps)
        file_idx, *_ = _build_slot_index(
            filepaths=fps,
            durations=durations,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            lufs_per_file=None,
            lufs_window_sec=1.0,
            lufs_cutoff=-40.0,
        )
        # Mirror the dataset's exact shuffle on an index-only pipeline.
        id_ds = (
            grain.MapDataset.source(range(len(file_idx)))
            .seed(0)
            .shuffle()
            .repeat()
            .map(lambda sid: int(file_idx[sid]))
        )
        draws = [id_ds[i] for i in range(60)]
        # All three files appear, and runs of the same file are short.
        assert len(set(draws)) == 3
        max_run = max(len(list(g)) for g in _runs(draws))
        assert max_run <= 3  # no lingering on one file


def _runs(seq):
    out, cur = [], [seq[0]]
    for x in seq[1:]:
        if x == cur[-1]:
            cur.append(x)
        else:
            out.append(cur)
            cur = [x]
    out.append(cur)
    return out


def test_requires_exactly_one_of_sources_or_filepaths():
    with pytest.raises(ValueError):
        create_windowed_audio_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 2.0})
        with pytest.raises(ValueError):
            create_windowed_audio_dataset(sources=tmp, filepaths=fps)


def test_alpha_out_of_range_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 2.0})
        with pytest.raises(ValueError):
            create_windowed_audio_dataset(filepaths=fps, alpha=1.5)


@pytest.mark.parametrize("bad_duration", [0.0, -1.0])
def test_nonpositive_duration_raises_with_explicit_hop(bad_duration):
    """duration <= 0 is a duration error even when an explicit hop is given."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0})
        with pytest.raises(ValueError, match="duration must be positive"):
            create_windowed_audio_dataset(filepaths=fps, duration=bad_duration, hop=1.0)


def test_nonpositive_duration_raises_when_hop_defaults():
    """A duration=0 mistake is blamed on duration, not on the hop it defaults to."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0})
        with pytest.raises(ValueError, match="duration must be positive"):
            create_windowed_audio_dataset(filepaths=fps, duration=0.0)


@pytest.mark.parametrize("bad_window", [0.0, -2.0])
def test_nonpositive_lufs_window_sec_raises_when_filtering(bad_window):
    """lufs_window_sec <= 0 raises when loudness filtering is active."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0}, sample_rate=8000)
        lpf = precompute_window_lufs(fps, lufs_window_sec=1.0, sample_rate=8000)
        with pytest.raises(ValueError, match="lufs_window_sec must be positive"):
            create_windowed_audio_dataset(
                filepaths=fps,
                duration=1.0,
                hop=1.0,
                sample_rate=8000,
                lufs_per_file=lpf,
                lufs_window_sec=bad_window,
            )


def test_num_epochs_replaces_the_repeat_flag():
    """`num_epochs` counts passes over the slot index; None is truly infinite."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0, "b": 12.0})
        common = dict(
            filepaths=fps,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            sample_rate=8000,
            shuffle=False,
        )
        slots = len(create_windowed_audio_dataset(num_epochs=1, **common))

        assert len(create_windowed_audio_dataset(num_epochs=3, **common)) == 3 * slots
        assert len(create_windowed_audio_dataset(num_epochs=None, **common)) == (
            sys.maxsize
        )
        # The default is a single pass over every slot.
        assert len(create_windowed_audio_dataset(**common)) == slots

        with pytest.raises(ValueError, match="num_epochs must be >= 1"):
            create_windowed_audio_dataset(num_epochs=0, **common)
        with pytest.raises(TypeError, match="num_epochs must be an int or None"):
            create_windowed_audio_dataset(num_epochs=True, **common)
        with pytest.raises(TypeError, match="repeat"):
            create_windowed_audio_dataset(repeat=True, **common)


def test_slot_shuffle_uses_a_derived_seed():
    """Slot order and jitter must not both come from the raw `shuffle_seed`."""
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 20.0, "b": 20.0})
        durations = scan_durations(fps)
        file_idx, *_ = _build_slot_index(
            filepaths=fps,
            durations=durations,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            lufs_per_file=None,
            lufs_window_sec=1.0,
            lufs_cutoff=-40.0,
        )
        shuffle_stream, excerpt_stream = _derive_seed_pair(7)
        assert shuffle_stream != excerpt_stream

        ds = create_windowed_audio_dataset(
            filepaths=fps,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            sample_rate=8000,
            shuffle=True,
            shuffle_seed=7,
            num_epochs=1,
        )
        order = [ds[i].filepath[0] for i in range(len(ds))]

        def slot_order(seed):
            ids = grain.MapDataset.source(range(len(file_idx))).seed(seed).shuffle()
            return [fps[int(file_idx[sid])] for sid in ids]

        assert order == slot_order(shuffle_stream)
        assert order != slot_order(7)


# --------------------------------------------------------------------------- #
# Windowed loudness + bagz cache
# --------------------------------------------------------------------------- #


def test_precompute_window_lufs_is_ragged_and_floors_silence():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 3.0, "b": 7.0}, sample_rate=8000)
        lpf = precompute_window_lufs(fps, lufs_window_sec=1.0)
        # Ragged: window counts scale with duration (3 and 7).
        assert lpf[fps[0]].shape == (3,)
        assert lpf[fps[1]].shape == (7,)
        assert lpf[fps[0]].dtype == np.float32
        # Noise at level 0.1 is well above the silence floor.
        assert np.all(lpf[fps[0]] > -100)


def test_loudness_filtering_keeps_only_loud_slots():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        sr = 8000
        # 10s file: first 5s silent, last 5s loud noise.
        audio = np.zeros((10 * sr, 1), dtype=np.float32)
        audio[5 * sr :] = np.random.default_rng(0).standard_normal((5 * sr, 1)) * 0.5
        fp = str(d / "halfsilent.wav")
        sf.write(fp, audio, sr)
        fps = [fp]

        lpf = precompute_window_lufs(fps, lufs_window_sec=1.0)
        durations = scan_durations(fps)

        full = _build_slot_index(
            filepaths=fps,
            durations=durations,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            lufs_per_file=None,
            lufs_window_sec=1.0,
            lufs_cutoff=-40.0,
        )[0]
        kept_idx, kept_off, *_ = _build_slot_index(
            filepaths=fps,
            durations=durations,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            lufs_per_file=lpf,
            lufs_window_sec=1.0,
            lufs_cutoff=-40.0,
        )
        assert 0 < len(kept_idx) < len(full)
        # Every surviving slot sits in the loud second half of the file.
        assert np.all(kept_off >= 4.0)


@requires_bagz
def test_bagz_cache_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 3.0, "b": 6.0}, sample_rate=8000)
        cache_dir = build_window_lufs_cache(
            fps,
            lufs_window_sec=1.0,
            out_dir=Path(tmp) / "cache",
            sample_rate=8000,
            mono=True,
        )
        cache = load_window_lufs(cache_dir)
        assert isinstance(cache, WindowLufsCache)
        assert cache.lufs_window_sec == 1.0
        assert cache.sample_rate == 8000
        assert cache.mono is True
        assert cache.durations[fps[1]] == pytest.approx(6.0, abs=0.01)
        # Ragged arrays survive the bagz round-trip exactly.
        ref = precompute_window_lufs(fps, 1.0, sample_rate=8000)
        for fp in fps:
            np.testing.assert_array_equal(cache.lufs[fp], ref[fp])


def test_save_window_lufs_temp_file_keeps_bagz_extension(tmp_path, monkeypatch):
    # bagz selects compression from the file extension, so the atomic-write
    # temp file must end in ".bagz" — a ".tmp" suffix would write uncompressed
    # records that the renamed lufs.bagz then fails to decompress. Stubbed so
    # the contract is enforced even where bagz has no wheel (macOS, Windows).
    import audiotree.sources.windowed as windowed_mod

    writer_paths = []

    class FakeWriter:
        def __init__(self, path):
            writer_paths.append(Path(path))
            self._f = open(path, "wb")

        def write(self, record):
            self._f.write(record)

        def close(self):
            self._f.close()

    class FakeBagz:
        Writer = FakeWriter

    monkeypatch.setattr(windowed_mod, "require_bagz", lambda purpose: FakeBagz)

    out = save_window_lufs(
        tmp_path / "c",
        {"x.wav": np.array([-14.0], np.float32)},
        lufs_window_sec=1.0,
    )
    assert writer_paths and all(p.suffix == ".bagz" for p in writer_paths)
    assert (out / "lufs.bagz").exists()
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []


@requires_bagz
def test_save_window_lufs_handles_empty_arrays():
    with tempfile.TemporaryDirectory() as tmp:
        lpf = {
            "x.wav": np.array([-14.0, -16.0], np.float32),
            "y.wav": np.zeros(0, np.float32),
        }
        out = save_window_lufs(
            Path(tmp) / "c", lpf, lufs_window_sec=1.0, sample_rate=8000, mono=True
        )
        cache = load_window_lufs(out)
        np.testing.assert_array_equal(cache.lufs["x.wav"], lpf["x.wav"])
        assert cache.lufs["y.wav"].shape == (0,)


def _write_cache_manifest(cache_dir: Path, **overrides):
    """Write a well-formed cache manifest, then apply ``overrides`` to it.

    An override of ``None`` deletes the key, simulating a truncated or
    hand-edited manifest.
    """
    import json

    from audiotree import _format

    manifest = {
        **_format.header(_format.LUFS_WINDOWS_CACHE),
        "lufs_window_sec": 1.0,
        "sample_rate": 8000,
        "mono": True,
        "filepaths": ["a.wav"],
        "durations": [2.0],
        "bagz_file": "lufs.bagz",
    }
    for key, value in overrides.items():
        if value is None:
            del manifest[key]
        else:
            manifest[key] = value
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache_dir


# Deliberately NOT @requires_bagz: the manifest is validated before the bagz
# file is opened, so a corrupt cache names itself even where bagz has no wheel.
@pytest.mark.parametrize(
    "missing", ["lufs_window_sec", "filepaths", "durations", "bagz_file"]
)
def test_load_window_lufs_names_a_manifest_missing_keys(tmp_path, missing):
    """A truncated/hand-edited manifest used to die with a bare KeyError."""
    cache_dir = _write_cache_manifest(tmp_path / "cache", **{missing: None})
    with pytest.raises(
        ValueError, match=f"missing required top-level key '{missing}'"
    ) as excinfo:
        load_window_lufs(cache_dir)
    assert "manifest.json" in str(excinfo.value)


def test_load_window_lufs_rejects_malformed_container_types(tmp_path):
    """'filepaths' and 'bagz_file' types are checked before bagz is touched."""
    cache_dir = _write_cache_manifest(tmp_path / "a", filepaths="not-a-list")
    with pytest.raises(ValueError, match="'filepaths' must be a list"):
        load_window_lufs(cache_dir)

    cache_dir = _write_cache_manifest(tmp_path / "b", bagz_file=7)
    with pytest.raises(ValueError, match="'bagz_file' must be a string"):
        load_window_lufs(cache_dir)


@requires_bagz
def test_lufs_cache_path_filters_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 6.0, "b": 6.0}, sample_rate=8000)
        cache_dir = build_window_lufs_cache(
            fps,
            lufs_window_sec=1.0,
            out_dir=Path(tmp) / "cache",
            sample_rate=8000,
            mono=True,
        )
        ds = create_windowed_audio_dataset(
            filepaths=fps,
            duration=1.0,
            hop=1.0,
            alpha=1.0,
            sample_rate=8000,
            mono=True,
            lufs_cache=cache_dir,
            lufs_cutoff=-200.0,
            shuffle=False,
            num_epochs=1,
        )
        # With a permissive cutoff, all noise slots survive.
        assert len(ds) == int(
            _index_counts(
                fps, scan_durations(fps), duration=1.0, hop=1.0, alpha=1.0
            ).sum()
        )


@requires_bagz
def test_lufs_cache_sample_rate_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0}, sample_rate=8000)
        cache_dir = build_window_lufs_cache(
            fps,
            lufs_window_sec=1.0,
            out_dir=Path(tmp) / "cache",
            sample_rate=8000,
            mono=True,
        )
        with pytest.raises(ValueError, match="sample_rate"):
            create_windowed_audio_dataset(
                filepaths=fps,
                sample_rate=16000,
                mono=True,
                lufs_cache=cache_dir,
            )


@requires_bagz
def test_lufs_cache_mono_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        fps = _mixed_corpus(tmp, {"a": 4.0}, sample_rate=8000)
        cache_dir = build_window_lufs_cache(
            fps,
            lufs_window_sec=1.0,
            out_dir=Path(tmp) / "cache",
            sample_rate=8000,
            mono=True,
        )
        with pytest.raises(ValueError, match="mono"):
            create_windowed_audio_dataset(
                filepaths=fps,
                sample_rate=8000,
                mono=False,
                lufs_cache=cache_dir,
            )


# --------------------------------------------------------------------------- #
# Composition with balanced datasets
# --------------------------------------------------------------------------- #


def test_window_params_in_balanced_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        a_dir = Path(tmp) / "groupA"
        b_dir = Path(tmp) / "groupB"
        _mixed_corpus(a_dir, {"a0": 4.0, "a1": 8.0}, sample_rate=8000)
        _mixed_corpus(b_dir, {"b0": 6.0}, sample_rate=8000)

        ds = create_balanced_audio_dataset(
            sources={"A": [str(a_dir)], "B": [str(b_dir)]},
            weights={"A": 0.5, "B": 0.5},
            sample_rate=8000,
            mono=True,
            window_params=WindowParams(duration=1.0, alpha=0.5),
        )
        item = ds[0]
        assert item.waveform.shape == (1, 1, 8000)
        assert item.source[0] in ("A", "B")


def test_window_params_and_saliency_params_mutually_exclusive():
    from audiotree.core import ExcerptConfig

    with tempfile.TemporaryDirectory() as tmp:
        a_dir = Path(tmp) / "g"
        _mixed_corpus(a_dir, {"a": 4.0}, sample_rate=8000)
        with pytest.raises(ValueError, match="window_params"):
            create_balanced_audio_dataset(
                sources={"A": [str(a_dir)]},
                window_params=WindowParams(),
                excerpt=ExcerptConfig(strategy="start"),
            )
