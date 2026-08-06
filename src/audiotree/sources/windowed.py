"""Length-aware windowed audio sampling for grain pipelines.

``create_windowed_audio_dataset`` changes the unit of sampling from the *file*
(as in :func:`audiotree.sources.create_audio_dataset`) to a *window* (a fixed
``duration`` excerpt). Every file is tiled into ``m_i`` evenly spaced slots,
where ``m_i`` scales with the file's length via a tunable power ``alpha``:

* ``alpha = 0`` -> one slot per file (uniform per file, the file-as-unit default).
* ``alpha = 1`` -> one slot per natural ``hop`` window (sampling frequency
  proportional to length; even coverage per second of audio).
* ``0 < alpha < 1`` -> in between (e.g. ``alpha = 0.5`` is ``sqrt(length)``).

The slots of all files are flattened into one index and globally shuffled by
grain, which scatters any single (possibly hour-long) file's slots uniformly
across the epoch -> diverse batches. Each slot owns a contiguous ``stride`` of
the file's valid-offset range, and ``jitter`` draws a random offset within that
stride, so coverage spans the whole file at every ``alpha`` while never
repeating an exact excerpt across epochs.

Because each slot's offset is confined to ``[0, duration_of_file - duration]``,
windows never run past end-of-file -- clamping is built into the tiling.
"""

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Mapping, Optional

import grain
import librosa
import numpy as np
import soundfile

from audiotree import AudioTree, _format
from audiotree._bagz import require_bagz
from audiotree._fs import safe_join, write_json_atomic

from .core import _default_extensions, find_audio_files

_LUFS_BAGZ = "lufs.bagz"
_LUFS_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class WindowParams:
    """Windowing knobs for length-aware sampling, bundled for reuse.

    Pass an instance as ``window_params=`` to
    :func:`~audiotree.sources.create_balanced_audio_dataset` to build every
    file-based group with :func:`create_windowed_audio_dataset` instead of the
    default one-excerpt-per-file behavior. The fields mirror that function's
    windowing arguments (``sample_rate``/``mono`` stay on the dataset call so a
    single value applies across groups).

    Attributes:
        duration: Length in seconds of each excerpt.
        hop: Stride in seconds between natural windows at ``alpha == 1``; defaults
            to ``duration`` (non-overlapping) when None.
        alpha: Length power for slots-per-file (0 = uniform per file, 1 =
            proportional to length).
        jitter: Randomize each draw's offset within its slot's stride.
        lufs_cache: Optional path to a cache from
            :func:`build_window_lufs_cache` for build-time saliency filtering.
        lufs_cutoff: Minimum per-window LUFS to keep a slot (when filtering).
        lufs_window_sec: Analysis window of the loudness cache; taken from the
            cache when ``lufs_cache`` is given.
    """

    duration: float = 1.0
    hop: Optional[float] = None
    alpha: float = 1.0
    jitter: bool = True
    lufs_cache: Optional[str] = None
    lufs_cutoff: float = -40.0
    lufs_window_sec: Optional[float] = None


def scan_durations(filepaths: List[str]) -> dict[str, float]:
    """Read each file's duration in seconds from its header (no decode).

    ``soundfile.info`` reads only the container header, so this is cheap enough
    to run over tens of thousands of files. Persist the result (e.g. JSON/NPZ)
    and pass it back as ``durations=`` to skip the scan on subsequent runs; the
    slot index is a pure function of these durations, so ``alpha``/``hop``/
    ``duration`` can be retuned without re-reading any headers.

    Args:
        filepaths: Audio file paths to inspect.

    Returns:
        A mapping from filepath to duration in seconds.
    """
    return {fp: soundfile.info(fp).duration for fp in filepaths}


def _file_windows(
    fp: str,
    lufs_window_sec: float,
    sample_rate: int | None,
    mono: bool,
) -> tuple[np.ndarray, int]:
    """Read ``fp`` and tile it into non-overlapping windows.

    Returns ``(windows, sr)`` where ``windows`` has shape
    ``(num_windows, channels, window_samples)`` (empty along axis 0 if the file
    is shorter than one window) and ``sr`` is the (possibly resampled) rate. The
    trailing partial window is dropped, matching per-window LUFS conventions.
    """
    audio, sr = soundfile.read(fp, dtype="float32", always_2d=True)  # (samples, ch)
    audio = audio.T  # (channels, samples)
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    if sample_rate is not None and sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    window_samples = int(lufs_window_sec * sr)
    channels = audio.shape[0]
    num_windows = audio.shape[1] // window_samples if window_samples > 0 else 0
    if num_windows == 0:
        return np.zeros((0, channels, max(window_samples, 1)), dtype=np.float32), sr

    audio = audio[:, : num_windows * window_samples]
    # (channels, num_windows * ws) -> (num_windows, channels, ws)
    windows = audio.reshape(channels, num_windows, window_samples).transpose(1, 0, 2)
    return np.ascontiguousarray(windows, dtype=np.float32), sr


def precompute_window_lufs(
    filepaths: List[str],
    lufs_window_sec: float,
    *,
    sample_rate: int | None = 44_100,
    mono: bool = True,
) -> dict[str, np.ndarray]:
    """Compute a per-file windowed-LUFS array for build-time saliency filtering.

    Each value is the integrated loudness (LUFS, ITU-R BS.1770) of one
    non-overlapping ``lufs_window_sec`` window, measured on the **CPU** with
    the upstream ``loudness`` library (``loudness.integrated_loudness``, the same
    kernel :meth:`AudioTree.replace_lufs` uses for NumPy waveforms). This is a
    one-time offline pass that runs entirely on the CPU -- it performs no
    JAX/GPU computation -- so it is safe to run before forking grain workers and
    keeps the data-source layer free of GPU work. For large corpora it
    parallelizes across files with a process pool.

    The arrays are **ragged** (length scales with file duration); files shorter
    than one window get an empty array, which downstream filtering treats as
    "keep". The result is suitable for passing as ``lufs_per_file=`` to
    :func:`create_windowed_audio_dataset`; use :func:`build_window_lufs_cache`
    to compute and persist it to disk (bagz) in one pass.

    The ``sample_rate`` and ``mono`` defaults match those of
    :func:`create_windowed_audio_dataset`, so by default the loudness reflects
    exactly the audio the model trains on; pass matching values if you change the
    dataset's. Files are normalized per these (resample to ``sample_rate``,
    average to mono); ``sample_rate=None`` keeps each file's native rate. Windows
    are measured independently, so files need not share a shape.

    TODO: very long files are read fully into RAM before windowing; segment-read
    them if hour-plus files strain memory.

    Args:
        filepaths: Audio file paths to analyze.
        lufs_window_sec: Analysis window length in seconds (independent of
            the training ``duration``). Should be at least 0.4s for a valid LUFS
            measurement.
        sample_rate: Resample every file to this rate before analysis, or None to
            keep each file's native rate. Defaults to the dataset's ``sample_rate``.
        mono: If True (the dataset default), average channels to mono first.

    Returns:
        A mapping from filepath to a 1-D ``float32`` array of per-window LUFS.
    """
    import loudness

    out: dict[str, np.ndarray] = {}
    for fp in filepaths:
        windows, sr = _file_windows(fp, lufs_window_sec, sample_rate, mono)
        if windows.shape[0] == 0:
            # Too short for even one analysis window; no saliency info.
            out[fp] = np.zeros((0,), dtype=np.float32)
            continue
        # ``integrated_loudness`` expects time-major ``(samples, channels)``.
        out[fp] = np.array(
            [
                loudness.integrated_loudness(np.ascontiguousarray(window.T), sr)
                for window in windows
            ],
            dtype=np.float32,
        )
    return out


@dataclass(frozen=True)
class WindowLufsCache:
    """A loaded windowed-loudness cache (see :func:`load_window_lufs`).

    Attributes:
        lufs: Mapping from filepath to its 1-D per-window LUFS array.
        durations: Mapping from filepath to duration in seconds, or ``None`` if
            durations were not stored.
        lufs_window_sec: The analysis window length the cache was built with.
        sample_rate: The sample rate the loudness was measured at (``None`` if
            native rates were used).
        mono: Whether channels were averaged to mono before measuring.
    """

    lufs: dict[str, np.ndarray]
    durations: Optional[dict[str, float]]
    lufs_window_sec: float
    sample_rate: Optional[int]
    mono: bool


def save_window_lufs(
    out_dir: str | Path,
    lufs_per_file: Mapping[str, np.ndarray],
    *,
    lufs_window_sec: float,
    durations: Optional[Mapping[str, float]] = None,
    sample_rate: Optional[int] = None,
    mono: bool = True,
) -> Path:
    """Persist a windowed-loudness cache to ``out_dir`` as bagz + JSON manifest.

    The ragged per-file LUFS arrays are stored as one ``float32`` record each in
    a single ``lufs.bagz`` file (no padding), in the order of
    ``lufs_per_file``. A ``manifest.json`` records the filepaths (parallel to
    the bagz records), the analysis window, optional durations, and the
    ``sample_rate``/``mono`` the loudness was measured at (so the dataset can
    verify the cache matches the audio it loads).

    Args:
        out_dir: Directory to write the cache into (created if missing).
        lufs_per_file: Mapping from filepath to its per-window LUFS array.
        lufs_window_sec: Analysis window length the arrays were computed with.
        durations: Optional mapping from filepath to duration in seconds; stored
            so the cache can also supply ``durations=`` to the dataset.
        sample_rate: The sample rate the loudness was measured at (``None`` if
            native). Stored for the dataset's consistency check.
        mono: Whether channels were averaged to mono before measuring.

    Returns:
        The cache directory as a ``Path``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepaths = list(lufs_per_file.keys())

    writer = require_bagz("writing windowed-LUFS caches").Writer(
        str(out_dir / _LUFS_BAGZ)
    )
    for fp in filepaths:
        writer.write(np.asarray(lufs_per_file[fp], dtype=np.float32).tobytes())
    writer.close()

    manifest = {
        **_format.header(_format.LUFS_WINDOWS_CACHE),
        "lufs_window_sec": float(lufs_window_sec),
        "sample_rate": int(sample_rate) if sample_rate is not None else None,
        "mono": bool(mono),
        "filepaths": filepaths,
        "durations": (
            [float(durations[fp]) for fp in filepaths]
            if durations is not None
            else None
        ),
        "bagz_file": _LUFS_BAGZ,
    }
    write_json_atomic(out_dir / _LUFS_MANIFEST, manifest, indent=None)
    return out_dir


def load_window_lufs(cache_dir: str | Path) -> WindowLufsCache:
    """Load a windowed-loudness cache written by :func:`save_window_lufs`.

    Args:
        cache_dir: Directory containing ``lufs.bagz`` and ``manifest.json``.

    Returns:
        A :class:`WindowLufsCache` with the per-file LUFS arrays, optional
        durations, and the analysis window length.
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / _LUFS_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    _format.check(
        manifest, _format.LUFS_WINDOWS_CACHE, source=str(cache_dir / _LUFS_MANIFEST)
    )

    filepaths = manifest["filepaths"]
    reader = require_bagz("reading windowed-LUFS caches").Reader(
        str(safe_join(cache_dir, manifest["bagz_file"], description="cache file"))
    )
    if len(reader) != len(filepaths):
        raise RuntimeError(
            f"Cache corruption: {len(reader)} bagz records for "
            f"{len(filepaths)} filepaths in {cache_dir}."
        )
    lufs_per_file = {
        fp: np.frombuffer(reader[i], dtype=np.float32).copy()
        for i, fp in enumerate(filepaths)
    }

    durations = None
    if manifest["durations"] is not None:
        durations = dict(zip(filepaths, manifest["durations"]))

    return WindowLufsCache(
        lufs=lufs_per_file,
        durations=durations,
        lufs_window_sec=manifest["lufs_window_sec"],
        sample_rate=manifest.get("sample_rate"),
        mono=manifest.get("mono", True),
    )


def build_window_lufs_cache(
    filepaths: List[str],
    lufs_window_sec: float,
    out_dir: str | Path,
    *,
    sample_rate: int | None = 44_100,
    mono: bool = True,
    durations: Optional[Mapping[str, float]] = None,
) -> Path:
    """Compute and persist a windowed-loudness cache in one preprocessing pass.

    Convenience wrapper that runs :func:`precompute_window_lufs`, scans
    durations (via :func:`scan_durations` unless ``durations`` is given), and
    writes both to ``out_dir`` with :func:`save_window_lufs`. Run this once
    offline, then pass ``lufs_cache=out_dir`` to
    :func:`create_windowed_audio_dataset`.

    Args:
        filepaths: Audio file paths to analyze.
        lufs_window_sec: Analysis window length in seconds.
        out_dir: Directory to write the cache into.
        sample_rate: If given, resample every file to this rate before analysis.
        mono: If True, average channels to mono before analysis.
        durations: Optional precomputed durations; scanned from headers if None.

    Returns:
        The cache directory as a ``Path``.
    """
    lufs_per_file = precompute_window_lufs(
        filepaths,
        lufs_window_sec,
        sample_rate=sample_rate,
        mono=mono,
    )
    if durations is None:
        durations = scan_durations(filepaths)
    return save_window_lufs(
        out_dir,
        lufs_per_file,
        lufs_window_sec=lufs_window_sec,
        durations=durations,
        sample_rate=sample_rate,
        mono=mono,
    )


def _build_slot_index(
    filepaths: List[str],
    durations: Mapping[str, float],
    duration: float,
    hop: float,
    alpha: float,
    lufs_per_file: Optional[Mapping[str, np.ndarray]],
    lufs_window_sec: float,
    lufs_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the flat, globally-shuffleable slot index.

    Returns four parallel arrays (one entry per slot): the file index, the
    nominal offset (seconds), the per-slot ``stride`` (the jitter span), and the
    maximum legal offset for that file (``duration_of_file - duration``, clamped
    at 0). See the module docstring for the tiling math.
    """
    file_idx_parts: List[np.ndarray] = []
    offset_parts: List[np.ndarray] = []
    stride_parts: List[np.ndarray] = []
    max_offset_parts: List[np.ndarray] = []

    for i, fp in enumerate(filepaths):
        file_duration = durations[fp]
        # Valid offsets keep the whole excerpt inside the file. A file shorter
        # than ``duration`` collapses to a single slot at offset 0 (padded by
        # ``from_file``'s ``pad_mode``).
        usable = max(0.0, file_duration - duration)
        natural_windows = max(1, int(math.floor(usable / hop)) + 1)
        num_slots = max(1, int(round(natural_windows**alpha)))
        stride = usable / num_slots  # slots tile [0, usable] with no gaps

        j = np.arange(num_slots)
        offsets = (j * stride).astype(np.float32)

        keep = np.ones(num_slots, dtype=bool)
        if lufs_per_file is not None:
            lufs = lufs_per_file[fp]  # KeyError if a file is missing: fail loud
            if lufs.size > 0:
                # Map each slot's center to its loudness-grid cell.
                centers = offsets + duration / 2.0
                cell = np.clip(
                    (centers / lufs_window_sec).astype(int), 0, lufs.size - 1
                )
                keep = lufs[cell] >= lufs_cutoff

        if not keep.any():
            continue

        file_idx_parts.append(np.full(int(keep.sum()), i, dtype=np.int32))
        offset_parts.append(offsets[keep])
        stride_parts.append(np.full(int(keep.sum()), stride, dtype=np.float32))
        max_offset_parts.append(np.full(int(keep.sum()), usable, dtype=np.float32))

    if not file_idx_parts:
        raise RuntimeError(
            "No slots were produced. If loudness filtering is enabled, the "
            f"cutoff ({lufs_cutoff} LUFS) may have rejected every window."
        )

    return (
        np.concatenate(file_idx_parts),
        np.concatenate(offset_parts),
        np.concatenate(stride_parts),
        np.concatenate(max_offset_parts),
    )


def _load_slot(
    slot_id: int,
    rng: np.random.Generator,
    *,
    filepaths: List[str],
    file_idx: np.ndarray,
    offset: np.ndarray,
    stride: np.ndarray,
    max_offset: np.ndarray,
    jitter: bool,
    sample_rate: int,
    duration: float,
    mono: bool,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None,
    source: str | None,
) -> AudioTree:
    """Load the excerpt for one slot, jittering the offset within its stride."""
    i = int(file_idx[slot_id])
    off = float(offset[slot_id])
    if jitter:
        off += float(rng.uniform(0.0, float(stride[slot_id])))
    # Clamp keeps the excerpt inside the file even with float round-off.
    off = min(off, float(max_offset[slot_id]))
    return AudioTree.from_file(
        filepaths[i],
        sample_rate=sample_rate,
        offset=off,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        source=source,
    )


def create_windowed_audio_dataset(
    sources: List[str] | str | None = None,
    filepaths: List[str] | None = None,
    *,
    duration: float = 1.0,
    hop: float | None = None,
    alpha: float = 1.0,
    jitter: bool = True,
    durations: Optional[Mapping[str, float]] = None,
    lufs_cache: str | Path | None = None,
    lufs_per_file: Optional[Mapping[str, np.ndarray]] = None,
    lufs_window_sec: float | None = None,
    lufs_cutoff: float = -40.0,
    shuffle: bool = True,
    repeat: bool = False,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    pad_mode: (
        Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None
    ) = "constant",
    extensions: Optional[List[str]] = None,
    source: str | None = None,
) -> grain.MapDataset:
    """Create a length-aware, evenly-covering MapDataset of audio windows.

    Unlike :func:`create_audio_dataset` (one random excerpt per file per epoch),
    this tiles every file into ``m_i`` slots with ``m_i`` proportional to
    ``length ** alpha``, flattens all slots into one globally shuffled index, and
    draws a jittered excerpt from each. The result samples long files more often
    than short ones (tunable via ``alpha``), covers each file evenly, and keeps
    batches diverse because grain scatters any one file's slots across the epoch.

    Args:
        sources: A directory path, file path, or glob pattern (e.g.
            ``"/data/*/mixture.wav"``), or a list of them. See
            :func:`find_audio_files` for how each entry is resolved. Mutually
            exclusive with ``filepaths`` -- provide exactly one.
        filepaths: An explicit list of audio file paths. Mutually exclusive with
            ``sources``.
        duration: Length in seconds of each excerpt.
        hop: Stride in seconds between natural windows at ``alpha == 1``. Defaults
            to ``duration`` (non-overlapping).
        alpha: Length power for slots-per-file (``0`` = uniform per file, ``1`` =
            proportional to length). See the module docstring.
        jitter: If True, randomize each draw's offset within its slot's stride so
            coverage spans the whole file and excerpts never repeat across epochs.
        durations: Optional precomputed ``{filepath: seconds}`` cache (see
            :func:`scan_durations`). If None, durations are scanned from headers
            (or taken from ``lufs_cache`` if it stored them).
        lufs_cache: Optional path to an on-disk cache written by
            :func:`build_window_lufs_cache`. When given, its per-file LUFS
            arrays, analysis window, and durations are loaded and used for
            build-time saliency filtering (overridable by the explicit
            ``lufs_per_file`` / ``lufs_window_sec`` / ``durations`` args).
        lufs_per_file: Optional precomputed per-file windowed-LUFS arrays (see
            :func:`precompute_window_lufs`). If given, slots whose center is
            below ``lufs_cutoff`` are dropped at build time. If None and no
            ``lufs_cache`` (the default), no loudness filtering is done --
            assume curated data.
        lufs_window_sec: Analysis window of ``lufs_per_file`` in seconds.
            Required when ``lufs_per_file`` is given without a ``lufs_cache``.
        lufs_cutoff: Minimum per-window LUFS to keep a slot.
        shuffle: Whether to globally shuffle slots (required for batch diversity).
        repeat: Whether to repeat infinitely (True for training).
        shuffle_seed: Seed for the global slot shuffle.
        excerpt_seed: Seed for jitter. Defaults to ``shuffle_seed``.
        sample_rate: Target sample rate for loaded audio.
        mono: Whether to convert audio to mono.
        pad_mode: Padding mode for files shorter than ``duration`` (numpy.pad
            modes), or None to not pad.
        extensions: Audio extensions to search when using ``sources``.
        source: Optional source group name stored in metadata.

    Returns:
        A grain.MapDataset over audio windows.
    """
    if hop is None:
        hop = duration
    if excerpt_seed is None:
        excerpt_seed = shuffle_seed
    if hop <= 0:
        raise ValueError(f"hop must be positive, got {hop}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    if (sources is None) == (filepaths is None):
        raise ValueError(
            "Provide exactly one of `sources` or `filepaths` "
            f"(got sources={sources!r}, filepaths={filepaths!r})."
        )

    if filepaths is None:
        if extensions is None:
            extensions = _default_extensions
        filepaths = find_audio_files(sources, extensions)
        if not filepaths:
            raise RuntimeError(
                f"No audio files found in sources {sources} with extensions {extensions}"
            )
    else:
        filepaths = [str(fp) for fp in filepaths]
        if not filepaths:
            raise ValueError("`filepaths` must be a non-empty list of file paths.")

    # Resolve the optional on-disk loudness cache; explicit args take precedence.
    if lufs_cache is not None:
        cache = load_window_lufs(lufs_cache)
        # The cached loudness only reflects the audio the dataset loads if it was
        # measured at the same sample_rate/mono; otherwise the cutoff is applied
        # to a different signal than the model sees.
        if cache.sample_rate is not None and cache.sample_rate != sample_rate:
            raise ValueError(
                f"lufs_cache was measured at sample_rate={cache.sample_rate} "
                f"but the dataset loads at sample_rate={sample_rate}. Rebuild the "
                "cache with a matching sample_rate."
            )
        if cache.mono != mono:
            raise ValueError(
                f"lufs_cache was measured with mono={cache.mono} but the "
                f"dataset loads with mono={mono}. Rebuild the cache with a "
                "matching mono setting."
            )
        if lufs_per_file is None:
            lufs_per_file = cache.lufs
        if lufs_window_sec is None:
            lufs_window_sec = cache.lufs_window_sec
        if durations is None:
            durations = cache.durations

    if lufs_per_file is not None and lufs_window_sec is None:
        raise ValueError(
            "lufs_window_sec is required when lufs_per_file is provided "
            "without a lufs_cache."
        )

    if durations is None:
        durations = scan_durations(filepaths)

    file_idx, offset, stride, max_offset = _build_slot_index(
        filepaths=filepaths,
        durations=durations,
        duration=duration,
        hop=hop,
        alpha=alpha,
        lufs_per_file=lufs_per_file,
        # Unused when lufs_per_file is None; coalesce to keep arithmetic valid.
        lufs_window_sec=lufs_window_sec if lufs_window_sec else 1.0,
        lufs_cutoff=lufs_cutoff,
    )

    # Source over slot ids; lookups go through the parallel arrays above.
    ds = grain.MapDataset.source(range(len(file_idx)))

    if shuffle:
        ds = ds.seed(shuffle_seed).shuffle()
    if repeat:
        ds = ds.repeat()

    load_fn = functools.partial(
        _load_slot,
        filepaths=filepaths,
        file_idx=file_idx,
        offset=offset,
        stride=stride,
        max_offset=max_offset,
        jitter=jitter,
        sample_rate=sample_rate,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        source=source,
    )
    ds = ds.seed(excerpt_seed).random_map(load_fn)

    return ds
