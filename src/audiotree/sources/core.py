import functools
import glob
import math
import os
import warnings
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Mapping, Optional

import audioread.exceptions
import grain
import numpy as np
import soundfile

from audiotree import AudioTree
from audiotree.core import ExcerptConfig, _leading_axis_size

if TYPE_CHECKING:
    from audiotree.sources.windowed import WindowConfig

_default_extensions = [".wav", ".flac"]

#: What to do when one file in the corpus cannot be read. See
#: :func:`create_audio_dataset`'s ``on_read_error`` for the semantics.
OnReadError = Literal["raise", "skip", "warn"]

_ON_READ_ERROR_VALUES = ("raise", "skip", "warn")


class AudioReadError(OSError):
    """One audio file could not be read.

    Raised by the dataset loaders in place of whatever the decoding stack threw,
    so that a corrupt file in a large corpus always names itself. The original
    exception is kept as ``__cause__`` and the path as :attr:`file_path`.

    The wrapping exists because the underlying exception frequently does *not*
    identify the file. ``soundfile`` puts the path in its message, but a
    truncated header sends ``librosa`` down its ``audioread`` fallback, which
    surfaces an :class:`EOFError` or an
    :class:`audioread.exceptions.NoBackendError` whose ``str()`` is empty --
    a blank traceback line at the end of a multi-hour run.

    .. note::
       On Windows, *holding* one of these keeps the offending file open. The
       traceback references librosa's ``audioread`` fallback frames, which hold
       a file handle, and Windows will not delete or replace an open file. It
       matters only if you accumulate errors rather than handling them --
       collect ``err.file_path`` and ``str(err)`` and let the exception go, or
       call :func:`gc.collect` after dropping it (exception and traceback
       reference each other, so refcounting alone does not free it).

    Attributes:
        file_path: The file that could not be read.
    """

    def __init__(self, message: str, file_path: str):
        super().__init__(message)
        self.file_path = file_path


#: Exceptions that mean "this file could not be read", as opposed to "this
#: dataset is misconfigured".
#:
#: The tuple is explicit rather than a bare ``Exception`` so that real bugs
#: still crash. Three of the entries are easy to get wrong:
#:
#: * ``audioread.exceptions.DecodeError`` (the base of ``NoBackendError``) is
#:   **not** a :class:`RuntimeError`, so an ``except RuntimeError`` misses every
#:   file that falls through to the ``audioread`` fallback.
#: * :class:`EOFError` is what ``audioread``'s stdlib backends raise on a
#:   header truncated mid-chunk, and it is not an :class:`OSError` either.
#: * :class:`soundfile.LibsndfileError` is listed instead of its
#:   :class:`RuntimeError` base on purpose: ``ExcerptConfig.on_failure="raise"``
#:   also raises a ``RuntimeError``, and that is a *policy* decision the caller
#:   already made, not a read failure to be papered over.
_READ_ERRORS: tuple[type[BaseException], ...] = (
    OSError,  # FileNotFoundError, IsADirectoryError, PermissionError, ...
    EOFError,
    soundfile.LibsndfileError,
    audioread.exceptions.DecodeError,
)


def _validate_on_read_error(on_read_error: str) -> OnReadError:
    """Check an ``on_read_error`` argument, returning it unchanged."""
    if on_read_error not in _ON_READ_ERROR_VALUES:
        raise ValueError(
            f"on_read_error must be one of {_ON_READ_ERROR_VALUES}, got "
            f"{on_read_error!r}."
        )
    return on_read_error


def find_audio_files(
    sources: str | Path | List[str | Path],
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Find audio files under one or more directories or glob patterns.

    Each entry in ``sources`` may be a directory, an individual file, or a glob
    pattern (any entry containing ``*``, ``?``, or ``[...]``). A directory is
    searched recursively; a glob is expanded, with each match then treated as a
    directory (searched recursively) or a file. ``**`` is supported for
    recursive glob matching, e.g. ``"/data/**/mixture.wav"``.

    Hidden files and directories (names starting with ``.``, such as ``.git``)
    are skipped when recursing into directories; glob patterns follow the usual
    shell rule that ``*`` does not match a leading ``.``. In every case a file is
    only kept if its extension is in ``extensions``. The returned paths are
    **sorted** and de-duplicated, so the order is deterministic across machines
    and filesystems — important for reproducible shuffling.

    A source that matches nothing (a typo, an unmounted drive, an extension that
    is not in ``extensions``) shrinks the corpus without failing, so each such
    source raises a :class:`UserWarning` naming it.

    Args:
        sources: A path or glob pattern, or a list of them. Each may be a
            :class:`~pathlib.Path` or ``str`` naming a directory (searched
            recursively) or a file, or a glob pattern such as
            ``"/mnt/d/musdb18hq/train/*/mixture.wav"``.
        extensions: File extensions to match (e.g. ``[".wav", ".flac"]``).
            Defaults to ``[".wav", ".flac"]``.

    Returns:
        A sorted, de-duplicated list of matching file paths.
    """
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    if extensions is None:
        extensions = _default_extensions
    extensions_lower = {ext.lstrip(".").lower() for ext in extensions}

    def _has_audio_extension(filename: str) -> bool:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in extensions_lower

    filepaths = []
    for source in sources:
        expanded = os.path.expandvars(str(Path(source).expanduser()))
        # A glob pattern is expanded to its matches; a plain path matches itself.
        matches = (
            glob.glob(expanded, recursive=True)
            if glob.has_magic(expanded)
            else [expanded]
        )
        found = 0
        for match in matches:
            if os.path.isdir(match):
                for root, dirs, files in os.walk(match):
                    # Prune hidden dirs in-place (prevents descent into .git, etc.)
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    for filename in files:
                        if filename.startswith("."):
                            continue
                        if _has_audio_extension(filename):
                            filepaths.append(os.path.join(root, filename))
                            found += 1
            elif os.path.isfile(match) and _has_audio_extension(match):
                filepaths.append(match)
                found += 1
        if not found:
            warnings.warn(
                f"Source {str(source)!r} (expanded to {expanded!r}) matched no "
                f"audio files with extensions {sorted(extensions_lower)}. It "
                "contributes nothing to the dataset -- check the path, the glob "
                "pattern, and `extensions`.",
                UserWarning,
                stacklevel=2,
            )
    return sorted(set(filepaths))


def _load_excerpt(
    file_path: str,
    rng: np.random.Generator,
    sample_rate: int,
    duration: float,
    mono: bool = True,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "reflect",
    excerpt: ExcerptConfig | None = None,
    source: str | None = None,
    channels: int | None = None,
    on_read_error: OnReadError = "raise",
) -> AudioTree | None:
    """Load one excerpt from ``file_path`` according to ``excerpt``.

    Uses the provided RNG for deterministic selection, so it composes with
    grain's ``random_map``.

    Args:
        file_path: Path to the audio file.
        rng: Random number generator from grain's random_map.
        sample_rate: Target sample rate for audio files.
        duration: Duration in seconds to load from each file.
        mono: Whether to convert audio to mono.
        pad_mode: Padding mode for files shorter than duration (numpy.pad modes).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        excerpt: Which part of the file to take; see :class:`ExcerptConfig`.
            Defaults to a random offset.
        source: Optional source group name (e.g., "music", "speech") stored as provenance.
        channels: Expected channel count. A file with a different count raises,
            naming the file, instead of letting the mismatch surface as an
            opaque shape error at batch time. ``None`` disables the check.
        on_read_error: What to do when the file cannot be read; see
            :func:`create_audio_dataset`.

    Returns:
        The loaded AudioTree, or ``None`` when the loudness search failed and
        ``excerpt.on_failure == "skip"`` (grain drops ``None`` elements at
        ``to_iter_dataset()``).

    Raises:
        AudioReadError: If the file cannot be read and ``on_read_error ==
            "raise"``.
        ValueError: If ``channels`` is given and the file has a different number
            of channels.
    """
    excerpt = excerpt or ExcerptConfig()
    common = dict(
        sample_rate=sample_rate,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        source=source,
    )

    try:
        if excerpt.strategy == "start":
            tree = AudioTree.from_file(file_path, offset=0, **common)
        elif excerpt.strategy == "random":
            tree = AudioTree.excerpt(file_path, rng=rng, **common)
        else:
            tree = AudioTree.loudest_excerpt(file_path, rng, excerpt=excerpt, **common)
    except _READ_ERRORS as exc:
        if on_read_error == "raise":
            raise AudioReadError(
                # Quoted, not {!r}: repr() escapes a Windows path's separators,
                # so the message doubles every backslash -- unreadable, not
                # copy-pasteable, and not findable by a plain substring search.
                f"Failed to read audio file '{file_path}': {type(exc).__name__}: {exc}",
                str(file_path),
            ) from exc
        if on_read_error == "warn":
            warnings.warn(
                f"Skipping unreadable audio file "
                f"'{file_path}': {type(exc).__name__}: {exc}. The item is "
                "dropped (grain skips None elements at iteration); pass "
                "on_read_error='raise' to fail on it instead.",
                UserWarning,
                stacklevel=2,
            )
        # ``None`` is grain's native "this slot is empty" convention: later
        # map stages are not called on it and ``to_iter_dataset()`` drops it,
        # so batches refill with real items instead of fabricated silence.
        return None

    if tree is not None and channels is not None and tree.num_channels != channels:
        raise ValueError(
            f"{file_path} has {tree.num_channels} channels, but this dataset "
            f"loads {channels}-channel audio. AudioTree.batch cannot collate a "
            "mixed-channel corpus: pass `mono=True` to mix everything down, "
            "pass `channels=` to declare the expected count, or exclude the file."
        )

    return tree


def _probe_channels(file_path: str) -> Optional[int]:
    """Read one file's channel count from its header, without decoding it.

    Returns ``None`` when the header cannot be read; the real load then reports
    whatever is wrong with the file, rather than this probe failing first.
    """
    try:
        return int(soundfile.info(file_path).channels)
    except Exception:  # noqa: BLE001 - any unreadable header just skips the check
        return None


#: The default excerpt policy: a random offset per draw. Shared because
#: ExcerptConfig is frozen, and used as the "did the caller customize this?"
#: reference by the windowed-sampling guard below.
_DEFAULT_EXCERPT = ExcerptConfig()


def _validate_num_epochs(num_epochs: int | None) -> int | None:
    """Check a ``num_epochs`` argument, returning it normalized.

    ``None`` means "repeat forever" and is passed straight through to grain's
    own infinite repeat, so an infinite dataset really is infinite rather than a
    large finite count. Anything else must be a positive integer: zero or a
    negative count would otherwise silently produce an empty dataset (or be
    quietly rounded up to one pass), which is never what the caller meant.

    Args:
        num_epochs: The caller's value.

    Returns:
        ``None``, or the count as a plain ``int``.

    Raises:
        TypeError: If ``num_epochs`` is neither ``None`` nor an integer.
        ValueError: If ``num_epochs`` is zero or negative.
    """
    if num_epochs is None:
        return None
    if isinstance(num_epochs, bool) or not isinstance(num_epochs, (int, np.integer)):
        raise TypeError(
            f"num_epochs must be an int or None, got {num_epochs!r}. Pass None "
            "to repeat forever, or a positive integer for that many passes."
        )
    if num_epochs < 1:
        raise ValueError(
            f"num_epochs must be >= 1, got {num_epochs}. Pass None to repeat "
            "forever; there is no way to ask for an empty dataset."
        )
    return int(num_epochs)


def _derive_seed_pair(base_seed: int) -> tuple[int, int]:
    """Split one user-facing seed into independent shuffle and excerpt seeds.

    ``excerpt_seed`` falls back to ``shuffle_seed``, so a caller who sets a
    single seed used to hand grain the *same* integer for the file-order stream
    and for the excerpt-offset stream, tying the two together: re-seeding a run
    moved both in lockstep instead of independently. Drawing two states from one
    :class:`numpy.random.SeedSequence` decorrelates them while keeping the whole
    thing a pure function of the caller's seed.

    Both streams are derived, never used verbatim, so the two roles stay
    symmetric -- the same reason :func:`_derive_group_seed` folds a ``role``
    string into its entropy.

    Args:
        base_seed: The caller's seed.

    Returns:
        ``(shuffle_seed, excerpt_seed)``, each in ``[0, 2**31)``.
    """
    state = np.random.SeedSequence(int(base_seed)).generate_state(2, dtype=np.uint32)
    return int(state[0]) & 0x7FFFFFFF, int(state[1]) & 0x7FFFFFFF


def create_audio_dataset(
    sources: str | Path | List[str | Path] | None = None,
    filepaths: List[str | Path] | None = None,
    *,
    shuffle: bool = True,
    num_epochs: int | None = 1,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    duration: float = 1.0,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "constant",
    extensions: Optional[List[str]] = None,
    excerpt: ExcerptConfig = _DEFAULT_EXCERPT,
    source: str | None = None,
    channels: Optional[int] = None,
    on_read_error: OnReadError = "raise",
) -> grain.MapDataset:
    """Create a simple MapDataset from audio files.

    This function creates a grain MapDataset that loads audio files from one or more
    directories. Unlike `create_balanced_audio_dataset`, this treats all files equally
    without balancing across groups.

    Args:
        sources: A directory path, file path, or glob pattern (e.g.
            ``"/data/*/mixture.wav"``), or a list of them, containing audio files.
            Each entry may be a ``str`` or a :class:`~pathlib.Path`.
            See :func:`find_audio_files` for how each entry is resolved.
            Mutually exclusive with ``filepaths`` — provide exactly one.
        filepaths: An explicit list of audio file paths to use instead of searching
            ``sources``. Mutually exclusive with ``sources`` — provide exactly one.
            Useful for custom splits (e.g. train/val) over a single directory without
            reorganizing it on disk. The given order is preserved (then shuffled if
            ``shuffle=True``).
        shuffle: Whether to shuffle files.
        num_epochs: How many passes over the corpus the dataset yields. ``None``
            repeats forever -- what training wants -- and makes ``len(ds)``
            report ``sys.maxsize``, grain's spelling of "infinite". An integer
            ``n >= 1`` yields exactly ``n`` passes, so ``len(ds)`` is ``n``
            times the file count; ``0`` or a negative count raises. Defaults to
            a single finite pass.
        shuffle_seed: Random seed for shuffling file order. The stream grain
            shuffles with is *derived* from this value, not used verbatim; see
            ``excerpt_seed``.
        excerpt_seed: Random seed for excerpt selection (random_map). If None,
            defaults to ``shuffle_seed``. The shuffle and excerpt streams are
            taken from two different draws of one
            :class:`numpy.random.SeedSequence`, so the file order and the
            excerpt offsets stay independent even when one seed feeds both. Use
            different values to create datasets that visit files in the same
            order but load different random excerpts.
        sample_rate: Target sample rate for audio files.
        mono: Whether to convert audio to mono.
        duration: Duration in seconds to load from each file.
        pad_mode: Padding mode for files shorter than duration (numpy.pad modes).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        extensions: List of audio file extensions to search for. Defaults to [".wav", ".flac"].
        excerpt: Which part of each file to take; see :class:`ExcerptConfig`.
            Defaults to a uniformly random offset.
        source: Optional source group name (e.g., "music", "speech") stored as
            provenance (``metadata["source"]``, read back via the ``source``
            property). If None, no ``source`` provenance is recorded.
        channels: Expected channel count of every file, so that a mixed-channel
            corpus fails at load time with the offending filename instead of at
            batch time with a shape error. Ignored when ``mono=True`` (everything
            is one channel then). When None, the count is taken from the header
            of the first file, which makes the odd stereo file in a mono corpus
            (or vice versa) name itself.
        on_read_error: What to do when one file cannot be read -- truncated,
            zero-byte, unreadable by this process. In a 100k-file corpus a
            single such file otherwise ends a multi-hour run.

            * ``"raise"`` (default): raise :class:`AudioReadError`, which
              always names the path even when the underlying decoder error
              does not.
            * ``"warn"``: drop the item (the loader returns ``None``, which
              grain skips at iteration) and emit a :class:`UserWarning` naming
              the file and the original error.
            * ``"skip"``: drop the item without warning, for a corpus already
              known to contain junk.

            Dropping uses grain's own convention: a ``None`` element passes
            through later ``map`` stages untouched and ``to_iter_dataset()``
            skips it, so batches stay full of real audio. Note that under a
            non-raising policy ``ds[i]`` can therefore return ``None`` for a
            broken file; iterate via ``to_iter_dataset()`` (or guard for
            ``None``) rather than assuming every index yields an item.

    Returns:
        A grain.MapDataset that loads audio files using random_map for proper RNG seeding.

    Example:
        Create a couple of short ``.wav`` files in a temporary directory to
        load from:

        >>> import os, tempfile
        >>> import numpy as np
        >>> import soundfile
        >>> data_dir = tempfile.mkdtemp()
        >>> for i in range(2):
        ...     soundfile.write(
        ...         os.path.join(data_dir, f"clip_{i}.wav"),
        ...         np.zeros((44100, 1), dtype=np.float32),
        ...         44100,
        ...     )

        Load all files from the directory. Each item is a single-example
        ``AudioTree`` shaped ``(Batch, Channels, Samples)``:

        >>> ds = create_audio_dataset(sources=data_dir, sample_rate=44100, duration=1.0)
        >>> len(ds)
        2
        >>> ds[0].waveform.shape
        (1, 1, 44100)

        Training dataset (shuffled, repeating forever):

        >>> train_ds = create_audio_dataset(
        ...     sources=data_dir,
        ...     shuffle=True,
        ...     num_epochs=None,
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )

        Validation dataset (deterministic, a single pass):

        >>> val_ds = create_audio_dataset(
        ...     sources=data_dir,
        ...     shuffle=False,
        ...     num_epochs=1,
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )
        >>> len(val_ds)
        2

        Any finite number of passes, which a boolean could not express:

        >>> len(create_audio_dataset(sources=data_dir, num_epochs=3))
        6

        Two datasets that visit files in the same order but load different
        random excerpts (same ``shuffle_seed``, different ``excerpt_seed``):

        >>> ds1 = create_audio_dataset(sources=data_dir, shuffle_seed=42, excerpt_seed=100)
        >>> ds2 = create_audio_dataset(sources=data_dir, shuffle_seed=42, excerpt_seed=200)
    """
    num_epochs = _validate_num_epochs(num_epochs)
    on_read_error = _validate_on_read_error(on_read_error)

    # Both streams are derived, so a lone `shuffle_seed` no longer drives the
    # file order and the excerpt offsets off one and the same integer.
    shuffle_stream_seed, derived_excerpt_seed = _derive_seed_pair(shuffle_seed)
    excerpt_stream_seed = (
        derived_excerpt_seed
        if excerpt_seed is None
        else _derive_seed_pair(excerpt_seed)[1]
    )

    if (sources is None) == (filepaths is None):
        raise ValueError(
            "Provide exactly one of `sources` or `filepaths` "
            f"(got sources={sources!r}, filepaths={filepaths!r})."
        )

    if filepaths is None:
        # Discover files under the given directories.
        if extensions is None:
            extensions = _default_extensions
        filepaths = find_audio_files(sources, extensions)
        if not filepaths:
            raise RuntimeError(
                f"No audio files found in sources {sources} with extensions {extensions}"
            )
    else:
        # Use the caller's explicit list as-is (order preserved).
        filepaths = [os.fspath(filepath) for filepath in filepaths]
        if not filepaths:
            raise ValueError("`filepaths` must be a non-empty list of file paths.")

    # Multi-channel loading needs every file to agree on the channel count, or
    # collation explodes far from the file that caused it. Take the expected
    # count from the first file's header (cheap: no decode) unless declared.
    if mono:
        channels = None
    elif channels is None:
        channels = _probe_channels(filepaths[0])

    # Create dataset from list of filepaths
    ds = grain.MapDataset.source(filepaths)

    if shuffle:
        ds = ds.seed(shuffle_stream_seed).shuffle()

    ds = ds.repeat() if num_epochs is None else ds.repeat(num_epochs)

    # Apply random_map so each index gets its own excerpt RNG
    load_fn = functools.partial(
        _load_excerpt,
        sample_rate=sample_rate,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        excerpt=excerpt,
        source=source,
        channels=channels,
        on_read_error=on_read_error,
    )
    ds = ds.seed(excerpt_stream_seed).random_map(load_fn)

    return ds


def _derive_group_seed(base_seed: int, group_name: str, role: str) -> int:
    """Derive a per-group seed from the group's *name* rather than its position.

    Drawing successive seeds from one RNG while iterating a mapping binds each
    group's stream to its position, so inserting or reordering a group silently
    swaps entire streams and changes what a run trains on. Hashing the name
    instead pins a group's stream to that group. ``role`` (``"shuffle"`` vs
    ``"excerpt"``) keeps the two streams independent even when the base seeds
    are equal, which is the default since ``excerpt_seed`` falls back to
    ``shuffle_seed``.

    Args:
        base_seed: The caller's seed for this role.
        group_name: Name of the group the seed is for.
        role: What the seed drives, e.g. ``"shuffle"`` or ``"excerpt"``.

    Returns:
        A seed in ``[0, 2**31)``.
    """
    entropy = [
        int(base_seed),
        zlib.crc32(group_name.encode("utf-8")),
        zlib.crc32(role.encode("utf-8")),
    ]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0]) & 0x7FFFFFFF


def _stamp_source(tree: AudioTree | None, group_name: str) -> AudioTree | None:
    """Give one item of a pre-built dataset group the file-group provenance schema.

    File-based groups produce items whose ``metadata`` holds all three
    provenance keys: ``filepath`` and ``offset`` (stamped by
    :meth:`AudioTree.from_file`) and ``source`` (from
    :func:`create_audio_dataset`'s ``source=`` argument). A pre-built
    ``datasets`` entry -- most naturally built with :meth:`AudioTree.create`
    from synthetic or preprocessed arrays -- may carry none of them.
    :meth:`AudioTree.batch` requires every item in a batch to expose the same
    ``metadata`` keys, so a batch spanning a file group and a pre-built group
    would fail its pytree check on the missing keys -- intermittently, since a
    single-group batch collates fine. Mapping this over each pre-built dataset
    gives every group the same schema.

    ``source`` is set to the group name the caller chose *here*, overwriting
    any ``source`` the dataset already carried -- exactly as a file group's
    name overrides whatever ``create_audio_dataset`` was told, since the user
    named the group at this call. A missing ``filepath`` is filled with empty
    strings and a missing ``offset`` with ``NaN`` -- "no source file" spelled
    honestly, without displacing real provenance the items may already carry.
    Encoding matches :meth:`AudioTree.create`'s own handling: one row per
    batch item.
    """
    if tree is None:
        # A skipped excerpt; grain drops it at to_iter_dataset() anyway.
        return None
    batch_size = _leading_axis_size(tree.waveform, tree.codes, tree.latents)
    metadata = {
        **tree._metadata,
        "source": AudioTree._encode_filepaths([group_name] * batch_size),
    }
    if metadata.get("filepath") is None:
        metadata["filepath"] = AudioTree._encode_filepaths([""] * batch_size)
    if metadata.get("offset") is None:
        metadata["offset"] = np.full((batch_size,), np.nan, dtype=np.float64)
    return tree.replace(_metadata=metadata)


def create_balanced_audio_dataset(
    sources: Mapping[str, str | Path | List[str | Path]] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    datasets: Optional[Mapping[str, grain.MapDataset]] = None,
    *,
    shuffle: bool = True,
    num_epochs: int | None = None,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    duration: Optional[float] = None,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "constant",
    extensions: Optional[List[str]] = None,
    excerpt: ExcerptConfig = _DEFAULT_EXCERPT,
    window: Optional["WindowConfig"] = None,
    channels: Optional[int] = None,
    on_read_error: OnReadError = "raise",
) -> grain.MapDataset:
    """Create a balanced MapDataset from multiple audio groups and/or pre-constructed datasets.

    This function creates a grain MapDataset that samples from multiple sources
    with specified weights. Sources can be either audio file directories or
    pre-constructed grain MapDatasets. It uses grain's random_map for excerpt
    loading, ensuring infinite variety in RNG seeds even when files are repeated.

    Args:
        sources: Optional dictionary mapping group names to directories (or globs,
            or lists of them, as ``str`` or :class:`~pathlib.Path`) of audio
            files. At least one of `sources` or `datasets` must be non-empty.
        weights: Optional dictionary mapping group names to sampling weights.
            Weights are normalized to sum to 1.0. Groups not in the dict
            default to weight 1.0. If None, all groups are weighted equally.
            Every key must name a group in `sources` or `datasets`; an unknown
            key raises rather than silently leaving its intended group at 1.0.
            Every weight must be a finite number greater than zero -- to
            disable a group, omit it from `sources`/`datasets` instead of
            weighting it 0.
        datasets: Optional dictionary mapping group names to pre-constructed grain MapDatasets.
            These datasets will be mixed with file-based sources. Useful for combining
            different data sources or including pre-processed datasets. Each item of a
            pre-built dataset is stamped with ``source=`` its group name (overwriting any
            ``source`` it already carried); a missing ``filepath`` is filled with empty
            strings and a missing ``offset`` with ``NaN``, so every item exposes the same
            provenance schema as the file-based groups and the two collate together under
            :meth:`AudioTree.batch`.
            IMPORTANT: Pre-constructed datasets MUST already be repeated (call `.repeat()`
            before passing them) to ensure infinite sampling. If a finite dataset is passed,
            grain.MapDataset.mix will truncate the mixed output to the shortest dataset length.
        shuffle: Whether to shuffle files within each file-based group. Set to False for
            deterministic iteration (e.g., pre-rendering). Does not affect pre-constructed datasets.
        num_epochs: How many passes each *file-based* group makes before it runs
            dry. ``None`` (the default) repeats every group forever, which is
            what balanced mixing normally wants: ``grain.MapDataset.mix``
            truncates its output to the shortest input, so any finite group caps
            the whole mixture. An integer ``n >= 1`` gives each group ``n``
            passes and therefore a finite mixture bounded by the smallest of
            them; ``0`` or a negative count raises. Note the default differs
            from :func:`create_audio_dataset`'s single pass -- there one pass
            over the corpus is exactly one epoch, whereas here a finite group
            silently truncates every other group.
        shuffle_seed: Random seed for shuffling file order. Each group's own seed is
            derived from this and the group's *name*, so adding or reordering
            groups leaves the other groups' streams untouched.
        excerpt_seed: Random seed for excerpt selection (random_map). If None, defaults
            to shuffle_seed. Derived per group the same way, and kept independent
            of the shuffle stream even when the two base seeds are equal.
        sample_rate: Target sample rate for audio files (only applies to file-based sources).
        mono: Whether to convert audio to mono (only applies to file-based sources, 0 or 1).
        duration: Duration in seconds to load from each file (only applies to
            file-based sources). Defaults to 1.0. Mutually exclusive with
            ``window``, which carries its own ``duration``.
        pad_mode: Padding mode for files shorter than duration (only applies to file-based sources).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        extensions: List of audio file extensions to search for (only applies to file-based sources).
        excerpt: Which part of each file to take; see :class:`ExcerptConfig`.
            Defaults to a uniformly random offset. Only applies to file-based sources.
        window: Optional :class:`~audiotree.sources.WindowConfig`. When
            given, each file-based group is built with
            :func:`~audiotree.sources.create_windowed_audio_dataset` (length-aware,
            evenly-covering window sampling) instead of one excerpt per file, using
            the group's ``duration``/``alpha``/etc. from the params and the shared
            ``sample_rate``/``mono``/``pad_mode`` here. The group ``weights`` still
            balance across groups, composing multiplicatively with the within-group
            length weighting. Mutually exclusive with a customized ``excerpt``
            and with ``duration``.
        channels: Expected channel count of every file (only applies to file-based
            sources built without ``window``); see
            :func:`create_audio_dataset`. When None (and ``mono=False``), one
            file is probed and the count is applied to *every* group, so two
            internally-consistent groups that disagree with each other fail at
            load time with a filename instead of at batch time with a shape
            error.
        on_read_error: What to do when one file cannot be read; see
            :func:`create_audio_dataset`. Applies to every file-based group
            (and, like ``channels``, is not supported alongside
            ``window``). Pre-constructed ``datasets`` keep whatever
            policy they were built with -- mixing a ``"raise"`` dataset with a
            ``"skip"`` one produces items with different extras keys, which
            :meth:`AudioTree.batch` cannot collate.

    Returns:
        A grain.MapDataset that interleaves items from source groups according
        to the specified weights -- infinite unless ``num_epochs`` is an integer
        or a finite dataset was passed in ``datasets``.

    Example:
        Set up two small groups of ``.wav`` files in temporary directories:

        >>> import os, tempfile
        >>> import numpy as np
        >>> import soundfile
        >>> speech_dir, music_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        >>> for d in (speech_dir, music_dir):
        ...     for i in range(2):
        ...         soundfile.write(
        ...             os.path.join(d, f"{i}.wav"),
        ...             np.zeros((44100, 1), dtype=np.float32),
        ...             44100,
        ...         )

        Equal weighting (the default). The returned dataset is infinite, so
        index it directly rather than calling ``len``:

        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": [speech_dir], "music": [music_dir]},
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )
        >>> ds[0].waveform.shape
        (1, 1, 44100)

        Custom weights (70% speech, 30% music):

        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": [speech_dir], "music": [music_dir]},
        ...     weights={"speech": 0.7, "music": 0.3},
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )

        For pre-rendering (deterministic, no shuffle):

        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": [speech_dir], "music": [music_dir]},
        ...     weights={"speech": 0.5, "music": 0.5},
        ...     shuffle=False,
        ...     shuffle_seed=42,
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )

        Mix file sources with a pre-constructed (already repeated) dataset:

        >>> preprocessed_ds = create_audio_dataset(sources=music_dir, num_epochs=None)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": [speech_dir]},
        ...     datasets={"preprocessed": preprocessed_ds},
        ...     weights={"speech": 0.7, "preprocessed": 0.3},
        ... )
    """
    num_epochs = _validate_num_epochs(num_epochs)
    on_read_error = _validate_on_read_error(on_read_error)

    if sources is None and datasets is None:
        raise ValueError("At least one of 'sources' or 'datasets' must be provided")

    if not sources and not datasets:
        raise ValueError(
            "No groups to mix: 'sources' and 'datasets' are both empty "
            f"(got sources={sources!r}, datasets={datasets!r})."
        )

    if window is not None and excerpt != _DEFAULT_EXCERPT:
        raise ValueError(
            "Pass at most one of `window` or a customized `excerpt`; "
            "windowed sampling chooses its own offsets and does its loudness "
            "filtering through `window` instead."
        )

    if window is not None and duration is not None:
        raise ValueError(
            "Pass at most one of `window` or `duration`; windowed "
            "sampling takes its excerpt length from `window.duration` "
            f"(got duration={duration!r}, window.duration="
            f"{window.duration!r})."
        )

    if window is not None and on_read_error != "raise":
        raise ValueError(
            "`window` does not support `on_read_error`: windowed "
            "sampling reads every file's duration up front to build its "
            "window index, so an unreadable file always raises there "
            f"(got on_read_error={on_read_error!r})."
        )

    if duration is None:
        duration = 1.0

    collisions = sorted(set(sources or {}) & set(datasets or {}))
    if collisions:
        raise ValueError(
            f"Group name(s) appear in both `sources` and `datasets`: {collisions}. "
            "Each group name must be unique across the two; a name in both would "
            "build two separate datasets sharing one weight, doubling that group's "
            "effective proportion."
        )

    group_names = list(sources or {}) + list(datasets or {})
    if weights is not None:
        unknown = [name for name in weights if name not in group_names]
        if unknown:
            raise ValueError(
                f"Unknown group name(s) in `weights`: {unknown}. Valid group "
                f"names are: {group_names}."
            )
        # A weight <= 0 would otherwise fall through to grain.MapDataset.mix,
        # which fails with messages naming neither the group nor this argument
        # (0.0 -> "Must specify all non-zero proportions for mixing.").
        invalid = {
            name: weight
            for name, weight in weights.items()
            if not (
                isinstance(weight, (int, float))
                and math.isfinite(weight)
                and weight > 0
            )
        }
        if invalid:
            raise ValueError(
                f"Invalid weight(s) in `weights`: {invalid}. Every weight must "
                "be a finite number > 0. To leave a group out of the mix, omit "
                "it from `sources`/`datasets` rather than giving it weight 0."
            )

    if excerpt_seed is None:
        excerpt_seed = shuffle_seed

    all_datasets = []
    all_proportions = []

    # Create datasets from file-based sources
    sources = sources or {}

    # Resolve the channel count once, across groups. Leaving each group to
    # probe its own first file would let two internally-consistent groups that
    # disagree with each other (a stereo music group, a mono speech group) both
    # pass, and the mismatch would surface at batch time, with no filename, and
    # only on a batch that happens to span both groups. Probing one file here
    # and handing the count to every group turns that into a load-time error
    # naming the offending file.
    if window is None and not mono and channels is None and sources:
        first_group_files = find_audio_files(
            next(iter(sources.values())), extensions or _default_extensions
        )
        if first_group_files:
            channels = _probe_channels(first_group_files[0])

    for group_name, folders in sources.items():
        if window is not None:
            # Length-aware windowed sampling within this group.
            from audiotree.sources.windowed import create_windowed_audio_dataset

            ds = create_windowed_audio_dataset(
                sources=folders,
                duration=window.duration,
                hop=window.hop,
                alpha=window.alpha,
                jitter=window.jitter,
                lufs_cache=window.lufs_cache,
                lufs_cutoff=window.lufs_cutoff,
                lufs_window_sec=window.lufs_window_sec,
                shuffle=shuffle,
                num_epochs=num_epochs,
                shuffle_seed=_derive_group_seed(shuffle_seed, group_name, "shuffle"),
                excerpt_seed=_derive_group_seed(excerpt_seed, group_name, "excerpt"),
                sample_rate=sample_rate,
                mono=mono,
                pad_mode=pad_mode,
                extensions=extensions,
                source=group_name,
            )
        else:
            ds = create_audio_dataset(
                sources=folders,
                shuffle=shuffle,
                num_epochs=num_epochs,
                shuffle_seed=_derive_group_seed(shuffle_seed, group_name, "shuffle"),
                excerpt_seed=_derive_group_seed(excerpt_seed, group_name, "excerpt"),
                sample_rate=sample_rate,
                mono=mono,
                duration=duration,
                pad_mode=pad_mode,
                extensions=extensions,
                excerpt=excerpt,
                source=group_name,  # Stamp provenance with the group name
                channels=channels,
                on_read_error=on_read_error,
            )

        all_datasets.append(ds)

        # Get weight for this group (default to 1.0)
        weight = 1.0
        if weights is not None:
            weight = weights.get(group_name, 1.0)
        all_proportions.append(weight)

    # Add pre-constructed datasets. Stamp each with its group name's `source`
    # so it batches with the file-based groups above (which already carry it);
    # see `_stamp_source`.
    if datasets is not None:
        for group_name, ds in datasets.items():
            all_datasets.append(
                ds.map(functools.partial(_stamp_source, group_name=group_name))
            )

            # Get weight for this group (default to 1.0)
            weight = 1.0
            if weights is not None:
                weight = weights.get(group_name, 1.0)
            all_proportions.append(weight)

    # Mix datasets with weights. Note: mix() produces an alternating pattern
    # (A, B, A, B, ...). If you need randomized order, call .shuffle() on the result.
    return grain.MapDataset.mix(all_datasets, weights=all_proportions)
