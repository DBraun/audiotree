import functools
import glob
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Mapping, Optional

import grain
import numpy as np

from audiotree import AudioTree
from audiotree.core import SaliencyParams

if TYPE_CHECKING:
    from audiotree.sources.windowed import WindowParams

_default_extensions = [".wav", ".flac"]


def find_audio_files(
    sources: str | List[str],
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

    Args:
        sources: A path or glob pattern, or a list of them. Each may be a
            directory (searched recursively), a file, or a glob pattern such as
            ``"/mnt/d/musdb18hq/train/*/mixture.wav"``.
        extensions: File extensions to match (e.g. ``[".wav", ".flac"]``).
            Defaults to ``[".wav", ".flac"]``.

    Returns:
        A sorted, de-duplicated list of matching file paths.
    """
    if isinstance(sources, str):
        sources = [sources]
    if extensions is None:
        extensions = _default_extensions
    extensions_lower = {ext.lstrip(".").lower() for ext in extensions}

    def _has_audio_extension(filename: str) -> bool:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in extensions_lower

    filepaths = []
    for source in sources:
        source = os.path.expandvars(str(Path(source).expanduser()))
        # A glob pattern is expanded to its matches; a plain path matches itself.
        matches = (
            glob.glob(source, recursive=True) if glob.has_magic(source) else [source]
        )
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
            elif os.path.isfile(match) and _has_audio_extension(match):
                filepaths.append(match)
    return sorted(set(filepaths))


def _load_audio_with_saliency(
    file_path: str,
    rng: np.random.Generator,
    sample_rate: int,
    duration: float,
    mono: bool = True,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "reflect",
    saliency_params: SaliencyParams | None = None,
    source: str | None = None,
) -> AudioTree:
    """Load audio file with optional saliency-based excerpt selection.

    Uses the provided RNG for deterministic random selection.
    This function is designed to work with grain's random_map.

    Args:
        file_path: Path to the audio file.
        rng: Random number generator from grain's random_map.
        sample_rate: Target sample rate for audio files.
        duration: Duration in seconds to load from each file.
        mono: Whether to convert audio to mono.
        pad_mode: Padding mode for files shorter than duration (numpy.pad modes).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        saliency_params: Optional saliency parameters for excerpt selection.
            If None or disabled: loads from beginning (deterministic)
            If enabled without loudness_cutoff: random excerpt using RNG
            If enabled with loudness_cutoff: multi-try saliency search for loud sections
        source: Optional source group name (e.g., "music", "speech") to store in metadata.

    Returns:
        AudioTree with the loaded audio data.
    """
    # Deterministic load from beginning if saliency is disabled
    if saliency_params is None or not saliency_params.enabled:
        return AudioTree.from_file(
            file_path,
            sample_rate=sample_rate,
            offset=0,
            duration=duration,
            mono=mono,
            pad_mode=pad_mode,
            source=source,
        )

    # Multi-try saliency search: find loud sections using multiple random samples
    if saliency_params.loudness_cutoff is not None:
        return AudioTree.salient_excerpt(
            file_path,
            rng,
            saliency_params=saliency_params,
            sample_rate=sample_rate,
            duration=duration,
            mono=mono,
            pad_mode=pad_mode,
            source=source,
        )

    # Simple random excerpt: use RNG for random position, no loudness filtering
    return AudioTree.excerpt(
        file_path,
        rng=rng,
        sample_rate=sample_rate,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        source=source,
    )


def create_audio_dataset(
    sources: List[str] | str | None = None,
    filepaths: List[str] | None = None,
    shuffle: bool = True,
    repeat: bool = False,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    duration: float = 1.0,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "constant",
    extensions: Optional[List[str]] = None,
    saliency_params: Optional[SaliencyParams] = None,
    source: str | None = None,
) -> grain.MapDataset:
    """Create a simple MapDataset from audio files.

    This function creates a grain MapDataset that loads audio files from one or more
    directories. Unlike `create_balanced_audio_dataset`, this treats all files equally
    without balancing across groups.

    Args:
        sources: A directory path, file path, or glob pattern (e.g.
            ``"/data/*/mixture.wav"``), or a list of them, containing audio files.
            See :func:`find_audio_files` for how each entry is resolved.
            Mutually exclusive with ``filepaths`` — provide exactly one.
        filepaths: An explicit list of audio file paths to use instead of searching
            ``sources``. Mutually exclusive with ``sources`` — provide exactly one.
            Useful for custom splits (e.g. train/val) over a single directory without
            reorganizing it on disk. The given order is preserved (then shuffled if
            ``shuffle=True``).
        shuffle: Whether to shuffle files.
        repeat: Whether to repeat the dataset infinitely. Set to True for training,
            False for validation/testing.
        shuffle_seed: Random seed for shuffling file order.
        excerpt_seed: Random seed for excerpt selection (random_map). If None, defaults
            to shuffle_seed. Use different values to create datasets that visit files
            in the same order but load different random excerpts.
        sample_rate: Target sample rate for audio files.
        mono: Whether to convert audio to mono.
        duration: Duration in seconds to load from each file.
        pad_mode: Padding mode for files shorter than duration (numpy.pad modes).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        extensions: List of audio file extensions to search for. Defaults to [".wav", ".flac"].
        saliency_params: Optional saliency parameters for excerpt selection.
        source: Optional source group name (e.g., "music", "speech") to store in metadata.
            If None, no source metadata is added.

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

        Training dataset (shuffle and repeat infinitely):

        >>> train_ds = create_audio_dataset(
        ...     sources=data_dir,
        ...     shuffle=True,
        ...     repeat=True,
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )

        Validation dataset (deterministic, no repeat):

        >>> val_ds = create_audio_dataset(
        ...     sources=data_dir,
        ...     shuffle=False,
        ...     repeat=False,
        ...     sample_rate=44100,
        ...     duration=1.0,
        ... )

        Two datasets that visit files in the same order but load different
        random excerpts (same ``shuffle_seed``, different ``excerpt_seed``):

        >>> ds1 = create_audio_dataset(sources=data_dir, shuffle_seed=42, excerpt_seed=100)
        >>> ds2 = create_audio_dataset(sources=data_dir, shuffle_seed=42, excerpt_seed=200)
    """
    if excerpt_seed is None:
        excerpt_seed = shuffle_seed

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
        filepaths = list(filepaths)
        if not filepaths:
            raise ValueError("`filepaths` must be a non-empty list of file paths.")

    # Create dataset from list of filepaths
    ds = grain.MapDataset.source(filepaths)

    if shuffle:
        ds = ds.seed(shuffle_seed).shuffle()

    if repeat:
        ds = ds.repeat()

    # Apply random_map for loading with saliency
    load_fn = functools.partial(
        _load_audio_with_saliency,
        sample_rate=sample_rate,
        duration=duration,
        mono=mono,
        pad_mode=pad_mode,
        saliency_params=saliency_params,
        source=source,
    )
    ds = ds.seed(excerpt_seed).random_map(load_fn)

    return ds


def create_balanced_audio_dataset(
    sources: Mapping[str, List[str]] | None = None,
    weights: Optional[Mapping[str, float]] = None,
    datasets: Optional[Mapping[str, grain.MapDataset]] = None,
    shuffle: bool = True,
    repeat: bool = True,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    duration: float = 1.0,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"]
    | None = "constant",
    extensions: Optional[List[str]] = None,
    saliency_params: Optional[SaliencyParams] = None,
    window_params: Optional["WindowParams"] = None,
) -> grain.MapDataset:
    """Create a balanced MapDataset from multiple audio groups and/or pre-constructed datasets.

    This function creates a grain MapDataset that samples from multiple sources
    with specified weights. Sources can be either audio file directories or
    pre-constructed grain MapDatasets. It uses grain's random_map for saliency-based
    loading, ensuring infinite variety in RNG seeds even when files are repeated.

    Args:
        sources: Optional dictionary mapping group names to lists of directories for
            audio files. At least one of `sources` or `datasets` must be provided.
        weights: Optional dictionary mapping group names to sampling weights.
            Weights are normalized to sum to 1.0. Groups not in the dict
            default to weight 1.0. If None, all groups are weighted equally.
            Group names can refer to keys in either `sources` or `datasets`.
        datasets: Optional dictionary mapping group names to pre-constructed grain MapDatasets.
            These datasets will be mixed with file-based sources. Useful for combining
            different data sources or including pre-processed datasets.
            IMPORTANT: Pre-constructed datasets MUST already be repeated (call `.repeat()`
            before passing them) to ensure infinite sampling. If a finite dataset is passed,
            grain.MapDataset.mix will truncate the mixed output to the shortest dataset length.
        shuffle: Whether to shuffle files within each file-based group. Set to False for
            deterministic iteration (e.g., pre-rendering). Does not affect pre-constructed datasets.
        repeat: Whether to repeat the dataset. If False, then the overall length is limited by smallest of the
            underlying datasets. See ``grain.MapDataset.mix``
        shuffle_seed: Random seed for shuffling file order. Used to initialize an RNG
            that derives independent seeds for each group and the final mix.
        excerpt_seed: Random seed for excerpt selection (random_map). If None, defaults
            to shuffle_seed. Used to initialize an RNG that derives independent seeds
            for each group.
        sample_rate: Target sample rate for audio files (only applies to file-based sources).
        mono: Whether to convert audio to mono (only applies to file-based sources, 0 or 1).
        duration: Duration in seconds to load from each file (only applies to file-based sources).
        pad_mode: Padding mode for files shorter than duration (only applies to file-based sources).
            Options: "constant" (zeros), "edge" (repeat edge), "reflect" (mirror),
            "symmetric" (mirror with edge), "wrap" (circular), or None (no padding).
        extensions: List of audio file extensions to search for (only applies to file-based sources).
        saliency_params: Optional saliency parameters for excerpt selection (only applies to file-based sources).
        window_params: Optional :class:`~audiotree.sources.WindowParams`. When
            given, each file-based group is built with
            :func:`~audiotree.sources.create_windowed_audio_dataset` (length-aware,
            evenly-covering window sampling) instead of one excerpt per file, using
            the group's ``duration``/``alpha``/etc. from the params and the shared
            ``sample_rate``/``mono``/``pad_mode`` here. The group ``weights`` still
            balance across groups, composing multiplicatively with the within-group
            length weighting. Mutually exclusive with ``saliency_params``.

    Returns:
        An infinite grain.MapDataset that interleaves items from source groups
        according to the specified weights.

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

        >>> preprocessed_ds = create_audio_dataset(sources=music_dir, repeat=True)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": [speech_dir]},
        ...     datasets={"preprocessed": preprocessed_ds},
        ...     weights={"speech": 0.7, "preprocessed": 0.3},
        ... )
    """
    if sources is None and datasets is None:
        raise ValueError("At least one of 'sources' or 'datasets' must be provided")

    if window_params is not None and saliency_params is not None:
        raise ValueError(
            "Pass at most one of `window_params` or `saliency_params`; windowed "
            "sampling does its loudness filtering through `window_params` instead."
        )

    if excerpt_seed is None:
        excerpt_seed = shuffle_seed

    # Create RNGs to derive independent seeds for each group
    shuffle_rng = np.random.default_rng(shuffle_seed)
    excerpt_rng = np.random.default_rng(excerpt_seed)

    all_datasets = []
    all_proportions = []

    # Create datasets from file-based sources
    sources = sources or {}
    for group_name, folders in sources.items():
        if window_params is not None:
            # Length-aware windowed sampling within this group.
            from audiotree.sources.windowed import create_windowed_audio_dataset

            ds = create_windowed_audio_dataset(
                sources=folders,
                duration=window_params.duration,
                hop=window_params.hop,
                alpha=window_params.alpha,
                jitter=window_params.jitter,
                lufs_cache=window_params.lufs_cache,
                lufs_cutoff=window_params.lufs_cutoff,
                lufs_window_sec=window_params.lufs_window_sec,
                shuffle=shuffle,
                repeat=repeat,
                shuffle_seed=int(shuffle_rng.integers(2**31)),
                excerpt_seed=int(excerpt_rng.integers(2**31)),
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
                repeat=repeat,
                shuffle_seed=int(shuffle_rng.integers(2**31)),
                excerpt_seed=int(excerpt_rng.integers(2**31)),
                sample_rate=sample_rate,
                mono=mono,
                duration=duration,
                pad_mode=pad_mode,
                extensions=extensions,
                saliency_params=saliency_params,
                source=group_name,  # Set source metadata to group name
            )

        all_datasets.append(ds)

        # Get weight for this group (default to 1.0)
        weight = 1.0
        if weights is not None:
            weight = weights.get(group_name, 1.0)
        all_proportions.append(weight)

    # Add pre-constructed datasets
    if datasets is not None:
        for group_name, ds in datasets.items():
            all_datasets.append(ds)

            # Get weight for this group (default to 1.0)
            weight = 1.0
            if weights is not None:
                weight = weights.get(group_name, 1.0)
            all_proportions.append(weight)

    # Mix datasets with weights. Note: mix() produces an alternating pattern
    # (A, B, A, B, ...). If you need randomized order, call .shuffle() on the result.
    return grain.MapDataset.mix(all_datasets, weights=all_proportions)
