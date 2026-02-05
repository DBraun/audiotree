import functools
import os
from pathlib import Path
from typing import AnyStr, List, Literal, Mapping, Optional

import grain
import numpy as np

from audiotree import AudioTree
from audiotree.core import SaliencyParams

_default_extensions = [".wav", ".flac"]


def _find_files_with_extensions(
    directory: str, extensions: List[str], max_depth=None, follow_symlinks=False
) -> list[AnyStr]:
    """
    Searches for files with specified extensions up to a maximum depth in the directory,
    without modifying dirs while iterating.

    Args:
        directory (str): The path to the directory to search.
        extensions (list): A list of file extensions to search for. Each extension should include a period.
        max_depth (int): The maximum depth to search for files.
        follow_symlinks (bool): Whether to follow symbolic links during the search.

    Returns:
        list (list[AnyStr]): A list of paths to files that match the extensions within the maximum depth.
    """
    matching_files = []
    extensions_set = {
        ext.lower() for ext in extensions
    }  # Normalize extensions to lowercase for matching

    # Expand environment variables and user home directory
    directory = os.path.expandvars(os.path.expanduser(directory))
    directory = os.path.abspath(directory)  # Ensure the directory path is absolute

    def recurse(current_dir, current_depth):
        if max_depth is not None and current_depth > max_depth:
            return
        with os.scandir(current_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=follow_symlinks) and any(
                    entry.name.lower().endswith(ext) for ext in extensions_set
                ):
                    matching_files.append(entry.path)
                elif entry.is_dir(follow_symlinks=follow_symlinks):
                    recurse(entry.path, current_depth + 1)

    recurse(directory, 0)
    return matching_files


def _load_audio_with_saliency(
    file_path: str,
    rng: np.random.Generator,
    sample_rate: int,
    duration: float,
    mono: bool = True,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None = "reflect",
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
    sources: List[str] | str,
    shuffle: bool = True,
    repeat: bool = False,
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: bool = True,
    duration: float = 1.0,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None = "constant",
    extensions: Optional[List[str]] = None,
    saliency_params: Optional[SaliencyParams] = None,
    source: str | None = None,
) -> grain.MapDataset:
    """Create a simple MapDataset from audio files.

    This function creates a grain MapDataset that loads audio files from one or more
    directories. Unlike `create_balanced_audio_dataset`, this treats all files equally
    without balancing across groups.

    Args:
        sources: A directory path or list of directory paths containing audio files.
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
        >>> # Load all files from a directory
        >>> ds = create_audio_dataset(
        ...     sources="/data/audio",
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Training dataset: shuffle and repeat infinitely
        >>> ds = create_audio_dataset(
        ...     sources=["/data/train1", "/data/train2"],
        ...     shuffle=True,
        ...     repeat=True,
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Validation dataset: deterministic, no repeat, limited records
        >>> ds = create_audio_dataset(
        ...     sources="/data/val",
        ...     shuffle=False,
        ...     repeat=False,
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Two datasets with same file order but different excerpts
        >>> ds1 = create_audio_dataset(
        ...     sources="/data/audio",
        ...     shuffle_seed=42,
        ...     excerpt_seed=100,
        ... )
        >>> ds2 = create_audio_dataset(
        ...     sources="/data/audio",
        ...     shuffle_seed=42,
        ...     excerpt_seed=200,
        ... )
    """
    if excerpt_seed is None:
        excerpt_seed = shuffle_seed

    if extensions is None:
        extensions = _default_extensions

    # Normalize sources to list
    if isinstance(sources, str):
        sources = [sources]

    # Collect all filepaths
    filepaths = []
    for folder in sources:
        folder_path = Path(folder)
        folder_path = Path(os.path.expandvars(os.path.expanduser(str(folder_path))))
        for ext in extensions:
            # Remove leading dot if present
            ext_clean = ext.lstrip('.')
            found_files = folder_path.rglob(f"*.{ext_clean}")
            filepaths.extend([str(p) for p in found_files])

    if not filepaths:
        raise RuntimeError(
            f"No audio files found in sources {sources} with extensions {extensions}"
        )

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
    shuffle_seed: int = 0,
    excerpt_seed: int | None = None,
    sample_rate: int = 44_100,
    mono: int = 1,
    duration: float = 1.0,
    pad_mode: Literal["constant", "edge", "reflect", "symmetric", "wrap"] | None = "constant",
    extensions: Optional[List[str]] = None,
    saliency_params: Optional[SaliencyParams] = None,
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

    Returns:
        An infinite grain.MapDataset that interleaves items from source groups
        according to the specified weights.

    Example:
        >>> # Equal weighting (default)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Custom weights: 70% speech, 30% music
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     weights={"speech": 0.7, "music": 0.3},
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # For pre-rendering (deterministic, no shuffle)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     weights={"speech": 0.5, "music": 0.5},
        ...     shuffle=False,
        ...     shuffle_seed=42,
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Mix file sources with a pre-constructed dataset
        >>> preprocessed_ds = create_audio_dataset(sources="/data/preprocessed", repeat=True)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"]},
        ...     datasets={"preprocessed": preprocessed_ds},
        ...     weights={"speech": 0.7, "preprocessed": 0.3},
        ... )
    """
    if sources is None and datasets is None:
        raise ValueError("At least one of 'sources' or 'datasets' must be provided")

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
        # Create dataset for this group with repeat=True (required for mixing)
        ds = create_audio_dataset(
            sources=folders,
            shuffle=shuffle,
            repeat=True,  # Always repeat before mixing
            shuffle_seed=int(shuffle_rng.integers(2**31)),
            excerpt_seed=int(excerpt_rng.integers(2**31)),
            sample_rate=sample_rate,
            mono=bool(mono),
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
