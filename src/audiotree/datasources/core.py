import glob
import math
import os
import warnings
from random import Random
from typing import AnyStr, List, Mapping, Optional, SupportsIndex, Union

from grain._src.python.dataset.transformations.mix import MixedIterDataset
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


class AudioDataSourceMixin:

    def load_audio(self, file_path, record_key: SupportsIndex) -> AudioTree:

        saliency_params: SaliencyParams = self.saliency_params

        if saliency_params is not None and saliency_params.enabled:
            if saliency_params.loudness_cutoff is not None:
                # Use salient_excerpt with loudness filtering
                return AudioTree.salient_excerpt(
                    file_path,
                    np.random.default_rng(int(record_key)),
                    saliency_params=saliency_params,
                    sample_rate=self.sample_rate,
                    duration=self.duration,
                    mono=self.mono,
                    pad_mode=self.pad_mode,
                )
            else:
                # Use excerpt for random offset without loudness filtering
                return AudioTree.excerpt(
                    file_path,
                    rng=np.random.default_rng(int(record_key)),
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                    mono=self.mono,
                    pad_mode=self.pad_mode,
                )
        else:
            # Load from beginning (deterministic)
            return AudioTree.from_file(
                file_path,
                sample_rate=self.sample_rate,
                offset=0,
                duration=self.duration,
                mono=self.mono,
                pad_mode=self.pad_mode,
            )


class AudioDataSimpleSource(grain.sources.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that aggregates all source files and weights them equally.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        num_records (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        pad_mode (str): The requested padding mode.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use. Defaults to None, meaning AudioTree.from_file
            will be used. If not None, either AudioTree.salient_excerpt will be used or AudioTree.excerpt will be used.
    """

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_records: int = None,
        sample_rate: int = 44_100,
        mono: int = 1,
        duration: float = 1.0,
        pad_mode: str = "constant",
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        self.pad_mode = pad_mode
        if extensions is None:
            extensions = _default_extensions
        self.saliency_params = saliency_params

        filepaths = []
        for group_name, folders in sources.items():
            filepaths_in_group = []
            for _folder in folders:
                folder = os.path.expandvars(os.path.expanduser(_folder))
                if os.path.isdir(folder):
                    filepaths_in_group += _find_files_with_extensions(
                        folder, extensions=extensions
                    )
                else:
                    filepaths_in_group += list(glob.glob(folder, recursive=True))

            if filepaths_in_group:
                filepaths += filepaths_in_group
            else:
                raise RuntimeError(
                    f"Group '{group_name}' is empty. "
                    f"The number of specified folders in the group was {len(folders)}. "
                    f"The approved file extensions were {extensions}."
                )

        if num_records is not None:
            filepaths = filepaths[:num_records]

        self.filepaths = filepaths

        self._length = len(filepaths)
        assert self._length > 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, record_key: SupportsIndex):
        file_path = self.filepaths[record_key]
        return self.load_audio(file_path, record_key)


class AudioDataBalancedSource(grain.sources.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that equally weights multiple sources, where each source is a list of directories.

    .. deprecated::
        Use :func:`create_balanced_audio_dataset` instead, which uses grain's
        public MapDataset.mix() API and supports custom weights.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        num_records (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        pad_mode (str): The requested padding mode.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
    """

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_records: int,
        sample_rate: int = 44_100,
        mono: int = 1,
        duration: float = 1.0,
        pad_mode: str = "constant",
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
    ):
        warnings.warn(
            "AudioDataBalancedSource is deprecated. Use create_balanced_audio_dataset() "
            "instead, which uses grain's public MapDataset.mix() API and supports custom weights.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        self.pad_mode = pad_mode
        if extensions is None:
            extensions = _default_extensions
        self.saliency_params = saliency_params

        groups = []

        for group_name, folders in sources.items():
            filepaths = []
            for _folder in folders:
                folder = os.path.expandvars(os.path.expanduser(_folder))
                if os.path.isdir(os.path.expandvars(os.path.expanduser(folder))):
                    filepaths += _find_files_with_extensions(
                        folder, extensions=extensions
                    )
                else:
                    filepaths += list(glob.glob(folder))

            if filepaths:
                groups.append(filepaths)
            else:
                raise RuntimeError(
                    f"Group '{group_name}' is empty. "
                    f"The number of specified folders in the group was {len(folders)}. "
                    f"The approved file extensions were {extensions}."
                )

        self._num_groups = len(groups)
        self._length = num_records

        ideal_group_length = math.ceil(num_records / self._num_groups)
        seed = 0
        lengthened_groups = []
        for group in groups:
            num_loops = math.ceil(ideal_group_length / len(group))
            lengthened_group = []
            for _ in range(num_loops):
                copied = group.copy()
                Random(seed).shuffle(copied)
                seed += 1
                lengthened_group += copied
            lengthened_groups.append(lengthened_group)
        self._groups = lengthened_groups

        assert self._length > 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, record_key: SupportsIndex):
        record_key = int(record_key)

        group_idx = record_key % self._num_groups
        idx = record_key // self._num_groups

        file_path = self._groups[group_idx][idx]

        return self.load_audio(file_path, record_key)


class AudioDataBalancedDataset(MixedIterDataset):
    """A Data Source that equally weights multiple sources, where each source is a list of directories.

    .. deprecated::
        Use :func:`create_balanced_audio_dataset` instead, which uses grain's
        public MapDataset.mix() API and returns a MapDataset with random access.
        This class uses internal grain APIs (MixedIterDataset) that may change.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        pad_mode (str): The requested padding mode.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
        weights (Mapping[str, float]): A dictionary mapping each source to its proportion in the dataset.
    """

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        sample_rate: int = 44_100,
        mono: int = 1,
        duration: float = 1.0,
        pad_mode: str = "constant",
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
        weights: Mapping[str, float] = None,
    ):
        warnings.warn(
            "AudioDataBalancedDataset is deprecated. Use create_balanced_audio_dataset() "
            "instead, which uses grain's public MapDataset.mix() API and returns a "
            "MapDataset with random access.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        self.pad_mode = pad_mode
        if extensions is None:
            extensions = _default_extensions
        self.saliency_params = saliency_params

        datasets = []

        seed = 0
        proportions = []
        for group_name, folders in sources.items():
            datasource = AudioDataSimpleSource(
                sources={group_name: folders},
                num_records=None,
                sample_rate=sample_rate,
                mono=mono,
                duration=duration,
                extensions=extensions,
                saliency_params=saliency_params,
            )
            dataset = (
                grain.MapDataset.source(datasource)
                .shuffle(seed=seed)
                .repeat()
                .to_iter_dataset()
            )
            seed += 1
            datasets.append(dataset)
            weight = 1.0
            if isinstance(weights, dict):
                weight = weights.get(group_name, 1.0)
            proportions.append(weight * 1000)

        super().__init__(datasets, proportions=proportions)


def create_balanced_audio_dataset(
    sources: Mapping[str, List[str]],
    num_records: int,
    weights: Optional[Mapping[str, float]] = None,
    shuffle: bool = True,
    seed: int = 0,
    sample_rate: int = 44_100,
    mono: int = 1,
    duration: float = 1.0,
    pad_mode: str = "constant",
    extensions: Optional[List[str]] = None,
    saliency_params: Optional[SaliencyParams] = None,
) -> grain.MapDataset:
    """Create a balanced MapDataset from multiple audio groups.

    This function creates a grain MapDataset that samples from multiple audio
    source groups with specified weights. It uses grain's public MapDataset.mix()
    API for weighted mixing.

    Args:
        sources: A dictionary mapping group names to lists of directories or
            glob expressions for audio files.
        num_records: Total number of records in the resulting dataset.
        weights: Optional dictionary mapping group names to sampling weights.
            Weights are normalized to sum to 1.0. Groups not in the dict
            default to weight 1.0. If None, all groups are weighted equally.
        shuffle: Whether to shuffle files within each group. Set to False for
            deterministic iteration (e.g., pre-rendering).
        seed: Random seed for shuffling. Each group uses seed + group_index
            for independent shuffling.
        sample_rate: Target sample rate for audio files.
        mono: Whether to convert audio to mono (0 or 1).
        duration: Duration in seconds to load from each file.
        pad_mode: Padding mode for files shorter than duration.
        extensions: List of audio file extensions to search for.
        saliency_params: Optional saliency parameters for excerpt selection.

    Returns:
        A grain.MapDataset with length num_records that samples from the
        source groups according to the specified weights.

    Example:
        >>> # Equal weighting (default)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     num_records=10000,
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # Custom weights: 70% speech, 30% music
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     num_records=10000,
        ...     weights={"speech": 0.7, "music": 0.3},
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )

        >>> # For pre-rendering (deterministic, no shuffle)
        >>> ds = create_balanced_audio_dataset(
        ...     sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        ...     num_records=72000,
        ...     weights={"speech": 0.5, "music": 0.5},
        ...     shuffle=False,
        ...     seed=42,
        ...     sample_rate=44100,
        ...     duration=3.0,
        ... )
    """
    if extensions is None:
        extensions = _default_extensions

    datasets = []
    proportions = []
    group_names = list(sources.keys())

    for i, group_name in enumerate(group_names):
        folders = sources[group_name]

        # Create a simple source for this group
        source = AudioDataSimpleSource(
            sources={group_name: folders},
            num_records=None,  # Load all files in group
            sample_rate=sample_rate,
            mono=mono,
            duration=duration,
            pad_mode=pad_mode,
            extensions=extensions,
            saliency_params=saliency_params,
        )

        # Wrap in MapDataset with optional shuffle and repeat
        ds = grain.MapDataset.source(source)
        if shuffle:
            ds = ds.shuffle(seed=seed + i)
        ds = ds.repeat()
        datasets.append(ds)

        # Get weight for this group (default to 1.0)
        weight = 1.0
        if weights is not None:
            weight = weights.get(group_name, 1.0)
        proportions.append(weight)

    # Mix datasets with weights and slice to num_records
    mixed = grain.MapDataset.mix(datasets, weights=proportions)
    return mixed.slice(slice(0, num_records))
