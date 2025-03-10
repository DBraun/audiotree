import glob
import math
import os
from random import Random
from typing import AnyStr, List, Mapping, SupportsIndex

from grain._src.python.dataset.transformations.mix import MixedIterDataset
from grain import python as grain
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

        if self.saliency_params is not None and self.saliency_params.enabled:
            saliency_params = self.saliency_params
            return AudioTree.salient_excerpt(
                file_path,
                np.random.default_rng(int(record_key)),
                saliency_params=saliency_params,
                sample_rate=self.sample_rate,
                duration=self.duration,
                mono=self.mono,
            )

        return AudioTree.from_file(
            file_path,
            sample_rate=self.sample_rate,
            offset=0,
            duration=self.duration,
            mono=self.mono,
        )


class AudioDataSimpleSource(grain.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that aggregates all source files and weights them equally.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        num_records (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
    """

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_records: int = None,
        sample_rate: int = 44_100,
        mono: int = 1,
        duration: float = 1.0,
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
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


class AudioDataBalancedSource(grain.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that equally weights multiple sources, where each source is a list of directories.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        num_records (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
    """

    # todo: make this algorithm work if the user specifies weights for the groups.
    #  Right now the groups are balanced uniformly.
    #  Eventually the __init__ should just take a list of ``AudioDataSimpleSource`` and
    #  the corresponding weights.
    #  AudioDataBalancedDataset accomplishes this, but since it's an IterDataset it doesn't have all the features
    #  of a plain RandomAccessDataSource.

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_records: int,
        sample_rate: int = 44_100,
        mono: int = 1,
        duration: float = 1.0,
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
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

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories or glob
            expressions involving a file extension.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
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
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
        weights: Mapping[str, float] = None,
    ):
        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
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
