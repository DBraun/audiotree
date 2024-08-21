import os
from random import Random
from typing import AnyStr, List, Mapping, SupportsIndex

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
                cpu=self.cpu,
            )

        return AudioTree.from_file(
            file_path,
            sample_rate=self.sample_rate,
            offset=0,
            duration=self.duration,
            mono=self.mono,
            cpu=self.cpu,
        )


class AudioDataSimpleSource(grain.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that aggregates all source files and weights them equally.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories.
        num_steps (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
        cpu (bool): Whether to load the audio data on the CPU.
    """

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_steps: int = None,
        sample_rate: int = 44100,
        mono: int = 1,
        duration: float = 1.0,
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
        cpu: bool = False,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        if extensions is None:
            extensions = _default_extensions
        self.saliency_params = saliency_params
        self.cpu = cpu

        filepaths = []
        for group_name, folders in sources.items():
            filepaths_in_group = []
            for folder in folders:
                filepaths_in_group += _find_files_with_extensions(
                    folder, extensions=extensions
                )

            if filepaths_in_group:
                filepaths += filepaths_in_group
            else:
                raise RuntimeError(
                    f"Group '{group_name}' is empty. "
                    f"The number of specified folders in the group was {len(folders)}. "
                    f"The approved file extensions were {extensions}."
                )

        if num_steps is not None:
            filepaths = filepaths[:num_steps]

        self._file_paths = filepaths

        self._length = len(filepaths)
        assert self._length > 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, record_key: SupportsIndex):
        file_path = self._file_paths[record_key]
        return self.load_audio(file_path, record_key)


class AudioDataBalancedSource(grain.RandomAccessDataSource, AudioDataSourceMixin):
    """A Data Source that equally weights multiple sources, where each source is a list of directories.

    Args:
        sources (Mapping[str, List[str]]): A dictionary mapping each source to a list of directories.
        num_steps (int): The requested length of the data source.
        sample_rate (int): The requested sample rate of the audio.
        mono (bool): Whether to force the audio to be mono.
        duration (float): The requested duration of the audio.
        extensions (List[str]): A list of file extensions to search for. Each extension should include a period.
        saliency_params (SaliencyParams): Saliency parameters to use.
        cpu (bool): Whether to load the audio data on the CPU.
    """

    # todo: make this algorithm work if the user specifies weights for the groups.
    #  Right now the groups are balanced uniformly.
    #  Eventually the __init__ should just take a list of ``AudioDataSimpleSource`` and
    #  the corresponding weights.

    def __init__(
        self,
        sources: Mapping[str, List[str]],
        num_steps: int,
        sample_rate: int = 44100,
        mono: int = 1,
        duration: float = 1.0,
        extensions: List[str] = None,
        saliency_params: SaliencyParams = None,
        cpu: bool = False,
    ):

        self.sample_rate = sample_rate
        self.mono = bool(mono)
        self.duration = duration
        if extensions is None:
            extensions = _default_extensions
        self.saliency_params = saliency_params
        self.cpu = cpu

        groups = []

        for group_name, folders in sources.items():
            filepaths = []
            for folder in folders:
                filepaths += _find_files_with_extensions(folder, extensions=extensions)

            if filepaths:
                groups.append(filepaths)
            else:
                raise RuntimeError(
                    f"Group '{group_name}' is empty. "
                    f"The number of specified folders in the group was {len(folders)}. "
                    f"The approved file extensions were {extensions}."
                )

        self._group_to_len = {i: len(group) for i, group in enumerate(groups)}
        self._groups = groups
        self._num_groups = len(groups)
        self._length = num_steps
        assert self._length > 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, record_key: SupportsIndex):
        record_key = int(record_key)

        group_idx = record_key % self._num_groups

        idx = record_key // self._num_groups

        x = idx % self._group_to_len[group_idx]
        y = idx // self._group_to_len[group_idx]

        file_paths_in_group = self._groups[group_idx].copy()

        Random(y + 4617 * group_idx).shuffle(
            file_paths_in_group
        )  # todo: 4617 is arbitrary and could probably be zero.

        file_path = file_paths_in_group[x]

        return self.load_audio(file_path, record_key)
