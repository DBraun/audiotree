"""TreeDataSource: pytree-native reader for memory-mapped datasets."""

import json
from pathlib import Path
from typing import Dict, List, SupportsIndex, Union

import numpy as np
from grain.sources import RandomAccessDataSource

from audiotree.core import AudioTree


def _reconstruct(node, leaf_arrays: Dict[str, np.ndarray]):
    """Reconstruct a pytree from its structure description and leaf arrays.

    Args:
        node: A structure node from the manifest. Strings are leaf references,
            dicts with "type" are internal nodes.
        leaf_arrays: Dict mapping leaf path strings to numpy arrays.

    Returns:
        Reconstructed pytree (AudioTree, dict, or array).
    """
    # String -> leaf reference
    if isinstance(node, str):
        return leaf_arrays[node]

    node_type = node["type"]

    if node_type == "dict":
        return {k: _reconstruct(v, leaf_arrays) for k, v in node["children"].items()}

    if node_type == "AudioTree":
        children = {
            k: _reconstruct(v, leaf_arrays) for k, v in node["children"].items()
        }
        children.setdefault("audio_data", None)
        return AudioTree(sample_rate=node["sample_rate"], **children)

    raise ValueError(f"Unknown structure node type: {node_type}")


class TreeDataSource(RandomAccessDataSource):
    """Read pytrees from memory-mapped files created by TreeWriter.

    Provides efficient random access to pre-rendered datasets without loading
    into RAM. Data is read from memory-mapped binary files and reconstructed
    into the original pytree structure (AudioTree, dict, etc.).

    Memmaps are opened lazily on first access and cached for the lifetime of
    the process. This avoids per-item mmap syscall overhead while remaining
    pickle-safe for grain's multiprocessing DataLoader.

    Args:
        manifest_path: Path to the manifest.json file
        raw: If True, return a flat Dict[str, np.ndarray] keyed by leaf path
            strings instead of reconstructing the pytree. Default False.

    Example:
        >>> ds = TreeDataSource("dataset/manifest.json")
        >>> sample = ds[0]
        >>> print(type(sample))  # <class 'AudioTree'>

        >>> ds = TreeDataSource.from_directory("dataset/")
        >>> print(len(ds))  # number of samples
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        raw: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.raw = raw

        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        version = self.manifest.get("version", "")
        if not version.startswith("2."):
            raise ValueError(
                f"Unsupported manifest version: {version!r}. "
                "TreeDataSource requires version 2.x manifests."
            )

        self._num_samples = self.manifest["num_samples"]
        self._structure = self.manifest["structure"]
        self._leaf_info = self.manifest["leaves"]

        # Lazily initialized per-process; not set here so the object stays
        # picklable for grain worker processes.
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []

    def _open_memmaps(self):
        """Open all memmap files. Called once per process on first access."""
        self._leaf_names = list(self._leaf_info.keys())
        for name in self._leaf_names:
            info = self._leaf_info[name]
            full_shape = tuple([self._num_samples] + info["shape_per_sample"])
            mm = np.memmap(
                self.data_dir / info["file"],
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=full_shape,
            )
            self._memmaps.append(mm)

    def __getstate__(self):
        """Drop memmaps before pickling (they reopen lazily in workers)."""
        state = self.__dict__.copy()
        state["_memmaps"] = []
        state["_leaf_names"] = []
        return state

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, record_key: SupportsIndex):
        """Load a single sample by index.

        Args:
            record_key: Index of the record to load

        Returns:
            Reconstructed pytree with batch dimension added to each leaf,
            or Dict[str, np.ndarray] if raw=True.

        Raises:
            IndexError: If index is out of range
        """
        idx = int(record_key)
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._num_samples})")

        if not self._memmaps:
            self._open_memmaps()

        leaf_arrays = {}
        for name, mm in zip(self._leaf_names, self._memmaps):
            leaf_arrays[name] = np.array(mm[idx])[np.newaxis, ...]

        if self.raw:
            return leaf_arrays

        return _reconstruct(self._structure, leaf_arrays)

    def get_metadata(self) -> Dict:
        """Get user metadata from the manifest.

        Returns:
            Dictionary containing user-provided metadata
        """
        return dict(self.manifest.get("metadata", {}))

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        **kwargs,
    ) -> "TreeDataSource":
        """Convenience constructor that finds manifest.json in directory.

        Args:
            directory: Directory containing memmap files and manifest.json
            **kwargs: Additional arguments passed to TreeDataSource

        Returns:
            TreeDataSource configured for the directory
        """
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return cls(manifest_path, **kwargs)
