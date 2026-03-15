"""TreeDataSource: pytree-native reader for memory-mapped datasets."""

import json
from pathlib import Path
from typing import Any, Dict, List, SupportsIndex, Union

import numpy as np
from grain.sources import RandomAccessDataSource

from audiotree.core import AudioTree


def _reconstruct(node, leaf_values: Dict[str, Any]):
    """Reconstruct a pytree from its structure description and leaf values.

    Args:
        node: A structure node from the manifest. Strings are array leaf
            references, dicts with "type" are internal nodes.
        leaf_values: Dict mapping leaf path strings to values (numpy arrays
            for array leaves, Python str for string leaves).

    Returns:
        Reconstructed pytree (AudioTree, dict, array, or str).
    """
    # String -> array leaf reference
    if isinstance(node, str):
        return leaf_values[node]

    node_type = node["type"]

    if node_type == "string_leaf":
        return leaf_values[node["leaf"]]

    if node_type == "dict":
        return {
            k: _reconstruct(v, leaf_values)
            for k, v in node["children"].items()
        }

    if node_type == "AudioTree":
        children = {
            k: _reconstruct(v, leaf_values)
            for k, v in node["children"].items()
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
        directory: Path to the directory containing manifest.json and
            memory-mapped data files.
        raw: If True, return a flat Dict[str, np.ndarray] keyed by leaf path
            strings instead of reconstructing the pytree. Default False.

    Example:
        >>> ds = TreeDataSource("dataset/")
        >>> sample = ds[0]
        >>> print(type(sample))  # <class 'AudioTree'>
    """

    def __init__(
        self,
        directory: Union[str, Path],
        raw: bool = False,
    ):
        self.data_dir = Path(directory)
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
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
        self._string_leaf_info = self.manifest.get("string_leaves", {})

        # Lazily initialized per-process; not set here so the object stays
        # picklable for grain worker processes.
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []
        self._bagz_readers: Dict = {}

    def _open_data_files(self):
        """Open all memmap and bagz files. Called once per process on first access."""
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

        if self._string_leaf_info:
            try:
                import bagz
            except ImportError:
                raise ImportError(
                    "The 'bagz' package is required for reading string leaves. "
                    "Install it with: pip install bagz"
                ) from None
            for name, info in self._string_leaf_info.items():
                self._bagz_readers[name] = bagz.Reader(
                    str(self.data_dir / info["file"])
                )

    def __getstate__(self):
        """Drop memmaps/readers before pickling (they reopen lazily in workers)."""
        state = self.__dict__.copy()
        state["_memmaps"] = []
        state["_leaf_names"] = []
        state["_bagz_readers"] = {}
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

        if not self._memmaps and not self._bagz_readers:
            self._open_data_files()

        leaf_values: Dict[str, Any] = {}
        for name, mm in zip(self._leaf_names, self._memmaps):
            leaf_values[name] = np.array(mm[idx])[np.newaxis, ...]

        for name, reader in self._bagz_readers.items():
            leaf_values[name] = reader[idx].decode("utf-8")

        if self.raw:
            return leaf_values

        return _reconstruct(self._structure, leaf_values)

    def get_metadata(self) -> Dict:
        """Get user metadata from the manifest.

        Returns:
            Dictionary containing user-provided metadata
        """
        return dict(self.manifest.get("metadata", {}))

