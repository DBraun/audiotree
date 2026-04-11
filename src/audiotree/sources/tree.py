"""TreeDataSource: pytree-native reader for memory-mapped datasets."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, SupportsIndex, Union

import numpy as np
from grain.sources import RandomAccessDataSource

from audiotree.core import AudioTree

# Sentinel for excluded leaves (distinct from None, which is a valid value).
_EXCLUDED = object()


def _reconstruct(node, leaf_values: Dict[str, Any]):
    """Reconstruct a pytree from its structure description and leaf values.

    Args:
        node: A structure node from the manifest. Strings are array leaf
            references, dicts with "type" are internal nodes.
        leaf_values: Dict mapping leaf path strings to values (numpy arrays
            for array leaves, Python str for string leaves). Excluded leaves
            are absent from the dict; they resolve to ``_EXCLUDED`` and are
            omitted from the reconstructed tree.

    Returns:
        Reconstructed pytree (AudioTree, dict, array, or str), or
        ``_EXCLUDED`` if the node itself is an excluded leaf.
    """
    # String -> array leaf reference
    if isinstance(node, str):
        return leaf_values.get(node, _EXCLUDED)

    node_type = node["type"]

    if node_type == "string_leaf":
        return leaf_values.get(node["leaf"], _EXCLUDED)

    if node_type == "dict":
        result = {}
        for k, v in node["children"].items():
            val = _reconstruct(v, leaf_values)
            if val is not _EXCLUDED:
                result[k] = val
        return result

    if node_type == "AudioTree":
        children = {}
        for k, v in node["children"].items():
            val = _reconstruct(v, leaf_values)
            if val is not _EXCLUDED:
                children[k] = val
        children.setdefault("audio_data", None)
        return AudioTree(sample_rate=node["sample_rate"], **children)

    raise ValueError(f"Unknown structure node type: {node_type}")


def _is_excluded(name: str, exclude_prefixes: List[str]) -> bool:
    """Check if a leaf name matches any exclude prefix.

    Matching semantics: ``"dry"`` matches ``"dry"`` exactly or any name
    starting with ``"dry."`` (e.g. ``"dry.audio_data"``), but does NOT
    match ``"dryness"``.
    """
    return any(name == p or name.startswith(p + ".") for p in exclude_prefixes)


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
        exclude_prefixes: List of dot-separated leaf name prefixes to skip
            loading. For example, ``["wet.audio_data"]`` skips the
            ``wet.audio_data`` memmap, and ``["dry"]`` skips all leaves
            under the ``dry`` subtree. Excluded leaves are omitted from
            the reconstructed pytree (AudioTree fields default to None).
            Default: load all leaves.
        load_into_memory: If True, load all non-excluded array leaves and
            string leaves into RAM at init time. Workers then read from
            pre-loaded numpy arrays instead of memmaps, eliminating disk
            I/O. With fork-based multiprocessing (default on Linux), the
            parent's data is shared with workers via copy-on-write.
            Default False.
        cache_memmaps: If True (default), memmap file handles are opened
            once and cached for the lifetime of the process. If False,
            memmaps are reopened on every ``__getitem__`` call, allowing
            the OS to reclaim pages between accesses and preventing page
            cache from growing unboundedly. The False setting trades a
            small CPU overhead (mmap syscalls) for controlled memory.
            Ignored when ``load_into_memory=True``.

    Example:
        >>> ds = TreeDataSource("dataset/")
        >>> sample = ds[0]
        >>> print(type(sample))  # <class 'AudioTree'>

        >>> ds = TreeDataSource("dataset/", exclude_prefixes=["wet.audio_data"])
        >>> sample = ds[0]
        >>> assert sample["wet"].audio_data is None  # excluded

        >>> ds = TreeDataSource("dataset/", load_into_memory=True)
        >>> sample = ds[0]  # reads from RAM, no disk I/O
    """

    def __init__(
        self,
        directory: Union[str, Path],
        raw: bool = False,
        exclude_prefixes: Optional[List[str]] = None,
        load_into_memory: bool = False,
        cache_memmaps: bool = True,
    ):
        self.data_dir = Path(directory)
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.raw = raw
        self.exclude_prefixes: List[str] = exclude_prefixes or []
        self.load_into_memory = load_into_memory
        self.cache_memmaps = cache_memmaps

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

        # In-memory storage (populated eagerly if load_into_memory=True).
        self._in_memory_arrays: Dict[str, np.ndarray] = {}
        self._in_memory_strings: Dict[str, List[str]] = {}

        if load_into_memory:
            self._load_all_into_memory()

        # Lazily initialized per-process; not set here so the object stays
        # picklable for grain worker processes.
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []
        self._bagz_readers: Dict = {}
        self._data_files_opened: bool = False

    def _load_all_into_memory(self):
        """Load all non-excluded leaves into RAM."""
        for name, info in self._leaf_info.items():
            if _is_excluded(name, self.exclude_prefixes):
                continue
            full_shape = tuple([self._num_samples] + info["shape_per_sample"])
            mm = np.memmap(
                self.data_dir / info["file"],
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=full_shape,
            )
            self._in_memory_arrays[name] = np.array(mm)
            del mm

        if self._string_leaf_info:
            try:
                import bagz
            except ImportError:
                raise ImportError(
                    "The 'bagz' package is required for reading string leaves. "
                    "Install it with: pip install bagz"
                ) from None
            for name, info in self._string_leaf_info.items():
                if _is_excluded(name, self.exclude_prefixes):
                    continue
                reader = bagz.Reader(str(self.data_dir / info["file"]))
                self._in_memory_strings[name] = [
                    reader[i].decode("utf-8") for i in range(self._num_samples)
                ]

        # Mark as opened so lazy path is skipped.
        self._data_files_opened = True

    def _open_data_files(self):
        """Open memmap and bagz files, skipping excluded leaves.

        When ``cache_memmaps=False``, only leaf names are collected (memmaps
        are reopened per-access in ``__getitem__`` instead).
        """
        self._leaf_names = []
        for name in self._leaf_info:
            if _is_excluded(name, self.exclude_prefixes):
                continue
            self._leaf_names.append(name)
            if self.cache_memmaps:
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
                if _is_excluded(name, self.exclude_prefixes):
                    continue
                self._bagz_readers[name] = bagz.Reader(
                    str(self.data_dir / info["file"])
                )

        self._data_files_opened = True

    def __getstate__(self):
        """Drop memmaps/readers before pickling (they reopen lazily in workers).

        In-memory data (``_in_memory_arrays``, ``_in_memory_strings``) is
        kept so that fork-based workers inherit the parent's pre-loaded
        data via copy-on-write.
        """
        state = self.__dict__.copy()
        state["_memmaps"] = []
        state["_leaf_names"] = []
        state["_bagz_readers"] = {}
        # If data is in memory, workers don't need to reopen files.
        if not self.load_into_memory:
            state["_data_files_opened"] = False
        return state

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, record_key: SupportsIndex):
        """Load a single sample by index.

        Args:
            record_key: Index of the record to load

        Returns:
            Reconstructed pytree with batch dimension added to each leaf,
            or Dict[str, np.ndarray] if raw=True. Excluded leaves are
            omitted from the result.

        Raises:
            IndexError: If index is out of range
        """
        idx = int(record_key)
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._num_samples})")

        if not self._data_files_opened:
            self._open_data_files()

        leaf_values: Dict[str, Any] = {}

        if self._in_memory_arrays:
            # Fast path: read from pre-loaded RAM arrays.
            for name, arr in self._in_memory_arrays.items():
                leaf_values[name] = arr[idx][np.newaxis, ...]
            for name, strings in self._in_memory_strings.items():
                leaf_values[name] = strings[idx]
        elif self.cache_memmaps:
            # Cached path: read from long-lived memmaps.
            for name, mm in zip(self._leaf_names, self._memmaps):
                leaf_values[name] = np.array(mm[idx])[np.newaxis, ...]
            for name, reader in self._bagz_readers.items():
                leaf_values[name] = reader[idx].decode("utf-8")
        else:
            # Uncached path: reopen memmaps per access to let the OS
            # reclaim pages, preventing page cache from growing.
            # See: https://github.com/karpathy/nanoGPT/blob/3adf61e/train.py#L117-L118
            for name in self._leaf_names:
                info = self._leaf_info[name]
                full_shape = tuple([self._num_samples] + info["shape_per_sample"])
                mm = np.memmap(
                    self.data_dir / info["file"],
                    dtype=np.dtype(info["dtype"]),
                    mode="r",
                    shape=full_shape,
                )
                leaf_values[name] = np.array(mm[idx])[np.newaxis, ...]
                del mm
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

