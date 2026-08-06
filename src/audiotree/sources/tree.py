"""TreeDataSource: pytree-native reader for memory-mapped datasets."""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, SupportsIndex, Union

import numpy as np
from grain.sources import RandomAccessDataSource

from audiotree import _format
from audiotree._bagz import require_bagz
from audiotree._fs import safe_join
from audiotree.core import AudioTree

# Sentinel for excluded leaves (distinct from None, which is a valid value).
_EXCLUDED = object()

# dtypes a manifest may name. All fixed-width numeric types; object/void and
# structured dtypes are excluded because memmapping them is either unsupported
# or an unpickling vector.
_ALLOWED_DTYPES = frozenset(
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]
)


def _validate_manifest(manifest: Dict, manifest_path: Path) -> None:
    """Check a manifest's declared shapes, dtypes and names before using them.

    A manifest travels with the dataset it describes, so its values reach
    ``np.memmap`` and the ``AudioTree`` constructor from a file the reader did
    not write. ``np.memmap(mode="r")`` does bound-check against the file size,
    so an over-large shape raises rather than reading out of bounds — but it
    surfaces as a bare ``ValueError`` naming neither the manifest nor the leaf,
    and a *smaller* shape silently truncates the dataset with no error at all.
    """

    def fail(message: str):
        raise ValueError(f"Invalid manifest {manifest_path}: {message}")

    num_samples = manifest.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        fail(f"num_samples must be an int, got {num_samples!r}")
    if num_samples < 0:
        fail(f"num_samples must be non-negative, got {num_samples}")

    for name, info in (manifest.get("leaves") or {}).items():
        shape = info.get("shape_per_sample")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
        ):
            fail(f"leaf {name!r} has an invalid shape_per_sample {shape!r}")
        if info.get("dtype") not in _ALLOWED_DTYPES:
            fail(
                f"leaf {name!r} declares dtype {info.get('dtype')!r}, which is not "
                f"one of {sorted(_ALLOWED_DTYPES)}"
            )
        if not isinstance(info.get("file"), str):
            fail(f"leaf {name!r} has a non-string 'file' entry")

    def check_node(node, path: str):
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        if node_type == "AudioTree":
            sample_rate = node.get("sample_rate")
            if (
                not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or sample_rate <= 0
            ):
                fail(f"AudioTree at {path or '<root>'} has sample_rate {sample_rate!r}")
            for key, child in (node.get("children") or {}).items():
                if key not in AudioTree.__dataclass_fields__:
                    fail(
                        f"AudioTree at {path or '<root>'} declares child {key!r}, "
                        f"which is not an AudioTree field"
                    )
                check_node(child, f"{path}.{key}" if path else key)
        elif node_type == "dict":
            for key, child in (node.get("children") or {}).items():
                check_node(child, f"{path}.{key}" if path else str(key))

    check_node(manifest.get("structure"), "")


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
        children.setdefault("waveform", None)
        return AudioTree(sample_rate=node["sample_rate"], **children)

    raise ValueError(f"Unknown structure node type: {node_type}")


def _is_excluded(name: str, exclude_prefixes: List[str]) -> bool:
    """Check if a leaf name matches any exclude prefix.

    Matching semantics: ``"dry"`` matches ``"dry"`` exactly or any name
    starting with ``"dry."`` (e.g. ``"dry.waveform"``), but does NOT
    match ``"dryness"``.
    """
    return any(name == p or name.startswith(p + ".") for p in exclude_prefixes)


class TreeDataSource(RandomAccessDataSource):
    """Read pytrees from memory-mapped files created by TreeWriter.

    Provides efficient random access to pre-rendered datasets without loading
    into RAM. Data is read from memory-mapped binary files and reconstructed
    into the original pytree structure (AudioTree, dict, etc.).

    Array memmaps are reopened on each access so the OS can reclaim pages
    between reads (keeping the page cache bounded during random access), while
    the source stays pickle-safe for grain's multiprocessing DataLoader. Pass
    ``load_into_memory=True`` to instead load every leaf into RAM up front.

    Args:
        directory: Path to the directory containing manifest.json and
            memory-mapped data files.
        exclude_prefixes: List of dot-separated leaf name prefixes to skip
            loading. For example, ``["wet.waveform"]`` skips the
            ``wet.waveform`` memmap, and ``["dry"]`` skips all leaves
            under the ``dry`` subtree. Excluded leaves are omitted from
            the reconstructed pytree (AudioTree fields default to None).
            Default: load all leaves.
        load_into_memory: If True, load all non-excluded array leaves and
            string leaves into RAM at init time. Workers then read from
            pre-loaded numpy arrays instead of memmaps, eliminating disk
            I/O. With fork-based multiprocessing (default on Linux), the
            parent's data is shared with workers via copy-on-write.
            Default False.

    Example:
        First, pre-render a small dataset with
        :class:`~audiotree.tree_writer.TreeWriter`. Here each sample is a dict
        with ``dry`` and ``wet`` :class:`~audiotree.AudioTree` branches:

        >>> import tempfile
        >>> import jax.numpy as jnp
        >>> from audiotree import AudioTree
        >>> from audiotree.tree_writer import TreeWriter
        >>> dataset_dir = tempfile.mkdtemp()
        >>> batch = {
        ...     "dry": AudioTree.create(jnp.zeros((4, 1, 16000)), 16000),
        ...     "wet": AudioTree.create(jnp.ones((4, 1, 16000)), 16000),
        ... }
        >>> with TreeWriter(dataset_dir, expected_samples=4) as w:
        ...     _ = w.write(batch)

        Read a sample back; the pytree structure is reconstructed:

        >>> ds = TreeDataSource(dataset_dir)
        >>> sample = ds[0]
        >>> sorted(sample.keys())
        ['dry', 'wet']
        >>> type(sample["dry"]).__name__
        'AudioTree'

        Skip loading some leaves with ``exclude_prefixes`` (they come back as
        ``None``):

        >>> ds = TreeDataSource(dataset_dir, exclude_prefixes=["wet.waveform"])
        >>> ds[0]["wet"].waveform is None
        True

        Load everything into RAM up front for I/O-free random access:

        >>> ds = TreeDataSource(dataset_dir, load_into_memory=True)
        >>> ds[0]["dry"].waveform.shape  # reads from RAM, no disk I/O
        (1, 1, 16000)
    """

    def __init__(
        self,
        directory: Union[str, Path],
        *,
        exclude_prefixes: Optional[List[str]] = None,
        load_into_memory: bool = False,
    ):
        self.data_dir = Path(directory)
        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        self.exclude_prefixes: List[str] = exclude_prefixes or []
        self.load_into_memory = load_into_memory

        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        _format.check(self.manifest, _format.TREE, source=str(self.manifest_path))

        _validate_manifest(self.manifest, self.manifest_path)

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
        self._leaf_names: List[str] = []
        self._bagz_readers: Dict = {}
        self._leaf_memmaps: Dict[str, np.memmap] = {}
        self._data_files_opened: bool = False
        # Grain's prefetch pool calls __getitem__ from many threads, so the
        # lazy open has to be serialized or a reader can observe a half-built
        # `_leaf_names`. Dropped and rebuilt around pickling -- see
        # __getstate__/__setstate__.
        self._open_lock = threading.Lock()

    def _load_all_into_memory(self):
        """Load all non-excluded leaves into RAM."""
        for name, info in self._leaf_info.items():
            if _is_excluded(name, self.exclude_prefixes):
                continue
            full_shape = tuple([self._num_samples] + info["shape_per_sample"])
            mm = np.memmap(
                safe_join(self.data_dir, info["file"], description="leaf file"),
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=full_shape,
            )
            self._in_memory_arrays[name] = np.array(mm)
            del mm

        for name, info in self._string_leaf_info.items():
            # Resolve bagz per *included* leaf. Requiring it up front meant that
            # excluding every string leaf still raised ImportError, so a dataset
            # written on Linux could not be opened at all elsewhere — not even
            # to read its waveforms.
            if _is_excluded(name, self.exclude_prefixes):
                continue
            bagz = require_bagz(f"reading string leaf {name!r} in TreeDataSource")
            reader = bagz.Reader(
                str(safe_join(self.data_dir, info["file"], description="string leaf"))
            )
            self._in_memory_strings[name] = [
                reader[i].decode("utf-8") for i in range(self._num_samples)
            ]

        # Mark as opened so lazy path is skipped.
        self._data_files_opened = True

    def _ensure_open(self):
        """Open this process's memmaps and readers once, safely under threads.

        Double-checked: the common case is one attribute read, and the slow
        path is serialized. ``_data_files_opened`` is assigned last, after every
        other attribute is fully built, so a thread that sees it ``True``
        without taking the lock cannot observe a half-built one.
        """
        if self._data_files_opened:
            return
        with self._open_lock:
            if not self._data_files_opened:
                self._open_data_files()

    def _open_data_files(self):
        """Open one memmap per array leaf and a reader per string leaf.

        The memmaps are held for the life of the process rather than rebuilt on
        every access. Rebuilding them was worth roughly 15x on a random-order
        read (4.4k -> 68k items/s measured), and it bought only the appearance
        of lower memory: the pages a mapping keeps resident are clean and
        file-backed, so the kernel evicts them under pressure. RSS does track
        the working set now, which is the trade -- ``load_into_memory=True``
        remains the option that makes that cost explicit and up front.

        Callers should use :meth:`_ensure_open`, which handles the locking.
        """
        leaf_names: List[str] = []
        memmaps: Dict[str, np.memmap] = {}
        for name, info in self._leaf_info.items():
            if _is_excluded(name, self.exclude_prefixes):
                continue
            leaf_names.append(name)
            memmaps[name] = np.memmap(
                safe_join(self.data_dir, info["file"], description="leaf file"),
                dtype=np.dtype(info["dtype"]),
                mode="r",
                shape=tuple([self._num_samples] + info["shape_per_sample"]),
            )

        readers: Dict = {}
        for name, info in self._string_leaf_info.items():
            # See _load_all_into_memory: bagz is resolved per included leaf.
            if _is_excluded(name, self.exclude_prefixes):
                continue
            bagz = require_bagz(f"reading string leaf {name!r} in TreeDataSource")
            readers[name] = bagz.Reader(
                str(safe_join(self.data_dir, info["file"], description="string leaf"))
            )

        self._leaf_names = leaf_names
        self._leaf_memmaps = memmaps
        self._bagz_readers = readers
        self._data_files_opened = True  # last: see _ensure_open

    def __getstate__(self):
        """Drop memmaps/readers before pickling (they reopen lazily in workers).

        In-memory data (``_in_memory_arrays``, ``_in_memory_strings``) is
        kept so that fork-based workers inherit the parent's pre-loaded
        data via copy-on-write.
        """
        state = self.__dict__.copy()
        state["_leaf_names"] = []
        state["_bagz_readers"] = {}
        # A memmap belongs to the process that made it, and a Lock cannot be
        # pickled at all; both are rebuilt by _ensure_open in the worker.
        state["_leaf_memmaps"] = {}
        del state["_open_lock"]
        # If data is in memory, workers don't need to reopen files.
        if not self.load_into_memory:
            state["_data_files_opened"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_lock = threading.Lock()

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, record_key: SupportsIndex):
        """Load a single sample by index.

        Args:
            record_key: Index of the record to load

        Returns:
            Reconstructed pytree with a batch dimension added to each leaf.
            Excluded leaves are omitted from the result.

        Raises:
            IndexError: If index is out of range
        """
        idx = int(record_key)
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._num_samples})")

        self._ensure_open()

        leaf_values: Dict[str, Any] = {}

        if self._in_memory_arrays:
            # Fast path: read from pre-loaded RAM arrays.
            for name, arr in self._in_memory_arrays.items():
                leaf_values[name] = arr[idx][np.newaxis, ...]
            for name, strings in self._in_memory_strings.items():
                leaf_values[name] = strings[idx]
        else:
            # `np.array` copies out of the mapping, so the returned tree never
            # aliases it and the held memmap stays an implementation detail.
            for name, mm in self._leaf_memmaps.items():
                leaf_values[name] = np.array(mm[idx])[np.newaxis, ...]
            for name, reader in self._bagz_readers.items():
                leaf_values[name] = reader[idx].decode("utf-8")

        return _reconstruct(self._structure, leaf_values)

    def get_metadata(self) -> Dict:
        """Get user metadata from the manifest.

        Returns:
            Dictionary containing user-provided metadata
        """
        return dict(self.manifest.get("metadata", {}))
