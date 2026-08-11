"""TreeDataSource: pytree-native reader for memory-mapped datasets."""

import json
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, SupportsIndex, Union

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

# AudioTree fields ``_reconstruct`` passes to the constructor itself (see
# ``_reconstruct``). A structure child of the same name is a valid dataclass
# field but reaches ``AudioTree(...)`` twice, so it must be rejected here rather
# than dying at the first ``__getitem__`` with "multiple values for keyword
# argument".
_RECONSTRUCTED_FIELDS = frozenset(["sample_rate"])

# Top-level keys ``TreeDataSource.__init__`` reads without a fallback. Missing
# any of them is a corrupt manifest, and validating them here turns a later raw
# ``KeyError`` into a named "Invalid manifest" error. ``string_leaves`` is
# deliberately absent: it is optional and defaults to ``{}``.
_REQUIRED_KEYS = ("num_samples", "structure", "leaves")

# The only children an AudioTree ``metadata`` node may declare. ``metadata`` is
# the library-managed provenance container, so its schema is closed: anything
# else under it is either user payload that belongs in ``extras`` (a dataset
# from the pre-1.0 era, when ``metadata`` was the user dict) or corruption, and
# both are rejected by name rather than silently reconstructed into the
# internal container. A node holding only these keys -- whichever era wrote it
# -- reads back with today's semantics.
_METADATA_CHILDREN = frozenset(["filepath", "source", "offset"])


def _validate_manifest(manifest: Dict, manifest_path: Path) -> None:
    """Check a manifest's declared shapes, dtypes and names before using them.

    A manifest travels with the dataset it describes, so its values reach
    ``np.memmap`` and the ``AudioTree`` constructor from a file the reader did
    not write. ``np.memmap(mode="r")`` bound-checks against the file size, so an
    over-large shape raises rather than reading out of bounds, but it surfaces
    as a bare ``ValueError`` naming neither the manifest nor the leaf, and only
    at the first read -- which under the default lazy mode may be inside a grain
    worker. So every leaf file is stat'd here instead, and a file too short for
    what the manifest claims (an over-large shape, an over-large
    ``num_samples``, a ``.bin`` truncated after the fact) is refused at
    construction, by name.

    Beyond sizes, the manifest's *shape* is pinned here too: the leaf tables
    must be dicts of well-formed entries, every structure node must be a known
    kind carrying its ``children``, and every leaf name the structure references
    must be declared in ``leaves``/``string_leaves``. A dangling reference would
    otherwise resolve to the excluded-leaf sentinel in ``_reconstruct`` and the
    field would just silently vanish from every sample.

    The bound is ``>=``, not ``==``. ``flush()`` is public and the writer
    publishes a manifest as soon as the schema is known, so from preallocation
    until ``close()`` every ``.bin`` is legitimately *longer* than
    ``num_samples`` implies; that prefix is exactly what the manifest promises
    is readable, and an equality check would reject a valid in-progress dataset.
    The price is the opposite corruption: a ``shape_per_sample`` too *small*
    reinterprets the file as more, shorter samples and so leaves it over-long,
    which is indistinguishable from a mid-write file by size alone. Catching
    that one needs a finality marker from the writer, not a bigger check here.
    """
    data_dir = manifest_path.parent

    def fail(message: str) -> NoReturn:
        raise ValueError(f"Invalid manifest {manifest_path}: {message}")

    for key in _REQUIRED_KEYS:
        if key not in manifest:
            fail(f"missing required top-level key {key!r}")

    num_samples = manifest.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        fail(f"num_samples must be an int, got {num_samples!r}")
    if num_samples < 0:
        fail(f"num_samples must be non-negative, got {num_samples}")

    # Container types first. `"leaves": []` is falsy, so a `... or {}` guard
    # would wave it through here and let the reader die on `[].items()` at the
    # first __getitem__ -- a raw AttributeError naming nothing.
    leaves = manifest["leaves"]
    if not isinstance(leaves, dict):
        fail(
            f"'leaves' must be a dict mapping leaf names to entries, got "
            f"{type(leaves).__name__}"
        )
    string_leaves = manifest.get("string_leaves", {})
    if not isinstance(string_leaves, dict):
        fail(
            f"'string_leaves' must be a dict mapping leaf names to entries, got "
            f"{type(string_leaves).__name__}"
        )

    for name, info in leaves.items():
        if not isinstance(info, dict):
            fail(f"leaf {name!r} entry must be a dict, got {type(info).__name__}")
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

        # Measure the file against the declaration. See the docstring for why
        # this is `>=` and what that deliberately does not catch.
        itemsize = np.dtype(info["dtype"]).itemsize
        # math.prod is arbitrary precision. np.prod(dtype=np.int64) silently
        # wraps, so a shape like [2**62, 4] multiplied out to 0, made `required`
        # 0, and let the size check pass -- the file then blew up at first read.
        required = num_samples * math.prod(shape) * itemsize
        leaf_path = safe_join(data_dir, info["file"], description="leaf file")
        try:
            actual = leaf_path.stat().st_size
        except OSError as e:
            fail(f"leaf {name!r} names {info['file']!r}, which cannot be read: {e}")
        if actual < required:
            fail(
                f"leaf {name!r} declares {num_samples} samples of shape "
                f"{tuple(shape)} and dtype {info['dtype']} ({required} bytes), but "
                f"{info['file']!r} is only {actual} bytes. The dataset is "
                f"truncated or the manifest does not describe it."
            )

    # String leaves live in bagz files, which cannot be opened everywhere (bagz
    # ships manylinux x86-64 wheels only). The type, safe_join and existence
    # checks need no bagz, so they run unconditionally; the record-count check
    # needs the file opened, so it runs only where bagz is importable.
    try:
        import bagz
    except ImportError:
        bagz = None
    for name, info in string_leaves.items():
        if not isinstance(info, dict):
            fail(
                f"string leaf {name!r} entry must be a dict, got {type(info).__name__}"
            )
        if not isinstance(info.get("file"), str):
            fail(f"string leaf {name!r} has a non-string 'file' entry")
        leaf_path = safe_join(data_dir, info["file"], description="string leaf")
        try:
            leaf_path.stat()
        except OSError as e:
            fail(
                f"string leaf {name!r} names {info['file']!r}, which cannot be "
                f"read: {e}"
            )
        if bagz is not None:
            records = len(bagz.Reader(str(leaf_path)))
            if records < num_samples:
                fail(
                    f"string leaf {name!r} declares {num_samples} samples but "
                    f"{info['file']!r} holds only {records} records. The dataset "
                    f"is truncated or the manifest does not describe it."
                )

    # Structure nodes are checked for shape *and* for reference integrity. A
    # leaf name the structure mentions but the leaf tables do not declare would
    # otherwise resolve through ``leaf_values.get(node, _EXCLUDED)`` in
    # ``_reconstruct`` -- indistinguishable from a deliberately excluded leaf, so
    # a corrupt manifest yields silently missing fields (and a dangling *root*
    # node leaks the ``_EXCLUDED`` sentinel object to the caller).
    known_leaf_names = set(leaves) | set(string_leaves)

    def check_leaf_reference(name, path: str, kind: str):
        if not isinstance(name, str):
            fail(f"{kind} at {path} must be a string leaf name, got {name!r}")
        if name not in known_leaf_names:
            fail(
                f"{kind} at {path} references {name!r}, which is not declared "
                f"in 'leaves' or 'string_leaves'"
            )

    def check_node(node, path: str):
        where = path or "<root>"
        if isinstance(node, str):
            check_leaf_reference(node, where, "structure leaf reference")
            return
        if not isinstance(node, dict):
            fail(
                f"structure node at {where} must be a leaf-name string or a "
                f"dict, got {node!r}"
            )
        node_type = node.get("type")
        if node_type == "string_leaf":
            check_leaf_reference(node.get("leaf"), where, "string_leaf node")
            return
        if node_type not in ("AudioTree", "dict"):
            fail(f"structure node at {where} has unknown type {node_type!r}")
        children = node.get("children")
        if not isinstance(children, dict):
            fail(
                f"{node_type} node at {where} must carry a 'children' dict, "
                f"got {children!r}"
            )
        if node_type == "AudioTree":
            sample_rate = node.get("sample_rate")
            if (
                not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or sample_rate <= 0
            ):
                fail(f"AudioTree at {where} has sample_rate {sample_rate!r}")
            for key in children:
                # The on-disk child ``metadata`` maps to the private
                # ``_metadata`` field (see ``tree_writer``); the underscore
                # spelling itself is not a valid on-disk name.
                field_name = "_metadata" if key == "metadata" else key
                if key.startswith("_") or field_name not in (
                    AudioTree.__dataclass_fields__
                ):
                    fail(
                        f"AudioTree at {where} declares child {key!r}, "
                        f"which is not an AudioTree field"
                    )
                if key in _RECONSTRUCTED_FIELDS:
                    fail(
                        f"AudioTree at {where} declares child {key!r}, "
                        f"which the reader sets itself -- it would reach the "
                        f"AudioTree constructor as a duplicate argument"
                    )
                if key == "metadata":
                    node_children = children[key]
                    if (
                        not isinstance(node_children, dict)
                        or node_children.get("type") != "dict"
                    ):
                        fail(
                            f"AudioTree metadata node at {where} must be a "
                            f"dict node, got {node_children!r}"
                        )
                    grandchildren = node_children.get("children", {})
                    if not isinstance(grandchildren, dict):
                        grandchildren = {}
                    unknown = sorted(set(grandchildren) - _METADATA_CHILDREN)
                    if unknown:
                        fail(
                            f"AudioTree metadata node at {where} declares "
                            f"child(ren) {', '.join(repr(n) for n in unknown)}. "
                            f"'metadata' is the library-managed provenance "
                            f"container and may hold only "
                            f"{sorted(_METADATA_CHILDREN)}; user payload "
                            f"belongs in 'extras'. The dataset was written by "
                            f"an incompatible version of audiotree, or is "
                            f"corrupt -- re-render it."
                        )
        for key, child in children.items():
            check_node(child, f"{path}.{key}" if path else str(key))

    check_node(manifest["structure"], "")


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
                # The on-disk child ``metadata`` fills the private
                # ``_metadata`` field (see ``tree_writer``).
                children["_metadata" if k == "metadata" else k] = val
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

    Each process holds one memmap per leaf, opened on first access and dropped
    on the way into a pickle, so the source stays safe to hand to grain's
    multiprocessing DataLoader however that DataLoader starts its workers. Pass
    ``load_into_memory=True`` to instead load every leaf into RAM up front; a
    source in that mode opens no files at all after construction.

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
            I/O. With fork-based multiprocessing (default on Linux) the
            parent's data is shared with workers via copy-on-write; with
            spawn it is pickled to them, so the RAM cost is per worker.
            Samples are copied out of the store on the way out, so a caller
            that writes into one cannot disturb the next reader.
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

        # A dataset whose root is a bare leaf (TreeWriter accepts a plain
        # array as the pytree; its leaf name is "") has nothing left when that
        # leaf is excluded -- and ``_reconstruct`` would hand the caller the
        # private ``_EXCLUDED`` sentinel object instead of a sample.
        root = self._structure
        root_leaf = (
            root
            if isinstance(root, str)
            else root.get("leaf")
            if root.get("type") == "string_leaf"
            else None
        )
        if root_leaf is not None and _is_excluded(root_leaf, self.exclude_prefixes):
            raise ValueError(
                f"exclude_prefixes={self.exclude_prefixes!r} excludes the "
                f"dataset's root leaf {root_leaf!r}, leaving nothing to "
                f"return. Drop the exclusion (or this dataset)."
            )

        # In-memory storage (populated eagerly if load_into_memory=True).
        self._in_memory_arrays: Dict[str, np.ndarray] = {}
        self._in_memory_strings: Dict[str, List[str]] = {}

        # Per-process file handles, opened lazily by _ensure_open and dropped
        # on the way into a pickle. Unused when load_into_memory=True.
        self._leaf_names: List[str] = []
        self._bagz_readers: Dict = {}
        self._leaf_memmaps: Dict[str, np.memmap] = {}
        self._data_files_opened: bool = False
        # Grain's prefetch pool calls __getitem__ from many threads, so the
        # lazy open has to be serialized or a reader can observe a half-built
        # `_leaf_names`. Dropped and rebuilt around pickling -- see
        # __getstate__/__setstate__.
        self._open_lock = threading.Lock()

        if load_into_memory:
            self._load_all_into_memory()

    def _load_all_into_memory(self):
        """Load all non-excluded leaves into RAM.

        Everything ``__getitem__`` needs then lives in ``_in_memory_arrays`` and
        ``_in_memory_strings``, both of which survive pickling, so a source in
        this mode never opens a file again -- not in this process and not in a
        worker.
        """
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

    def _ensure_open(self):
        """Open this process's memmaps and readers once, safely under threads.

        Returns the ``(memmaps, readers)`` pair, fetched under the lock:
        handing the caller a snapshot is what makes a concurrent
        :meth:`close` safe. Without it, a reader that passed an unlocked
        opened-check and then iterated ``self._leaf_memmaps`` after close()
        swapped the dicts would find them empty and return a structurally
        valid but *empty* sample -- silent corruption, not an error. The
        snapshot keeps the handles alive for the duration of that one read;
        the mappings are released when the last reference drops.
        """
        with self._open_lock:
            if not self._data_files_opened:
                self._open_data_files()
            return self._leaf_memmaps, self._bagz_readers

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

        In-memory data (``_in_memory_arrays``, ``_in_memory_strings``) is kept,
        so a ``load_into_memory=True`` source arrives in the worker already able
        to answer -- by copy-on-write under fork, by pickle under spawn.
        """
        state = self.__dict__.copy()
        state["_leaf_names"] = []
        state["_bagz_readers"] = {}
        # A memmap belongs to the process that made it, and a Lock cannot be
        # pickled at all; both are rebuilt by _ensure_open in the worker.
        state["_leaf_memmaps"] = {}
        del state["_open_lock"]
        # The flag has to travel with the handles it describes: leaving it True
        # in a state whose handles were just dropped told the worker's
        # _ensure_open there was nothing to do, and it read from empty dicts.
        state["_data_files_opened"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_lock = threading.Lock()

    def close(self) -> None:
        """Release this process's memmaps and readers.

        Holding the mappings open is what makes reads fast, but a mapped file
        cannot be deleted, moved, or replaced on Windows until it is unmapped --
        so a run that reads a dataset and then tries to clean it up fails with
        ``PermissionError: [WinError 32]`` while the source is alive. POSIX
        allows the unlink and hides the problem entirely.

        Reading again reopens transparently, so this is a release rather than a
        teardown; ``TreeDataSource`` is also a context manager, which is the
        tidier way to scope the handles::

            with TreeDataSource(directory) as source:
                tree = source[0]
        """
        with self._open_lock:
            self._leaf_memmaps = {}
            self._bagz_readers = {}
            self._leaf_names = []
            self._data_files_opened = False

    def __enter__(self) -> "TreeDataSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self):
        # A backstop, not the contract: interpreter shutdown may already have
        # torn down what close() touches, and refcount timing is not something
        # to rely on for releasing OS handles.
        try:
            self.close()
        except Exception:
            pass

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

        leaf_values: Dict[str, Any] = {}

        # Branch on the *mode*, not on whether the RAM store happens to hold
        # anything: a source that excludes every array leaf has an empty
        # `_in_memory_arrays` and no open files, and testing the store sent it
        # down the file path to read from dicts that mode never fills -- losing
        # its string leaves and returning an empty sample.
        if self.load_into_memory:
            # `np.array` copies, as on the file path below: the store is shared
            # by every sample this source ever returns, so handing out views
            # would let one caller's in-place write rewrite the dataset for all
            # the readers after it.
            for name, arr in self._in_memory_arrays.items():
                leaf_values[name] = np.array(arr[idx])[np.newaxis, ...]
            for name, strings in self._in_memory_strings.items():
                leaf_values[name] = strings[idx]
        else:
            # Iterate the snapshot _ensure_open returned, not the attributes:
            # a concurrent close() swaps the attribute dicts for empty ones,
            # and reading those would silently yield an empty sample.
            memmaps, readers = self._ensure_open()
            # `np.array` copies out of the mapping, so the returned tree never
            # aliases it and the held memmap stays an implementation detail.
            for name, mm in memmaps.items():
                leaf_values[name] = np.array(mm[idx])[np.newaxis, ...]
            for name, reader in readers.items():
                leaf_values[name] = reader[idx].decode("utf-8")

        return _reconstruct(self._structure, leaf_values)

    def get_metadata(self) -> Dict:
        """Get user metadata from the manifest.

        Returns:
            Dictionary containing user-provided metadata
        """
        return dict(self.manifest.get("metadata", {}))
