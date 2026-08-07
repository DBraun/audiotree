"""TreeWriter: pytree-native writer for memory-mapped datasets."""

import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import jax
import jax.tree_util
import numpy as np

from audiotree import _format
from audiotree._bagz import require_bagz
from audiotree._fs import refuse_to_clobber, safe_join, write_json_atomic
from audiotree.core import PYTREE_FIELDS, AudioTree

# dtypes a leaf may be written with. Must stay in sync with the reader's
# ``audiotree.sources.tree._ALLOWED_DTYPES``: a dataset whose manifest names a
# dtype outside this set is refused at read time, and discovering that after an
# hours-long pre-render is the worst possible moment. It is duplicated rather
# than imported because ``audiotree.sources`` pulls in grain, which the writer
# deliberately does not require.
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


OnOverflow = Literal["error", "trim", "grow"]

#: How much bigger a reallocation makes the memmaps, relative to their current
#: size. Growing to exactly what the current batch needs would re-mmap every
#: leaf on every subsequent batch; overshooting geometrically amortizes that,
#: and ``close()`` truncates the slack away.
_GROWTH_FACTOR = 1.5


def _native_dtype(dtype: np.dtype) -> np.dtype:
    """Return *dtype* in this machine's byte order.

    ``.bin`` files carry no byte-order mark, and the manifest records a plain
    dtype name (``"float32"``), so a big-endian leaf on a little-endian host
    would be read back byte-swapped. Normalizing on write keeps the one encoding
    the reader knows how to name.
    """
    if dtype.byteorder in (">", "<"):  # "=" and "|" are already native/N.A.
        return dtype.newbyteorder("=")
    return dtype


def _path_to_string(path: Tuple) -> str:
    """Convert a JAX key path to a dot-separated string.

    Examples:
        (GetAttrKey('waveform'),) -> "waveform"
        (GetAttrKey('metadata'), DictKey('mel')) -> "metadata.mel"
        (DictKey('dry'), GetAttrKey('waveform')) -> "dry.waveform"
    """
    parts = []
    for key in path:
        if isinstance(key, jax.tree_util.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jax.tree_util.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jax.tree_util.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def _serialize_structure(pytree) -> Tuple[Any, List[str], List[str]]:
    """Walk a pytree to build a JSON-serializable structure and collect leaf names.

    Returns:
        (structure, array_leaf_names, string_leaf_names) where structure is a
        JSON-compatible description, array_leaf_names lists dot-separated paths
        for array leaves, and string_leaf_names lists paths for string leaves.
    """
    array_leaf_names: List[str] = []
    string_leaf_names: List[str] = []

    def _walk(node, prefix: str):
        # Leaf: numpy or JAX array
        if isinstance(node, (np.ndarray, jax.Array)):
            array_leaf_names.append(prefix)
            return prefix  # string = leaf reference

        # String leaf (single string, batch size 1)
        if isinstance(node, str):
            string_leaf_names.append(prefix)
            return {"type": "string_leaf", "leaf": prefix}

        # List of strings (batch of strings)
        if isinstance(node, list) and node and all(isinstance(x, str) for x in node):
            string_leaf_names.append(prefix)
            return {"type": "string_leaf", "leaf": prefix}

        # AudioTree node
        if isinstance(node, AudioTree):
            children = {}
            for fname in PYTREE_FIELDS:
                value = getattr(node, fname)
                if value is None:
                    continue
                child_prefix = f"{prefix}.{fname}" if prefix else fname
                if fname == "metadata":
                    if not value:  # empty dict
                        children["metadata"] = {"type": "dict", "children": {}}
                    else:
                        children["metadata"] = _walk(value, child_prefix)
                else:
                    children[fname] = _walk(value, child_prefix)
            return {
                "type": "AudioTree",
                "sample_rate": node.sample_rate,
                "children": children,
            }

        # Dict node
        if isinstance(node, dict):
            children = {}
            for key in sorted(node.keys()):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                children[str(key)] = _walk(node[key], child_prefix)
            return {"type": "dict", "children": children}

        raise TypeError(
            f"Unsupported pytree node type: {type(node)}. "
            "Leaves must be numpy/JAX arrays, str, or List[str]."
        )

    structure = _walk(pytree, "")
    return structure, array_leaf_names, string_leaf_names


def _extract_leaves(
    pytree, array_leaf_names: List[str], string_leaf_names: List[str]
) -> Tuple[List[np.ndarray], Dict[str, List[str]]]:
    """Walk a pytree and extract array leaves and string data.

    Returns:
        (array_leaves, string_data) where array_leaves is a list of numpy arrays
        in the same order as array_leaf_names, and string_data maps string leaf
        names to lists of strings.
    """
    array_leaves: List[np.ndarray] = []
    string_data: Dict[str, List[str]] = {}

    def _walk(node, prefix: str):
        if isinstance(node, (np.ndarray, jax.Array)):
            array_leaves.append(np.asarray(node))
            return

        if isinstance(node, str):
            string_data[prefix] = [node]
            return

        if isinstance(node, list) and node and all(isinstance(x, str) for x in node):
            string_data[prefix] = list(node)
            return

        if isinstance(node, AudioTree):
            for fname in PYTREE_FIELDS:
                value = getattr(node, fname)
                if value is None:
                    continue
                child_prefix = f"{prefix}.{fname}" if prefix else fname
                if fname == "metadata":
                    if value:
                        _walk(value, child_prefix)
                else:
                    _walk(value, child_prefix)
            return

        if isinstance(node, dict):
            for key in sorted(node.keys()):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(node[key], child_prefix)
            return

        raise TypeError(f"Unsupported pytree node type: {type(node)}.")

    _walk(pytree, "")
    return array_leaves, string_data


class TreeWriter:
    """Write any pytree to memory-mapped binary files.

    Each leaf in the pytree becomes a separate .bin file. The tree structure
    is stored in manifest.json for reconstruction by TreeDataSource.

    Accepts AudioTree objects, dicts of arrays, dicts of AudioTrees, or any
    combination. The schema is inferred from the first write() call.

    Args:
        directory: Directory where memmap files will be written
        expected_samples: Number of samples to pre-allocate. This is an
            allocation hint, not a cap: what happens when more samples are
            offered is decided by ``on_overflow``. Under-shooting it is always
            fine -- :meth:`close` truncates the files to what was written.
        on_overflow: What to do with a batch that does not fit the current
            allocation.

            * ``"grow"`` (default) reallocates every leaf file to fit and
              carries on, so ``expected_samples`` really is only a hint.
            * ``"error"`` raises :class:`ValueError`, naming how many samples
              were written, allocated and offered.
            * ``"trim"`` writes as much of the batch as fits and drops the
              rest, warning with the same counts. Once the allocation is full,
              every further ``write()`` warns and returns 0.
        metadata: Optional dict of user metadata to store in manifest
        pbar: Optional tqdm progress bar instance. Updated by ``batch_size``
            after each ``write()`` call.
        close_pbar: If True, close the progress bar when the writer closes.
            Default False.
        manifest_interval: Seconds between refreshes of the manifest's
            ``num_samples`` while writing. The manifest is what tells a reader
            how much of the dataset is real, so a hard kill must not leave it
            claiming zero; refreshing on a timer bounds how stale that count can
            be without rewriting the JSON on every batch. Pass ``0`` to refresh
            only on :meth:`flush` and :meth:`close`.

    Example:
        Pre-render a few batches of one-second mono ``AudioTree`` objects into a
        memory-mapped dataset, then read one back with
        :class:`~audiotree.sources.TreeDataSource`:

        >>> import tempfile
        >>> import jax.numpy as jnp
        >>> from audiotree import AudioTree
        >>> from audiotree.sources import TreeDataSource
        >>> out_dir = tempfile.mkdtemp()
        >>> batches = [AudioTree.create(jnp.zeros((8, 1, 16000)), 16000) for _ in range(3)]
        >>> with TreeWriter(out_dir, expected_samples=8 * len(batches)) as w:
        ...     for batch in batches:
        ...         _ = w.write(batch)
        >>> ds = TreeDataSource(out_dir)
        >>> len(ds)
        24
        >>> ds[0].waveform.shape  # one sample, batch dim added back
        (1, 1, 16000)
    """

    def __init__(
        self,
        directory: Union[str, Path],
        expected_samples: int,
        *,
        on_overflow: OnOverflow = "grow",
        metadata: Optional[Dict[str, Any]] = None,
        pbar=None,
        close_pbar: bool = False,
        exist_ok: bool = False,
        manifest_interval: float = 5.0,
    ):
        if on_overflow not in ("error", "trim", "grow"):
            raise ValueError(
                f"on_overflow must be 'error', 'trim' or 'grow', got {on_overflow!r}."
            )
        if expected_samples < 0:
            raise ValueError(
                f"expected_samples must be non-negative, got {expected_samples}."
            )
        self.directory = Path(directory)
        self.expected_samples = expected_samples
        self.on_overflow = on_overflow
        self.metadata = metadata or {}
        self.exist_ok = exist_ok
        self.manifest_interval = manifest_interval
        self._pbar = pbar
        self._close_pbar = close_pbar

        self._manifest_written_at = 0.0
        self._current_index = 0
        self._allocated_samples = expected_samples
        self._broken: Optional[str] = None
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []
        self._leaf_info: Dict[str, Dict] = {}
        self._string_leaf_names: List[str] = []
        self._string_leaf_info: Dict[str, Dict] = {}
        self._bagz_writers: Dict = {}
        self._structure = None
        self._is_open = False
        self._is_closed = False

    def open(self) -> "TreeWriter":
        """Open the writer and create output directory.

        Returns:
            self for method chaining

        Raises:
            RuntimeError: If the writer is already open, or has been closed.
        """
        if self._is_open:
            raise RuntimeError("Writer is already open")
        if self._is_closed:
            raise RuntimeError(
                "Writer has been closed and cannot be reopened: its memmaps are "
                "gone and its files truncated, so further writes would be "
                "dropped. Create a new TreeWriter."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.exist_ok:
            refuse_to_clobber(self.directory, ("manifest.json", "*.bin", "*.bagz"))
        self._is_open = True
        return self

    def _check_structure_matches(self, structure, leaf_names, string_leaf_names):
        """Raise if a later write's structure differs from the first write's.

        Leaf extraction is positional, so a renamed, added, reordered or dropped
        leaf would otherwise write into whichever ``.bin`` happens to sit at that
        index. ``structure`` also carries each AudioTree's ``sample_rate``, so a
        rate change is caught here too rather than being silently mislabelled by
        the value captured on the first write.
        """
        if (structure, leaf_names, string_leaf_names) == (
            self._structure,
            self._leaf_names,
            self._string_leaf_names,
        ):
            return

        expected = set(self._leaf_names) | set(self._string_leaf_names)
        got = set(leaf_names) | set(string_leaf_names)
        details = []
        if got - expected:
            details.append(f"unexpected leaves {sorted(got - expected)}")
        if expected - got:
            details.append(f"missing leaves {sorted(expected - got)}")
        if not details and (
            leaf_names != self._leaf_names
            or string_leaf_names != self._string_leaf_names
        ):
            details.append("leaf ordering changed")
        if not details:
            details.append("structure differs (e.g. a changed sample_rate or nesting)")
        raise ValueError(
            "Pytree structure must match the first write: "
            + "; ".join(details)
            + f". Expected leaves {sorted(expected)}."
        )

    def _leaf_path(self, name: str, extension: str, claimed: Dict[str, str]) -> Path:
        """Resolve one leaf's output file, refusing collisions and escapes.

        Filenames are dot-joined leaf paths, which is not injective: the leaves
        ``{"a.b": x}`` and ``{"a": {"b": y}}`` both name ``a.b.bin``, so one
        silently overwrote the other. And a leaf named ``"../escaped"`` wrote
        outside the dataset directory entirely, so every filename goes through
        :func:`~audiotree._fs.safe_join` -- the same check the reader applies to
        the manifest it reads back.
        """
        filename = f"{name}{extension}"
        if filename in claimed:
            raise ValueError(
                f"Leaves {claimed[filename]!r} and {name!r} both map to the file "
                f"{filename!r}. Leaf filenames are dot-joined pytree paths, so a "
                f"key containing '.' can collide with a nested one; rename one of "
                f"them."
            )
        path = safe_join(self.directory, filename, description="leaf file")
        if path.parent != self.directory.resolve():
            raise ValueError(
                f"Leaf {name!r} maps to the file {filename!r}, which is not a "
                f"plain name inside {self.directory}. Leaf keys must not contain "
                f"path separators."
            )
        claimed[filename] = name
        return path

    def _init_from_pytree(self, pytree, structure, leaf_names, string_leaf_names):
        """Infer schema from the first pytree and create memmap/bagz files."""
        array_leaves, _ = _extract_leaves(pytree, leaf_names, string_leaf_names)

        # Resolve and validate every output file *before* creating any of them
        # or recording any schema, so a pytree the reader could not read back
        # fails on the first write -- leaving neither half-written files nor a
        # half-initialized writer -- rather than after hours of rendering.
        claimed: Dict[str, str] = {}
        array_paths = []
        for name, leaf in zip(leaf_names, array_leaves):
            dtype = _native_dtype(leaf.dtype)
            if str(dtype) not in _ALLOWED_DTYPES:
                raise ValueError(
                    f"Leaf {name!r} has dtype {leaf.dtype!s}, which TreeDataSource "
                    f"cannot read back. Supported dtypes: "
                    f"{sorted(_ALLOWED_DTYPES)}. Cast the leaf before writing."
                )
            array_paths.append((self._leaf_path(name, ".bin", claimed), dtype))
        string_paths = [
            self._leaf_path(name, ".bagz", claimed) for name in string_leaf_names
        ]

        self._structure = structure
        self._leaf_names = leaf_names
        self._string_leaf_names = string_leaf_names

        # Create memmaps for array leaves
        for (path, dtype), name, leaf in zip(
            array_paths, self._leaf_names, array_leaves
        ):
            shape_per_sample = leaf.shape[1:]  # exclude batch dim

            self._leaf_info[name] = {
                "dtype": str(dtype),
                "shape_per_sample": list(shape_per_sample),
                "file": path.name,
            }

            full_shape = (self._allocated_samples,) + shape_per_sample
            # Size the file first, then map it read-write. `mode="w+"` would do
            # both, but it cannot create a zero-byte mapping -- and
            # `expected_samples=0` is legal when the writer may grow. numpy 2.2+
            # pads such a file to one byte itself; the declared floor (2.1.3)
            # hands `mmap` a length of 0 and gets "cannot mmap an empty file",
            # so only the lowest-direct CI leg ever saw it. Same `max(..., 1)`
            # the grow path already applies.
            nbytes = int(np.prod(full_shape, dtype=np.int64)) * dtype.itemsize
            with open(path, "wb") as fh:
                fh.truncate(max(nbytes, 1))
            mm = np.memmap(path, dtype=dtype, mode="r+", shape=full_shape)
            self._memmaps.append(mm)

        # Create bagz writers for string leaves
        if self._string_leaf_names:
            bagz = require_bagz("storing string leaves in TreeWriter")
            for name, path in zip(self._string_leaf_names, string_paths):
                self._string_leaf_info[name] = {"file": path.name}
                self._bagz_writers[name] = bagz.Writer(str(path))

        # Publish the manifest as soon as the schema is known, so a pre-render
        # killed partway through leaves a readable prefix rather than a
        # directory of orphaned .bin files. `flush()` keeps it current.
        self._write_manifest()

    def _leaf_file(self, name: str) -> Path:
        """Resolve an already-recorded leaf's file, re-checking containment."""
        return safe_join(
            self.directory, self._leaf_info[name]["file"], description="leaf file"
        )

    def _bytes_per_sample(self, name: str) -> int:
        """Size on disk of one sample of leaf *name*."""
        info = self._leaf_info[name]
        shape_per_sample = info["shape_per_sample"]
        elems = int(np.prod(shape_per_sample)) if shape_per_sample else 1
        return elems * np.dtype(info["dtype"]).itemsize

    def _grow(self, required_samples: int):
        """Reallocate every array leaf so *required_samples* samples fit.

        A memmap is a fixed-size view of a fixed-size file, so growing means
        extending each ``.bin`` and remapping it. Extending is done with
        :func:`os.truncate`, which appends zeros without touching -- or even
        reading -- the bytes already there, so the prefix a reader can see stays
        exactly what was written: a crash part-way through leaves some leaves
        extended and some not, all of them still valid for the
        ``num_samples`` the manifest claims.

        Each leaf's old mapping is released before its file is resized, because
        Windows refuses to resize a file that is currently mapped.

        Raises:
            OSError: If a file cannot be extended (a full disk, typically). The
                writer is then poisoned: everything written so far is still
                readable and :meth:`close` still finalizes it, but further
                ``write()`` calls raise rather than silently skipping the leaves
                whose mappings were lost.
        """
        new_allocated = max(
            required_samples, int(self._allocated_samples * _GROWTH_FACTOR)
        )
        try:
            for i, name in enumerate(self._leaf_names):
                mm = self._memmaps[i]
                mm.flush()
                # Drop every reference so the mapping is closed before the
                # resize; `del mm` alone would leave the list holding it.
                del self._memmaps[i]
                del mm

                path = self._leaf_file(name)
                shape_per_sample = tuple(self._leaf_info[name]["shape_per_sample"])
                # max(..., 1): a leaf with a zero-sized dimension needs no bytes,
                # and an empty file cannot be mapped.
                os.truncate(path, max(new_allocated * self._bytes_per_sample(name), 1))
                self._memmaps.insert(
                    i,
                    np.memmap(
                        path,
                        dtype=np.dtype(self._leaf_info[name]["dtype"]),
                        mode="r+",
                        shape=(new_allocated,) + shape_per_sample,
                    ),
                )
        except BaseException:
            self._broken = (
                f"TreeWriter failed to grow its memmap files to hold "
                f"{new_allocated} samples, so some leaves are no longer mapped. "
                f"The {self._current_index} samples already written remain "
                f"readable and close() will finalize them, but this writer "
                f"cannot accept further writes."
            )
            raise
        self._allocated_samples = new_allocated

    def _handle_overflow(self, batch_size: int) -> int:
        """Apply the ``on_overflow`` policy to a batch that does not fit.

        Returns:
            How many of the *batch_size* offered samples may be written.
        """
        room = max(self._allocated_samples - self._current_index, 0)
        counts = (
            f"{self._current_index} written, {self._allocated_samples} allocated "
            f"(expected_samples={self.expected_samples}), {batch_size} offered"
        )
        if self.on_overflow == "error":
            raise ValueError(
                f"Batch does not fit the writer's allocation: {counts}. Raise "
                f"expected_samples, or pass on_overflow='grow' to reallocate as "
                f"needed or 'trim' to drop the excess."
            )
        if self.on_overflow == "grow":
            self._grow(self._current_index + batch_size)
            return batch_size
        warnings.warn(
            f"TreeWriter is dropping {batch_size - room} of {batch_size} offered "
            f"samples: {counts}. Pass on_overflow='grow' to reallocate instead, "
            f"or 'error' to fail.",
            stacklevel=3,
        )
        return room

    def write(self, pytree) -> int:
        """Write a batch of samples to the memmap and bagz files.

        The first call infers the schema from the pytree structure.
        Array leaves must have the same batch size (first dimension).
        String leaves (``str`` or ``List[str]``) are stored in bagz files.

        Args:
            pytree: Any JAX-compatible pytree (AudioTree, dict, nested).
                Array leaves must have a batch dimension. String leaves
                can be a single ``str`` (batch size 1) or ``List[str]``.

        Returns:
            Number of samples written. This is the batch size unless
            ``on_overflow="trim"`` dropped part of the batch, in which case it
            is smaller (possibly 0) and a warning is issued.

        Raises:
            RuntimeError: If writer is not open, has been closed, or could not
                grow its files for an earlier batch.
            ValueError: If shapes don't match, or if the batch does not fit the
                allocation and ``on_overflow="error"``.
        """
        if self._is_closed:
            raise RuntimeError(
                "Writer is closed; further writes would be silently dropped. "
                "Create a new TreeWriter."
            )
        if self._broken is not None:
            raise RuntimeError(self._broken)
        if not self._is_open:
            raise RuntimeError("Writer is not open. Call open() first.")

        structure, leaf_names, string_leaf_names = _serialize_structure(pytree)

        # Initialize schema on first write, then require every later write to
        # match it by name -- not just by leaf count.
        if self._structure is None:
            self._init_from_pytree(pytree, structure, leaf_names, string_leaf_names)
        else:
            self._check_structure_matches(structure, leaf_names, string_leaf_names)

        array_leaves, string_data = _extract_leaves(
            pytree, self._leaf_names, self._string_leaf_names
        )

        # Determine batch size from first available leaf
        batch_size = None
        if array_leaves:
            batch_size = array_leaves[0].shape[0]
        elif string_data:
            batch_size = len(next(iter(string_data.values())))

        if batch_size is None:
            raise ValueError("Pytree has no leaves.")

        # The allocation is a hint, not a cap: `on_overflow` decides whether an
        # overshooting batch grows the files, raises, or is trimmed away.
        if self._current_index + batch_size > self._allocated_samples:
            writable = self._handle_overflow(batch_size)
            if writable < batch_size:
                if writable <= 0:
                    return 0
                array_leaves = [leaf[:writable] for leaf in array_leaves]
                string_data = {k: v[:writable] for k, v in string_data.items()}
                batch_size = writable
        end_idx = self._current_index + batch_size

        # Validate every leaf -- batch size, per-sample shape, exact dtype, and
        # (for string leaves) utf-8 encodability -- BEFORE mutating any memmap or
        # bagz file. A write() must be atomic: either the whole batch commits or
        # none of it does. Validating and writing in one interleaved pass let a
        # check that tripped on a later leaf leave earlier ones already written
        # -- and, for string leaves, orphaned bagz records that positionally
        # misbind every later label. Exact-dtype matching also closes the gap
        # where dtype was recorded on the first write but never re-checked, so
        # numpy's default unsafe cast silently corrupted a later batch's values.
        for i, leaf in enumerate(array_leaves):
            name = self._leaf_names[i]
            if leaf.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"leaf '{name}' has {leaf.shape[0]}"
                )
            expected_shape = tuple(self._leaf_info[name]["shape_per_sample"])
            if leaf.shape[1:] != expected_shape:
                raise ValueError(
                    f"Shape mismatch for leaf '{name}': "
                    f"expected {expected_shape}, got {leaf.shape[1:]}"
                )
            expected_dtype = self._leaf_info[name]["dtype"]
            got_dtype = str(_native_dtype(leaf.dtype))
            if got_dtype != expected_dtype:
                raise ValueError(
                    f"Dtype mismatch for leaf '{name}': expected {expected_dtype}, "
                    f"got {got_dtype}. numpy would cast it into the schema dtype "
                    f"without warning, silently corrupting values; cast the leaf "
                    f"to {expected_dtype} before writing."
                )

        encoded_string_data: Dict[str, List[bytes]] = {}
        for name, strings in string_data.items():
            if len(strings) != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"string leaf '{name}' has {len(strings)}"
                )
            try:
                encoded_string_data[name] = [s.encode("utf-8") for s in strings]
            except UnicodeEncodeError as e:
                raise ValueError(
                    f"String leaf '{name}' has a value that is not utf-8 "
                    f"encodable: {e}. The batch is rejected with nothing written."
                ) from e

        # Validation passed: commit every leaf. Array writes are positional at
        # _current_index (a failed one is overwritten by the next batch or
        # truncated by close), but a bagz append cannot be un-appended, so a
        # failure part-way through the commit -- a disk error, say -- can leave
        # the leaf files inconsistent. Poison the writer so subsequent writes
        # and any finalization fail loudly rather than silently producing a
        # dataset whose later string labels are shifted.
        try:
            for mm, leaf in zip(self._memmaps, array_leaves):
                mm[self._current_index : end_idx] = leaf
            for name, records in encoded_string_data.items():
                writer = self._bagz_writers[name]
                for record in records:
                    writer.write(record)
        except BaseException:
            self._broken = (
                f"TreeWriter failed part-way through committing a batch, so its "
                f"leaf files may be inconsistent (some leaves written, some not). "
                f"The {self._current_index} samples already written remain "
                f"readable and close() will finalize them, but this writer cannot "
                f"accept further writes."
            )
            raise

        self._current_index += batch_size
        if self._pbar is not None:
            self._pbar.update(batch_size)

        # Keep the on-disk sample count honest as the render progresses. The
        # manifest is written first with num_samples=0, so without this a
        # SIGKILL left a directory of real data that reads back as empty.
        if (
            self.manifest_interval
            and time.monotonic() - self._manifest_written_at >= self.manifest_interval
        ):
            self._write_manifest()
        return batch_size

    def _write_manifest(self):
        """Write ``manifest.json`` atomically (temp file + rename).

        Called on schema init, from :meth:`flush`, and from :meth:`close`, so the
        on-disk manifest always describes a prefix that is actually readable and
        a reader never observes a half-written file.
        """
        manifest = {
            **_format.header(_format.TREE),
            "num_samples": self._current_index,
            "expected_samples": self.expected_samples,
            "created_at": datetime.now().isoformat(),
            "structure": self._structure,
            "leaves": self._leaf_info,
            "string_leaves": self._string_leaf_info,
            "metadata": self.metadata,
        }
        write_json_atomic(self.directory / "manifest.json", manifest)
        self._manifest_written_at = time.monotonic()

    def flush(self):
        """Flush all memmap files to disk and refresh the manifest."""
        for mm in self._memmaps:
            mm.flush()
        if self._structure is not None:
            self._write_manifest()

    def close(self):
        """Close all memmap and bagz files and write manifest.

        If fewer samples were written than were allocated -- because
        ``expected_samples`` overshot, or because ``on_overflow="grow"``
        reallocated with slack -- the memmap files are truncated to the actual
        sample count so no disk space is wasted and readers see the correct
        size.

        Closing is terminal: the writer cannot be reopened, because its files
        have been truncated to what was already written and its memmaps
        released.
        """
        if not self._is_open:
            return
        self._is_closed = True

        actual_samples = self._current_index

        # Flush and release memmaps
        for mm in self._memmaps:
            mm.flush()
            del mm
        self._memmaps.clear()

        # Truncate memmap files if we wrote fewer samples than allocated. Each
        # file is measured rather than compared against `_allocated_samples`,
        # so a growth that failed part-way -- leaving leaves at different
        # allocations -- still ends up with every file exactly num_samples long.
        if actual_samples < self._allocated_samples:
            logger = logging.getLogger(__name__)
            logger.info(
                "Wrote %d / %d allocated samples; truncating memmap files.",
                actual_samples,
                self._allocated_samples,
            )
        for name in self._leaf_names:
            filepath = safe_join(
                self.directory,
                self._leaf_info[name]["file"],
                description="leaf file",
            )
            actual_bytes = actual_samples * self._bytes_per_sample(name)
            if filepath.stat().st_size > actual_bytes:
                os.truncate(filepath, actual_bytes)

        # Close bagz writers
        for writer in self._bagz_writers.values():
            writer.close()
        self._bagz_writers.clear()

        # Final manifest, now reflecting the truncated sample count.
        if self._structure is not None:
            self._write_manifest()

        if self._close_pbar and self._pbar is not None:
            self._pbar.close()

        self._is_open = False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about written data."""
        return {
            "samples_written": self._current_index,
            "expected_samples": self.expected_samples,
            "allocated_samples": self._allocated_samples,
            "output_directory": str(self.directory),
            "leaves": list(self._leaf_names),
            "string_leaves": list(self._string_leaf_names),
            "is_open": self._is_open,
        }

    def __enter__(self) -> "TreeWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
