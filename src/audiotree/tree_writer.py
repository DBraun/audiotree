"""TreeWriter: pytree-native writer for memory-mapped datasets."""

import json
import logging
import math
import os
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import jax
import numpy as np

from audiotree import _format
from audiotree._bagz import limits_path, require_bagz
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
_ArrayLeaf = tuple[str, np.ndarray]
_StringLeaf = tuple[str, list[str] | list[list[str]], int | None]

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


def _describe_batch(pytree) -> tuple[Any, list[_ArrayLeaf], list[_StringLeaf], bool]:
    """Build the on-disk structure and collect each leaf in storage order."""
    arrays: list[_ArrayLeaf] = []
    strings: list[_StringLeaf] = []
    mini_batched = False

    def walk(node, prefix: str, group_size: int | None = None):
        nonlocal mini_batched
        if isinstance(node, (np.ndarray, jax.Array)):
            arrays.append((prefix, np.asarray(node)))
            return prefix

        if isinstance(node, (str, list)):
            if group_size is None:
                values = [node] if isinstance(node, str) else node
                if not all(isinstance(s, str) for s in values):
                    raise TypeError(
                        f"String leaf {prefix!r} needs a str or list of strings."
                    )
            else:
                values = node
                if not isinstance(values, list) or not all(
                    isinstance(group, list)
                    and len(group) == group_size
                    and all(isinstance(s, str) for s in group)
                    for group in values
                ):
                    raise ValueError(
                        f"Mini-batched string leaf {prefix!r} needs a list of "
                        f"groups of {group_size} strings."
                    )
            strings.append((prefix, values, group_size))
            return {"type": "string_leaf", "leaf": prefix}

        if isinstance(node, AudioTree):
            if (
                not isinstance(node.sample_rate, int)
                or isinstance(node.sample_rate, bool)
                or node.sample_rate <= 0
            ):
                raise ValueError(
                    f"AudioTree at {prefix or '<root>'} has sample_rate {node.sample_rate!r}."
                )
            reference = node.waveform
            grouped = reference is not None and reference.ndim >= 4
            if reference is None and node._mini_batched:
                reference = node.codes if node.codes is not None else node.latents
                grouped = reference is not None and reference.ndim >= 2
            group_size = reference.shape[1] if grouped else None
            children = {}
            for name in PYTREE_FIELDS:
                value = getattr(node, name)
                if value is None:
                    continue
                # The provenance field serializes under its historical name.
                key = "metadata" if name == "_metadata" else name
                path = f"{prefix}.{key}" if prefix else key
                children[key] = walk(value, path, group_size)
            structure = {
                "type": "AudioTree",
                "sample_rate": node.sample_rate,
                "children": children,
            }
            if node._mini_batched:
                mini_batched = True
                structure["mini_batched"] = True
            return structure

        if isinstance(node, dict):
            children = {}
            for key in sorted(node):
                path = f"{prefix}.{key}" if prefix else str(key)
                if str(key) in children:
                    raise ValueError(
                        f"Pytree keys at {prefix or '<root>'} must have distinct string names."
                    )
                children[str(key)] = walk(node[key], path, group_size)
            return {"type": "dict", "children": children}

        raise TypeError(
            f"Unsupported pytree node type: {type(node)}. "
            "Leaves must be numpy/JAX arrays, str, or a list of strings."
        )

    structure = walk(pytree, "")
    return structure, arrays, strings, mini_batched


class TreeWriter:
    """Write any pytree to memory-mapped binary files.

    Each leaf in the pytree becomes a separate .bin file. The tree structure
    is stored in manifest.json for reconstruction by TreeDataSource.

    Accepts AudioTree objects, dicts of arrays, dicts of AudioTrees, or any
    combination. The schema is inferred from the first accepted batch.
    String leaves have a companion ``limits.<name>.bagz`` file, which stays
    separate so the dataset can be read before the writer closes.

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
        # The metadata dict only meets json.dump inside _write_manifest -- at
        # the end of the first write(), at interval refreshes (after batches
        # have committed), and at close(). A numpy scalar in it would fail
        # every one of those, leaving all the leaf data committed but no
        # manifest at all: an unreadable dataset, discovered after the render.
        # A cheap dumps here moves that discovery to construction time.
        try:
            json.dumps(self.metadata)
        except TypeError as e:
            raise TypeError(
                f"TreeWriter metadata must be JSON-serializable (it is stored "
                f"in manifest.json): {e}. Convert numpy values with .item() or "
                f".tolist() first."
            ) from e
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
        self._format_version = _format.CURRENT_VERSION
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

    def _leaf_path(self, name: str, filename: str, claimed: Dict[str, str]) -> Path:
        """Resolve a leaf file and reserve its name on case-insensitive volumes."""
        path = safe_join(self.directory, filename, description="leaf file")
        if (
            path.parent != self.directory.resolve()
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(
                f"Leaf {name!r} maps to the file {filename!r}, which is not a "
                f"plain name inside {self.directory}. Leaf keys must not contain "
                f"path separators."
            )
        # Normalize Unicode as well as case to catch equivalent names on macOS.
        key = unicodedata.normalize("NFD", path.name).casefold()
        if key in claimed:
            raise ValueError(
                f"Leaves {claimed[key]!r} and {name!r} both map to the file "
                f"{filename!r}. Leaf filenames and Bagz limits files must have "
                f"distinct names on case-insensitive filesystems."
            )
        claimed[key] = name
        return path

    def _initialize_storage(self, structure, arrays, strings, allocation, version):
        """Allocate storage for the validated first batch and publish its schema."""
        claimed: Dict[str, str] = {}
        leaf_info: Dict[str, Dict] = {}
        array_paths = []
        for name, leaf in arrays:
            dtype = _native_dtype(leaf.dtype)
            path = self._leaf_path(name, f"{name}.bin", claimed)
            array_paths.append((path, dtype, leaf.shape[1:]))
            leaf_info[name] = {
                "dtype": str(dtype),
                "shape_per_sample": list(leaf.shape[1:]),
                "file": path.name,
            }

        string_leaf_info: Dict[str, Dict] = {}
        string_paths = []
        for name, _, group_size in strings:
            path = self._leaf_path(name, f"{name}.bagz", claimed)
            sidecar = limits_path(path)
            self._leaf_path(name, sidecar.name, claimed)
            string_paths.append((name, path, sidecar))
            info = {"file": path.name, "limits_placement": "separate"}
            if group_size is not None:
                info["group_size"] = group_size
            string_leaf_info[name] = info
        if leaf_info.keys() & string_leaf_info.keys():
            raise ValueError("Array and string leaves cannot share a dot-joined path.")

        if strings:
            bagz = require_bagz("storing string leaves in TreeWriter")
            options = bagz.Writer.Options(
                limits_placement=bagz.LimitsPlacement.SEPARATE
            )
        bagz_writers: Dict = {}
        memmaps: List[np.memmap] = []
        created: List[Path] = []
        try:
            for path, dtype, shape_per_sample in array_paths:
                full_shape = (allocation,) + shape_per_sample
                nbytes = math.prod(full_shape) * dtype.itemsize
                with open(path, "wb") as fh:
                    created.append(path)
                    # Zero-sized arrays still need a mappable file on NumPy 2.1.
                    fh.truncate(max(nbytes, 1))
                memmaps.append(
                    np.memmap(path, dtype=dtype, mode="r+", shape=full_shape)
                )

            for name, path, sidecar in string_paths:
                created.extend((path, sidecar))
                bagz_writers[name] = bagz.Writer(str(path), options)
        except BaseException:
            for writer in bagz_writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
            memmaps.clear()
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._broken = (
                "TreeWriter failed while creating its leaf files, so it was "
                "never initialized. The files it had created were removed; "
                "create a new TreeWriter."
            )
            raise

        self._structure = structure
        self._format_version = version
        self._allocated_samples = allocation
        self._leaf_names = [name for name, _ in arrays]
        self._string_leaf_names = [name for name, _, _ in strings]
        self._leaf_info = leaf_info
        self._memmaps = memmaps
        self._string_leaf_info = string_leaf_info
        self._bagz_writers = bagz_writers
        self.flush()

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

    def _writable_samples(self, batch_size: int) -> int:
        """Apply the overflow policy before growing or allocating any files."""
        room = self._allocated_samples - self._current_index
        if batch_size <= room or self.on_overflow == "grow":
            return batch_size
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
        warnings.warn(
            f"TreeWriter is dropping {batch_size - room} of {batch_size} offered "
            f"samples: {counts}. Pass on_overflow='grow' to reallocate instead, "
            f"or 'error' to fail.",
            stacklevel=3,
        )
        return room

    def _validate_batch(self, arrays, strings):
        """Check the offered batch and encode all string records before I/O."""
        for name, leaf in arrays:
            if leaf.ndim == 0:
                raise ValueError(
                    f"Array leaf {name!r} is 0-dimensional; every array leaf "
                    f"needs a leading batch dimension. Wrap a scalar as "
                    f"np.asarray([value])."
                )
        if arrays:
            batch_size = arrays[0][1].shape[0]
        elif strings:
            batch_size = len(strings[0][1])
        else:
            raise ValueError("Pytree has no leaves.")

        for name, leaf in arrays:
            if leaf.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"leaf {name!r} has {leaf.shape[0]}"
                )
            dtype = str(_native_dtype(leaf.dtype))
            if self._structure is None:
                if dtype not in _ALLOWED_DTYPES:
                    raise ValueError(
                        f"Leaf {name!r} has dtype {leaf.dtype!s}, which TreeDataSource "
                        f"cannot read back. Supported dtypes: "
                        f"{sorted(_ALLOWED_DTYPES)}. Cast the leaf before writing."
                    )
                continue
            info = self._leaf_info[name]
            expected_shape = tuple(info["shape_per_sample"])
            if leaf.shape[1:] != expected_shape:
                raise ValueError(
                    f"Shape mismatch for leaf {name!r}: "
                    f"expected {expected_shape}, got {leaf.shape[1:]}"
                )
            expected_dtype = info["dtype"]
            if dtype != expected_dtype:
                raise ValueError(
                    f"Dtype mismatch for leaf {name!r}: expected {expected_dtype}, "
                    f"got {dtype}. Cast the leaf to {expected_dtype} before writing."
                )

        encoded = {}
        for name, values, group_size in strings:
            if len(values) != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"string leaf {name!r} has {len(values)}"
                )
            if self._structure is not None:
                expected_group_size = self._string_leaf_info[name].get("group_size")
                if group_size != expected_group_size:
                    raise ValueError(
                        f"Mini-batch grouping changed for string leaf {name!r}."
                    )
            try:
                if group_size is None:
                    encoded[name] = [value.encode("utf-8") for value in values]
                else:
                    encoded[name] = [
                        json.dumps(group, ensure_ascii=False).encode("utf-8")
                        for group in values
                    ]
            except UnicodeEncodeError as e:
                raise ValueError(
                    f"String leaf {name!r} has a value that is not utf-8 "
                    f"encodable: {e}. The batch is rejected with nothing written."
                ) from e
        return batch_size, encoded

    def write(self, pytree) -> int:
        """Write a batch of samples to the memmap and bagz files.

        The first call infers the schema from the pytree structure.
        Array leaves must have the same batch size (first dimension).
        String leaves are stored in Bagz files with separate record limits.

        Args:
            pytree: Any JAX-compatible pytree (AudioTree, dict, nested).
                Array leaves must have a batch dimension. String leaves
                can be a single ``str`` (batch size 1) or ``List[str]``. Mini-batched
                AudioTree extras can also contain a list of string groups.

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

        structure, arrays, strings, mini_batched = _describe_batch(pytree)
        if self._structure is not None:
            self._check_structure_matches(
                structure,
                [name for name, _ in arrays],
                [name for name, _, _ in strings],
            )
        batch_size, encoded_string_data = self._validate_batch(arrays, strings)
        writable = self._writable_samples(batch_size)
        if batch_size and writable == 0:
            return 0
        end_idx = self._current_index + writable

        if self._structure is None:
            version = _format.CURRENT_VERSION
            if strings:
                version = _format.TREE_STRING_LIMITS_VERSION
            elif mini_batched:
                version = _format.TREE_MINI_BATCH_VERSION
            allocation = max(self._allocated_samples, end_idx)
            self._initialize_storage(structure, arrays, strings, allocation, version)
        elif end_idx > self._allocated_samples:
            self._grow(end_idx)

        # Validation passed: commit every leaf. Array writes are positional at
        # _current_index (a failed one is overwritten by the next batch or
        # truncated by close), but a bagz append cannot be un-appended, so a
        # failure part-way through the commit -- a disk error, say -- can leave
        # the leaf files inconsistent. Poison the writer so subsequent writes
        # and any finalization fail loudly rather than silently producing a
        # dataset whose later string labels are shifted.
        try:
            for mm, (_, leaf) in zip(self._memmaps, arrays):
                mm[self._current_index : end_idx] = leaf[:writable]
            for name, records in encoded_string_data.items():
                writer = self._bagz_writers[name]
                for record in records[:writable]:
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

        self._current_index = end_idx
        if self._pbar is not None:
            self._pbar.update(writable)

        # Keep the on-disk sample count honest as the render progresses. The
        # manifest is written first with num_samples=0, so without this a
        # SIGKILL left a directory of real data that reads back as empty.
        if (
            self.manifest_interval
            and time.monotonic() - self._manifest_written_at >= self.manifest_interval
        ):
            self.flush()
        return writable

    def _write_manifest(self):
        """Write ``manifest.json`` atomically (temp file + rename).

        Called on schema init, from :meth:`flush`, and from :meth:`close`, so the
        on-disk manifest always describes a prefix that is actually readable and
        a reader never observes a half-written file.
        """

        manifest = {
            **_format.header(
                _format.TREE,
                format_version=self._format_version,
                min_reader_version=self._format_version,
            ),
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
        """Flush array and string files, then publish their readable prefix."""
        for mm in self._memmaps:
            mm.flush()
        for writer in self._bagz_writers.values():
            writer.flush()
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
            # max(..., 1): a zero-size-per-sample leaf needs no bytes, but an
            # empty file cannot be mapped, so never truncate below the one-byte
            # floor that _initialize_storage and _grow allocate to.
            actual_bytes = max(actual_samples * self._bytes_per_sample(name), 1)
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
