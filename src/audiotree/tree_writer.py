"""TreeWriter: pytree-native writer for memory-mapped datasets."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        expected_samples: Total number of samples to pre-allocate
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
        metadata: Optional[Dict[str, Any]] = None,
        pbar=None,
        close_pbar: bool = False,
        exist_ok: bool = False,
        manifest_interval: float = 5.0,
    ):
        self.directory = Path(directory)
        self.expected_samples = expected_samples
        self.metadata = metadata or {}
        self.exist_ok = exist_ok
        self.manifest_interval = manifest_interval
        self._pbar = pbar
        self._close_pbar = close_pbar

        self._manifest_written_at = 0.0
        self._current_index = 0
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

            full_shape = (self.expected_samples,) + shape_per_sample
            mm = np.memmap(
                path,
                dtype=dtype,
                mode="w+",
                shape=full_shape,
            )
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
            Number of samples written (batch size)

        Raises:
            RuntimeError: If writer is not open, or has been closed
            ValueError: If shapes don't match or would exceed expected_samples
        """
        if self._is_closed:
            raise RuntimeError(
                "Writer is closed; further writes would be silently dropped. "
                "Create a new TreeWriter."
            )
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

        end_idx = self._current_index + batch_size

        # Trim batch to fit within expected_samples (last batch may overshoot)
        if end_idx > self.expected_samples:
            actual_batch = self.expected_samples - self._current_index
            if actual_batch <= 0:
                return 0
            array_leaves = [leaf[:actual_batch] for leaf in array_leaves]
            string_data = {k: v[:actual_batch] for k, v in string_data.items()}
            batch_size = actual_batch
            end_idx = self._current_index + batch_size

        # Write array leaves to memmaps
        for i, (mm, leaf) in enumerate(zip(self._memmaps, array_leaves)):
            if leaf.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"leaf '{self._leaf_names[i]}' has {leaf.shape[0]}"
                )

            expected_shape = tuple(
                self._leaf_info[self._leaf_names[i]]["shape_per_sample"]
            )
            if leaf.shape[1:] != expected_shape:
                raise ValueError(
                    f"Shape mismatch for leaf '{self._leaf_names[i]}': "
                    f"expected {expected_shape}, got {leaf.shape[1:]}"
                )

            mm[self._current_index : end_idx] = leaf

        # Write string leaves to bagz files
        for name, strings in string_data.items():
            if len(strings) != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: expected {batch_size}, "
                    f"string leaf '{name}' has {len(strings)}"
                )
            writer = self._bagz_writers[name]
            for s in strings:
                writer.write(s.encode("utf-8"))

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

        If fewer samples were written than ``expected_samples``, the memmap
        files are truncated to the actual sample count so no disk space is
        wasted and readers see the correct size.

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

        # Truncate memmap files if we wrote fewer samples than allocated
        if actual_samples < self.expected_samples:
            logger = logging.getLogger(__name__)
            logger.info(
                "Wrote %d / %d expected samples; truncating memmap files.",
                actual_samples,
                self.expected_samples,
            )
            for name in self._leaf_names:
                filepath = safe_join(
                    self.directory,
                    self._leaf_info[name]["file"],
                    description="leaf file",
                )
                shape_per_sample = self._leaf_info[name]["shape_per_sample"]
                dtype = np.dtype(self._leaf_info[name]["dtype"])
                elems = int(np.prod(shape_per_sample)) if shape_per_sample else 1
                actual_bytes = actual_samples * elems * dtype.itemsize
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
            "output_directory": str(self.directory),
            "leaves": list(self._leaf_names),
            "string_leaves": list(self._string_leaf_names),
            "is_open": self._is_open,
        }

    def __enter__(self) -> "TreeWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
