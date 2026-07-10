"""TreeWriter: pytree-native writer for memory-mapped datasets."""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.tree_util
import numpy as np

from audiotree._bagz import require_bagz
from audiotree.core import AudioTree

# AudioTree pytree field names in declaration order (matching Flax flatten order).
# sample_rate is excluded (pytree_node=False).
_AUDIOTREE_FIELD_ORDER = [
    "waveform",
    "lufs",
    "lufs_windows",
    "pitch",
    "velocity",
    "note_duration",
    "codes",
    "latents",
    "metadata",
]


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
            for fname in _AUDIOTREE_FIELD_ORDER:
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
            for fname in _AUDIOTREE_FIELD_ORDER:
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
        output_dir: Directory where memmap files will be written
        expected_samples: Total number of samples to pre-allocate
        metadata: Optional dict of user metadata to store in manifest
        pbar: Optional tqdm progress bar instance. Updated by ``batch_size``
            after each ``write()`` call.
        close_pbar: If True, close the progress bar when the writer closes.
            Default False.

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
        output_dir: Union[str, Path],
        expected_samples: int,
        metadata: Optional[Dict[str, Any]] = None,
        pbar=None,
        close_pbar: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.expected_samples = expected_samples
        self.metadata = metadata or {}
        self._pbar = pbar
        self._close_pbar = close_pbar

        self._current_index = 0
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []
        self._leaf_info: Dict[str, Dict] = {}
        self._string_leaf_names: List[str] = []
        self._string_leaf_info: Dict[str, Dict] = {}
        self._bagz_writers: Dict = {}
        self._structure = None
        self._is_open = False

    def open(self) -> "TreeWriter":
        """Open the writer and create output directory.

        Returns:
            self for method chaining
        """
        if self._is_open:
            raise RuntimeError("Writer is already open")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._is_open = True
        return self

    def _init_from_pytree(self, pytree):
        """Infer schema from the first pytree and create memmap/bagz files."""
        self._structure, self._leaf_names, self._string_leaf_names = (
            _serialize_structure(pytree)
        )

        array_leaves, _ = _extract_leaves(
            pytree, self._leaf_names, self._string_leaf_names
        )

        # Create memmaps for array leaves
        for name, leaf in zip(self._leaf_names, array_leaves):
            shape_per_sample = leaf.shape[1:]  # exclude batch dim
            dtype = leaf.dtype
            filename = f"{name}.bin"

            self._leaf_info[name] = {
                "dtype": str(dtype),
                "shape_per_sample": list(shape_per_sample),
                "file": filename,
            }

            full_shape = (self.expected_samples,) + shape_per_sample
            mm = np.memmap(
                self.output_dir / filename,
                dtype=dtype,
                mode="w+",
                shape=full_shape,
            )
            self._memmaps.append(mm)

        # Create bagz writers for string leaves
        if self._string_leaf_names:
            bagz = require_bagz("storing string leaves in TreeWriter")
            for name in self._string_leaf_names:
                filename = f"{name}.bagz"
                self._string_leaf_info[name] = {"file": filename}
                self._bagz_writers[name] = bagz.Writer(str(self.output_dir / filename))

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
            RuntimeError: If writer is not open
            ValueError: If shapes don't match or would exceed expected_samples
        """
        if not self._is_open:
            raise RuntimeError("Writer is not open. Call open() first.")

        # Initialize schema on first write
        if self._structure is None:
            self._init_from_pytree(pytree)

        array_leaves, string_data = _extract_leaves(
            pytree, self._leaf_names, self._string_leaf_names
        )

        if len(array_leaves) != len(self._memmaps):
            raise ValueError(
                f"Pytree has {len(array_leaves)} array leaves but schema "
                f"expects {len(self._memmaps)}. Structure must match the "
                f"first write."
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
        return batch_size

    def flush(self):
        """Flush all memmap files to disk."""
        for mm in self._memmaps:
            mm.flush()

    def close(self):
        """Close all memmap and bagz files and write manifest.

        If fewer samples were written than ``expected_samples``, the memmap
        files are truncated to the actual sample count so no disk space is
        wasted and readers see the correct size.
        """
        if not self._is_open:
            return

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
                filepath = self.output_dir / self._leaf_info[name]["file"]
                shape_per_sample = self._leaf_info[name]["shape_per_sample"]
                dtype = np.dtype(self._leaf_info[name]["dtype"])
                elems = int(np.prod(shape_per_sample)) if shape_per_sample else 1
                actual_bytes = actual_samples * elems * dtype.itemsize
                os.truncate(filepath, actual_bytes)

        # Close bagz writers
        for writer in self._bagz_writers.values():
            writer.close()
        self._bagz_writers.clear()

        # Write manifest
        manifest = {
            "version": "2.0",
            "num_samples": self._current_index,
            "expected_samples": self.expected_samples,
            "created_at": datetime.now().isoformat(),
            "structure": self._structure,
            "leaves": self._leaf_info,
            "string_leaves": self._string_leaf_info,
            "metadata": self.metadata,
        }

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if self._close_pbar and self._pbar is not None:
            self._pbar.close()

        self._is_open = False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about written data."""
        return {
            "samples_written": self._current_index,
            "expected_samples": self.expected_samples,
            "output_directory": str(self.output_dir),
            "leaves": list(self._leaf_names),
            "string_leaves": list(self._string_leaf_names),
            "is_open": self._is_open,
        }

    def __enter__(self) -> "TreeWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
