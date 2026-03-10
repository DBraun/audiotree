"""TreeWriter: pytree-native writer for memory-mapped datasets."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.tree_util
import numpy as np

from audiotree.core import AudioTree

# AudioTree pytree field names in declaration order (matching Flax flatten order).
# sample_rate is excluded (pytree_node=False).
_AUDIOTREE_FIELD_ORDER = [
    "audio_data",
    "loudness",
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
        (GetAttrKey('audio_data'),) -> "audio_data"
        (GetAttrKey('metadata'), DictKey('mel')) -> "metadata.mel"
        (DictKey('dry'), GetAttrKey('audio_data')) -> "dry.audio_data"
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


def _serialize_structure(pytree) -> Tuple[Any, List[str]]:
    """Walk a pytree to build a JSON-serializable structure and collect leaf names.

    Returns:
        (structure, leaf_names) where structure is a JSON-compatible description
        and leaf_names is an ordered list of dot-separated leaf path strings.
        The leaf order matches JAX's tree_flatten order.
    """
    leaf_names: List[str] = []

    def _walk(node, prefix: str):
        # Leaf: numpy or JAX array
        if isinstance(node, (np.ndarray, jax.Array)):
            leaf_names.append(prefix)
            return prefix  # string = leaf reference

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
            "All leaves must be numpy or JAX arrays."
        )

    structure = _walk(pytree, "")
    return structure, leaf_names


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

    Example:
        >>> tree = AudioTree(audio_data=audio, sample_rate=44100, loudness=loud)
        >>> with TreeWriter("output/", expected_samples=10000) as w:
        ...     for batch in dataloader:
        ...         w.write(batch)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        expected_samples: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.expected_samples = expected_samples
        self.metadata = metadata or {}

        self._current_index = 0
        self._memmaps: List[np.memmap] = []
        self._leaf_names: List[str] = []
        self._leaf_info: Dict[str, Dict] = {}
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
        """Infer schema from the first pytree and create memmap files."""
        # Build structure and get leaf names via manual walk
        self._structure, self._leaf_names = _serialize_structure(pytree)

        # Verify leaf order matches JAX's flatten order
        paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(pytree)
        jax_names = [_path_to_string(path) for path, _ in paths_and_leaves]
        assert self._leaf_names == jax_names, (
            f"Leaf order mismatch between manual walk and JAX flatten. "
            f"Manual: {self._leaf_names}, JAX: {jax_names}"
        )

        # Create memmaps
        leaves = [leaf for _, leaf in paths_and_leaves]
        for name, leaf in zip(self._leaf_names, leaves):
            arr = np.asarray(leaf)
            shape_per_sample = arr.shape[1:]  # exclude batch dim
            dtype = arr.dtype
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

    def write(self, pytree) -> int:
        """Write a batch of samples to the memmap files.

        The first call infers the schema from the pytree structure.
        All leaves must have the same batch size (first dimension).

        Args:
            pytree: Any JAX-compatible pytree (AudioTree, dict, nested).
                All leaves must be arrays with a batch dimension.

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

        leaves = jax.tree.leaves(pytree)

        if len(leaves) != len(self._memmaps):
            raise ValueError(
                f"Pytree has {len(leaves)} leaves but schema expects "
                f"{len(self._memmaps)}. Structure must match the first write."
            )

        batch_size = np.asarray(leaves[0]).shape[0]
        end_idx = self._current_index + batch_size

        if end_idx > self.expected_samples:
            raise ValueError(
                f"Writing {batch_size} samples would exceed expected_samples "
                f"({self._current_index} + {batch_size} > {self.expected_samples})"
            )

        for i, (mm, leaf) in enumerate(zip(self._memmaps, leaves)):
            arr = np.asarray(leaf)

            if arr.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch sizes: leaf 0 has {batch_size}, "
                    f"leaf {i} ({self._leaf_names[i]}) has {arr.shape[0]}"
                )

            expected_shape = tuple(
                self._leaf_info[self._leaf_names[i]]["shape_per_sample"]
            )
            if arr.shape[1:] != expected_shape:
                raise ValueError(
                    f"Shape mismatch for leaf '{self._leaf_names[i]}': "
                    f"expected {expected_shape}, got {arr.shape[1:]}"
                )

            mm[self._current_index:end_idx] = arr

        self._current_index += batch_size
        return batch_size

    def flush(self):
        """Flush all memmap files to disk."""
        for mm in self._memmaps:
            mm.flush()

    def close(self):
        """Close all memmap files and write manifest."""
        if not self._is_open:
            return

        # Flush and release memmaps
        for mm in self._memmaps:
            mm.flush()
            del mm
        self._memmaps.clear()

        # Write manifest
        manifest = {
            "version": "2.0",
            "num_samples": self._current_index,
            "expected_samples": self.expected_samples,
            "created_at": datetime.now().isoformat(),
            "structure": self._structure,
            "leaves": self._leaf_info,
            "metadata": self.metadata,
        }

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self._is_open = False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about written data."""
        return {
            "samples_written": self._current_index,
            "expected_samples": self.expected_samples,
            "output_directory": str(self.output_dir),
            "leaves": list(self._leaf_names),
            "is_open": self._is_open,
        }

    def __enter__(self) -> "TreeWriter":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
