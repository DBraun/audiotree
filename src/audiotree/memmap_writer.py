"""MemmapWriter class for writing arrays to memory-mapped files."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np

# Avoid circular import: import AudioTree utilities lazily
HAS_AUDIOTREE = False
AudioTree = None
AudioTreeFieldExtractor = None

def _ensure_audiotree_imported():
    """Lazy import to avoid circular dependency."""
    global HAS_AUDIOTREE, AudioTree, AudioTreeFieldExtractor
    if HAS_AUDIOTREE or AudioTree is not None:
        return True
    try:
        from audiotree.core import AudioTree as _AudioTree
        from audiotree.audiotree_utils import AudioTreeFieldExtractor as _Extractor
        AudioTree = _AudioTree
        AudioTreeFieldExtractor = _Extractor
        HAS_AUDIOTREE = True
        return True
    except ImportError:
        return False


@dataclass
class FieldSpec:
    """Specification for a single field to be written to memmap.

    Args:
        name: Name of the field (used for filename and manifest key)
        dtype: NumPy dtype for the field
        shape_per_sample: Shape of each sample, excluding batch dimension
    """

    name: str
    dtype: np.dtype
    shape_per_sample: Tuple[int, ...]

    def __post_init__(self):
        # Ensure dtype is a numpy dtype
        self.dtype = np.dtype(self.dtype)
        # Ensure shape is a tuple
        if isinstance(self.shape_per_sample, int):
            self.shape_per_sample = (self.shape_per_sample,)
        else:
            self.shape_per_sample = tuple(self.shape_per_sample)


class MemmapWriter:
    """Write arrays to disk using memory-mapped files for efficient large dataset creation.

    Unlike AudioWriter which writes individual WAV files, MemmapWriter creates
    contiguous binary files that can be memory-mapped for fast random access.
    This is ideal for pre-rendering large training datasets.

    Args:
        output_dir: Directory where memmap files will be written
        field_specs: List of FieldSpec defining what fields to write
        expected_samples: Total number of samples to write (required for pre-allocation)
        metadata: Optional dict of additional metadata to store in manifest
        pbar: Optional tqdm progress bar instance
        close_pbar: Whether to close progress bar on exit

    Example:
        >>> specs = [
        ...     FieldSpec("audio", np.float32, (2, 132300)),
        ...     FieldSpec("params", np.float32, (12,)),
        ...     FieldSpec("label", np.int32, ()),
        ... ]
        >>> with MemmapWriter("output", specs, expected_samples=10000) as writer:
        ...     for batch in dataloader:
        ...         writer.write_batch({
        ...             "audio": batch["audio"],
        ...             "params": batch["params"],
        ...             "label": batch["label"],
        ...         })
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        field_specs: Optional[List[FieldSpec]] = None,
        expected_samples: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        pbar: Optional[Any] = None,
        close_pbar: bool = False,
        infer_schema: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.field_specs = {spec.name: spec for spec in field_specs} if field_specs else None
        self.expected_samples = expected_samples
        self.metadata = metadata or {}
        self.pbar = pbar
        self.close_pbar = close_pbar
        # If field_specs are provided, don't infer schema
        self.infer_schema = infer_schema if field_specs is None else False

        self._current_index = 0
        self._memmaps: Dict[str, np.memmap] = {}
        self._string_buffers: Dict[str, List[str]] = {}
        self._is_open = False
        self._schema_inferred = False
        self._audiotree_fields: Dict[str, bool] = {}

    def _create_memmaps(self):
        """Create memory-mapped files for all fields."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name, spec in self.field_specs.items():
            full_shape = (self.expected_samples,) + spec.shape_per_sample
            filepath = self.output_dir / f"{name}.bin"

            self._memmaps[name] = np.memmap(
                filepath,
                dtype=spec.dtype,
                mode="w+",
                shape=full_shape,
            )

    def _infer_schema_from_batch(self, data: Dict):
        """Infer schema from first batch.

        Args:
            data: First batch dict (may contain AudioTree objects)
        """
        _ensure_audiotree_imported()

        if not HAS_AUDIOTREE:
            raise RuntimeError("AudioTree not available but infer_schema=True with AudioTree data")

        extracted = {}

        # Extract AudioTree objects and track which keys are AudioTrees
        for key, value in data.items():
            if AudioTree is not None and isinstance(value, AudioTree):
                self._audiotree_fields[key] = True
                tree_fields = AudioTreeFieldExtractor.extract_fields(value, prefix=f"{key}_")
                extracted.update(tree_fields)
            elif isinstance(value, np.ndarray):
                extracted[key] = value
            # Strings are handled separately, skip for now

        # Infer FieldSpec from extracted fields
        field_specs_list = AudioTreeFieldExtractor.infer_field_specs(extracted)
        self.field_specs = {spec.name: spec for spec in field_specs_list}

        # Infer expected_samples from batch size if not provided
        if self.expected_samples is None:
            batch_size = next(iter(extracted.values())).shape[0]
            raise ValueError(
                "expected_samples must be provided when infer_schema=True. "
                f"First batch has {batch_size} samples."
            )

        self._schema_inferred = True

    def _extract_batch_arrays(self, data: Dict) -> Dict[str, np.ndarray]:
        """Extract numpy arrays from batch, decomposing AudioTree objects.

        Args:
            data: Batch dict (may contain AudioTree objects and numpy arrays)

        Returns:
            Flat dict with only numpy arrays
        """
        _ensure_audiotree_imported()

        extracted = {}

        for key, value in data.items():
            if AudioTree is not None and isinstance(value, AudioTree):
                # Decompose AudioTree to flat fields
                tree_fields = AudioTreeFieldExtractor.extract_fields(value, prefix=f"{key}_")
                extracted.update(tree_fields)
            elif isinstance(value, np.ndarray):
                extracted[key] = value
            # Skip non-array types (will be in strings dict)

        return extracted

    def open(self) -> "MemmapWriter":
        """Open the writer and create memmap files.

        Returns:
            self for method chaining
        """
        if self._is_open:
            raise RuntimeError("Writer is already open")

        # Only create memmaps if schema is already known
        # If inferring, memmaps will be created on first write
        if self.field_specs is not None:
            self._create_memmaps()

        self._is_open = True
        return self

    def write_batch(
        self,
        data: Dict[str, Union[np.ndarray, "AudioTree"]],
        strings: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """Write a batch of data to the memmap files.

        Args:
            data: Dict mapping field names to numpy arrays or AudioTree objects
            strings: Optional dict mapping string field names to lists of strings

        Returns:
            Number of samples written

        Raises:
            RuntimeError: If writer is not open
            ValueError: If field names don't match specs or shapes are wrong
        """
        if not self._is_open:
            raise RuntimeError("Writer is not open. Call open() first.")

        # Infer schema on first write if needed
        if self.infer_schema and not self._schema_inferred:
            self._infer_schema_from_batch(data)
            self._create_memmaps()  # Create memmaps after inferring schema

        # Extract numpy arrays from AudioTree objects (only if inferring schema)
        if self.infer_schema or self._audiotree_fields:
            extracted_data = self._extract_batch_arrays(data)
        else:
            # Use data as-is (backward compatible with explicit field_specs)
            extracted_data = data

        # Validate and write each field
        batch_size = None
        for name, arr in extracted_data.items():
            if name not in self._memmaps:
                raise ValueError(f"Unknown field: {name}. Expected one of: {list(self._memmaps.keys())}")

            arr = np.asarray(arr)

            if batch_size is None:
                batch_size = arr.shape[0]
            elif arr.shape[0] != batch_size:
                raise ValueError(f"Inconsistent batch sizes: {batch_size} vs {arr.shape[0]}")

            # Check expected shape
            spec = self.field_specs[name]
            expected_sample_shape = spec.shape_per_sample
            actual_sample_shape = arr.shape[1:]
            if actual_sample_shape != expected_sample_shape:
                raise ValueError(
                    f"Shape mismatch for field '{name}': "
                    f"expected sample shape {expected_sample_shape}, "
                    f"got {actual_sample_shape}"
                )

            # Check if we have room
            end_idx = self._current_index + batch_size
            if end_idx > self.expected_samples:
                raise ValueError(
                    f"Writing {batch_size} samples would exceed expected_samples "
                    f"({self._current_index} + {batch_size} > {self.expected_samples})"
                )

            # Write to memmap
            self._memmaps[name][self._current_index : end_idx] = arr

        # Handle string fields
        if strings:
            for name, str_list in strings.items():
                if len(str_list) != batch_size:
                    raise ValueError(
                        f"String list '{name}' has length {len(str_list)}, "
                        f"expected {batch_size}"
                    )
                if name not in self._string_buffers:
                    self._string_buffers[name] = []
                self._string_buffers[name].extend(str_list)

        self._current_index += batch_size

        # Update progress bar
        if self.pbar is not None:
            self.pbar.update(batch_size)

        return batch_size

    def write_sample(
        self,
        data: Dict[str, np.ndarray],
        strings: Optional[Dict[str, str]] = None,
    ) -> int:
        """Write a single sample to the memmap files.

        Convenience method that adds batch dimension and calls write_batch.

        Args:
            data: Dict mapping field names to numpy arrays (no batch dimension)
            strings: Optional dict mapping string field names to single strings

        Returns:
            1 (number of samples written)
        """
        # Add batch dimension to each array
        batched_data = {name: arr[np.newaxis, ...] for name, arr in data.items()}

        # Convert single strings to lists
        batched_strings = None
        if strings:
            batched_strings = {name: [s] for name, s in strings.items()}

        return self.write_batch(batched_data, batched_strings)

    def flush(self):
        """Flush all memmap files to disk."""
        for mm in self._memmaps.values():
            mm.flush()

    def close(self):
        """Close all memmap files and write manifest."""
        if not self._is_open:
            return

        # Flush and close memmaps
        for mm in self._memmaps.values():
            mm.flush()
            del mm
        self._memmaps.clear()

        # Write string fields to JSON
        for name, str_list in self._string_buffers.items():
            json_path = self.output_dir / f"{name}.json"
            with open(json_path, "w") as f:
                json.dump(str_list, f)

        # Write manifest
        manifest = {
            "version": "1.0",
            "num_samples": self._current_index,
            "expected_samples": self.expected_samples,
            "created_at": datetime.now().isoformat(),
            **self.metadata,
            "fields": {},
            "string_fields": {},
        }

        # Store AudioTree reconstruction info if any AudioTree objects were written
        if self._audiotree_fields:
            manifest["audiotree_fields"] = self._audiotree_fields

        for name, spec in self.field_specs.items():
            full_shape = (self._current_index,) + spec.shape_per_sample
            manifest["fields"][name] = {
                "dtype": str(spec.dtype),
                "shape": list(full_shape),
                "filename": f"{name}.bin",
            }

        for name in self._string_buffers:
            manifest["string_fields"][name] = f"{name}.json"

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self._is_open = False

        # Close progress bar
        if self.pbar is not None and self.close_pbar:
            self.pbar.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about written data.

        Returns:
            Dictionary containing write statistics
        """
        return {
            "samples_written": self._current_index,
            "expected_samples": self.expected_samples,
            "output_directory": str(self.output_dir),
            "fields": list(self.field_specs.keys()),
            "string_fields": list(self._string_buffers.keys()),
            "is_open": self._is_open,
        }

    def __enter__(self) -> "MemmapWriter":
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - calls close() to save manifest."""
        self.close()
