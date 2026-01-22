"""DataSource for reading memory-mapped files created by MemmapWriter."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, SupportsIndex, Tuple, Union

import numpy as np
from grain.sources import RandomAccessDataSource

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


class MemmapDataSource(RandomAccessDataSource):
    """A DataSource that reads from memory-mapped files created by MemmapWriter.

    This provides efficient random access to large pre-rendered datasets
    without loading the entire dataset into RAM. Data is read directly
    from memory-mapped binary files.

    Args:
        manifest_path: Path to the manifest.json file
        num_records: Optional limit on number of records (applied after split)
        fields: Optional list of field names to load (default: all)
        transform_fn: Optional function to transform each sample dict
        split: Which split to use ("train", "val", "test", or None for all data)
        split_ratios: Tuple of (train, val, test) ratios, must sum to 1.0
        split_seed: Random seed for reproducible split assignment
        load_into_memory: If True, load entire dataset into RAM at init time.
            Faster access but uses more memory. Default False.
        reconstruct_audiotree: If True, automatically reconstruct AudioTree objects
            from flat fields based on manifest metadata. Fields with matching prefixes
            are grouped and assembled into AudioTree objects. Default True.

    Example:
        >>> # With AudioTree reconstruction (default)
        >>> source = MemmapDataSource("dataset/manifest.json")
        >>> sample = source[0]
        >>> print(type(sample["audio"]))  # <class 'AudioTree'>
        >>> print(sample["audio"].audio_data.shape)  # (1, 2, 48000)
        >>> print(sample["audio"].metadata.keys())  # dict_keys(['feature'])

        >>> # Without AudioTree reconstruction (raw arrays)
        >>> source = MemmapDataSource(
        ...     "dataset/manifest.json",
        ...     reconstruct_audiotree=False
        ... )
        >>> sample = source[0]
        >>> print(sample["audio_audio_data"].shape)  # (1, 2, 48000) - flat fields

        >>> # With train/val/test split
        >>> train_source = MemmapDataSource(
        ...     "dataset/manifest.json",
        ...     split="train",
        ...     split_ratios=(0.8, 0.1, 0.1),
        ... )
        >>> val_source = MemmapDataSource(
        ...     "dataset/manifest.json",
        ...     split="val",
        ...     split_ratios=(0.8, 0.1, 0.1),
        ... )

        >>> # With field filtering (load only specific fields)
        >>> source = MemmapDataSource(
        ...     "dataset/manifest.json",
        ...     fields=["audio_audio_data", "labels"]
        ... )

        >>> # With transform
        >>> source = MemmapDataSource(
        ...     "dataset/manifest.json",
        ...     transform_fn=lambda x: {**x, "volume_scaled": x["audio"].audio_data * 0.5}
        ... )
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        num_records: Optional[int] = None,
        fields: Optional[List[str]] = None,
        transform_fn: Optional[Callable[[Dict], Dict]] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        split_seed: int = 42,
        load_into_memory: bool = False,
        reconstruct_audiotree: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_dir = self.manifest_path.parent
        self.transform_fn = transform_fn
        self.reconstruct_audiotree = reconstruct_audiotree

        # Load manifest
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        # Load AudioTree metadata from manifest
        self._audiotree_fields = self.manifest.get("audiotree_fields", {})
        self._sample_rate = self.manifest.get("sample_rate", 48000)

        total_samples = self.manifest["num_samples"]

        # Compute split indices if split is specified
        if split is not None:
            # Validate split_ratios
            if len(split_ratios) != 3:
                raise ValueError("split_ratios must have exactly 3 values (train, val, test)")
            if abs(sum(split_ratios) - 1.0) > 1e-6:
                raise ValueError(f"split_ratios must sum to 1.0, got {sum(split_ratios)}")

            # Create reproducible shuffled indices
            rng = np.random.default_rng(split_seed)
            all_indices = rng.permutation(total_samples)

            # Compute split boundaries
            train_end = int(total_samples * split_ratios[0])
            val_end = train_end + int(total_samples * split_ratios[1])

            # Select indices for the requested split
            if split == "train":
                self._indices = all_indices[:train_end]
            elif split == "val":
                self._indices = all_indices[train_end:val_end]
            elif split == "test":
                self._indices = all_indices[val_end:]
            else:
                raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

            self._num_samples = len(self._indices)
        else:
            # No split - use all samples in order
            self._indices = None
            self._num_samples = total_samples

        # Apply num_records limit (after split)
        if num_records is not None:
            self._num_samples = min(self._num_samples, num_records)
            if self._indices is not None:
                self._indices = self._indices[:self._num_samples]

        # Determine which fields to load
        all_fields = set(self.manifest["fields"].keys())
        if fields is not None:
            missing = set(fields) - all_fields
            if missing:
                raise ValueError(f"Unknown fields: {missing}. Available: {all_fields}")
            self._fields = fields
        else:
            self._fields = list(all_fields)

        # Store field info for lazy memmap creation
        self._field_info = {}
        for name in self._fields:
            info = self.manifest["fields"][name]
            self._field_info[name] = {
                "filepath": self.data_dir / info["filename"],
                "dtype": np.dtype(info["dtype"]),
                "shape": tuple(info["shape"]),
            }

        # Load string fields into memory (usually small)
        self._string_data = {}
        for name, filename in self.manifest.get("string_fields", {}).items():
            json_path = self.data_dir / filename
            if json_path.exists():
                with open(json_path) as f:
                    self._string_data[name] = json.load(f)

        # Optionally load all data into memory for faster access
        self._load_into_memory = load_into_memory
        self._in_memory_data = {}
        if load_into_memory:
            for name, info in self._field_info.items():
                mm = np.memmap(
                    info["filepath"],
                    dtype=info["dtype"],
                    mode="r",
                    shape=info["shape"],
                )
                self._in_memory_data[name] = np.array(mm)
                del mm

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return self._num_samples

    def __getitem__(self, record_key: SupportsIndex) -> Dict[str, np.ndarray]:
        """Load a single sample by index.

        Following best practices, we recreate the memmap on each access
        to avoid memory leaks with long-running data loaders.

        Args:
            record_key: Index of the record to load

        Returns:
            Dictionary mapping field names to numpy arrays with batch dimension

        Raises:
            IndexError: If index is out of range
        """
        idx = int(record_key)
        if idx < 0 or idx >= self._num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self._num_samples})")

        # Map through split indices if using splits
        actual_idx = int(self._indices[idx]) if self._indices is not None else idx

        sample = {}

        if self._load_into_memory:
            # Use preloaded in-memory data
            for name in self._field_info:
                sample[name] = self._in_memory_data[name][actual_idx][np.newaxis, ...]
        else:
            # Recreate memmap on each access to avoid memory leak
            for name, info in self._field_info.items():
                mm = np.memmap(
                    info["filepath"],
                    dtype=info["dtype"],
                    mode="r",
                    shape=info["shape"],
                )
                # Copy data to regular ndarray and add batch dimension
                sample[name] = np.array(mm[actual_idx])[np.newaxis, ...]
                del mm

        # Add string data
        for name, str_list in self._string_data.items():
            if actual_idx < len(str_list):
                sample[name] = str_list[actual_idx]

        # Reconstruct AudioTree objects if requested
        if self.reconstruct_audiotree and self._audiotree_fields:
            sample = self._reconstruct_audiotrees(sample)

        # Apply transform if provided
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)

        return sample

    def _reconstruct_audiotrees(self, raw_sample: Dict) -> Dict:
        """Reconstruct AudioTree objects from flat fields.

        Args:
            raw_sample: Dict with flat fields (AudioTree fields prefixed)

        Returns:
            Dict with AudioTree objects reconstructed and non-AudioTree fields preserved
        """
        _ensure_audiotree_imported()

        if not HAS_AUDIOTREE:
            return raw_sample

        output = {}

        # Reconstruct each AudioTree
        for tree_name in self._audiotree_fields:
            reconstructed = AudioTreeFieldExtractor.reconstruct_audiotree(
                raw_sample,
                tree_name,
                self._sample_rate
            )
            output[tree_name] = reconstructed

        # Add non-AudioTree fields (remove batch dimension)
        for field_name, value in raw_sample.items():
            # Skip if this field belongs to an AudioTree
            if any(field_name.startswith(f"{tree}_") for tree in self._audiotree_fields):
                continue
            # Remove batch dimension for consistency
            if isinstance(value, np.ndarray):
                output[field_name] = value[0]
            else:
                output[field_name] = value

        return output

    def get_slice(
        self,
        start: int,
        end: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load a slice of samples efficiently.

        This is more efficient than calling __getitem__ in a loop
        as it reads contiguous data from the memmap.

        Note: This method is not supported when using splits, since the
        indices are shuffled and not contiguous in the memmap.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            fields: Optional list of field names to load (default: all loaded fields)

        Returns:
            Dictionary mapping field names to numpy arrays with batch dimension

        Raises:
            NotImplementedError: If called when a split is active
        """
        if self._indices is not None:
            raise NotImplementedError(
                "get_slice() is not supported when using splits. "
                "Use __getitem__() in a loop instead, or use split=None."
            )

        if start < 0 or end > self._num_samples or start >= end:
            raise IndexError(
                f"Invalid slice [{start}:{end}] for dataset of size {self._num_samples}"
            )

        target_fields = fields if fields is not None else self._fields

        sample = {}
        for name in target_fields:
            if name not in self._field_info:
                raise ValueError(f"Unknown field: {name}")

            if self._load_into_memory:
                sample[name] = self._in_memory_data[name][start:end]
            else:
                info = self._field_info[name]
                mm = np.memmap(
                    info["filepath"],
                    dtype=info["dtype"],
                    mode="r",
                    shape=info["shape"],
                )
                sample[name] = np.array(mm[start:end])
                del mm

        # Add string data slice
        for name, str_list in self._string_data.items():
            sample[name] = str_list[start:end]

        return sample

    def get_metadata(self) -> Dict:
        """Get the manifest metadata.

        Returns:
            Dictionary containing manifest metadata (excludes field specs)
        """
        metadata = dict(self.manifest)
        metadata.pop("fields", None)
        metadata.pop("string_fields", None)
        return metadata

    def get_field_info(self, field_name: str) -> Dict:
        """Get information about a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Dictionary with dtype, shape, and filepath

        Raises:
            ValueError: If field doesn't exist
        """
        if field_name not in self._field_info:
            raise ValueError(
                f"Unknown field: {field_name}. Available: {list(self._field_info.keys())}"
            )
        return dict(self._field_info[field_name])

    @property
    def fields(self) -> List[str]:
        """Get list of available field names."""
        return list(self._field_info.keys())

    @property
    def string_fields(self) -> List[str]:
        """Get list of available string field names."""
        return list(self._string_data.keys())

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        **kwargs,
    ) -> "MemmapDataSource":
        """Convenience constructor that finds manifest.json in directory.

        Args:
            directory: Directory containing memmap files and manifest.json
            **kwargs: Additional arguments passed to MemmapDataSource

        Returns:
            MemmapDataSource configured for the directory
        """
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return cls(manifest_path, **kwargs)
