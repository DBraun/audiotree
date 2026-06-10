"""DataSource for reading AudioWriter outputs with manifest support."""

from pathlib import Path
from typing import Dict, List, Literal, Optional, SupportsIndex, Union

import numpy as np
from grain import python as grain

from audiotree import AudioTree
from audiotree.writer import _AUDIOTREE_FIELDS


class ManifestDataSource(grain.RandomAccessDataSource):
    """A DataSource that reads audio files based on a manifest file created by AudioWriter.

    This DataSource is designed to work seamlessly with the output of AudioWriter, reading
    audio files and restoring their associated metadata from NPZ manifests.

    NPZ format provides:
    - Efficient binary storage with 20x+ compression vs JSON for large datasets
    - Fast loading without text parsing
    - Exact numeric type preservation
    - Support for both compressed and uncompressed variants

    Args:
        manifest_path: Path to the manifest file (NPZ format)
        audio_dir: Optional directory containing audio files. If None, uses manifest directory
        num_records: Optional limit on number of records to load
        sample_rate: Optional target sample rate for resampling
        mono: Whether to convert audio to mono
        duration: Optional duration to trim/pad audio to (in seconds)
        pad_mode: Padding mode if duration is specified ("constant" or "wrap")
        filter_fn: Optional function to filter manifest entries

    Example:
        >>> # Read from NPZ manifest
        >>> source = ManifestDataSource("output/manifest.npz")
        >>>
        >>> # Filter by metadata
        >>> source = ManifestDataSource(
        ...     "output/manifest.npz",
        ...     filter_fn=lambda entry: entry.get('loudness', -float('inf')) > -20
        ... )
        >>>
        >>> # Use convenience constructor for AudioWriter output
        >>> source = ManifestDataSource.from_writer_output(
        ...     "output_dir",
        ... )
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        audio_dir: Optional[Union[str, Path]] = None,
        num_records: Optional[int] = None,
        sample_rate: Optional[int] = None,
        mono: bool = False,
        duration: Optional[float] = None,
        pad_mode: Literal["constant", "wrap"] = "constant",
        filter_fn: Optional[callable] = None,
    ):
        self.manifest_path = Path(manifest_path)

        # Default audio_dir to manifest directory
        if audio_dir is None:
            self.audio_dir = self.manifest_path.parent
        else:
            self.audio_dir = Path(audio_dir)

        self.sample_rate = sample_rate
        self.mono = mono
        self.duration = duration
        self.pad_mode = pad_mode

        # Load and process manifest
        self.entries = self._load_manifest(filter_fn)

        # Limit records if specified
        if num_records is not None:
            self.entries = self.entries[:num_records]

        self._length = len(self.entries)
        if self._length == 0:
            raise ValueError(f"No valid entries found in manifest: {manifest_path}")

    def _load_manifest(self, filter_fn: Optional[callable] = None) -> List[Dict]:
        """Load manifest file and optionally filter entries.

        Args:
            filter_fn: Optional function to filter entries

        Returns:
            List of manifest entry dictionaries
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Detect format from extension
        suffix = self.manifest_path.suffix.lower()

        if suffix == '.npz':
            entries = self._load_npz_manifest()
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}. Use .npz")

        # Apply filter if provided
        if filter_fn:
            entries = [entry for entry in entries if filter_fn(entry)]

        return entries

    def _load_npz_manifest(self) -> List[Dict]:
        """Load manifest from NPZ format using vectorized operations.

        Returns:
            List of manifest entry dictionaries
        """
        data = np.load(self.manifest_path, allow_pickle=True)

        # Pre-classify keys outside the loop - O(k) instead of O(n×k)
        regular_keys = []
        metadata_keys = []
        tag_keys = []

        for key in data.keys():
            if key.startswith('tags_'):
                tag_keys.append(key)
            elif key.startswith('metadata_'):
                metadata_keys.append(key)
            else:
                regular_keys.append(key)

        # Get number of entries
        if not regular_keys:
            return []
        num_entries = len(data[regular_keys[0]])

        # Pre-fetch all arrays - single numpy operation per key
        regular_arrays = {k: data[k] for k in regular_keys}
        metadata_arrays = {k: data[k] for k in metadata_keys}
        tag_arrays = {k: data[k] for k in tag_keys}

        # Build entries efficiently
        entries = []
        for i in range(num_entries):
            entry = {}

            # Process regular fields
            for key in regular_keys:
                value = regular_arrays[key][i]

                # Handle missing values
                if isinstance(value, (np.integer, int)) and value == -1:
                    continue
                elif isinstance(value, (np.floating, float)) and np.isnan(value):
                    continue
                elif isinstance(value, (str, np.str_)) and value == '':
                    continue

                # Convert scalar arrays and numpy types
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()

                if isinstance(value, np.integer):
                    entry[key] = int(value)
                elif isinstance(value, np.floating):
                    entry[key] = float(value)
                else:
                    entry[key] = value

            # Process metadata fields - keep as-is for batching
            for key in metadata_keys:
                value = metadata_arrays[key][i]
                entry[key] = value

            # Process tag fields
            tags = {}
            for key in tag_keys:
                tag_key = key[5:]  # Remove 'tags_' prefix
                value = tag_arrays[key][i]
                if value is not None and value != '':
                    tags[tag_key] = value

            if tags:
                entry['tags'] = tags

            entries.append(entry)

        return entries

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return self._length

    def __getitem__(self, record_key: SupportsIndex) -> AudioTree:
        """Load an AudioTree for the given record index.

        Args:
            record_key: Index of the record to load

        Returns:
            AudioTree with audio data and restored metadata
        """
        entry = self.entries[int(record_key)]

        # Prepare metadata from manifest
        metadata = {}

        # Add metadata arrays from manifest (these can be batched properly)
        for key, value in entry.items():
            if key.startswith('metadata_'):
                # Remove 'metadata_' prefix and add to metadata
                metadata_key = key[9:]  # len('metadata_') = 9
                # Wrap in array with batch dimension for batching
                if isinstance(value, np.ndarray):
                    # Add batch dimension if needed
                    if value.ndim == 0:
                        metadata[metadata_key] = np.array([value.item()])
                    else:
                        metadata[metadata_key] = value[np.newaxis, ...]  # Add batch dim
                else:
                    # Convert scalar to array with batch dim
                    metadata[metadata_key] = np.array([value])

        # Check if audio files were actually written
        files_written = entry.get('files_written', True)

        if files_written:
            # Audio files exist - load from disk
            filename = entry['filename']
            audio_path = self.audio_dir / filename

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Build kwargs for AudioTree.from_file with all available fields
            tree_kwargs = {
                'sample_rate': self.sample_rate or entry.get('sample_rate'),
                'duration': self.duration,
                'mono': self.mono,
                'pad_mode': self.pad_mode if self.duration else None,
                'metadata': metadata,
            }

            # Add AudioTree fields dynamically from manifest
            for field_name in _AUDIOTREE_FIELDS:
                if field_name in entry:
                    tree_kwargs[field_name] = entry[field_name]

            # Load audio file with all properties
            audio_tree = AudioTree.from_file(audio_path, **tree_kwargs)
        else:
            # No audio files - create AudioTree from manifest metadata only
            sample_rate = self.sample_rate or entry.get('sample_rate')
            channels = entry.get('channels', 1)
            samples = entry.get('samples', 0)

            # Create zero audio data with correct shape
            waveform = np.zeros((1, channels, samples), dtype=np.float32)

            # Build kwargs for AudioTree.create
            tree_kwargs = {
                'sample_rate': sample_rate,
                'metadata': metadata,
            }

            # Add AudioTree fields dynamically from manifest
            for field_name in _AUDIOTREE_FIELDS:
                if field_name in entry:
                    # Wrap scalar values in array with batch dimension
                    value = entry[field_name]
                    if isinstance(value, (np.ndarray, list)):
                        tree_kwargs[field_name] = np.array([value])
                    else:
                        tree_kwargs[field_name] = np.array([value])

            # Create AudioTree with zero audio data
            audio_tree = AudioTree.create(waveform, **tree_kwargs)

        return audio_tree

    def get_entry(self, index: int) -> Dict:
        """Get the raw manifest entry for a given index.

        Args:
            index: Index of the entry

        Returns:
            Dictionary containing the manifest entry
        """
        return self.entries[index]

    def get_all_entries(self) -> List[Dict]:
        """Get all manifest entries.

        Returns:
            List of all manifest entry dictionaries
        """
        return self.entries.copy()

    def filter_by_tag(self, tag_name: str, tag_value) -> 'ManifestDataSource':
        """Create a new ManifestDataSource filtered by a specific tag value.

        Args:
            tag_name: Name of the tag to filter by
            tag_value: Value the tag must have

        Returns:
            New ManifestDataSource with filtered entries
        """
        def filter_fn(entry):
            tags = entry.get('tags', {})
            return tags.get(tag_name) == tag_value

        return ManifestDataSource(
            manifest_path=self.manifest_path,
            audio_dir=self.audio_dir,
            sample_rate=self.sample_rate,
            mono=self.mono,
            duration=self.duration,
            pad_mode=self.pad_mode,
            filter_fn=filter_fn
        )

    def filter_by_loudness(self, min_lufs: float = None, max_lufs: float = None) -> 'ManifestDataSource':
        """Create a new ManifestDataSource filtered by loudness range.

        Filters entries based on the 'loudness' field in the manifest.
        Works with manifests created by AudioWriter in NPZ format.

        Args:
            min_lufs: Minimum loudness in LUFS (inclusive)
            max_lufs: Maximum loudness in LUFS (inclusive)

        Returns:
            New ManifestDataSource with filtered entries

        Example:
            >>> # Keep only samples louder than -20 dB
            >>> loud_source = source.filter_by_loudness(min_lufs=-20.0)
            >>>
            >>> # Keep samples in specific loudness range
            >>> mid_source = source.filter_by_loudness(min_lufs=-30.0, max_lufs=-15.0)
        """
        def filter_fn(entry):
            loudness = entry.get('loudness')
            if loudness is None:
                return False
            if min_lufs is not None and loudness < min_lufs:
                return False
            if max_lufs is not None and loudness > max_lufs:
                return False
            return True

        return ManifestDataSource(
            manifest_path=self.manifest_path,
            audio_dir=self.audio_dir,
            sample_rate=self.sample_rate,
            mono=self.mono,
            duration=self.duration,
            pad_mode=self.pad_mode,
            filter_fn=filter_fn
        )

    @classmethod
    def from_writer_output(
        cls,
        output_dir: Union[str, Path],
        **kwargs
    ) -> 'ManifestDataSource':
        """Convenience constructor for reading AudioWriter output.

        This method automatically locates the manifest file in the output directory
        based on the specified format and creates a ManifestDataSource configured
        to read the audio files and metadata.

        Args:
            output_dir: Directory containing AudioWriter output
            **kwargs: Additional arguments passed to ManifestDataSource
                     (e.g., sample_rate, mono, note_duration, filter_fn)

        Returns:
            ManifestDataSource configured for the output directory

        Example:
            >>> # Read NPZ manifest
            >>> source = ManifestDataSource.from_writer_output("output")
            >>>
            >>> # Read with resampling
            >>> source = ManifestDataSource.from_writer_output(
            ...     "output",
            ...     sample_rate=16000
            ... )
        """
        output_dir = Path(output_dir)
        manifest_path = output_dir / f"manifest.npz"
        return cls(manifest_path=manifest_path, audio_dir=output_dir, **kwargs)