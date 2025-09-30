"""DataSource for reading AudioWriter outputs with manifest support."""

from pathlib import Path
from typing import Dict, List, Literal, Optional, SupportsIndex, Union

import numpy as np
from grain import python as grain

from audiotree import AudioTree


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
        """Load manifest from NPZ format.

        Returns:
            List of manifest entry dictionaries
        """
        data = np.load(self.manifest_path, allow_pickle=True)
        entries = []

        # Get the number of entries from the first array
        num_entries = 0
        for key in data.keys():
            if not key.startswith('tags_'):
                num_entries = len(data[key])
                break

        # Reconstruct entries from arrays
        for i in range(num_entries):
            entry = {}

            # Process regular fields
            for key in data.keys():
                if key.startswith('tags_'):
                    # Handle tag fields separately
                    continue

                value = data[key][i]

                # Handle missing values
                if isinstance(value, (np.integer, int)) and value == -1:
                    continue  # Skip missing integer values
                elif isinstance(value, (np.floating, float)) and np.isnan(value):
                    continue  # Skip NaN values
                elif isinstance(value, (str, np.str_)) and value == '':
                    continue  # Skip empty strings

                # Handle metadata fields - keep them as numpy arrays for batching
                if key.startswith('metadata_'):
                    # Keep metadata arrays as-is for proper batching
                    entry[key] = value
                else:
                    # Convert regular fields
                    if isinstance(value, np.ndarray) and value.size == 1:
                        value = value.item()  # Convert scalar arrays to Python types

                    # Convert numpy types to Python types
                    if isinstance(value, np.integer):
                        entry[key] = int(value)
                    elif isinstance(value, np.floating):
                        entry[key] = float(value)
                    else:
                        entry[key] = value

            # Process tag fields
            tags = {}
            for key in data.keys():
                if key.startswith('tags_'):
                    tag_key = key[5:]  # Remove 'tags_' prefix
                    value = data[key][i]
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

        # Construct audio file path
        filename = entry['filename']
        audio_path = self.audio_dir / filename

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

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

        # Load audio file with AudioTree properties
        tree = AudioTree.from_file(
            audio_path,
            sample_rate=self.sample_rate or entry.get('sample_rate'),
            duration=self.duration,
            mono=self.mono,
            pad_mode=self.pad_mode if self.duration else None,
            metadata=metadata,  # Pass metadata back
            # Pass AudioTree properties directly from manifest
            loudness=entry.get('loudness', None),
            pitch=entry.get('pitch', None),
            velocity=entry.get('velocity', None),
            note_duration=entry.get('note_duration', None),
            codes=entry.get('codes', None),
            latents=entry.get('latents', None),
        )

        return tree

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