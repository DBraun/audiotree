"""AudioWriter class for writing AudioTree objects to disk with manifest support."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import soundfile

from .core import AudioTree


# AudioTree fields that should be tracked (excluding audio_data, sample_rate, metadata)
_AUDIOTREE_FIELDS = ['loudness', 'pitch', 'velocity', 'note_duration', 'codes', 'latents']


class AudioWriter:
    """Write AudioTree objects sequentially to disk with optional manifest generation.

    The AudioWriter provides a stateful way to write multiple AudioTree objects,
    maintaining consistent naming and optionally generating manifest files that
    track all written audio files and their metadata.

    Args:
        output_dir: Directory where audio files will be written
        pattern: Filename pattern with {index} placeholder for sequential numbering
        sample_rate: Optional target sample rate for resampling all audio
        include_timestamp: Whether to include timestamps in manifest entries
        compress_manifest: Whether to compress NPZ manifest files (only applies to npz format)
        write_audio: Whether to write audio files to disk (default True). When False,
            only manifest is generated with metadata
        pbar: Optional tqdm progress bar instance to update during writing
        close_pbar: Whether to close the progress bar on exit (default False)
        show_progress: Create an internal tqdm progress bar (requires tqdm installed)
        progress_desc: Description for internal progress bar (default "Writing audio")

    Example:
        >>> # With external progress bar
        >>> from tqdm import tqdm
        >>> pbar = tqdm(total=100, desc="Processing")
        >>> with AudioWriter("output", pbar=pbar) as writer:
        ...     for audio_tree in audio_trees:
        ...         writer.write(audio_tree)

        >>> # With internal progress bar
        >>> with AudioWriter("output", show_progress=True) as writer:
        ...     for audio_tree in audio_trees:
        ...         writer.write(audio_tree)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        pattern: str = "audio_{index:04d}.wav",
        sample_rate: Optional[int] = None,
        include_timestamp: bool = False,
        compress_manifest: bool = True,
        write_audio: bool = True,
        pbar: Optional[Any] = None,
        close_pbar: bool = False,
        show_progress: bool = False,
        progress_desc: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pattern = pattern
        self.sample_rate = sample_rate
        self.include_timestamp = include_timestamp
        self.compress_manifest = compress_manifest
        self.write_audio = write_audio
        self.index = 0
        self.written_paths = []
        self.manifest_data = []
        self._expected_fields = None  # Track which AudioTree fields should be present

        # Progress bar support
        self.pbar = pbar
        self.close_pbar = close_pbar
        self.show_progress = show_progress
        self.progress_desc = progress_desc or "Writing audio"
        self._internal_pbar = None

        # Create internal progress bar if requested
        if self.show_progress and self.pbar is None:
            try:
                from tqdm import tqdm
                self._internal_pbar = tqdm(desc=self.progress_desc, unit="files")
                self.pbar = self._internal_pbar
                self.close_pbar = True  # Always close internal progress bars
            except ImportError:
                # tqdm not available, silently continue without progress bar
                pass

    def _get_present_fields(self, tree: AudioTree) -> set:
        """Get the set of AudioTree fields that are not None.

        Args:
            tree: AudioTree to inspect

        Returns:
            Set of field names that are present (not None)
        """
        present = set()
        for field_name in _AUDIOTREE_FIELDS:
            if getattr(tree, field_name, None) is not None:
                present.add(field_name)
        return present

    def write(self, tree: AudioTree, tags: Optional[Dict] = None) -> List[Path]:
        """Write all items in an AudioTree batch to disk.

        Args:
            tree: AudioTree containing one or more audio items in batch dimension
            tags: Optional dictionary of custom metadata to include in manifest

        Returns:
            List of Path objects for all written files

        Raises:
            ValueError: If AudioTree fields don't match previously written trees
        """
        # Optionally resample if target sample rate specified
        if self.sample_rate and tree.sample_rate != self.sample_rate:
            tree = tree.resample(self.sample_rate)

        # Validate field consistency
        present_fields = self._get_present_fields(tree)
        if self._expected_fields is None:
            # First write - record which fields are present
            self._expected_fields = present_fields
        else:
            # Subsequent writes - validate fields match
            if present_fields != self._expected_fields:
                missing = self._expected_fields - present_fields
                extra = present_fields - self._expected_fields
                error_parts = []
                if missing:
                    error_parts.append(f"missing fields: {sorted(missing)}")
                if extra:
                    error_parts.append(f"extra fields: {sorted(extra)}")
                raise ValueError(
                    f"AudioTree fields don't match previous writes. "
                    f"{', '.join(error_parts)}. "
                    f"All AudioTrees written to the same manifest must have consistent fields."
                )

        batch_size = tree.audio_data.shape[0]
        paths = []

        for i in range(batch_size):
            # Generate filename
            filename = self.pattern.format(index=self.index)
            filepath = self.output_dir / filename

            # Write audio file if requested
            if self.write_audio:
                # Convert to numpy and transpose for soundfile (channels, samples) -> (samples, channels)
                audio = np.array(tree.audio_data[i].T)
                soundfile.write(filepath, audio, tree.sample_rate)
                self.written_paths.append(filepath)

            paths.append(filepath)

            # Collect manifest entry
            entry = self._create_manifest_entry(tree, i, filename, tags)
            self.manifest_data.append(entry)

            self.index += 1

        # Update progress bar if available
        if self.pbar is not None:
            self.pbar.update(batch_size)

        return paths

    def _create_manifest_entry(
        self,
        tree: AudioTree,
        batch_index: int,
        filename: str,
        tags: Optional[Dict] = None
    ) -> Dict:
        """Create a manifest entry for a single audio file.

        Args:
            tree: Source AudioTree
            batch_index: Index within the batch
            filename: Output filename
            tags: Optional custom metadata

        Returns:
            Dictionary containing manifest entry data
        """
        entry = {
            'index': np.int32(self.index),
            'filename': filename,
            'sample_rate': np.int32(tree.sample_rate),
            'channels': np.int32(tree.audio_data.shape[1]),
            'samples': np.int32(tree.audio_data.shape[2]),
            'duration_seconds': np.float32(tree.audio_data.shape[2] / tree.sample_rate),
            'files_written': self.write_audio
        }

        # Add timestamp only if requested
        if self.include_timestamp:
            entry['timestamp'] = datetime.now().isoformat()

        # Add AudioTree fields dynamically, preserving dtypes
        for field_name in _AUDIOTREE_FIELDS:
            field_value = getattr(tree, field_name, None)
            if field_value is not None:
                # Extract the value for this batch index
                if isinstance(field_value, np.ndarray):
                    if field_value.ndim > 0 and batch_index < len(field_value):
                        val = field_value[batch_index]
                        # Keep as numpy scalar to preserve dtype
                        entry[field_name] = val if isinstance(val, np.generic) else np.array(val, dtype=field_value.dtype)
                    elif field_value.ndim == 0:
                        # Scalar array
                        entry[field_name] = field_value
                else:
                    # Non-array value (shouldn't normally happen for these fields)
                    entry[field_name] = field_value

        # Add source filepath if available (consistent naming)
        filepaths = tree.filepath
        if filepaths and batch_index < len(filepaths):
            entry['filepath'] = filepaths[batch_index]

        # Add custom tags
        if tags:
            entry['tags'] = tags

        # Add metadata arrays if present
        if tree.metadata:
            for key, value in tree.metadata.items():
                # Skip certain internal metadata keys that shouldn't be saved
                if key in ['filepath', 'offset', 'duration', 'manifest_index']:
                    continue

                # For arrays in metadata, extract the batch_index element
                if isinstance(value, np.ndarray):
                    if value.ndim > 0 and len(value) > batch_index:
                        # Save the value for this batch item
                        entry[f'metadata_{key}'] = value[batch_index]
                    elif value.ndim == 0:
                        # Scalar array
                        entry[f'metadata_{key}'] = value
                # For non-array metadata, only include if it's meant to be saved
                elif not isinstance(value, (dict, list)) or key == 'tags':
                    # Skip complex objects unless they're tags
                    if key != 'tags':  # tags already handled above
                        entry[f'metadata_{key}'] = value

        return entry

    def save_manifest(self) -> Optional[Path]:
        """Save the manifest file to disk.

        Returns:
            Path to the saved manifest file, or None if no data to save
        """
        if not self.manifest_data:
            return None

        manifest_path = self.output_dir / f"manifest.npz"

        # Convert manifest data to arrays for efficient NPZ storage
        arrays_dict = self._manifest_to_arrays()

        # Save as compressed or uncompressed NPZ
        if self.compress_manifest:
            np.savez_compressed(manifest_path, **arrays_dict)
        else:
            np.savez(manifest_path, **arrays_dict)

        return manifest_path

    def _manifest_to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert manifest data to numpy arrays for NPZ storage.

        Returns:
            Dictionary of numpy arrays ready for NPZ storage
        """
        if not self.manifest_data:
            return {}

        # Collect all unique fields across entries
        all_fields = set()
        for entry in self.manifest_data:
            all_fields.update(entry.keys())

        # Separate scalar fields from tag fields
        scalar_fields = {f for f in all_fields if f != 'tags'}

        # Initialize result dictionary
        arrays = {}

        # Process scalar fields
        for field in scalar_fields:
            # Collect values from all entries
            values = [entry[field] for entry in self.manifest_data]

            # Infer dtype from first value
            first_val = values[0]

            # Convert to appropriate numpy array type
            if isinstance(first_val, str):
                # String fields - use object dtype
                arrays[field] = np.array(values, dtype=object)
            elif isinstance(first_val, bool):
                # Boolean fields
                arrays[field] = np.array(values, dtype=bool)
            elif isinstance(first_val, np.ndarray):
                # Array fields (e.g., metadata arrays, codes, latents)
                arrays[field] = np.stack(values)
            elif isinstance(first_val, (np.generic, int, float)):
                # Numeric scalars - preserve dtype
                if isinstance(first_val, np.generic):
                    # numpy scalar - use its dtype
                    arrays[field] = np.array(values, dtype=first_val.dtype)
                else:
                    # Python scalar - convert to numpy type
                    if isinstance(first_val, int):
                        arrays[field] = np.array(values, dtype=np.int32)
                    else:
                        arrays[field] = np.array(values, dtype=np.float32)
            else:
                # Fallback for other types
                arrays[field] = np.concatenate(values, axis=0)

        # Process tags if present
        if any('tags' in entry for entry in self.manifest_data):
            # Collect all unique tag keys
            all_tag_keys = set()
            for entry in self.manifest_data:
                if 'tags' in entry and isinstance(entry['tags'], dict):
                    all_tag_keys.update(entry['tags'].keys())

            # Store each tag as a separate array
            for tag_key in all_tag_keys:
                tag_values = []
                for entry in self.manifest_data:
                    if 'tags' in entry and tag_key in entry['tags']:
                        tag_values.append(entry['tags'][tag_key])
                    else:
                        tag_values.append(None)

                # Store with 'tags_' prefix
                arrays[f'tags_{tag_key}'] = np.array(tag_values, dtype=object)

        return arrays

    def get_stats(self) -> Dict:
        """Get statistics about written files.

        Returns:
            Dictionary containing write statistics
        """
        stats = {
            'total_files': len(self.written_paths),
            'output_directory': str(self.output_dir),
            'current_index': self.index,
            'write_audio': self.write_audio
        }

        # Only include batch count if timestamps are being tracked
        if self.include_timestamp:
            stats['total_batches'] = len(set(entry.get('timestamp', '') for entry in self.manifest_data))

        return stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - calls close() to save manifest and close progress bar."""
        self.close()

    def close(self):
        """Manually close the writer, save manifest, and close progress bar."""
        self.save_manifest()

        # Close progress bar if requested
        if self.pbar is not None and self.close_pbar:
            self.pbar.close()