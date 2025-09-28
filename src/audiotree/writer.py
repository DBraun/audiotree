"""AudioWriter class for writing AudioTree objects to disk with manifest support."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import soundfile

from .core import AudioTree


class AudioWriter:
    """Write AudioTree objects sequentially to disk with optional manifest generation.

    The AudioWriter provides a stateful way to write multiple AudioTree objects,
    maintaining consistent naming and optionally generating manifest files that
    track all written audio files and their metadata.

    Args:
        output_dir: Directory where audio files will be written
        pattern: Filename pattern with {index} placeholder for sequential numbering
        sample_rate: Optional target sample rate for resampling all audio
        manifest_format: Format for manifest file ("npz" or None to disable)
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
        ...     for tree in audio_trees:
        ...         writer.write(tree)

        >>> # With internal progress bar
        >>> with AudioWriter("output", show_progress=True) as writer:
        ...     for tree in audio_trees:
        ...         writer.write(tree)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        pattern: str = "audio_{index:04d}.wav",
        sample_rate: Optional[int] = None,
        manifest_format: Optional[Literal["npz"]] = "npz",
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
        self.manifest_format = manifest_format
        self.include_timestamp = include_timestamp
        self.compress_manifest = compress_manifest
        self.write_audio = write_audio
        self.index = 0
        self.written_paths = []
        self.manifest_data = []

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

    def write(self, tree: AudioTree, tags: Optional[Dict] = None) -> List[Path]:
        """Write all items in an AudioTree batch to disk.

        Args:
            tree: AudioTree containing one or more audio items in batch dimension
            tags: Optional dictionary of custom metadata to include in manifest

        Returns:
            List of Path objects for all written files
        """
        # Optionally resample if target sample rate specified
        if self.sample_rate and tree.sample_rate != self.sample_rate:
            tree = tree.resample(self.sample_rate)

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
            if self.manifest_format:
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

        # Add optional AudioTree metadata with consistent naming
        # Keep original dtypes - don't convert to Python float
        if tree.loudness is not None and batch_index < len(tree.loudness):
            val = tree.loudness[batch_index]
            # Keep as numpy scalar to preserve dtype
            entry['loudness'] = val if isinstance(val, np.generic) else np.float32(val)

        if tree.pitch is not None and batch_index < len(tree.pitch):
            val = tree.pitch[batch_index]
            entry['pitch'] = val if isinstance(val, np.generic) else np.float32(val)

        if tree.velocity is not None and batch_index < len(tree.velocity):
            val = tree.velocity[batch_index]
            # Velocity should stay as int16
            entry['velocity'] = val if isinstance(val, np.generic) else np.int16(val)

        if tree.note_duration is not None and batch_index < len(tree.note_duration):
            val = tree.note_duration[batch_index]
            entry['note_duration'] = val if isinstance(val, np.generic) else np.float32(val)

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
            Path to the saved manifest file, or None if manifest is disabled
        """
        if not self.manifest_format or not self.manifest_data:
            return None

        manifest_path = self.output_dir / f"manifest.{self.manifest_format}"

        if self.manifest_format == "npz":
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
            values = []
            for entry in self.manifest_data:
                if field in entry:
                    value = entry[field]
                    # Convert strings to object array for proper storage
                    if isinstance(value, str):
                        values.append(value)
                    else:
                        values.append(value)
                else:
                    # Use appropriate default based on field type
                    if field in ['index', 'sample_rate', 'channels', 'samples']:
                        values.append(-1)  # Use -1 as missing value for integers
                    elif field in ['loudness', 'pitch', 'velocity', 'note_duration']:
                        values.append(np.nan)  # Use NaN for floats
                    else:
                        values.append('')  # Empty string for text fields

            # Convert to appropriate numpy array type
            if field in ['filename', 'filepath', 'timestamp']:
                # String fields - use object dtype
                arrays[field] = np.array(values, dtype=object)
            elif field in ['index', 'sample_rate', 'channels', 'samples']:
                # Integer fields
                arrays[field] = np.array(values, dtype=np.int32)
            elif field in ['velocity']:
                # MIDI velocity is typically 0-127, can be stored as int16
                arrays[field] = np.array(values, dtype=np.int16)
            elif field in ['files_written']:
                # Boolean fields
                arrays[field] = np.array(values, dtype=bool)
            elif field.startswith('metadata_'):
                # Metadata fields - preserve original dtype if possible
                # Check the first non-None value to determine dtype
                first_val = None
                for v in values:
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        first_val = v
                        break

                if first_val is not None:
                    if isinstance(first_val, np.ndarray):
                        # For arrays, stack them
                        arrays[field] = np.stack(values)
                    elif isinstance(first_val, str):
                        # String metadata
                        arrays[field] = np.array(values, dtype=object)
                    elif isinstance(first_val, (np.integer, int)):
                        arrays[field] = np.array(values, dtype=np.int32)
                    elif isinstance(first_val, np.floating):
                        arrays[field] = np.array(values, dtype=first_val.dtype)
                    else:
                        # Try to preserve the original type
                        arrays[field] = np.array(values)
                else:
                    # Default to object array if all values are None
                    arrays[field] = np.array(values, dtype=object)
            else:
                # Default float fields
                arrays[field] = np.array(values, dtype=np.float32)

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
            'manifest_format': self.manifest_format,
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
        if self.manifest_format:
            self.save_manifest()

        # Close progress bar if requested
        if self.pbar is not None and self.close_pbar:
            self.pbar.close()