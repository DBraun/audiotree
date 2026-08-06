"""AudioWriter class for writing AudioTree objects to disk with manifest support."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import soundfile

from . import _format
from ._fs import refuse_to_clobber
from .core import LABEL_FIELDS, AudioTree

# Manifest columns hold one value per written item. Metadata keys that describe
# *where* an item came from are reconstructed by the reader instead.
_SKIPPED_METADATA_KEYS = frozenset(
    {"filepath", "offset", "duration", "manifest_index", "tags"}
)


def _column_value(name: str, value: Any, batch_index: int) -> Any:
    """Pick the ``batch_index``-th row out of a per-item manifest column.

    ``value`` arrives as whatever the caller put on the AudioTree: a
    ``jax.Array``, a NumPy array, a list, or a scalar. Dispatching on
    ``isinstance(value, np.ndarray)`` missed every array-like that is not a
    NumPy array -- a ``jax.Array`` being the common case -- and stored the whole
    batch in *every* row, so labels ended up bound to the wrong audio. Coerce
    first, then let one code path handle all of them.

    Raises:
        ValueError: If *value* is neither a scalar nor an array-like covering
            the batch.
    """
    # Scalars label the whole batch, so they are stored verbatim in every row.
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value

    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(
            f"Manifest column {name!r} holds a {type(value).__name__}, which has no "
            f"manifest representation. Columns must be scalars, strings, or "
            f"array-likes with one row per batch item."
        )
    if array.ndim == 0:
        return array
    if batch_index >= len(array):
        raise ValueError(
            f"Manifest column {name!r} has {len(array)} rows, too few for batch "
            f"index {batch_index}. Every per-item column must cover the batch."
        )
    return array[batch_index]


class AudioWriter:
    """Write AudioTree objects sequentially to disk with optional manifest generation.

    The AudioWriter provides a stateful way to write multiple AudioTree objects,
    maintaining consistent naming and optionally generating manifest files that
    track all written audio files and their metadata.

    Args:
        directory: Directory where audio files will be written
        pattern: Filename pattern with {index} placeholder for sequential numbering
        include_timestamp: Whether to include timestamps in manifest entries
        compress_manifest: Whether to compress NPZ manifest files (only applies to npz format)
        write_audio: Whether to write audio files to disk (default True). When False,
            only manifest is generated with metadata
        subtype: Optional soundfile subtype string (e.g. ``"PCM_16"``, ``"PCM_24"``,
            ``"FLOAT"``). Forwarded to ``soundfile.write``. When ``None`` (default),
            soundfile picks its format default — ``PCM_16`` for WAV. Use ``"PCM_24"``
            or ``"FLOAT"`` when writing quiet / high-dynamic-range material that will
            be read back after further processing, to avoid 16-bit quantization
            artifacts.
        pbar: Optional tqdm progress bar instance to update during writing
        close_pbar: Whether to close the progress bar on exit (default False)
        show_progress: Create an internal tqdm progress bar. Raises ``ImportError``
            if tqdm is not installed — install it with ``audiotree[progress]``.
        progress_desc: Description for internal progress bar (default "Writing audio")
        exist_ok: Whether to write into a directory that already holds a manifest.
            Defaults to ``False``, which raises ``FileExistsError`` rather than
            overwriting an existing dataset (and catches two writers aimed at one
            directory). Pass ``True`` to deliberately overwrite or append.
        manifest_every: Rewrite ``manifest.npz`` once this many entries have
            accumulated since the last write, so a run killed after 100k files
            leaves a manifest describing (nearly) all of them instead of none.
            Each rewrite costs one pass over every entry so far, hence the
            throttle; pass ``0`` to write the manifest only at ``close()``.

    Example:
        Write a handful of (silent, one-second mono) ``AudioTree`` objects to a
        temporary directory. A ``manifest.npz`` is refreshed every
        ``manifest_every`` entries and finalized on context exit.

        >>> import tempfile
        >>> from pathlib import Path
        >>> from audiotree import AudioTree
        >>> import jax.numpy as jnp
        >>> out_dir = tempfile.mkdtemp()
        >>> audio_trees = [AudioTree.create(jnp.zeros((1, 44100)), 44100) for _ in range(3)]
        >>> with AudioWriter(out_dir) as writer:
        ...     for audio_tree in audio_trees:
        ...         _ = writer.write(audio_tree)
        >>> sorted(p.name for p in Path(out_dir).glob("*"))
        ['audio_0000.wav', 'audio_0001.wav', 'audio_0002.wav', 'manifest.npz']

        Writing again to a directory that already holds a manifest raises, so a
        finished dataset is never silently overwritten:

        >>> AudioWriter(out_dir)
        Traceback (most recent call last):
            ...
        FileExistsError: ... already contains a dataset (manifest.npz). ...

        Pass an external progress bar with ``pbar=...``, or have the writer
        create its own with ``show_progress=True``. ``exist_ok=True`` opts in to
        reusing the directory:

        >>> from tqdm import tqdm  # doctest: +SKIP
        >>> pbar = tqdm(total=len(audio_trees), desc="Processing")  # doctest: +SKIP
        >>> with AudioWriter(out_dir, pbar=pbar, exist_ok=True) as writer:  # doctest: +SKIP
        ...     for audio_tree in audio_trees:
        ...         _ = writer.write(audio_tree)
        >>> with AudioWriter(out_dir, show_progress=True, exist_ok=True) as writer:  # doctest: +SKIP
        ...     for audio_tree in audio_trees:
        ...         _ = writer.write(audio_tree)
    """

    def __init__(
        self,
        directory: Union[str, Path] = ".",
        *,
        pattern: str = "audio_{index:04d}.wav",
        include_timestamp: bool = False,
        compress_manifest: bool = True,
        write_audio: bool = True,
        subtype: Optional[str] = None,
        pbar: Optional[Any] = None,
        close_pbar: bool = False,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        exist_ok: bool = False,
        manifest_every: int = 1000,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not exist_ok:
            refuse_to_clobber(self.directory, ("manifest.npz", "manifest.json"))
        self.exist_ok = exist_ok
        self.pattern = pattern
        # Inferred from the first written tree; every later write must match it.
        self.sample_rate = None
        self.include_timestamp = include_timestamp
        self.compress_manifest = compress_manifest
        self.write_audio = write_audio
        self.subtype = subtype
        self.index = 0
        self.written_paths = []
        self.manifest_data = []
        self._expected_fields = None  # Track which AudioTree fields should be present
        self.manifest_every = manifest_every
        self._manifest_index = 0  # self.index as of the last manifest write

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
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm, which is not installed. "
                    'Install it with `pip install "audiotree[progress]"`.'
                ) from exc

    def _get_present_fields(self, tree: AudioTree) -> set:
        """Get the set of AudioTree fields that are not None.

        Args:
            tree: AudioTree to inspect

        Returns:
            Set of field names that are present (not None)
        """
        present = set()
        for field_name in LABEL_FIELDS:
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
        # Take the sample rate from the first written tree and require every later
        # tree to match it (resample beforehand with AudioTree.resample if needed).
        if self.sample_rate is None:
            self.sample_rate = tree.sample_rate
        elif tree.sample_rate != self.sample_rate:
            raise ValueError(
                f"AudioTree sample_rate {tree.sample_rate} does not match the "
                f"writer's sample_rate {self.sample_rate} (set by the first write). "
                f"All AudioTrees written to one manifest must share a sample rate; "
                f"resample beforehand with AudioTree.resample()."
            )

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

        batch_size = tree.waveform.shape[0]
        paths = []

        for i in range(batch_size):
            # Generate filename
            # todo: need a way to pass more kwargs to this formatter
            filename = self.pattern.format(index=self.index)
            filepath = self.directory / filename

            # Collect the manifest entry first: a column this writer cannot
            # store raises, and doing that before the WAV exists keeps the
            # directory free of audio no manifest row points at.
            entry = self._create_manifest_entry(tree, i, filename, tags)

            # Write audio file if requested
            if self.write_audio:
                # Convert to numpy and transpose for soundfile (channels, samples) -> (samples, channels)
                audio = np.array(tree.waveform[i].T)
                soundfile.write(
                    str(filepath), audio, tree.sample_rate, subtype=self.subtype
                )
                self.written_paths.append(filepath)

            paths.append(filepath)
            self.manifest_data.append(entry)

            self.index += 1

        # Update progress bar if available
        if self.pbar is not None:
            self.pbar.update(batch_size)

        # Publish the manifest as writing progresses, so a killed run leaves a
        # readable dataset rather than a directory of unindexed WAVs.
        if (
            self.manifest_every
            and self.index - self._manifest_index >= self.manifest_every
        ):
            self.save_manifest()

        return paths

    def _create_manifest_entry(
        self,
        tree: AudioTree,
        batch_index: int,
        filename: str,
        tags: Optional[Dict] = None,
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
            "index": np.int32(self.index),
            "filename": filename,
            "sample_rate": np.int32(tree.sample_rate),
            "channels": np.int32(tree.waveform.shape[1]),
            "samples": np.int32(tree.waveform.shape[2]),
            "duration_seconds": np.float32(tree.waveform.shape[2] / tree.sample_rate),
            "files_written": self.write_audio,
        }

        # Add timestamp only if requested
        if self.include_timestamp:
            entry["timestamp"] = datetime.now().isoformat()

        # Add AudioTree fields dynamically, preserving dtypes
        for field_name in LABEL_FIELDS:
            field_value = getattr(tree, field_name, None)
            if field_value is not None:
                entry[field_name] = _column_value(field_name, field_value, batch_index)

        # Add source filepath if available (consistent naming)
        filepaths = tree.filepath
        if filepaths and batch_index < len(filepaths):
            entry["filepath"] = filepaths[batch_index]

        # Add custom tags
        if tags:
            entry["tags"] = tags

        # Add metadata arrays if present
        if tree.metadata:
            for key, value in tree.metadata.items():
                # Skip internal keys, and `tags`, which is handled above.
                if key in _SKIPPED_METADATA_KEYS:
                    continue
                # A nested dict has no NPZ column representation; write nested
                # pytrees with TreeWriter instead.
                if isinstance(value, dict):
                    continue
                column = f"metadata_{key}"
                entry[column] = _column_value(column, value, batch_index)

        return entry

    def save_manifest(self) -> Optional[Path]:
        """Write ``manifest.npz`` atomically (temp file + rename).

        A reader -- or a retry after a crash -- sees either the previous
        manifest or the new one, never the half-written NPZ that a kill during
        ``savez`` would otherwise leave behind (which reads as a corrupt zip and
        blocks re-rendering with "already contains a dataset").

        Returns:
            Path to the saved manifest file, or None if no data to save
        """
        if not self.manifest_data:
            return None

        manifest_path = self.directory / "manifest.npz"

        # Convert manifest data to arrays for efficient NPZ storage
        arrays_dict = self._manifest_to_arrays()

        # Stamp the format header. NPZ has no place for scalars, so each value
        # is a 0-d array under the reserved `__audiotree_` prefix, which the
        # reader strips before classifying the per-entry columns.
        for key, value in _format.header(_format.MANIFEST).items():
            arrays_dict[f"{_format.NPZ_HEADER_PREFIX}{key}"] = np.array(
                json.dumps(value)
            )

        # Save as compressed or uncompressed NPZ. The temp name is dotted so it
        # does not look like a dataset to `refuse_to_clobber`.
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        with open(tmp_path, "wb") as f:
            if self.compress_manifest:
                np.savez_compressed(f, **arrays_dict)
            else:
                np.savez(f, **arrays_dict)
        os.replace(tmp_path, manifest_path)

        self._manifest_index = self.index
        return manifest_path

    def _manifest_to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert manifest data to numpy arrays for NPZ storage.

        Returns:
            Dictionary of numpy arrays ready for NPZ storage
        """
        if not self.manifest_data:
            return {}

        n_entries = len(self.manifest_data)

        # Collect all unique fields across entries. Sorting matters: NPZ keys are
        # written in insertion order, so iterating the set directly made the file
        # bytes depend on the interpreter's string hash seed.
        all_fields = set()
        for entry in self.manifest_data:
            all_fields.update(entry.keys())

        # Separate scalar fields from tag fields
        scalar_fields = sorted(f for f in all_fields if f != "tags")

        # Initialize result dictionary
        arrays = {}

        # Process scalar fields
        for field in scalar_fields:
            # Collect values from all entries
            values = [entry[field] for entry in self.manifest_data if field in entry]
            if len(values) != n_entries:
                raise ValueError(
                    f"Manifest column {field!r} covers {len(values)} of "
                    f"{n_entries} entries. A field present on some writes and "
                    f"absent on others cannot be stored as a manifest column; "
                    f"write every entry with the same fields."
                )

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
                raise ValueError(
                    f"Manifest column {field!r} holds "
                    f"{type(first_val).__name__} values, which cannot be stored "
                    f"as a manifest column."
                )

            # Every column indexes the manifest by row, so a column of any other
            # length would silently misalign labels with audio.
            if len(arrays[field]) != n_entries:
                raise ValueError(
                    f"Manifest column {field!r} became {len(arrays[field])} rows "
                    f"for {n_entries} entries."
                )

        # Process tags if present
        if any("tags" in entry for entry in self.manifest_data):
            # Collect all unique tag keys (sorted, for a deterministic NPZ)
            all_tag_keys = set()
            for entry in self.manifest_data:
                if "tags" in entry and isinstance(entry["tags"], dict):
                    all_tag_keys.update(entry["tags"].keys())

            # Store each tag as a separate array
            for tag_key in sorted(all_tag_keys):
                tag_values = []
                for entry in self.manifest_data:
                    if "tags" in entry and tag_key in entry["tags"]:
                        tag_values.append(entry["tags"][tag_key])
                    else:
                        tag_values.append(None)

                # Store with 'tags_' prefix
                arrays[f"tags_{tag_key}"] = np.array(tag_values, dtype=object)

        return arrays

    def get_stats(self) -> Dict:
        """Get statistics about written files.

        Returns:
            Dictionary containing write statistics
        """
        stats = {
            "total_files": len(self.written_paths),
            "output_directory": str(self.directory),
            "current_index": self.index,
            "write_audio": self.write_audio,
        }

        # Only include batch count if timestamps are being tracked
        if self.include_timestamp:
            stats["total_batches"] = len(
                set(entry.get("timestamp", "") for entry in self.manifest_data)
            )

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
