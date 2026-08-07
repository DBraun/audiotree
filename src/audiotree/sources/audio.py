"""DataSource for reading AudioWriter outputs with manifest support."""

import copy
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, SupportsIndex, Union

import numpy as np
from grain import python as grain

from audiotree import AudioTree, _manifest
from audiotree._fs import safe_join
from audiotree.core import LABEL_FIELDS
from audiotree.sources.core import (
    READ_ERROR_KEY,
    _READ_ERRORS,
    _validate_on_read_error,
    AudioReadError,
    OnReadError,
)


def _with_batch_axis(value) -> np.ndarray:
    """Restore the leading batch axis a manifest row drops.

    A manifest stores one row per item, so a read-back value is the *contents*
    of a batch-of-1 field: ``lufs`` is a scalar, ``lufs_windows`` is ``(W,)``,
    ``codes`` is ``(codebooks, frames)``. Every AudioTree field must carry the
    batch axis, or ``AudioTree.batch`` concatenates along the wrong one.
    """
    return np.asarray(value)[np.newaxis, ...]


class AudioDataSource(grain.RandomAccessDataSource):
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
        on_read_error: What to do when one entry's audio file is missing or
            cannot be decoded. ``"raise"`` (default) raises
            :class:`~audiotree.sources.core.AudioReadError`, naming the path;
            ``"warn"`` substitutes digital silence of the entry's shape and
            emits a :class:`UserWarning`; ``"skip"`` substitutes the same
            silence quietly. Under either non-raising policy *every* item
            carries ``metadata["read_error"]`` (``True`` on a substitute), so
            the failure stays visible and the items still batch together. See
            :func:`~audiotree.sources.create_audio_dataset` for the reasoning.

    Example:
        First write some audio with :class:`~audiotree.AudioWriter` so there is
        a ``manifest.npz`` to read back:

        >>> import tempfile
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from audiotree import AudioTree, AudioWriter
        >>> out_dir = tempfile.mkdtemp()
        >>> lufs = np.full((5,), -10.0, dtype=np.float32)
        >>> with AudioWriter(out_dir) as writer:
        ...     _ = writer.write(
        ...         AudioTree.create(jnp.zeros((5, 1, 44100)), 44100, lufs=lufs)
        ...     )
        >>> manifest_path = f"{out_dir}/manifest.npz"

        Read straight from the NPZ manifest:

        >>> source = AudioDataSource(manifest_path)
        >>> len(source)
        5
        >>> source[0].waveform.shape
        (1, 1, 44100)

        Filter entries by metadata while loading:

        >>> source = AudioDataSource(
        ...     manifest_path,
        ...     filter_fn=lambda entry: entry.get('lufs', -float('inf')) > -20
        ... )

        Or use the convenience constructor that points at the output directory:

        >>> source = AudioDataSource.from_writer_output(out_dir)
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        *,
        audio_dir: Optional[Union[str, Path]] = None,
        num_records: Optional[int] = None,
        sample_rate: Optional[int] = None,
        mono: bool = False,
        duration: Optional[float] = None,
        pad_mode: Literal["constant", "wrap"] = "constant",
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        on_read_error: OnReadError = "raise",
    ):
        self.manifest_path = Path(manifest_path)
        self.on_read_error = _validate_on_read_error(on_read_error)

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

    def _load_manifest(
        self, filter_fn: Optional[Callable[[Dict], bool]] = None
    ) -> List[Dict]:
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

        if suffix == ".npz":
            entries = self._load_npz_manifest()
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}. Use .npz")

        # Apply filter if provided
        if filter_fn:
            entries = [entry for entry in entries if filter_fn(entry)]

        return entries

    def _load_npz_manifest(self) -> List[Dict]:
        """Load manifest entries from NPZ, then shape them for this source.

        Parsing the file -- header check, presence masks, string decoding -- is
        :func:`audiotree._manifest.read_entries`. All that is left here is the
        scalar convention this source's callers rely on.

        Returns:
            List of manifest entry dictionaries
        """
        entries = _manifest.read_entries(self.manifest_path)

        for entry in entries:
            for key, value in entry.items():
                # AudioTree fields are user data: keep the stored array and its
                # dtype exactly. Demoting them to Python scalars loses both the
                # dtype and, for a size-1 array, the shape (a one-window
                # ``lufs_windows`` would come back 0-d). Metadata columns are
                # kept whole for the same reason, and tags are already decoded.
                if key in LABEL_FIELDS or key.startswith("metadata_") or key == "tags":
                    continue

                # Bookkeeping columns (filename, sample_rate, channels, ...) are
                # consumed as Python scalars, including as ``sample_rate``, which
                # is a static pytree field and must not be a NumPy integer.
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()

                if isinstance(value, np.integer):
                    entry[key] = int(value)
                elif isinstance(value, np.floating):
                    entry[key] = float(value)
                else:
                    entry[key] = value

        return entries

    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        return self._length

    def _synthetic_geometry(self, entry: Dict):
        """Shape a synthetic zero waveform to match a real load of ``entry``.

        A synthetic item (a silence substitute for an unreadable file, or a
        token-only manifest with no audio on disk) must collate with the real
        items around it, so it has to carry the *target* geometry a real
        :meth:`AudioTree.from_file` would produce -- not the manifest's stored
        original geometry. The manifest records ``channels``/``samples`` at the
        written (original) rate; this mirrors ``from_file``'s output:

        - ``sample_rate``: the requested target rate (or the entry's own).
        - ``channels``: one when ``mono`` is set, else the entry's channels.
        - ``samples``: ``round(duration * target_sr)`` when a duration is
          requested (``from_file`` pads/trims to exactly that); otherwise the
          stored count rescaled by the rate ratio the same way ``librosa``
          resamples -- ``ceil(samples * target_sr / original_sr)`` -- so a
          substitute matches a resampled real item to the sample. With no
          duration and no resampling this is just the stored count.

        Returns:
            ``(channels, samples, sample_rate)`` for the synthetic waveform.
        """
        original_sr = entry.get("sample_rate")
        sample_rate = self.sample_rate or original_sr
        channels = 1 if self.mono else int(entry.get("channels", 1))
        if self.duration is not None and sample_rate:
            samples = max(0, round(self.duration * sample_rate))
        else:
            samples = int(entry.get("samples", 0))
            if (
                self.sample_rate is not None
                and original_sr
                and self.sample_rate != original_sr
            ):
                samples = int(np.ceil(samples * (self.sample_rate / original_sr)))
        return channels, samples, sample_rate

    def _substitute_silence(
        self, entry: Dict, audio_path, exc: BaseException, metadata: Dict
    ) -> AudioTree:
        """Build the stand-in returned for an entry whose audio cannot be read.

        Shaped by :meth:`_synthetic_geometry` to the *target* rate, channel
        count, and length a real load would produce, so a substitute collates
        with the real items around it, and tagged with the offending path plus
        ``metadata[READ_ERROR_KEY] == True``.
        """
        channels, samples, sample_rate = self._synthetic_geometry(entry)

        tree_kwargs = {
            "sample_rate": sample_rate,
            # ``from_file`` records the excerpt offset; match it so a substitute
            # and a real load carry the same metadata keys.
            "metadata": {
                **metadata,
                "offset": np.array([0.0]),
                READ_ERROR_KEY: np.array([True]),
            },
            "filepath": entry.get("filepath") or str(audio_path),
        }
        for field_name in LABEL_FIELDS:
            if field_name in entry:
                tree_kwargs[field_name] = _with_batch_axis(entry[field_name])

        if self.on_read_error == "warn":
            warnings.warn(
                f"Substituting silence for unreadable audio file "
                f"{str(audio_path)!r}: {type(exc).__name__}: {exc}. Every "
                f"substitute carries metadata[{READ_ERROR_KEY!r}] == True; pass "
                "on_read_error='raise' to fail on it instead.",
                UserWarning,
                stacklevel=3,
            )

        return AudioTree.create(
            np.zeros((1, channels, samples), dtype=np.float32), **tree_kwargs
        )

    def __getitem__(self, record_key: SupportsIndex) -> AudioTree:
        """Load an AudioTree for the given record index.

        Args:
            record_key: Index of the record to load

        Returns:
            AudioTree with audio data and restored metadata

        Raises:
            AudioReadError: If the entry's audio file is missing or undecodable
                and ``on_read_error == "raise"``.
        """
        entry = self.entries[int(record_key)]

        # Prepare metadata from manifest
        metadata = {}

        # Add metadata arrays from manifest (these can be batched properly)
        for key, value in entry.items():
            if key.startswith("metadata_"):
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
        files_written = entry.get("files_written", True)

        # The original source path is stored as a top-level ``filepath`` column
        # (a decoded string), distinct from the on-disk output ``filename``.
        # Restore it so ``.filepath`` reports where the item came from, matching
        # AudioTree.from_manifest. When the tree was written without a source
        # path the column is absent and ``.filepath`` stays unset.
        source_filepath = entry.get("filepath")

        if files_written:
            # Audio files exist - load from disk
            filename = entry["filename"]
            audio_path = safe_join(self.audio_dir, filename, description="audio file")

            # Build kwargs for AudioTree.from_file with all available fields
            tree_kwargs = {
                "sample_rate": self.sample_rate or entry.get("sample_rate"),
                "duration": self.duration,
                "mono": self.mono,
                "pad_mode": self.pad_mode if self.duration else None,
                "metadata": metadata,
            }

            # Prefer the recorded source path over the output audio path.
            if source_filepath is not None:
                tree_kwargs["filepath"] = source_filepath

            # Add AudioTree fields dynamically from manifest, each with an
            # explicit leading batch axis. ``from_file`` only adds one to a
            # scalar, so an array-valued field (``lufs_windows``, ``codes``,
            # ``latents``) would otherwise arrive unbatched and be concatenated
            # along the wrong axis by ``AudioTree.batch``.
            for field_name in LABEL_FIELDS:
                if field_name in entry:
                    tree_kwargs[field_name] = _with_batch_axis(entry[field_name])

            # Load audio file with all properties. A missing or undecodable
            # file is a read error like any other, so it goes through the same
            # policy rather than ending the run on the spot.
            try:
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                audio_tree = AudioTree.from_file(audio_path, **tree_kwargs)
            except _READ_ERRORS as exc:
                if self.on_read_error == "raise":
                    raise AudioReadError(
                        f"Failed to read audio file {str(audio_path)!r} for "
                        f"manifest entry {int(record_key)}: "
                        f"{type(exc).__name__}: {exc}",
                        str(audio_path),
                    ) from exc
                return self._substitute_silence(entry, audio_path, exc, metadata)
        else:
            # No audio files - create AudioTree from manifest metadata only.
            # Size the synthetic waveform to the *target* geometry (honoring
            # sample_rate, mono, and duration) so it collates with real items,
            # exactly as a silence substitute does.
            channels, samples, sample_rate = self._synthetic_geometry(entry)
            waveform = np.zeros((1, channels, samples), dtype=np.float32)

            # Build kwargs for AudioTree.create
            tree_kwargs = {
                "sample_rate": sample_rate,
                "metadata": metadata,
            }

            # Restore the recorded source path (there is no audio file here).
            if source_filepath is not None:
                tree_kwargs["filepath"] = source_filepath

            # Add AudioTree fields dynamically from manifest, each with an
            # explicit leading batch axis (see the files_written branch above).
            for field_name in LABEL_FIELDS:
                if field_name in entry:
                    tree_kwargs[field_name] = _with_batch_axis(entry[field_name])

            # Create AudioTree with zero audio data
            audio_tree = AudioTree.create(waveform, **tree_kwargs)

        # Under a non-raising policy every item is marked, so that a substitute
        # and a real load agree on their metadata keys and still batch together.
        if self.on_read_error != "raise":
            audio_tree = audio_tree.replace(
                metadata={
                    **audio_tree.metadata,
                    READ_ERROR_KEY: np.array([False]),
                }
            )

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

    def _with_entries(self, entries: List[Dict], description: str) -> "AudioDataSource":
        """Return a copy of this source restricted to ``entries``.

        The copy shares every loading option (``audio_dir``, ``sample_rate``,
        ``mono``, ...) but owns its own entry list, so it is a *view* over the
        already-loaded entries rather than a fresh read of the manifest. That is
        what makes the ``filter_*`` helpers compose: each one narrows whatever
        the receiver already contains, including a constructor ``filter_fn`` and
        a ``num_records`` cap.

        Args:
            entries: The subset of ``self.entries`` to keep
            description: Human-readable description of the narrowing, used in the
                error raised when nothing is left

        Returns:
            New AudioDataSource over ``entries``
        """
        if not entries:
            raise ValueError(
                f"No entries left after {description} "
                f"({len(self.entries)} entries before filtering)."
            )
        view = copy.copy(self)
        view.entries = list(entries)
        view._length = len(view.entries)
        return view

    def filter(self, predicate: Callable[[Dict], bool]) -> "AudioDataSource":
        """Create a new AudioDataSource keeping the entries matching ``predicate``.

        Filtering narrows the *current* entries, so filters compose:
        ``source.filter(a).filter(b)`` keeps the entries matching both.

        Args:
            predicate: Function called with a manifest entry, returning whether
                to keep it

        Returns:
            New AudioDataSource with the matching entries

        Raises:
            ValueError: If no entry matches
        """
        return self._with_entries(
            [entry for entry in self.entries if predicate(entry)],
            "filtering by the given predicate",
        )

    def filter_by_tag(self, tag_name: str, tag_value) -> "AudioDataSource":
        """Create a new AudioDataSource filtered by a specific tag value.

        Narrows the receiver's entries, so this composes with any other filter
        already applied (including a constructor ``filter_fn`` and a
        ``num_records`` cap).

        Args:
            tag_name: Name of the tag to filter by
            tag_value: Value the tag must have

        Returns:
            New AudioDataSource with filtered entries

        Raises:
            ValueError: If no entry has that tag value
        """

        def filter_fn(entry):
            tags = entry.get("tags", {})
            return tags.get(tag_name) == tag_value

        return self._with_entries(
            [entry for entry in self.entries if filter_fn(entry)],
            f"filtering on tag {tag_name!r} == {tag_value!r}",
        )

    def filter_by_lufs(
        self, min_lufs: Optional[float] = None, max_lufs: Optional[float] = None
    ) -> "AudioDataSource":
        """Create a new AudioDataSource filtered by loudness range.

        Filters entries based on the 'lufs' field in the manifest.
        Works with manifests created by AudioWriter in NPZ format.

        Narrows the receiver's entries, so this composes with any other filter
        already applied (including a constructor ``filter_fn`` and a
        ``num_records`` cap).

        Args:
            min_lufs: Minimum loudness in LUFS (inclusive)
            max_lufs: Maximum loudness in LUFS (inclusive)

        Returns:
            New AudioDataSource with filtered entries

        Raises:
            ValueError: If no entry falls in the range

        Example:
            Write four items with known per-item loudness so the manifest
            records a ``lufs`` field to filter on:

            >>> import tempfile
            >>> import numpy as np
            >>> import jax.numpy as jnp
            >>> from audiotree import AudioTree, AudioWriter
            >>> out_dir = tempfile.mkdtemp()
            >>> tree = AudioTree.create(
            ...     jnp.zeros((4, 1, 44100)), 44100,
            ...     lufs=np.array([-30.0, -18.0, -10.0, -25.0], dtype=np.float32),
            ... )
            >>> with AudioWriter(out_dir) as writer:
            ...     _ = writer.write(tree)
            >>> source = AudioDataSource.from_writer_output(out_dir)

            Keep only samples louder than -20 LUFS:

            >>> loud_source = source.filter_by_lufs(min_lufs=-20.0)
            >>> len(loud_source)
            2

            Keep samples within a specific loudness range:

            >>> mid_source = source.filter_by_lufs(min_lufs=-30.0, max_lufs=-15.0)
            >>> len(mid_source)
            3

            Filters compose, so chaining keeps only what matches both:

            >>> len(loud_source.filter_by_lufs(max_lufs=-15.0))
            1
        """

        def filter_fn(entry):
            lufs = entry.get("lufs")
            if lufs is None:
                return False
            if min_lufs is not None and lufs < min_lufs:
                return False
            if max_lufs is not None and lufs > max_lufs:
                return False
            return True

        return self._with_entries(
            [entry for entry in self.entries if filter_fn(entry)],
            f"filtering on lufs in [{min_lufs}, {max_lufs}]",
        )

    @classmethod
    def from_writer_output(
        cls, output_dir: Union[str, Path], **kwargs
    ) -> "AudioDataSource":
        """Convenience constructor for reading AudioWriter output.

        This method automatically locates the manifest file in the output directory
        based on the specified format and creates a AudioDataSource configured
        to read the audio files and metadata.

        Args:
            output_dir: Directory containing AudioWriter output
            **kwargs: Additional arguments passed to AudioDataSource
                     (e.g., sample_rate, mono, note_duration, filter_fn)

        Returns:
            AudioDataSource configured for the output directory

        Example:
            Write some audio, then read it back from the output directory:

            >>> import tempfile
            >>> import jax.numpy as jnp
            >>> from audiotree import AudioTree, AudioWriter
            >>> out_dir = tempfile.mkdtemp()
            >>> with AudioWriter(out_dir) as writer:
            ...     _ = writer.write(AudioTree.create(jnp.zeros((3, 1, 44100)), 44100))
            >>> source = AudioDataSource.from_writer_output(out_dir)
            >>> len(source)
            3

            Read with on-the-fly resampling to 16 kHz:

            >>> source = AudioDataSource.from_writer_output(out_dir, sample_rate=16000)
            >>> source[0].waveform.shape
            (1, 1, 16000)
        """
        output_dir = Path(output_dir)
        manifest_path = output_dir / "manifest.npz"
        return cls(manifest_path=manifest_path, audio_dir=output_dir, **kwargs)
