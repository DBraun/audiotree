"""AudioWriter class for writing AudioTree objects to disk with manifest support."""

import string
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile

from . import _manifest
from ._fs import refuse_to_clobber
from .core import LABEL_FIELDS, AudioTree, _require_batched_rank

# Manifest columns hold one value per written item. Provenance lives in the
# AudioTree ``metadata`` container (its own ``filepath``/``source`` columns),
# so ``extras_*`` columns are purely user payload -- except for these
# bookkeeping keys, which describe the *read* that produced the item rather
# than the item itself, and are not written as columns.
_SKIPPED_EXTRAS_KEYS = frozenset({"offset", "duration", "manifest_index", "tags"})

# The provenance strings ``AudioTree.metadata`` carries, each written to a
# dedicated top-level manifest column of the same name.
_PROVENANCE_COLUMNS = ("filepath", "source")

# Subtypes that store a sample as written. Every other subtype quantizes onto a
# fixed-point grid and hard-clips anything outside [-1, 1].
_FLOATING_SUBTYPES = frozenset({"FLOAT", "DOUBLE"})


def _widest_subtype(suffix: str) -> Optional[str]:
    """Pick the least destructive subtype a container supports.

    Float model output routinely leaves [-1, 1], and libsndfile's own default is
    ``PCM_16`` for most containers, which silently destroys it. Prefer 32-bit
    float where the container allows it (WAV, AIFF, CAF, W64, RF64...), then
    24-bit PCM (FLAC), and only then fall back to the container's default
    (compressed containers such as OGG, which admit no PCM subtype at all).

    Args:
        suffix: Output filename suffix, with or without its leading dot.

    Returns:
        A soundfile subtype, or ``None`` if libsndfile does not know the
        container -- in which case ``soundfile.write`` is left to report it.
    """
    fmt = suffix.lstrip(".").upper()
    if fmt not in soundfile.available_formats():
        return None
    for candidate in ("FLOAT", "PCM_24"):
        if soundfile.check_format(fmt, candidate):
            return candidate
    return soundfile.default_subtype(fmt)


def _pattern_has_index(pattern: str) -> bool:
    """Whether ``pattern`` has an ``{index}`` replacement field.

    ``str.format`` silently ignores kwargs a pattern does not reference, so a
    pattern lacking ``{index}`` formats to the *same* filename for every item --
    each write clobbering the last. Detect that by parsing the pattern's fields
    rather than trusting ``.format`` to complain.
    """
    for _, field_name, _, _ in string.Formatter().parse(pattern):
        if field_name is None:
            continue
        # Field names may be dotted/indexed (``index[0]``); the root is what
        # ``.format(index=...)`` binds.
        root = field_name.split(".")[0].split("[")[0]
        if root == "index":
            return True
    return False


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
    track all written audio files and their extras.

    Args:
        directory: Directory where audio files will be written
        pattern: Filename pattern with {index} placeholder for sequential numbering
        include_timestamp: Whether to record a ``timestamp`` manifest column.
            One timestamp is minted per ``write()`` call and shared by every
            entry in that batch, so it records when the batch was written and
            distinct timestamps count ``write()`` calls.
        compress_manifest: Whether to compress NPZ manifest files (only applies to npz format)
        write_audio: Whether to write audio files to disk (default True). When False,
            only manifest is generated with extras
        subtype: soundfile subtype string (e.g. ``"PCM_16"``, ``"PCM_24"``,
            ``"FLOAT"``), forwarded to ``soundfile.write`` and recorded in the
            manifest's ``subtype`` column. When ``None`` (default) the writer
            picks the least destructive subtype the container supports —
            ``"FLOAT"`` for WAV/AIFF/CAF/…, ``"PCM_24"`` for FLAC, the
            container's own default otherwise. That is deliberately *not*
            libsndfile's default of ``PCM_16``, which hard-clips the
            out-of-[-1, 1] samples that float model output routinely contains.
            Passing a fixed-point subtype explicitly is fine; the writer then
            warns (``RuntimeWarning``) whenever an item it clips actually
            exceeds the representable range.
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
        # Without an {index} field the pattern formats to one filename for the
        # whole run, so every item would overwrite the last on disk while the
        # manifest still recorded a distinct row per (lost) item. A manifest-only
        # run writes no audio, so the filename is a bookkeeping label there and
        # need not be unique.
        if write_audio and not _pattern_has_index(pattern):
            raise ValueError(
                f"pattern {pattern!r} has no '{{index}}' field, so every item "
                f"would be written to the same file and all but the last lost. "
                f"Include '{{index}}' (e.g. 'audio_{{index:04d}}.wav') so each "
                f"item gets a unique filename."
            )
        self.pattern = pattern
        # Inferred from the first written tree; every later write must match it.
        self.sample_rate = None
        self.include_timestamp = include_timestamp
        self.compress_manifest = compress_manifest
        self.write_audio = write_audio
        self.subtype = subtype
        self._resolved_subtypes: Dict[str, Optional[str]] = {}
        self.index = 0
        self.written_paths = []
        self.manifest_data = []
        self._expected_fields = None  # Track which AudioTree fields should be present
        # The extras-column schema (which extras_* columns, and which
        # provenance columns) is fixed by the first write; later writes must
        # match.
        self._expected_extras_keys = None
        self._expected_provenance = None
        # Each column's logical value kind, pinned by its first value. Kind
        # drift (int rows, then a str row) is rejected at the offending write;
        # see _check_entry_kinds.
        self._column_kinds: Dict[str, str] = {}
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

    def _get_extras_column_keys(self, tree: AudioTree) -> set:
        """The extras keys that become ``extras_*`` manifest columns.

        Mirrors the column-selection logic in :meth:`_create_manifest_entry`:
        internal keys are skipped, and nested dicts have no column representation.

        Args:
            tree: AudioTree to inspect

        Returns:
            Set of extras keys that will be written as columns
        """
        keys = set()
        for key, value in tree.extras.items():
            if key in _SKIPPED_EXTRAS_KEYS:
                continue
            if isinstance(value, dict):
                continue
            keys.add(key)
        return keys

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
        # One row per batch item only makes sense at rank 3. A mini-batched tree
        # has two leading axes, and every axis below shifts by one: a 6-item
        # (3, 2, 1, 800) tree wrote *three* rows claiming channels=2, samples=1.
        # `write_audio=True` was saved only by soundfile rejecting the shape;
        # manifest-only runs -- the mode where nothing else looks at the audio --
        # recorded nonsense extras and said nothing.
        _require_batched_rank(tree.waveform, "AudioWriter.write")

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

        # Validate the extras-column schema eagerly, exactly as LABEL_FIELDS
        # are validated above. An extras key present on some writes and absent
        # on others cannot be stored as one column; caught here, the drifting
        # write fails at its own call -- before its WAVs land -- rather than at
        # the next save/close, which would abort with the manifest unwritten.
        extras_keys = self._get_extras_column_keys(tree)
        provenance = {
            column: bool(getattr(tree, column)) for column in _PROVENANCE_COLUMNS
        }
        if self._expected_extras_keys is None:
            self._expected_extras_keys = extras_keys
            self._expected_provenance = provenance
        else:
            if extras_keys != self._expected_extras_keys:
                missing = self._expected_extras_keys - extras_keys
                extra = extras_keys - self._expected_extras_keys
                error_parts = []
                if missing:
                    error_parts.append(
                        f"missing keys: {sorted('extras_' + k for k in missing)}"
                    )
                if extra:
                    error_parts.append(
                        f"extra keys: {sorted('extras_' + k for k in extra)}"
                    )
                raise ValueError(
                    f"AudioTree extras keys don't match previous writes. "
                    f"{', '.join(error_parts)}. "
                    f"All AudioTrees written to the same manifest must carry the "
                    f"same extras keys."
                )
            for column in _PROVENANCE_COLUMNS:
                if provenance[column] != self._expected_provenance[column]:
                    had = "had" if self._expected_provenance[column] else "had no"
                    now = "has" if provenance[column] else "has no"
                    raise ValueError(
                        f"AudioTree {column!r} presence doesn't match previous "
                        f"writes: the first write {had} {column}s but this one "
                        f"{now}. The {column!r} column must cover every entry "
                        f"or none."
                    )

        batch_size = tree.waveform.shape[0]

        # A provenance list shorter than the batch would populate its column
        # for only some items in this very write, producing a ragged column.
        # Reject it here rather than at save time, with the WAVs unwritten.
        for column in _PROVENANCE_COLUMNS:
            values = getattr(tree, column)
            if values and len(values) < batch_size:
                raise ValueError(
                    f"AudioTree has {len(values)} {column}s for a batch of "
                    f"{batch_size}; the {column!r} column would cover only part "
                    f"of this write. Provide one {column} per item, or none."
                )
        paths = []
        # (filename, peak, subtype) for every item this write clips.
        clipped: List[Tuple[str, float, Optional[str]]] = []

        # One timestamp for the whole call: the batch lands together, so its
        # entries share their provenance, and get_stats() can count batches as
        # distinct timestamps. Minting one per item made total_batches count
        # items instead.
        timestamp = datetime.now().isoformat() if self.include_timestamp else None

        for i in range(batch_size):
            # Generate filename
            # todo: need a way to pass more kwargs to this formatter
            filename = self.pattern.format(index=self.index)
            filepath = self.directory / filename
            # Only a file that exists has a subtype; a manifest-only run records
            # none rather than claiming an encoding nothing was written in.
            subtype = self._resolve_subtype(filepath) if self.write_audio else None

            # Collect the manifest entry first: a column this writer cannot
            # store raises, and doing that before the WAV exists keeps the
            # directory free of audio no manifest row points at.
            entry = self._create_manifest_entry(
                tree, i, filename, tags, subtype, timestamp
            )
            self._check_entry_kinds(entry)

            # Write audio file if requested
            if self.write_audio:
                # Convert to numpy and transpose for soundfile (channels, samples) -> (samples, channels)
                audio = np.array(tree.waveform[i].T)
                # A fixed-point subtype silently hard-clips out-of-range samples,
                # so measure the peak before handing the audio to libsndfile and
                # report it rather than let the data disappear.
                if subtype not in _FLOATING_SUBTYPES and audio.size:
                    peak = float(np.abs(audio).max())
                    if peak > 1.0:
                        clipped.append((filename, peak, subtype))
                soundfile.write(str(filepath), audio, tree.sample_rate, subtype=subtype)
                self.written_paths.append(filepath)

            paths.append(filepath)
            self.manifest_data.append(entry)

            self.index += 1

        if clipped:
            worst = max(clipped, key=lambda item: item[1])
            warnings.warn(
                f"AudioWriter clipped {len(clipped)} of {batch_size} items: subtype "
                f"{worst[2]!r} represents only [-1, 1], and {worst[0]} peaks at "
                f"{worst[1]:.4g}. Those samples are gone from the file on disk. "
                f"Pass subtype='FLOAT' to store the audio as written, or scale it "
                f"down before writing.",
                RuntimeWarning,
                stacklevel=2,
            )

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

    def _resolve_subtype(self, filepath: Path) -> Optional[str]:
        """The soundfile subtype this writer will encode ``filepath`` with.

        An explicit ``subtype=`` is used verbatim; otherwise the container's
        widest subtype is chosen (see :func:`_widest_subtype`). Results are
        cached per suffix, since ``pattern`` fixes the container for a run.
        """
        if self.subtype is not None:
            return self.subtype
        suffix = filepath.suffix.lower()
        if suffix not in self._resolved_subtypes:
            self._resolved_subtypes[suffix] = _widest_subtype(suffix)
        return self._resolved_subtypes[suffix]

    def _create_manifest_entry(
        self,
        tree: AudioTree,
        batch_index: int,
        filename: str,
        tags: Optional[Dict] = None,
        subtype: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict:
        """Create a manifest entry for a single audio file.

        Args:
            tree: Source AudioTree
            batch_index: Index within the batch
            filename: Output filename
            tags: Optional custom metadata
            subtype: soundfile subtype the audio was encoded with, or ``None``
                when no audio file was written
            timestamp: ISO timestamp shared by every entry of this ``write()``
                call, or ``None`` when timestamps are disabled

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
            # Absent (masked out) for a manifest-only run; a reader must not
            # guess PCM_16 the way libsndfile's default would.
            "subtype": subtype,
        }

        # Add timestamp only if requested (None when include_timestamp=False)
        if timestamp is not None:
            entry["timestamp"] = timestamp

        # Add AudioTree fields dynamically, preserving dtypes
        for field_name in LABEL_FIELDS:
            field_value = getattr(tree, field_name, None)
            if field_value is not None:
                entry[field_name] = _column_value(field_name, field_value, batch_index)

        # Provenance (from the AudioTree metadata container), one dedicated
        # column per string: the original source path and the source group.
        for column in _PROVENANCE_COLUMNS:
            values = getattr(tree, column)
            if values and batch_index < len(values):
                entry[column] = values[batch_index]

        # Add custom tags
        if tags:
            entry["tags"] = tags

        # Add extras arrays if present
        if tree.extras:
            for key, value in tree.extras.items():
                # Skip internal keys, and `tags`, which is handled above.
                if key in _SKIPPED_EXTRAS_KEYS:
                    continue
                # A nested dict has no NPZ column representation; write nested
                # pytrees with TreeWriter instead.
                if isinstance(value, dict):
                    continue
                column = f"extras_{key}"
                entry[column] = _column_value(column, value, batch_index)

        return entry

    def _check_entry_kinds(self, entry: Dict) -> None:
        """Pin each column's value kind at its first value, rejecting drift here.

        A column whose values change logical kind across writes (int rows, then
        a str row) passes the schema checks in :meth:`write` -- the *keys* still
        match -- and would otherwise only be caught by the encoder at
        save/close, aborting with the manifest unwritten and every WAV already
        on disk, orphaned. Checked as each entry is collected, before its audio
        is written, so the offending ``write()`` fails at its own call. The
        encoder's per-column homogeneity checks remain the backstop.

        Args:
            entry: A manifest entry from :meth:`_create_manifest_entry`.

        Raises:
            ValueError: If a value's kind differs from the kind established by
                the column's first value.
        """

        def cells():
            for column, value in entry.items():
                if column == "tags":
                    # Pivoted into tags_* columns by _manifest_to_columns.
                    for tag_key, tag_value in value.items():
                        yield f"{_manifest.TAG_PREFIX}{tag_key}", tag_value
                else:
                    yield column, value

        for column, value in cells():
            if value is None:
                # A missing value (masked subtype, absent tag) fixes no kind.
                continue
            kind = _manifest._value_kind(value)
            established = self._column_kinds.setdefault(column, kind)
            if kind != established:
                raise ValueError(
                    f"Manifest column {column!r} holds {established} values "
                    f"from previous writes, but this write supplies a {kind} "
                    f"value ({value!r}). A column must hold one type; rejected "
                    f"at this write so the run's earlier audio is not orphaned "
                    f"by a failed manifest save at close()."
                )

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

        manifest_path = _manifest.write(
            self.directory / "manifest.npz",
            self._manifest_to_columns(),
            len(self.manifest_data),
            compress=self.compress_manifest,
        )

        self._manifest_index = self.index
        return manifest_path

    def _manifest_to_columns(self) -> Dict[str, List[Any]]:
        """Pivot the accumulated entries into one list of values per column.

        Encoding those values -- dtypes, fixed-width strings, presence masks --
        belongs to :mod:`audiotree._manifest`, which is also what reads them
        back. This method only decides *which* columns exist and in what order.

        Returns:
            Column name to its per-item values, ``None`` where an item has no
            value for that column (only tags can be missing this way).
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

        columns: Dict[str, List[Any]] = {}

        # Per-item columns, tags excepted: those are pivoted out below.
        for field in sorted(f for f in all_fields if f != "tags"):
            values = [entry[field] for entry in self.manifest_data if field in entry]
            if len(values) != n_entries:
                raise ValueError(
                    f"Manifest column {field!r} covers {len(values)} of "
                    f"{n_entries} entries. A field present on some writes and "
                    f"absent on others cannot be stored as a manifest column; "
                    f"write every entry with the same fields."
                )
            columns[field] = values

        # Tags, in contrast, are genuinely per-item: an item may carry a tag its
        # neighbours lack, and `None` here becomes a False in that column's
        # presence mask rather than a sentinel value.
        all_tag_keys = set()
        for entry in self.manifest_data:
            if isinstance(entry.get("tags"), dict):
                all_tag_keys.update(entry["tags"].keys())

        for tag_key in sorted(all_tag_keys):
            columns[f"{_manifest.TAG_PREFIX}{tag_key}"] = [
                entry.get("tags", {}).get(tag_key) for entry in self.manifest_data
            ]

        return columns

    def get_stats(self) -> Dict:
        """Get statistics about written files.

        Returns:
            Dictionary containing write statistics. When ``include_timestamp``
            is set, ``total_batches`` counts :meth:`write` calls: each call
            mints one timestamp shared by its batch's entries, so distinct
            timestamps are distinct batches.
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
