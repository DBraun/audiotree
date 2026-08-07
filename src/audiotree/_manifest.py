"""The AudioWriter manifest format: one writer, one reader.

An ``AudioWriter`` manifest is a single ``manifest.npz`` holding one *column* per
recorded field and one *row* per written item, plus the format header described
in :mod:`audiotree._format`. This module owns that encoding end to end:
:func:`write` turns per-row Python values into NPZ arrays, and :func:`read_entries`
turns them back into per-row dicts. Every reader in audiotree goes through here,
so the file's rules are stated once rather than re-derived per call site.

Two properties of the encoding are load-bearing:

**Nothing is pickled.** A manifest travels with the data it describes, so anyone
who shares, mirrors, or downloads a pre-rendered dataset hands the reader a file
they did not write. ``np.load(..., allow_pickle=True)`` on such a file is
arbitrary code execution in every data worker. Strings are therefore stored as
fixed-width unicode (``<U``) arrays and bytes as fixed-width ``|S`` arrays --
never as ``dtype=object`` -- and the reader passes ``allow_pickle=False``.

**Absence is carried by a mask, never by a value.** A column whose value is
missing for some rows is written alongside a boolean array named
``__mask_<column>``; a ``False`` cell means "this row has no value", and the
cell's contents are meaningless filler. ``-1``, ``NaN`` and ``""`` are ordinary
data that a reader must hand back unchanged, so no value is ever interpreted as
a "missing" sentinel. Columns with no missing rows carry no mask.

The header additionally records ``num_entries``, so the row count is stated by
the file rather than inferred from whichever column happens to be listed first.
"""

import json
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from . import _format

#: Prefix of the boolean presence mask belonging to ``<column>``. Distinct from
#: ``_format.NPZ_HEADER_PREFIX`` so header, mask and data keys never collide:
#: data columns are named after AudioTree fields, ``metadata_*`` or ``tags_*``,
#: none of which can start with an underscore.
MASK_PREFIX = "__mask_"

#: Nested under ``entry["tags"]`` by :func:`read_entries`.
TAG_PREFIX = "tags_"


class ManifestColumns(NamedTuple):
    """A manifest as it is stored: whole columns, plus their presence masks.

    Attributes:
        header: The decoded format header (``format``, ``format_version``,
            ``min_reader_version``, ``producer``, ``num_entries``).
        num_entries: Number of rows. Every column has this many.
        columns: One array per column, indexed by row along the first axis.
            String columns are ``<U`` arrays, bytes columns ``|S``; everything
            else keeps the dtype it was written with.
        present: The boolean mask of the columns that have one, by column name.
            A column absent from this mapping is present in every row.
    """

    header: Dict[str, Any]
    num_entries: int
    columns: Dict[str, np.ndarray]
    present: Dict[str, np.ndarray]

    def is_present(self, column: str, row: int) -> bool:
        """Whether ``column`` holds a value for ``row``."""
        mask = self.present.get(column)
        return True if mask is None else bool(mask[row])


def _scalar_kind(value: Any) -> str:
    """The logical type of a scalar manifest value, for homogeneity checks.

    A column stores one logical type: ``bool`` is distinct from ``int`` (``True``
    is an ``int`` in Python, but a bool column stores booleans) and ``int`` is
    distinct from ``float`` (an int column cannot hold a float without truncating
    it). A numpy scalar reports the kind of its dtype, so an ``int16`` and an
    ``int32`` both count as ``"int"`` -- differing widths are legitimate within a
    column, differing kinds are not. Likewise ``np.str_`` counts as ``"str"``
    and ``np.bytes_`` as ``"bytes"``, matching their Python counterparts.
    """
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, np.generic):
        return {"i": "int", "u": "int", "f": "float", "U": "str", "S": "bytes"}.get(
            value.dtype.kind, value.dtype.kind
        )
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


#: Human-readable names for numpy dtype kind codes, for the array-row
#: homogeneity check. Unlike :func:`_scalar_kind`, signed and unsigned integers
#: are kept distinct here: stacking an ``int64`` row with a ``uint64`` row
#: promotes to ``float64``, which changes the kind, so array rows are held to
#: raw dtype-kind equality.
_DTYPE_KIND_NAMES = {
    "b": "bool",
    "i": "int",
    "u": "uint",
    "f": "float",
    "c": "complex",
    "U": "str",
    "S": "bytes",
    "M": "datetime",
    "m": "timedelta",
    "O": "object",
}


def _value_kind(value: Any) -> str:
    """The logical kind of one manifest cell value, scalar or array row.

    An array row reports its dtype's kind with an ``" array"`` suffix, so an
    int scalar and an int-array row are distinct kinds -- they are stored
    differently and cannot share a column. ``AudioWriter`` uses this to pin a
    column's kind at its first value and reject drift at the offending
    ``write()`` rather than at close, with :func:`_encode_column`'s own
    homogeneity checks as the backstop.
    """
    if isinstance(value, np.ndarray):
        kind = _DTYPE_KIND_NAMES.get(value.dtype.kind, value.dtype.kind)
        return f"{kind} array"
    return _scalar_kind(value)


def _encode_column(
    name: str, values: Sequence[Any]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Encode one column's per-row values into an array and optional mask.

    Args:
        name: Column name, used in error messages.
        values: One value per row. ``None`` means *this row has no value* and is
            the only way to say so -- every other value is stored verbatim.

    Returns:
        ``(array, mask)``. ``mask`` is ``None`` when no row is missing.

    Raises:
        ValueError: If the column holds values with no NPZ representation,
            mixes logical types (strings with non-strings, array rows of
            differing dtype kinds or shapes), or holds a string/bytes value
            ending in a NUL, which fixed-width storage cannot round-trip.
            Array rows of the *same* dtype kind but differing widths are fine:
            they stack to the widest width with every value intact.
    """
    count = len(values)
    present = np.fromiter(
        (value is not None for value in values), dtype=bool, count=count
    )
    mask = None if bool(present.all()) else present
    provided = [value for value in values if value is not None]

    if not provided:
        # Nothing to infer a dtype from. An empty string column plus an
        # all-False mask says "no row has a value" without inventing a sentinel.
        return np.zeros(count, dtype="<U1"), present

    sample = provided[0]

    if isinstance(sample, str):
        if not all(isinstance(value, str) for value in provided):
            raise ValueError(
                f"Manifest column {name!r} mixes strings with "
                f"{type(next(v for v in provided if not isinstance(v, str))).__name__} "
                f"values. A column must hold one type."
            )
        offender = next((value for value in provided if value.endswith("\x00")), None)
        if offender is not None:
            raise ValueError(
                f"Manifest column {name!r} holds a string ending in a NUL "
                f"character ({offender!r}). NumPy fixed-width '<U' storage cannot "
                f"tell a trailing NUL from padding, so it is dropped on read and "
                f"the value would not round-trip; store it without a trailing NUL."
            )
        filled = [value if value is not None else "" for value in values]
        return np.array(filled, dtype=np.str_), mask

    if isinstance(sample, bytes):
        if not all(isinstance(value, bytes) for value in provided):
            raise ValueError(
                f"Manifest column {name!r} mixes bytes with "
                f"{type(next(v for v in provided if not isinstance(v, bytes))).__name__} "
                f"values. A column must hold one type."
            )
        offender = next((value for value in provided if value.endswith(b"\x00")), None)
        if offender is not None:
            raise ValueError(
                f"Manifest column {name!r} holds a bytes value ending in a NUL "
                f"byte ({offender!r}). NumPy fixed-width '|S' storage cannot tell "
                f"a trailing NUL from padding, so it is dropped on read and the "
                f"value would not round-trip; store it without a trailing NUL."
            )
        filled = [value if value is not None else b"" for value in values]
        return np.array(filled, dtype=np.bytes_), mask

    if isinstance(sample, (bool, np.bool_)):
        offender = next((v for v in provided if _scalar_kind(v) != "bool"), None)
        if offender is not None:
            raise ValueError(
                f"Manifest column {name!r} mixes bool values with "
                f"{type(offender).__name__} values. A column must hold one type."
            )
        filled = [bool(value) if value is not None else False for value in values]
        return np.array(filled, dtype=bool), mask

    if isinstance(sample, np.ndarray):
        reference = np.asarray(sample)
        kind = reference.dtype.kind
        # np.stack promotes mixed dtypes to a common one instead of raising: an
        # int32 row next to a float32 row becomes float64, and an int row next
        # to a string row becomes the int's str() text. Either way the values
        # come back with a dtype nobody wrote, so rows must share a dtype kind.
        # Widths *within* a kind are fine -- int32+int64 stacks to int64 and
        # <U3+<U8 to <U8, a widening with every written value intact, just as a
        # scalar string column already stores every row at the widest width.
        offender = next(
            (
                array
                for array in (np.asarray(value) for value in provided)
                if array.dtype.kind != kind
            ),
            None,
        )
        if offender is not None:
            raise ValueError(
                f"Manifest column {name!r} mixes "
                f"{_DTYPE_KIND_NAMES.get(kind, kind)} arrays with "
                f"{_DTYPE_KIND_NAMES.get(offender.dtype.kind, offender.dtype.kind)} "
                f"arrays. Stacking them would silently promote every row to a "
                f"common dtype, so a column's rows must share one dtype kind."
            )
        if kind in ("U", "S"):
            # Same trailing-NUL rule as the scalar branches, applied to the
            # rows this branch converts itself. A row that is already a numpy
            # array is exempt -- not skipped, but genuinely unable to offend:
            # fixed-width element access strips trailing NULs, so the value the
            # caller can observe is exactly what read_entries returns. A
            # str/bytes leaf inside a list/tuple row, though, still holds its
            # NUL, and np.asarray below would be what truncates it.
            nul: Any = "\x00" if kind == "U" else b"\x00"
            offender = next(
                (
                    element
                    for value in provided
                    if not isinstance(value, np.ndarray)
                    for element in np.asarray(value, dtype=object).ravel()
                    if isinstance(element, (str, bytes)) and element.endswith(nul)
                ),
                None,
            )
            if offender is not None:
                raise ValueError(
                    f"Manifest column {name!r} holds an array row with a value "
                    f"ending in a NUL ({offender!r}). NumPy fixed-width "
                    f"'{'<U' if kind == 'U' else '|S'}' storage cannot tell a "
                    f"trailing NUL from padding, so it is dropped on read and "
                    f"the value would not round-trip; store it without a "
                    f"trailing NUL."
                )
        filled = [
            np.asarray(value)
            if value is not None
            else np.zeros(reference.shape, reference.dtype)
            for value in values
        ]
        try:
            stacked = np.stack(filled)
        except ValueError as exc:
            raise ValueError(
                f"Manifest column {name!r} holds arrays of differing shapes "
                f"({exc}). Every row of a column must have the same shape."
            ) from exc
        return stacked, mask

    if isinstance(sample, (np.generic, int, float)):
        kind = _scalar_kind(sample)
        offender = next((v for v in provided if _scalar_kind(v) != kind), None)
        if offender is not None:
            raise ValueError(
                f"Manifest column {name!r} mixes {kind} values with "
                f"{type(offender).__name__} values. A column must hold one type."
            )
        if isinstance(sample, np.generic):
            # A numpy scalar carries its own width; keep it (an int16 velocity
            # stays int16). A bare Python int/float has no width, so widen to
            # 64-bit rather than silently narrowing: int32 overflows on a large
            # byte offset or hash, and float32 rounds a float64 value away.
            dtype = sample.dtype
        elif isinstance(sample, int):
            dtype = np.dtype(np.int64)
        else:
            dtype = np.dtype(np.float64)
        filled = [value if value is not None else dtype.type(0) for value in values]
        return np.array(filled, dtype=dtype), mask

    raise ValueError(
        f"Manifest column {name!r} holds {type(sample).__name__} values, which "
        f"cannot be stored as a manifest column."
    )


def encode(
    columns: Mapping[str, Sequence[Any]], num_entries: int
) -> Dict[str, np.ndarray]:
    """Build the full NPZ payload -- data, masks and header -- for a manifest.

    Args:
        columns: Column name to its per-row values, ``None`` for a missing row.
        num_entries: Number of rows every column must have.

    Returns:
        The mapping to hand to ``np.savez``. Key order follows ``columns``, each
        column immediately followed by its mask, with the header last, so the
        bytes are reproducible for a given input.

    Raises:
        ValueError: If a column does not cover every row, or holds values with
            no NPZ representation.
    """
    payload: Dict[str, np.ndarray] = {}
    for name, values in columns.items():
        if name.startswith(MASK_PREFIX) or name.startswith(_format.NPZ_HEADER_PREFIX):
            raise ValueError(
                f"Manifest column {name!r} uses a reserved prefix "
                f"({MASK_PREFIX!r} marks presence masks and "
                f"{_format.NPZ_HEADER_PREFIX!r} marks the format header)."
            )
        if name == "tags":
            raise ValueError(
                f"Manifest column name 'tags' is reserved: read_entries "
                f"synthesizes the 'tags' key from the {TAG_PREFIX!r} columns, so a "
                f"data column named 'tags' would be clobbered. Store the value "
                f"under a '{TAG_PREFIX}...' name (or a different column name)."
            )
        if len(values) != num_entries:
            raise ValueError(
                f"Manifest column {name!r} covers {len(values)} of {num_entries} "
                f"entries. Every column indexes the manifest by row, so a short "
                f"column would silently misalign labels with audio."
            )
        array, mask = _encode_column(name, values)
        payload[name] = array
        if mask is not None:
            payload[f"{MASK_PREFIX}{name}"] = mask

    header = _format.header(_format.MANIFEST)
    header["num_entries"] = num_entries
    for key, value in header.items():
        # NPZ has no place for scalars, so each header value is a JSON-encoded
        # 0-d string array under the reserved header prefix.
        payload[f"{_format.NPZ_HEADER_PREFIX}{key}"] = np.array(json.dumps(value))
    return payload


def write(
    path: Union[str, Path],
    columns: Mapping[str, Sequence[Any]],
    num_entries: int,
    *,
    compress: bool = True,
) -> Path:
    """Write a manifest atomically (temp file + rename).

    A reader -- or a retry after a crash -- sees either the previous manifest or
    the new one, never the half-written NPZ that a kill during ``savez`` would
    otherwise leave behind (which reads as a corrupt zip and blocks re-rendering
    with "already contains a dataset").

    Args:
        path: Destination ``manifest.npz``.
        columns: Column name to its per-row values, ``None`` for a missing row.
        num_entries: Number of rows every column must have.
        compress: Whether to deflate the NPZ.

    Returns:
        ``path``, for convenience.
    """
    path = Path(path)
    payload = encode(columns, num_entries)
    # The temp name is dotted so it does not look like a dataset to
    # `refuse_to_clobber`.
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "wb") as f:
        if compress:
            np.savez_compressed(f, **payload)
        else:
            np.savez(f, **payload)
    os.replace(tmp_path, path)
    return path


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Read every array out of an NPZ without unpickling, and close the archive.

    An open ``NpzFile`` holds the zip open, which leaks a descriptor per source
    and blocks deletion on Windows, so the arrays are materialized eagerly.
    """
    try:
        with np.load(path, allow_pickle=False) as npz:
            return dict(npz)
    except ValueError as exc:
        if "allow_pickle" not in str(exc):
            raise
        raise ValueError(
            f"{path}: this AudioWriter manifest stores pickled object arrays, so "
            f"it was written by a pre-1.0 audiotree. Loading it would execute "
            f"whatever the file contains, which audiotree will not do -- "
            f"re-render the dataset."
        ) from exc


def read_columns(path: Union[str, Path]) -> ManifestColumns:
    """Read a manifest into whole columns, validating its header and shapes.

    This is the column-oriented primitive; :func:`read_entries` builds per-row
    dicts on top of it. Callers that want to slice a manifest without
    materializing every row (``AudioTree.from_manifest`` selecting a filtered
    subset, say) should use this.

    Args:
        path: Path to a ``manifest.npz``.

    Returns:
        A :class:`ManifestColumns`.

    Raises:
        ValueError: If the file carries no audiotree header, is a different
            format, needs a newer reader, stores pickled objects, or declares a
            row count its columns do not match.
    """
    path = Path(path)
    data = _load_npz(path)
    source = str(path)

    header = {
        key[len(_format.NPZ_HEADER_PREFIX) :]: json.loads(str(data[key]))
        for key in data
        if key.startswith(_format.NPZ_HEADER_PREFIX)
    }
    _format.check(header, _format.MANIFEST, source=source)

    num_entries = header.get("num_entries")
    if (
        not isinstance(num_entries, int)
        or isinstance(num_entries, bool)
        or num_entries < 0
    ):
        raise ValueError(
            f"{source}: manifest header declares num_entries={num_entries!r}, "
            f"which is not a row count."
        )

    columns: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for key, array in data.items():
        if key.startswith(_format.NPZ_HEADER_PREFIX):
            continue
        if key.startswith(MASK_PREFIX):
            masks[key[len(MASK_PREFIX) :]] = array
        else:
            columns[key] = array

    # A manifest is untrusted input: a column shorter than the declared row
    # count would read past its end (or, worse, silently pair row i of one
    # column with row i of another that means something else).
    for name, array in columns.items():
        if array.ndim == 0 or len(array) != num_entries:
            raise ValueError(
                f"{source}: manifest column {name!r} has "
                f"{'no rows' if array.ndim == 0 else len(array)} rows but the "
                f"header declares {num_entries}."
            )
    for name, mask in masks.items():
        if name not in columns:
            raise ValueError(
                f"{source}: manifest holds a presence mask for {name!r}, which "
                f"is not a column of this manifest."
            )
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != num_entries:
            raise ValueError(
                f"{source}: presence mask for column {name!r} must be "
                f"{num_entries} booleans, got {mask.dtype} of shape {mask.shape}."
            )

    return ManifestColumns(
        header=header, num_entries=num_entries, columns=columns, present=masks
    )


def _decode_cell(array: np.ndarray, row: int) -> Any:
    """Take one row out of a column, undoing the fixed-width string encoding.

    A scalar string/bytes cell comes back as a Python ``str``/``bytes``. An
    array-valued cell -- ``array[row]`` is a sub-array, as it is for any
    multi-dimensional column -- is handed back as that sub-array unchanged;
    ``str()``/``bytes()`` on it would yield a numpy repr or the raw fixed-width
    buffer, destroying the stored value.
    """
    value = array[row]
    if np.ndim(value) == 0:
        if array.dtype.kind == "U":
            return str(value)
        if array.dtype.kind == "S":
            return bytes(value)
    return value


def read_entries(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a manifest into one dict per written item.

    This is the reader every consumer of an ``AudioWriter`` manifest should use:
    :class:`~audiotree.sources.AudioDataSource` for its entry list, and
    ``AudioTree.from_manifest`` when it does not need whole columns.

    Args:
        path: Path to a ``manifest.npz``.

    Returns:
        One dict per row, in written order, holding:

        * every column that has a value for that row, under its stored name --
          so ``"filename"``, ``"sample_rate"``, the AudioTree label fields, and
          ``"metadata_*"`` keys keep their prefix. Values keep the dtype they
          were written with (a NumPy scalar, or a sub-array for an
          array-valued column); string columns come back as ``str`` and bytes
          columns as ``bytes``.
        * ``"tags"``, a dict of the ``tags_*`` columns with the prefix stripped
          and their values as plain Python scalars, present only when the row
          has at least one tag.

        A column with no value for a row is simply absent from that row's dict.
        No value is treated as a missing marker: a stored ``-1``, ``NaN`` or
        ``""`` comes back as itself.

    Raises:
        ValueError: If the manifest is unreadable (see :func:`read_columns`), or
            stores a non-scalar tag.
    """
    manifest = read_columns(path)
    entries: List[Dict[str, Any]] = []

    for row in range(manifest.num_entries):
        entry: Dict[str, Any] = {}
        tags: Dict[str, Any] = {}
        for name, array in manifest.columns.items():
            if not manifest.is_present(name, row):
                continue
            value = _decode_cell(array, row)
            if name.startswith(TAG_PREFIX):
                tag_name = name[len(TAG_PREFIX) :]
                # A tag column is meant to hold scalars. A container cell
                # (accepted by ``AudioWriter(..., tags={...})`` without
                # complaint) reaches user comparisons as an array and raises an
                # opaque "truth value of an array is ambiguous", so name the
                # offender here instead.
                if isinstance(value, np.ndarray):
                    raise ValueError(
                        f"Manifest {path} stores a non-scalar value for tag "
                        f"{tag_name!r} in entry {row} (shape {value.shape}). Tag "
                        f"values must be scalars (str, int, float, bool or None); "
                        f"store array-valued information as AudioTree metadata "
                        f"instead."
                    )
                # Tags are plain scalars the caller handed in, not typed audio
                # data, so hand back what was written (`True`, not `np.True_`).
                tags[tag_name] = (
                    value.item() if isinstance(value, np.generic) else value
                )
            else:
                entry[name] = value
        if tags:
            entry["tags"] = tags
        entries.append(entry)

    return entries
