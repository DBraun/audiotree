"""Round-trip regression tests for the NPZ manifest encoder/decoder.

Covers two bugs in :mod:`audiotree._manifest`:

* **M4** -- ``_decode_cell`` corrupted array-valued ``<U``/``|S`` columns by
  applying ``str()``/``bytes()`` to a sub-array (yielding a numpy repr string
  or the raw fixed-width buffer). Such a cell must round-trip to an equal
  numpy sub-array.
* **M5** -- ``_encode_column`` narrowed bare Python ``int``/``float`` samples to
  ``int32``/``float32``: a large int raised ``OverflowError`` at save time and a
  float64 value was silently rounded. Python ints/floats must widen to 64-bit,
  while numpy scalars keep their own width.
* **m10** -- only the string branch of ``_encode_column`` validated type
  homogeneity; the bool and numeric branches coerced contaminants (``[True, 5]``
  stored ``[True, True]``, ``[1, 2.5]`` truncated to ``[1, 2]``). A mixed-type
  column must be rejected at write time.
* **m9** -- NumPy fixed-width ``<U``/``|S`` storage drops trailing NULs, so a
  value ending in a NUL cannot round-trip. Such values must be rejected at write
  time rather than silently truncated.
* **n5** -- ``encode`` reserved the ``__mask_``/``__audiotree_`` prefixes but not
  the name ``"tags"``, which ``read_entries`` synthesizes from ``tags_*``
  columns and would silently clobber. A column named exactly ``"tags"`` must be
  rejected.
* **m14** -- the ndarray branch of ``_encode_column`` checked only shape
  homogeneity, so ``np.stack`` silently promoted mixed dtype kinds (an int32
  row next to a float32 row became float64; an int row next to a string row
  became its ``str()`` text), violating ``read_entries``' dtype-fidelity
  contract. Rows must share a dtype kind; widths within a kind may differ and
  stack to the widest, every value intact. The trailing-NUL rule also applies
  to str/bytes leaves in list rows, which the encoder itself converts.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree, AudioWriter, _manifest


# --- M4: array-valued string / bytes columns round-trip as sub-arrays ---


def test_array_valued_string_column_round_trips_direct(tmp_path):
    """A ``<U`` column of per-row string arrays comes back as equal sub-arrays."""
    path = tmp_path / "manifest.npz"
    rows = [np.array(["alice", "bob"]), np.array(["carol", "dan"])]
    _manifest.write(path, {"extras_speakers": rows}, len(rows))

    entries = _manifest.read_entries(path)
    for entry, row in zip(entries, rows):
        cell = entry["extras_speakers"]
        assert isinstance(cell, np.ndarray), (
            f"got {type(cell).__name__}, not a sub-array"
        )
        assert cell.dtype.kind == "U"
        np.testing.assert_array_equal(cell, row)


def test_array_valued_bytes_column_round_trips_direct(tmp_path):
    """A ``|S`` column of per-row bytes arrays comes back as equal sub-arrays."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([b"x", b"yy"]), np.array([b"zzz", b"w"])]
    _manifest.write(path, {"extras_raw": rows}, len(rows))

    entries = _manifest.read_entries(path)
    for entry, row in zip(entries, rows):
        cell = entry["extras_raw"]
        assert isinstance(cell, np.ndarray), (
            f"got {type(cell).__name__}, not a sub-array"
        )
        assert cell.dtype.kind == "S"
        np.testing.assert_array_equal(cell, row)


def test_scalar_string_and_bytes_cells_still_decode_to_python(tmp_path):
    """A 1-D string/bytes column still yields plain ``str``/``bytes`` scalars."""
    path = tmp_path / "manifest.npz"
    _manifest.write(
        path,
        {"filepath": ["a.wav", "b.wav"], "extras_blob": [b"one", b"two"]},
        2,
    )

    entries = _manifest.read_entries(path)
    assert entries[0]["filepath"] == "a.wav"
    assert isinstance(entries[0]["filepath"], str)
    assert entries[1]["extras_blob"] == b"two"
    assert isinstance(entries[1]["extras_blob"], bytes)


def test_array_valued_string_column_round_trips_end_to_end():
    """Array-valued string extras survives AudioWriter -> read_entries."""
    speakers = np.array([["alice", "bob"], ["carol", "dan"]])
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((2, 1, 8), dtype=np.float32),
            sample_rate=16000,
            extras={"speakers": speakers},
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        entries = _manifest.read_entries(output_dir / "manifest.npz")

    for entry, expected in zip(entries, speakers):
        cell = entry["extras_speakers"]
        assert isinstance(cell, np.ndarray)
        assert cell.dtype.kind == "U"
        np.testing.assert_array_equal(cell, expected)


def test_array_valued_bytes_column_round_trips_end_to_end():
    """Array-valued bytes extras survives AudioWriter -> read_entries."""
    raw = np.array([[b"x", b"yy"], [b"zzz", b"w"]])
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((2, 1, 8), dtype=np.float32),
            sample_rate=16000,
            extras={"raw": raw},
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        entries = _manifest.read_entries(output_dir / "manifest.npz")

    for entry, expected in zip(entries, raw):
        cell = entry["extras_raw"]
        assert isinstance(cell, np.ndarray)
        assert cell.dtype.kind == "S"
        np.testing.assert_array_equal(cell, expected)


# --- M5: Python int/float widen to 64-bit; numpy scalars keep their width ---


def test_large_python_int_does_not_overflow_and_round_trips(tmp_path):
    """A byte offset past int32 must save without OverflowError and read back exactly."""
    path = tmp_path / "manifest.npz"
    value = 2**31  # OverflowError against int32 on numpy 2.x
    _manifest.write(path, {"tags_byte_offset": [value]}, 1)

    columns = _manifest.read_columns(path)
    assert columns.columns["tags_byte_offset"].dtype == np.int64
    entries = _manifest.read_entries(path)
    assert entries[0]["tags"]["byte_offset"] == value


def test_even_larger_python_int_survives(tmp_path):
    """A value well past int32 (a hash-sized int) round-trips through int64."""
    path = tmp_path / "manifest.npz"
    value = 2**62 + 12345
    _manifest.write(path, {"tags_hash": [value]}, 1)

    entries = _manifest.read_entries(path)
    assert entries[0]["tags"]["hash"] == value


def test_python_float_round_trips_without_rounding(tmp_path):
    """A Python float (float64) must not be narrowed to float32 and rounded."""
    path = tmp_path / "manifest.npz"
    value = 100000.001  # not representable in float32
    _manifest.write(path, {"tags_start_seconds": [value]}, 1)

    columns = _manifest.read_columns(path)
    assert columns.columns["tags_start_seconds"].dtype == np.float64
    entries = _manifest.read_entries(path)
    assert entries[0]["tags"]["start_seconds"] == value


def test_numpy_int32_scalar_keeps_its_dtype(tmp_path):
    """An explicit ``np.int32`` sample must stay int32 -- widening is for bare ints."""
    path = tmp_path / "manifest.npz"
    _manifest.write(path, {"extras_narrow": [np.int32(5)]}, 1)

    columns = _manifest.read_columns(path)
    assert columns.columns["extras_narrow"].dtype == np.int32
    entries = _manifest.read_entries(path)
    assert entries[0]["extras_narrow"] == 5


# --- m10: bool/numeric columns reject non-homogeneous contaminants ---


def test_bool_column_with_non_bool_raises(tmp_path):
    """A bool column contaminated by an int must not coerce it to ``True``."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"column 'c' mixes bool"):
        _manifest.write(path, {"c": [True, 5]}, 2)


def test_int_column_with_float_raises(tmp_path):
    """An int column contaminated by a float must not truncate it."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"column 'n' mixes int"):
        _manifest.write(path, {"n": [1, 2.5]}, 2)


def test_float_column_with_int_raises(tmp_path):
    """Homogeneity is symmetric: a float column may not hold a bare int."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"column 'n' mixes float"):
        _manifest.write(path, {"n": [2.5, 1]}, 2)


def test_uniform_bool_int_float_columns_still_round_trip(tmp_path):
    """The homogeneity check must not reject legitimately-uniform columns."""
    path = tmp_path / "manifest.npz"
    _manifest.write(
        path,
        {
            "extras_flag": [True, False],
            "extras_count": [1, 2],
            "extras_amount": [1.5, 2.5],
        },
        2,
    )

    columns = _manifest.read_columns(path)
    assert columns.columns["extras_flag"].dtype == np.bool_
    assert columns.columns["extras_count"].dtype == np.int64
    assert columns.columns["extras_amount"].dtype == np.float64

    entries = _manifest.read_entries(path)
    assert [e["extras_flag"] for e in entries] == [True, False]
    assert [e["extras_count"] for e in entries] == [1, 2]
    assert [e["extras_amount"] for e in entries] == [1.5, 2.5]


# --- m9: trailing NULs in string/bytes cells are rejected, not truncated ---


def test_string_with_trailing_nul_raises(tmp_path):
    """A ``<U`` cell ending in a NUL cannot round-trip, so it is rejected."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"column 's'.*NUL"):
        _manifest.write(path, {"s": ["x\x00", "y"]}, 2)


def test_bytes_with_trailing_nul_raises(tmp_path):
    """A ``|S`` cell ending in a NUL byte cannot round-trip, so it is rejected."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"column 'b'.*NUL"):
        _manifest.write(path, {"b": [b"ab\x00", b"\x00\x00"]}, 2)


def test_strings_and_bytes_without_trailing_nul_unaffected(tmp_path):
    """Ordinary values -- including internal NULs -- still round-trip exactly."""
    path = tmp_path / "manifest.npz"
    _manifest.write(
        path,
        {"extras_s": ["a\x00b", "plain"], "extras_b": [b"a\x00b", b"plain"]},
        2,
    )

    entries = _manifest.read_entries(path)
    assert [e["extras_s"] for e in entries] == ["a\x00b", "plain"]
    assert [e["extras_b"] for e in entries] == [b"a\x00b", b"plain"]


# --- n5: a column literally named 'tags' is reserved ---


def test_column_named_tags_raises(tmp_path):
    """A data column named exactly 'tags' would be clobbered, so it is rejected."""
    path = tmp_path / "manifest.npz"
    with pytest.raises(ValueError, match=r"'tags' is reserved"):
        _manifest.write(path, {"tags": ["hello"], "tags_genre": ["rock"]}, 1)


def test_tags_prefix_pivot_still_works(tmp_path):
    """Reserving 'tags' must not disturb the ``tags_*`` -> ``tags`` synthesis."""
    path = tmp_path / "manifest.npz"
    _manifest.write(path, {"tags_genre": ["rock"], "tags_year": [1994]}, 1)

    entries = _manifest.read_entries(path)
    assert entries[0]["tags"] == {"genre": "rock", "year": 1994}


# --- m14: array-valued columns enforce dtype-kind homogeneity across rows ---


def test_array_column_mixing_int_and_float_rows_raises(tmp_path):
    """An int row next to a float row would stack to float64; rejected instead."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([1, 2], dtype=np.int32), np.array([1.5, 2.5], dtype=np.float32)]
    with pytest.raises(ValueError, match=r"column 'emb' mixes int arrays.*float"):
        _manifest.write(path, {"emb": rows}, 2)


def test_array_column_mixing_int_and_string_rows_raises(tmp_path):
    """An int row next to a string row would be stringified; rejected instead."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([1, 2]), np.array(["a", "b"])]
    with pytest.raises(ValueError, match=r"column 'ids' mixes int arrays.*str"):
        _manifest.write(path, {"ids": rows}, 2)


def test_array_column_mixing_signed_and_unsigned_rows_raises(tmp_path):
    """int64 + uint64 rows would stack to float64 -- a kind change -- so signed
    and unsigned integer rows are held apart."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.uint64)]
    with pytest.raises(ValueError, match=r"column 'n' mixes int arrays.*uint"):
        _manifest.write(path, {"n": rows}, 2)


def test_float32_embedding_rows_round_trip_with_dtype(tmp_path):
    """A legitimate array column -- per-row float32 embeddings -- is not caught
    by the kind check, and its dtype survives the round trip."""
    path = tmp_path / "manifest.npz"
    rows = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.4, 0.5, 0.6], dtype=np.float32),
    ]
    _manifest.write(path, {"extras_embedding": rows}, len(rows))

    columns = _manifest.read_columns(path)
    assert columns.columns["extras_embedding"].dtype == np.float32

    entries = _manifest.read_entries(path)
    for entry, row in zip(entries, rows):
        cell = entry["extras_embedding"]
        assert isinstance(cell, np.ndarray)
        assert cell.dtype == np.float32
        np.testing.assert_array_equal(cell, row)


def test_array_rows_of_same_kind_promote_to_the_widest_width(tmp_path):
    """Widths within a kind stack to the widest with every value intact:
    int32+int64 -> int64, <U3+<U8 -> <U8."""
    path = tmp_path / "manifest.npz"
    int_rows = [np.array([1, 2], dtype=np.int32), np.array([3, 2**40])]
    str_rows = [np.array(["ab", "c"]), np.array(["longer", "strings!"])]
    _manifest.write(path, {"extras_n": int_rows, "extras_s": str_rows}, 2)

    columns = _manifest.read_columns(path)
    assert columns.columns["extras_n"].dtype == np.int64
    assert columns.columns["extras_s"].dtype.kind == "U"

    entries = _manifest.read_entries(path)
    np.testing.assert_array_equal(entries[1]["extras_n"], [3, 2**40])
    np.testing.assert_array_equal(entries[1]["extras_s"], ["longer", "strings!"])


def test_string_list_row_with_trailing_nul_raises(tmp_path):
    """A str leaf inside a list row still holds its NUL, and the encoder's own
    np.asarray would be what truncates it -- so it is rejected. (An ndarray row
    cannot offend: numpy strips trailing NULs at element access, so the
    observable value round-trips exactly.)"""
    path = tmp_path / "manifest.npz"
    rows = [np.array(["ok", "fine"]), ["bad\x00", "fine"]]
    with pytest.raises(ValueError, match=r"column 's'.*NUL"):
        _manifest.write(path, {"s": rows}, 2)


def test_bytes_list_row_with_trailing_nul_raises(tmp_path):
    """Same trailing-NUL rule for bytes leaves in list rows."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([b"ok", b"fine"]), [b"bad\x00", b"fine"]]
    with pytest.raises(ValueError, match=r"column 'b'.*NUL"):
        _manifest.write(path, {"b": rows}, 2)


# --- strict column schema: unrecognized columns fail by name at read time ---


def _write_dataset_with_stray_column(directory: Path) -> Path:
    """Write a real one-item dataset, then smuggle a stray column into its NPZ.

    The writer can only produce schema-valid manifests, so the stray key is
    spliced in after the fact -- exactly the shape of corruption (or of a file
    from an incompatible format revision) the reader must refuse.
    """
    tree = AudioTree.create(np.zeros((1, 1, 80), dtype=np.float32), 8000)
    with AudioWriter(directory) as writer:
        writer.write(tree)
    manifest_path = directory / "manifest.npz"
    with np.load(manifest_path, allow_pickle=False) as npz:
        payload = dict(npz)
    payload["bogus_col"] = np.array([1])
    with open(manifest_path, "wb") as fh:
        np.savez(fh, **payload)
    return manifest_path


def test_unrecognized_column_is_rejected_by_name(tmp_path):
    """The format's column schema is closed: no consumer routes an unknown
    column anywhere, so it would silently vanish from every loaded tree.
    ``read_columns`` refuses it by name instead."""
    path = tmp_path / "manifest.npz"
    _manifest.write(path, {"extras_ok": [1.0], "bogus_col": [1]}, 1)
    with pytest.raises(ValueError, match=r"unrecognized column.*'bogus_col'"):
        _manifest.read_columns(path)
    with pytest.raises(ValueError, match=r"unrecognized column.*'bogus_col'"):
        _manifest.read_entries(path)


def test_mask_of_unrecognized_column_is_rejected(tmp_path):
    """A presence mask names its column, so a stray ``__mask_`` key is held to
    the same schema as the column it claims to describe."""
    path = tmp_path / "manifest.npz"
    payload = _manifest.encode({"extras_x": [1.0, 2.0]}, 2)
    payload[f"{_manifest.MASK_PREFIX}bogus_col"] = np.array([True, False])
    with open(path, "wb") as fh:
        np.savez(fh, **payload)
    with pytest.raises(ValueError, match=r"unrecognized column.*'bogus_col'"):
        _manifest.read_columns(path)


def test_mask_of_absent_column_is_rejected(tmp_path):
    """A mask for a schema-valid name that is not a column of this manifest is
    still refused: it describes nothing."""
    path = tmp_path / "manifest.npz"
    payload = _manifest.encode({"extras_x": [1.0, 2.0]}, 2)
    payload[f"{_manifest.MASK_PREFIX}lufs"] = np.array([True, False])
    with open(path, "wb") as fh:
        np.savez(fh, **payload)
    with pytest.raises(
        ValueError, match=r"presence mask for 'lufs', which is not a column"
    ):
        _manifest.read_columns(path)


def test_from_manifest_goes_through_the_strict_schema(tmp_path):
    """``AudioTree.from_manifest`` reads via the shared parser, so a stray
    column fails there too rather than being silently dropped."""
    manifest_path = _write_dataset_with_stray_column(tmp_path)
    with pytest.raises(ValueError, match=r"unrecognized column.*'bogus_col'"):
        AudioTree.from_manifest(manifest_path)


def test_audio_data_source_goes_through_the_strict_schema(tmp_path):
    """``AudioDataSource`` reads via the shared parser as well."""
    from audiotree.sources import AudioDataSource

    manifest_path = _write_dataset_with_stray_column(tmp_path)
    with pytest.raises(ValueError, match=r"unrecognized column.*'bogus_col'"):
        AudioDataSource(manifest_path)


def test_every_writer_column_passes_the_strict_schema(tmp_path):
    """The reader's allow-list must cover everything the writer can record:
    every bookkeeping column, every label field, filepath, extras and tags."""
    from audiotree.core import LABEL_FIELDS

    tree = AudioTree.create(
        np.zeros((2, 1, 80), dtype=np.float32),
        8000,
        lufs=np.array([-20.0, -18.0], dtype=np.float32),
        lufs_windows=np.zeros((2, 1), dtype=np.float32),
        pitch=np.array([60, 61], dtype=np.int16),
        velocity=np.array([100, 99], dtype=np.int16),
        note_duration=np.array([0.5, 0.6], dtype=np.float32),
        codes=np.zeros((2, 2, 4), dtype=np.int32),
        latents=np.zeros((2, 3), dtype=np.float32),
        extras={"energy": np.array([0.1, 0.2], dtype=np.float32)},
        filepath=["a.wav", "b.wav"],
    )
    with AudioWriter(tmp_path, include_timestamp=True) as writer:
        writer.write(tree, tags={"split": "train"})

    columns = _manifest.read_columns(tmp_path / "manifest.npz").columns
    assert set(LABEL_FIELDS) <= set(columns)
    assert "extras_energy" in columns
    assert "tags_split" in columns
    assert "timestamp" in columns
    assert "filepath" in columns
