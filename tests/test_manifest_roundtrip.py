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
"""

import tempfile
from pathlib import Path

import numpy as np

from audiotree import AudioTree, AudioWriter, _manifest


# --- M4: array-valued string / bytes columns round-trip as sub-arrays ---


def test_array_valued_string_column_round_trips_direct(tmp_path):
    """A ``<U`` column of per-row string arrays comes back as equal sub-arrays."""
    path = tmp_path / "manifest.npz"
    rows = [np.array(["alice", "bob"]), np.array(["carol", "dan"])]
    _manifest.write(path, {"metadata_speakers": rows}, len(rows))

    entries = _manifest.read_entries(path)
    for entry, row in zip(entries, rows):
        cell = entry["metadata_speakers"]
        assert isinstance(cell, np.ndarray), (
            f"got {type(cell).__name__}, not a sub-array"
        )
        assert cell.dtype.kind == "U"
        np.testing.assert_array_equal(cell, row)


def test_array_valued_bytes_column_round_trips_direct(tmp_path):
    """A ``|S`` column of per-row bytes arrays comes back as equal sub-arrays."""
    path = tmp_path / "manifest.npz"
    rows = [np.array([b"x", b"yy"]), np.array([b"zzz", b"w"])]
    _manifest.write(path, {"metadata_raw": rows}, len(rows))

    entries = _manifest.read_entries(path)
    for entry, row in zip(entries, rows):
        cell = entry["metadata_raw"]
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
        {"filepath": ["a.wav", "b.wav"], "metadata_blob": [b"one", b"two"]},
        2,
    )

    entries = _manifest.read_entries(path)
    assert entries[0]["filepath"] == "a.wav"
    assert isinstance(entries[0]["filepath"], str)
    assert entries[1]["metadata_blob"] == b"two"
    assert isinstance(entries[1]["metadata_blob"], bytes)


def test_array_valued_string_column_round_trips_end_to_end():
    """Array-valued string metadata survives AudioWriter -> read_entries."""
    speakers = np.array([["alice", "bob"], ["carol", "dan"]])
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((2, 1, 8), dtype=np.float32),
            sample_rate=16000,
            metadata={"speakers": speakers},
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        entries = _manifest.read_entries(output_dir / "manifest.npz")

    for entry, expected in zip(entries, speakers):
        cell = entry["metadata_speakers"]
        assert isinstance(cell, np.ndarray)
        assert cell.dtype.kind == "U"
        np.testing.assert_array_equal(cell, expected)


def test_array_valued_bytes_column_round_trips_end_to_end():
    """Array-valued bytes metadata survives AudioWriter -> read_entries."""
    raw = np.array([[b"x", b"yy"], [b"zzz", b"w"]])
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        tree = AudioTree.create(
            np.zeros((2, 1, 8), dtype=np.float32),
            sample_rate=16000,
            metadata={"raw": raw},
        )
        with AudioWriter(output_dir, write_audio=False) as writer:
            writer.write(tree)

        entries = _manifest.read_entries(output_dir / "manifest.npz")

    for entry, expected in zip(entries, raw):
        cell = entry["metadata_raw"]
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
    _manifest.write(path, {"metadata_narrow": [np.int32(5)]}, 1)

    columns = _manifest.read_columns(path)
    assert columns.columns["metadata_narrow"].dtype == np.int32
    entries = _manifest.read_entries(path)
    assert entries[0]["metadata_narrow"] == 5
