"""Format-compatibility tests against committed fixtures.

Every other reader test writes its input with the code under test, so the writer
and reader always change together — a repacking that breaks every dataset on
disk keeps the suite green. These fixtures were written once, are committed, and
are read here with **hardcoded** expected values. If a change makes them fail,
either the change is a format break (bump `format_version` major) or the reader
regressed.

The fixtures are deliberately tiny and bagz-free, so they are portable to every
platform audiotree supports.

To regenerate after an intentional format break, rewrite the fixtures with the
new writer, bump `_format.CURRENT_VERSION`, and update the values below --
regenerating them to make a red test go green defeats the purpose.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree, _format, _manifest
from audiotree.sources import AudioDataSource, TreeDataSource


def load_manifest(path):
    """Read an NPZ manifest into a plain dict, closing the archive.

    A bare ``np.load`` on an NPZ returns a lazy ``NpzFile`` that keeps the zip
    open. POSIX lets you unlink an open file, so the leak is invisible there;
    Windows refuses, and ``TemporaryDirectory`` cleanup fails with WinError 32.

    ``allow_pickle=False`` is the point of the manifest encoding, not an
    incidental argument: a committed fixture that could only be read with
    unpickling would mean the format regressed.
    """
    with np.load(path, allow_pickle=False) as npz:
        return dict(npz)


GOLDEN = Path(__file__).parent / "assets" / "golden"
TREE_DIR = GOLDEN / "tree_v1_0"
MANIFEST_DIR = GOLDEN / "manifest_v1_0"


def test_golden_fixtures_are_present():
    """A missing fixture must fail loudly rather than skip the whole module."""
    assert (TREE_DIR / "manifest.json").is_file()
    assert (MANIFEST_DIR / "manifest.npz").is_file()


# === TreeWriter format ===


def test_golden_tree_header():
    manifest = json.loads((TREE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == _format.TREE
    assert manifest["format_version"] == [1, 0]
    assert manifest["min_reader_version"] == [1, 0]
    assert manifest["num_samples"] == 4
    assert manifest["metadata"] == {"note": "golden"}
    assert sorted(manifest["leaves"]) == ["audio.waveform", "label"]
    assert manifest["leaves"]["audio.waveform"]["dtype"] == "float32"
    assert manifest["leaves"]["audio.waveform"]["shape_per_sample"] == [1, 8]
    assert manifest["leaves"]["audio.waveform"]["file"] == "audio.waveform.bin"
    # The AudioTree node's extras dict is stored under the children key
    # "extras" -- the on-disk spelling is part of the format contract.
    audio_children = manifest["structure"]["children"]["audio"]["children"]
    assert sorted(audio_children) == ["extras", "waveform"]


def test_golden_tree_reads_with_expected_values():
    """The committed memmaps decode to exactly the bytes they were written with."""
    source = TreeDataSource(TREE_DIR)
    assert len(source) == 4

    for index in range(4):
        sample = source[index]
        assert sorted(sample) == ["audio", "label"]
        assert isinstance(sample["audio"], AudioTree)
        assert sample["audio"].sample_rate == 16000
        assert sample["audio"].waveform.shape == (1, 1, 8)

        expected = np.arange(index * 8, (index + 1) * 8, dtype=np.float32) / 100.0
        np.testing.assert_allclose(
            sample["audio"].waveform.ravel(), expected, rtol=0, atol=1e-7
        )
        np.testing.assert_array_equal(sample["label"].ravel(), [index])
        assert sample["label"].dtype == np.int32
    source.close()


def test_golden_tree_exclude_prefixes():
    """Selective loading resolves against the committed leaf names."""
    source = TreeDataSource(TREE_DIR, exclude_prefixes=["label"])
    sample = source[0]
    assert "label" not in sample
    assert sample["audio"].waveform.shape == (1, 1, 8)
    source.close()


# === AudioWriter manifest format ===


def test_golden_manifest_header():
    manifest = _manifest.read_columns(MANIFEST_DIR / "manifest.npz")
    assert manifest.header["format"] == _format.MANIFEST
    assert manifest.header["format_version"] == [1, 0]
    assert manifest.header["min_reader_version"] == [1, 0]
    assert manifest.header["num_entries"] == 3
    assert manifest.num_entries == 3


def test_golden_manifest_stores_no_object_arrays():
    """Every column is loadable with ``allow_pickle=False``, strings included.

    A manifest travels with the data it describes, so a reader that has to
    unpickle it executes whatever a downloaded dataset contains.
    """
    data = load_manifest(MANIFEST_DIR / "manifest.npz")
    assert data["filename"].dtype.kind == "U"
    assert data["subtype"].dtype.kind == "U"
    assert all(array.dtype != object for array in data.values())


def test_golden_manifest_stores_extras_under_their_prefix():
    """Per-item ``AudioTree.extras`` land as ``extras_<key>`` columns -- the
    on-disk spelling is part of the format contract."""
    data = load_manifest(MANIFEST_DIR / "manifest.npz")
    assert data["extras_energy"].dtype == np.float32
    np.testing.assert_allclose(data["extras_energy"], [0.5, 0.25, 0.125])


def test_golden_manifest_records_a_lossless_subtype():
    """The committed audio is float WAV, and the manifest says so."""
    data = load_manifest(MANIFEST_DIR / "manifest.npz")
    assert list(data["subtype"]) == ["FLOAT", "FLOAT", "FLOAT"]


def test_golden_manifest_masks_the_absent_tag():
    """Absence is carried by the mask; the third row's empty cell is filler."""
    data = load_manifest(MANIFEST_DIR / "manifest.npz")
    mask_key = f"{_manifest.MASK_PREFIX}tags_split"
    np.testing.assert_array_equal(data[mask_key], [True, True, False])

    entries = _manifest.read_entries(MANIFEST_DIR / "manifest.npz")
    assert [entry.get("tags") for entry in entries] == [
        {"split": "train"},
        {"split": "test"},
        None,
    ]


@pytest.mark.parametrize(
    "index,peak,pitch,velocity,energy",
    [(0, 0.25, 60, 100, 0.5), (1, 0.5, 61, 99, 0.25), (2, 0.75, 62, 98, 0.125)],
)
def test_golden_manifest_reads_with_expected_values(
    index, peak, pitch, velocity, energy
):
    """Audio, dtypes and per-item scalars all survive the committed manifest."""
    source = AudioDataSource.from_writer_output(MANIFEST_DIR)
    assert len(source) == 3

    item = source[index]
    assert item.sample_rate == 8000
    assert item.waveform.shape == (1, 1, 400)
    # Written as FLOAT, so the peak survives exactly -- no quantization step.
    np.testing.assert_allclose(np.abs(item.waveform).max(), peak, atol=1e-7)

    np.testing.assert_array_equal(item.pitch, [pitch])
    np.testing.assert_array_equal(item.velocity, [velocity])
    assert item.pitch.dtype == np.int32
    assert item.velocity.dtype == np.int32

    # The extras_energy column comes back under tree.extras, batch axis intact.
    np.testing.assert_allclose(item.extras["energy"], [energy])
    assert item.extras["energy"].dtype == np.float32


def test_golden_manifest_batches_without_corruption():
    """Batching the committed items stacks them along the batch axis."""
    source = AudioDataSource.from_writer_output(MANIFEST_DIR)
    batched = AudioTree.batch([source[i] for i in range(3)])
    assert batched.waveform.shape == (3, 1, 400)
    np.testing.assert_array_equal(batched.pitch.ravel(), [60, 61, 62])
    np.testing.assert_array_equal(batched.velocity.ravel(), [100, 99, 98])
