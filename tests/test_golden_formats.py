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

from audiotree import AudioTree, _format
from audiotree.sources import ManifestDataSource, TreeDataSource

GOLDEN = Path(__file__).parent / "assets" / "golden"
TREE_DIR = GOLDEN / "tree_v1_0"
MANIFEST_DIR = GOLDEN / "manifest_v1_0"


def test_golden_fixtures_are_present():
    """A missing fixture must fail loudly rather than skip the whole module."""
    assert (TREE_DIR / "manifest.json").is_file()
    assert (MANIFEST_DIR / "manifest.npz").is_file()


# === TreeWriter format ===


def test_golden_tree_header():
    manifest = json.loads((TREE_DIR / "manifest.json").read_text())
    assert manifest["format"] == _format.TREE
    assert manifest["format_version"] == [1, 0]
    assert manifest["min_reader_version"] == [1, 0]
    assert manifest["num_samples"] == 4
    assert manifest["metadata"] == {"note": "golden"}
    assert sorted(manifest["leaves"]) == ["audio.waveform", "label"]
    assert manifest["leaves"]["audio.waveform"]["dtype"] == "float32"
    assert manifest["leaves"]["audio.waveform"]["shape_per_sample"] == [1, 8]
    assert manifest["leaves"]["audio.waveform"]["file"] == "audio.waveform.bin"


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


def test_golden_tree_exclude_prefixes():
    """Selective loading resolves against the committed leaf names."""
    source = TreeDataSource(TREE_DIR, exclude_prefixes=["label"])
    sample = source[0]
    assert "label" not in sample
    assert sample["audio"].waveform.shape == (1, 1, 8)


# === AudioWriter manifest format ===


def test_golden_manifest_header():
    data = np.load(MANIFEST_DIR / "manifest.npz", allow_pickle=True)
    header = {
        key[len(_format.NPZ_HEADER_PREFIX) :]: json.loads(str(data[key]))
        for key in data.files
        if key.startswith(_format.NPZ_HEADER_PREFIX)
    }
    assert header["format"] == _format.MANIFEST
    assert header["format_version"] == [1, 0]
    assert header["min_reader_version"] == [1, 0]


@pytest.mark.parametrize(
    "index,peak,pitch,velocity",
    [(0, 0.25, 60, 100), (1, 0.5, 61, 99), (2, 0.75, 62, 98)],
)
def test_golden_manifest_reads_with_expected_values(index, peak, pitch, velocity):
    """Audio, dtypes and per-item scalars all survive the committed manifest."""
    source = ManifestDataSource.from_writer_output(MANIFEST_DIR)
    assert len(source) == 3

    item = source[index]
    assert item.sample_rate == 8000
    assert item.waveform.shape == (1, 1, 400)
    # Written as PCM_16, so allow a quantization step.
    np.testing.assert_allclose(np.abs(item.waveform).max(), peak, atol=1e-4)

    np.testing.assert_array_equal(item.pitch, [pitch])
    np.testing.assert_array_equal(item.velocity, [velocity])
    assert item.pitch.dtype == np.int32
    assert item.velocity.dtype == np.int32


def test_golden_manifest_batches_without_corruption():
    """Batching the committed items stacks them along the batch axis."""
    source = ManifestDataSource.from_writer_output(MANIFEST_DIR)
    batched = AudioTree.batch([source[i] for i in range(3)])
    assert batched.waveform.shape == (3, 1, 400)
    np.testing.assert_array_equal(batched.pitch.ravel(), [60, 61, 62])
    np.testing.assert_array_equal(batched.velocity.ravel(), [100, 99, 98])
