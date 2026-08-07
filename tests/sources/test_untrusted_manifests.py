"""A dataset directory is untrusted input.

Manifests travel with the data they describe, so anyone who shares, mirrors, or
downloads a pre-rendered dataset hands the reader a file that names the paths it
will open and the shapes and dtypes it will memmap. These tests pin the guards
that keep a hostile or merely corrupt manifest from escaping the dataset
directory or reaching a constructor with nonsense.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from audiotree import AudioTree, TreeWriter
from audiotree._fs import safe_join
from audiotree.sources import TreeDataSource


@pytest.fixture
def dataset(tmp_path):
    """A real one-sample dataset, plus a secret file outside its directory."""
    data_dir = tmp_path / "dataset"
    with TreeWriter(str(data_dir), expected_samples=1) as writer:
        writer.write(AudioTree.create(np.zeros((1, 1, 12), dtype=np.float32), 16000))
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"LEAKED-SECRET-OUTSIDE-THE-DATASET-DIR-0123456789")
    return data_dir, secret


def _retamper(data_dir: Path, mutate) -> None:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    mutate(copy.deepcopy(manifest), manifest)
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_absolute_leaf_path_is_refused(dataset):
    """`Path("/data") / "/etc/passwd"` is `/etc/passwd` — pathlib drops the base."""
    data_dir, secret = dataset
    _retamper(
        data_dir,
        lambda _, m: m["leaves"][next(iter(m["leaves"]))].update(file=str(secret)),
    )
    with pytest.raises(ValueError, match="Refusing to open an absolute"):
        TreeDataSource(data_dir)[0]


def test_parent_traversal_is_refused(dataset):
    """`..` segments are not normalized away by a plain join either."""
    data_dir, _ = dataset
    _retamper(
        data_dir,
        lambda _, m: m["leaves"][next(iter(m["leaves"]))].update(
            file="../../secret.bin"
        ),
    )
    with pytest.raises(ValueError, match=r"containing '\.\.'"):
        TreeDataSource(data_dir)[0]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda m: m["leaves"][next(iter(m["leaves"]))].update(dtype="object"),
            "not one of",
        ),
        (
            lambda m: m["leaves"][next(iter(m["leaves"]))].update(
                shape_per_sample=[1, -5]
            ),
            "invalid shape_per_sample",
        ),
        (
            lambda m: m["structure"]["children"].update(
                evil={"type": "dict", "children": {}}
            ),
            "not an AudioTree field",
        ),
        (lambda m: m["structure"].update(sample_rate=0), "sample_rate"),
        (lambda m: m.update(num_samples=-1), "non-negative"),
    ],
)
def test_manifest_schema_is_validated(dataset, mutate, match):
    """Declared dtypes, shapes, sample rates and field names are checked."""
    data_dir, _ = dataset
    _retamper(data_dir, lambda _, m: mutate(m))
    with pytest.raises(ValueError, match=match):
        TreeDataSource(data_dir)


@pytest.mark.parametrize(
    "mutate,match",
    [
        # Container types: '"leaves": []' is falsy, so it used to slip past a
        # `... or {}` guard and crash at the first __getitem__ with a raw
        # AttributeError from _open_data_files.
        (lambda m: m.update(leaves=[]), "'leaves' must be a dict"),
        (
            lambda m: m["leaves"].update({next(iter(m["leaves"])): "junk"}),
            "entry must be a dict",
        ),
        (lambda m: m.update(string_leaves=[]), "'string_leaves' must be a dict"),
        # Structure nodes must be well-formed: a dict node without 'children'
        # used to die with a raw KeyError inside _reconstruct.
        (lambda m: m["structure"].pop("children"), "must carry a 'children' dict"),
        (
            lambda m: m["structure"]["children"].update(metadata={"type": "dict"}),
            "must carry a 'children' dict",
        ),
        (
            lambda m: m["structure"]["children"].update(
                metadata={"type": "banana", "children": {}}
            ),
            "unknown type",
        ),
        (
            lambda m: m["structure"]["children"].update(metadata=42),
            "must be a leaf-name string or a dict",
        ),
    ],
)
def test_manifest_container_types_are_validated(dataset, mutate, match):
    """Malformed containers fail at construction with a named error."""
    data_dir, _ = dataset
    _retamper(data_dir, lambda _, m: mutate(m))
    with pytest.raises(ValueError, match=match):
        TreeDataSource(data_dir)


@pytest.mark.parametrize(
    "mutate,match",
    [
        # A leaf name the structure mentions but the leaf tables don't declare
        # used to resolve to the excluded-leaf sentinel in _reconstruct --
        # silently missing fields, indistinguishable from deliberate exclusion.
        (
            lambda m: m["structure"]["children"].update(waveform="ghost"),
            "not declared",
        ),
        (
            lambda m: m["structure"]["children"].update(
                metadata={"type": "string_leaf", "leaf": "ghost"}
            ),
            "not declared",
        ),
        (
            lambda m: m["structure"]["children"].update(
                metadata={"type": "string_leaf", "leaf": 3}
            ),
            "must be a string leaf name",
        ),
        # A dangling ROOT node used to leak the _EXCLUDED sentinel object
        # itself to the caller.
        (lambda m: m.update(structure="ghost"), "not declared"),
        # Renaming a declared leaf dangles the structure's reference to it.
        (
            lambda m: m.update(
                leaves={"renamed": m["leaves"][next(iter(m["leaves"]))]}
            ),
            "not declared",
        ),
    ],
)
def test_structure_leaf_references_must_resolve(dataset, mutate, match):
    """Every structure leaf reference must name a declared (string) leaf."""
    data_dir, _ = dataset
    _retamper(data_dir, lambda _, m: mutate(m))
    with pytest.raises(ValueError, match=match):
        TreeDataSource(data_dir)


def test_untampered_dataset_still_reads(dataset):
    """The guards must not reject a manifest the writer actually produced."""
    data_dir, _ = dataset
    assert TreeDataSource(data_dir)[0].waveform.shape == (1, 1, 12)


def test_safe_join_accepts_ordinary_relative_paths(tmp_path):
    """Nested relative paths inside the directory remain legal."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "leaf.bin").write_bytes(b"")
    assert (
        safe_join(tmp_path, "sub/leaf.bin") == (tmp_path / "sub" / "leaf.bin").resolve()
    )


def test_safe_join_rejects_symlink_escape(tmp_path):
    """A symlink inside the dataset pointing outside it is still an escape."""
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "link.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside"):
        safe_join(data_dir, "link.bin")


def test_backslash_traversal_is_named_not_merely_caught(tmp_path):
    """A Windows-style ``..`` traversal is refused for the right reason.

    ``safe_join`` checked ``".." in PurePosixPath(relative).parts``, but to
    PurePosixPath ``..\\..\\etc\\hosts`` is one opaque part -- so the named check
    missed it and only the resolve()-based containment check below caught it.
    Refused either way, but with a message that did not say why, on the one
    platform where backslashes are the native separator.
    """
    with pytest.raises(ValueError, match=r"containing '\.\.'"):
        safe_join(tmp_path, r"..\..\etc\hosts", description="audio file")

    # The POSIX spelling keeps naming itself too.
    with pytest.raises(ValueError, match=r"containing '\.\.'"):
        safe_join(tmp_path, "../../etc/hosts", description="audio file")

    # A backslash that is not a traversal stays a legal (if odd) filename.
    assert safe_join(tmp_path, r"weird\name.wav", description="audio file")
