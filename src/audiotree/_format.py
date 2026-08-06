"""Versioning for audiotree's on-disk formats.

audiotree writes three kinds of artifact that users keep for years — a
``TreeWriter`` directory, an ``AudioWriter`` manifest, and a windowed-LUFS cache
— so at 1.0 their layouts become a compatibility contract. This module holds the
one header those artifacts carry and the one check every reader performs, so the
rules live in a single place rather than being re-improvised per format.

Each artifact records:

``format``
    Which of the three it is, so pointing a reader at the wrong directory fails
    by name instead of by ``KeyError``.
``format_version``
    ``[major, minor]``. A **major** bump means an incompatible layout; a reader
    refuses it. A **minor** bump means additive change — new fields a reader may
    ignore — so old readers keep working.
``min_reader_version``
    The oldest reader that can make sense of this artifact. This is what makes
    additive minors safe: a writer that adds a field *readers must honor* raises
    this instead of bumping major, and older readers refuse rather than silently
    ignoring it.
``producer``
    The writing audiotree version, for debugging.

Artifacts written by a pre-1.0 audiotree carry no header and are refused with an
explicit "re-render this dataset" message. None of these formats shipped in a
released version (1.0 is the first release to contain ``TreeWriter`` at all), so
there is nothing in the wild to stay compatible with, and starting the contract
clean is worth more than accommodating development-time artifacts.
"""

from typing import Dict, Tuple

# The formats, by their `format` string.
TREE = "audiotree-tree"
MANIFEST = "audiotree-manifest"
LUFS_WINDOWS_CACHE = "audiotree-lufs-windows"

# What this build of audiotree writes, and the newest layout it can read.
CURRENT_VERSION: Tuple[int, int] = (1, 0)

# NPZ has no place for scalars alongside per-entry columns, so the AudioWriter
# manifest stores its header as JSON-encoded 0-d arrays under this prefix. The
# rest of that layout -- columns, fixed-width strings, presence masks -- lives in
# `audiotree._manifest`, which is the only module that reads or writes it.
NPZ_HEADER_PREFIX = "__audiotree_"

_HUMAN_NAMES = {
    TREE: "TreeWriter dataset",
    MANIFEST: "AudioWriter manifest",
    LUFS_WINDOWS_CACHE: "windowed-LUFS cache",
}


def header(
    format_name: str,
    *,
    format_version: Tuple[int, int] = CURRENT_VERSION,
    min_reader_version: Tuple[int, int] = (1, 0),
) -> Dict[str, object]:
    """Build the header a writer stamps onto an artifact."""
    from audiotree import __version__

    return {
        "format": format_name,
        "format_version": list(format_version),
        "min_reader_version": list(min_reader_version),
        "producer": f"audiotree {__version__}",
    }


def _as_pair(value, field: str, source: str) -> Tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError(
            f"{source}: {field} must be a [major, minor] pair of ints, got {value!r}"
        )
    return (value[0], value[1])


def check(manifest: Dict, expected_format: str, *, source: str) -> Tuple[int, int]:
    """Validate an artifact's header and return its ``(major, minor)`` version.

    Args:
        manifest: The parsed header/manifest mapping.
        expected_format: Which format the caller is prepared to read.
        source: Path or description used in error messages.

    Returns:
        The artifact's ``format_version``.

    Raises:
        ValueError: If the artifact carries no header (pre-1.0), is a different
            format, was written by a newer incompatible audiotree, or requires a
            newer reader.
    """
    human = _HUMAN_NAMES.get(expected_format, expected_format)
    found_format = manifest.get("format")

    if found_format is None:
        raise ValueError(
            f"{source}: this {human} carries no format header, so it was written "
            f"by a pre-1.0 audiotree. Those layouts were never released and are "
            f"not read by 1.0 — re-render the dataset."
        )

    if found_format != expected_format:
        raise ValueError(
            f"{source}: expected a {human} (format {expected_format!r}) but found "
            f"format {found_format!r}."
        )

    version = _as_pair(manifest.get("format_version"), "format_version", source)
    min_reader = _as_pair(
        manifest.get("min_reader_version", [1, 0]), "min_reader_version", source
    )

    if version[0] != CURRENT_VERSION[0]:
        raise ValueError(
            f"{source}: this {human} is format_version {version[0]}.{version[1]}, "
            f"but this audiotree reads {CURRENT_VERSION[0]}.x. "
            f"It was written by {manifest.get('producer', 'an unknown version')}."
        )
    if min_reader > CURRENT_VERSION:
        raise ValueError(
            f"{source}: this {human} requires a reader of at least "
            f"{min_reader[0]}.{min_reader[1]}, but this audiotree is "
            f"{CURRENT_VERSION[0]}.{CURRENT_VERSION[1]}. "
            f"It was written by {manifest.get('producer', 'an unknown version')}."
        )
    return version
