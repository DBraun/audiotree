"""Filesystem helpers shared by the writers and the manifest-driven readers."""

import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def safe_join(base: Path, relative: str, *, description: str = "path") -> Path:
    """Join a manifest-supplied *relative* onto *base*, refusing to escape it.

    A dataset directory is untrusted input: manifests travel with the data they
    describe, so anyone who shares, mirrors, or downloads a pre-rendered dataset
    is handing the reader a file that names the paths it will open. ``pathlib``
    discards the left operand when the right is absolute (``Path("/data") /
    "/etc/passwd"`` is ``/etc/passwd``) and does not normalize ``..``, so an
    unchecked join reads whatever the manifest asks for.

    Symlinks are resolved before the containment check, so a link inside the
    dataset pointing outside it is also rejected.

    Raises:
        ValueError: If *relative* is absolute, contains ``..``, or resolves
            outside *base*.
    """
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ValueError(
            f"Refusing to open an absolute {description} from a manifest: "
            f"{relative!r}. Paths must be relative to the dataset directory."
        )
    # Both flavours, mirroring the absoluteness check above. To PurePosixPath,
    # `..\..\etc\hosts` is a single opaque part, so a backslash traversal would
    # slip past this named check and be caught only by the containment check
    # below -- refused either way, but with a message that does not say why.
    if ".." in PurePosixPath(relative).parts or ".." in PureWindowsPath(relative).parts:
        raise ValueError(
            f"Refusing to open a {description} containing '..' from a manifest: "
            f"{relative!r}."
        )
    base_resolved = base.resolve()
    resolved = (base_resolved / relative).resolve()
    if resolved != base_resolved and base_resolved not in resolved.parents:
        raise ValueError(
            f"Manifest {description} {relative!r} resolves outside the dataset "
            f"directory {base}."
        )
    return resolved


def refuse_to_clobber(directory: Path, patterns: tuple[str, ...]) -> None:
    """Raise if *directory* already holds dataset files matching *patterns*.

    Both writers open their outputs in truncating mode, so pointing one at a
    directory that already holds a dataset destroys it with no warning — and two
    processes aiming at one directory (the natural way to parallelize a
    pre-render) interleave into the same files, last close winning.
    """
    for pattern in patterns:
        for existing in sorted(directory.glob(pattern)):
            raise FileExistsError(
                f"{directory} already contains a dataset ({existing.name}). "
                f"Writing here would overwrite it. Choose another directory, "
                f"remove the existing one, or pass exist_ok=True to overwrite."
            )


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write JSON to *path* via a temp file and a rename.

    A reader either sees the previous file or the new one, never a partial
    write — which matters because these manifests are refreshed while the
    dataset they describe is still being written.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        # A failed dump (an unserializable payload, a full disk) must not
        # strand a partial .tmp next to the real file.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
