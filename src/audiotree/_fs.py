"""Filesystem helpers shared by the writers."""

import json
import os
from pathlib import Path
from typing import Any


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
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=indent)
    os.replace(tmp, path)
