"""Optional ``bagz`` dependency, resolved lazily.

``bagz`` publishes manylinux x86-64 wheels only — no macOS, no Linux aarch64,
and nothing for Python 3.14 — so it is an optional extra (``audiotree[bagz]``)
rather than a hard requirement, and must not be imported at module scope.
audiotree works fine without it as long as no feature that stores bagz records
(string leaves in TreeWriter/TreeDataSource, windowed-LUFS caches) is used.
Call :func:`require_bagz` at the point of use — and only for leaves that are
actually being read, so ``exclude_prefixes`` can skip string leaves on a
platform where bagz is unavailable.
"""


def require_bagz(purpose: str):
    """Import and return ``bagz``, or raise a clear error naming the feature.

    Args:
        purpose: Human-readable description of the feature needing bagz,
            used in the error message.
    """
    try:
        import bagz
    except ImportError as e:
        raise ImportError(
            f"The 'bagz' package is required for {purpose}, but it is not "
            "installed. Install it with `pip install audiotree[bagz]`. Note "
            "that bagz publishes manylinux x86-64 wheels only, so on macOS, "
            "Linux aarch64, or Python 3.14 you may need to build it from "
            "source — or exclude the string leaves that need it."
        ) from e
    return bagz
