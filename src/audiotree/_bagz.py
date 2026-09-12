"""Platform-dependent ``bagz`` dependency, resolved lazily.

``bagz`` is installed by default on macOS and Linux. Resolve it lazily so
other platforms can use features that do not need record storage.
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
            "installed. Install it with `pip install 'bagz>=0.3.8'`. "
            "Bagz 0.3.8+ provides wheels for Linux x86-64 and macOS Apple "
            "Silicon; other platforms may need a source build, or you can "
            "exclude the string leaves that need it."
        ) from e
    return bagz
