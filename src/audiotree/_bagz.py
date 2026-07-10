"""Optional ``bagz`` dependency, resolved lazily.

``bagz`` ships Linux-only wheels (see ``pyproject.toml``), so it must not be
imported at module scope: audiotree works fine on other platforms as long as
no feature that stores bagz records (string leaves, windowed-LUFS caches) is
used. Call :func:`require_bagz` at the point of use.
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
            "installed. bagz ships Linux-only wheels; on other platforms "
            "install it manually or avoid this feature."
        ) from e
    return bagz
