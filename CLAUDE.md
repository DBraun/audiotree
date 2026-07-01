# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioTree is an audio data loading and augmentation library built on JAX.
It provides a clean API for handling audio data in machine learning pipelines, with support for various transformations and integration with Google's Grain data loading library.

## Common Development Commands

This project uses [uv](https://docs.astral.sh/uv/). Dev dependencies live in the
`dev` [dependency group](https://peps.python.org/pep-0735/) (installed by `uv sync`).

```bash
# Install for development (project + the `dev` dependency group)
uv sync

# Run tests (doctests in src/ + the tests/ suite)
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run a single test file
uv run pytest tests/test_resample.py

# Build documentation
uv run make -C docs html

# Build package
uv build
```

## High-Level Architecture

The codebase is organized into three main components:

1. **Core (`audiotree.core`)**: The `AudioTree` dataclass is the central data structure, representing audio as JAX arrays with shape (batch × channels × samples).
It includes metadata like sample rate and provides methods for loading files, resampling, and computing loudness. `replace_lufs()` fills two fields: `lufs`, the gated BS.1770 integrated loudness (NumPy via the `loudness` library, JAX via a vmapped `jaxloudnorm` meter), and `lufs_windows`, the ungated per-window loudness-over-time curve with an optional `hop_duration_sec` for overlap (computed natively so it needs no unreleased `loudness` build — NumPy uses exact K-weighting IIR biquads via `scipy`, JAX uses `jaxloudnorm`'s FIR-approximated K-weighting; see `audiotree.loudness`).

1. **Datasources (`audiotree.sources`)**: Provides integration with Google's Grain library for ML data pipelines.
The key functions are `create_audio_dataset()` for simple loading and `create_balanced_audio_dataset()` for multi-group balanced sampling.
Both use grain's `random_map` for proper RNG seeding.
The `load_audio_with_saliency()` function handles saliency-based excerpt selection with infinite RNG variety even when files repeat.
`create_windowed_audio_dataset()` (in `audiotree.sources.windowed`) instead makes the *window* the unit of sampling: each file is tiled into `round(n_windows ** alpha)` jittered slots, globally shuffled for even coverage, length-aware frequency (`alpha`), and batch diversity. Optional build-time loudness filtering reads a `build_window_lufs_cache()` bagz cache of ragged per-file windowed-LUFS arrays. It composes with `create_balanced_audio_dataset()` via `WindowParams`.

1. **Transforms (`audiotree.transforms`)**: Audio augmentations with dual backends: NumPy (`audiotree.transforms`) for CPU grain pipelines using `np.random.Generator`, and JAX (`audiotree.transforms.jax`) for GPU/JIT training using `jax.random.key`.

1. **Writer (`audiotree.writer`)**: The `AudioWriter` class provides sequential writing of AudioTree batches to disk with automatic manifest generation.
Manifests can be saved as NPZ (default, best for large datasets), JSON, or CSV formats, tracking metadata like loudness, pitch, and file paths.
The `TreeWriter` class (`audiotree.tree_writer`) writes arbitrary pytrees — AudioTrees, dicts of AudioTrees, or nested structures — to memory-mapped binary files (with `bagz` for string leaves), enabling zero-copy random access via `audiotree.sources.TreeDataSource` as a Grain `RandomAccessDataSource`.

## Changelog

User-facing changes (public API, behavior, or dependencies) are tracked in `CHANGELOG.md`. Add a bullet under its `## Unreleased` section, following the category conventions documented at the top of that file.
