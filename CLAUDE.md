# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AudioTree is an audio data loading and augmentation library built on JAX.
It provides a clean API for handling audio data in machine learning pipelines, with support for various transformations and integration with Google's Grain data loading library.

## Common Development Commands

```bash
# Install for development (includes all dev dependencies)
pip install ".[dev]"

# Run tests
python3 -m pytest tests

# Run tests with coverage
pytest --cov

# Run a single test file
python3 -m pytest tests/test_resample.py

# Build documentation
cd docs && make html

# Build package
python3 -m build
```

## High-Level Architecture

The codebase is organized into three main components:

1. **Core (`audiotree.core`)**: The `AudioTree` dataclass is the central data structure, representing audio as JAX arrays with shape (batch × channels × samples).
It includes metadata like sample rate and provides methods for loading files, resampling, and computing loudness.

1. **Datasources (`audiotree.sources`)**: Provides integration with Google's Grain library for ML data pipelines.
The key functions are `create_audio_dataset()` for simple loading and `create_balanced_audio_dataset()` for multi-group balanced sampling.
Both use grain's `random_map` for proper RNG seeding. The `load_audio_with_saliency()` function handles saliency-based excerpt selection with infinite RNG variety even when files repeat.

1. **Transforms (`audiotree.transforms`)**: Audio augmentations with dual backends: NumPy (`audiotree.transforms`) for CPU grain pipelines using `np.random.Generator`, and JAX (`audiotree.transforms.jax`) for GPU/JIT training using `jax.random.key`.

1. **Writer (`audiotree.writer`)**: The `AudioWriter` class provides sequential writing of AudioTree batches to disk with automatic manifest generation.
Manifests can be saved as NPZ (default, best for large datasets), JSON, or CSV formats, tracking metadata like loudness, pitch, and file paths.
The NPZ format offers significant compression benefits for datasets with 100+ items.

## Key Design Patterns

- All audio data is represented as JAX arrays, enabling GPU acceleration and integration with JAX-based ML frameworks
- The library uses dataclasses extensively for clean, immutable data structures
- Audio loading is lazy when possible to optimize memory usage in data pipelines
