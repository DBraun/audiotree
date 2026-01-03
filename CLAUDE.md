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

1. **Datasources (`audiotree.datasources`)**: Provides integration with Google's Grain library for ML data pipelines.
The key abstraction is `AudioDataSourceMixin` which other sources inherit from.
Sources handle loading audio files and can balance sampling across multiple data sources.

1. **Transforms (`audiotree.transforms`)**: A collection of audio augmentations that operate on AudioTree objects.
Transforms can be applied in any order and include volume changes, phase manipulation, and neural codec encoding.
All transforms follow a consistent interface inheriting from `BaseTransform`.

1. **Writer (`audiotree.writer`)**: The `AudioWriter` class provides sequential writing of AudioTree batches to disk with automatic manifest generation.
Manifests can be saved as NPZ (default, best for large datasets), JSON, or CSV formats, tracking metadata like loudness, pitch, and file paths.
The NPZ format offers significant compression benefits for datasets with 100+ items.

## Key Design Patterns

- All audio data is represented as JAX arrays, enabling GPU acceleration and integration with JAX-based ML frameworks
- The library uses dataclasses extensively for clean, immutable data structures
- Audio loading is lazy when possible to optimize memory usage in data pipelines
