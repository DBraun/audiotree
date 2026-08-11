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

# Run tests: unit suite + src/ docstring examples (via --doctest-modules).
# NOTE: this does NOT run the docs/source/*.rst examples — see the doctest
# command below.
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run a single test file
uv run pytest tests/test_resample.py

# Run the docs/source/*.rst examples (.. testcode:: / .. testoutput::).
# pytest does not collect these; the Sphinx doctest builder does. This is the
# exact command CI runs in its build_docs job.
uv run make -C docs doctest

# Build documentation (HTML)
uv run make -C docs html

# Build package
uv build
```

## High-Level Architecture

The codebase is organized into three main components:

1. **Core (`audiotree.core`)**: The `AudioTree` dataclass is the central data structure, representing audio as JAX arrays with shape (batch × channels × samples).
It carries the sample rate, per-item label fields, an `extras` dict of user-supplied per-item arrays, and a library-internal `_metadata` provenance container (the encoded `filepath`/`source` arrays plus the plain `offset` array — closed schema, read via the `.filepath`/`.source`/`.offset` properties; private, and serialized on disk under the name `metadata`), and provides methods for loading files, resampling, and computing loudness. `replace_lufs()` fills two fields: `lufs`, the gated BS.1770 integrated loudness (NumPy via the `loudness` library, JAX via a vmapped `jaxloudnorm` meter), and `lufs_windows`, the ungated per-window loudness-over-time curve with an optional `hop_duration_sec` for overlap (computed natively so it needs no unreleased `loudness` build — NumPy uses exact K-weighting IIR biquads via `scipy`, JAX uses `jaxloudnorm`'s FIR-approximated K-weighting; see `audiotree.loudness`).

1. **Datasources (`audiotree.sources`)**: Provides integration with Google's Grain library for ML data pipelines.
The key functions are `create_audio_dataset()` for simple loading and `create_balanced_audio_dataset()` for multi-group balanced sampling.
Both use grain's `random_map` for proper RNG seeding.
Excerpt selection is configured with an `ExcerptConfig` passed to these constructors — a fresh random offset per read gives infinite RNG variety even when files repeat.
`create_windowed_audio_dataset()` (in `audiotree.sources.windowed`) instead makes the *window* the unit of sampling: each file is tiled into `round(n_windows ** alpha)` jittered slots, globally shuffled for even coverage, length-aware frequency (`alpha`), and batch diversity. Optional build-time loudness filtering reads a `build_window_lufs_cache()` bagz cache of ragged per-file windowed-LUFS arrays. It composes with `create_balanced_audio_dataset()` via `WindowConfig`.

1. **Transforms (`audiotree.transforms`)**: Audio augmentations with dual backends: NumPy (`audiotree.transforms`) for CPU grain pipelines using `np.random.Generator`, and JAX (`audiotree.transforms.jax`) for GPU/JIT training using `jax.random.key`.

1. **Writer (`audiotree.writer`)**: The `AudioWriter` class provides sequential writing of AudioTree batches to disk as individual audio files plus an NPZ manifest with one row per item (loudness, pitch, the dedicated `filepath`/`source`/`offset` bookkeeping columns, custom tags, and `extras_*` columns that are purely user payload). `write_audio=False` writes the manifest alone (e.g. embeddings/features in `extras`). Its output is read back with `audiotree.sources.AudioDataSource` (a Grain `RandomAccessDataSource`) or `AudioTree.from_manifest()` (the whole manifest as one batched AudioTree, with an optional `filter_fn`); both restore the recorded `extras` arrays and the `filepath`/`source`/`offset` provenance.
The `TreeWriter` class (`audiotree.tree_writer`) writes arbitrary pytrees — AudioTrees, dicts of AudioTrees, or nested structures — to memory-mapped binary files (with `bagz` for string leaves), enabling zero-copy random access via `audiotree.sources.TreeDataSource` as a Grain `RandomAccessDataSource`. An AudioTree node serializes both per-item dicts as separate children: a `metadata` node holding only the `filepath`/`source`/`offset` leaves, and an `extras` node holding the user's leaves.

Three "metadata"-adjacent concepts stay distinct — keep them straight in docs and code: `AudioTree.extras` is the *user's* per-item payload dict (entirely the user's; the library plants nothing there); `AudioTree._metadata` is the *library-internal* per-item provenance container (private; the encoded `filepath`/`source` plus the plain `offset` array; closed schema — anything else in a stored `metadata` node, its on-disk spelling, is rejected by name at read; read via the `.filepath`/`.source`/`.offset` properties); `TreeWriter(metadata=...)` is *dataset-level* metadata, stored once under the manifest's top-level `"metadata"` key and read via `TreeDataSource.get_metadata()`.

## Documentation & learning the library

`docs/source/` is the library's teaching material — narrative guides with **runnable**
examples (published at <https://dirt.design/audiotree>). Reading them in order is the
fastest way to onboard, and they are the reference to reach for when a change touches
a subsystem:

- `introduction/introduction.rst` — the `AudioTree` container: construction, batching,
  loudness, indexing, and pytree operations.
- `introduction/sources.rst` — loading audio into Grain pipelines; saliency; time-aligned
  annotations; resumable and accelerator-prefetched training.
- `introduction/transform_chaining.rst` + `introduction/transforms.rst` — augmentations,
  and the NumPy (CPU data-loader) vs JAX (jitted, on-device) transform backends.
- `introduction/writer.rst` — writing datasets: `AudioWriter`/`AudioDataSource` (audio
  files + NPZ manifest) vs `TreeWriter`/`TreeDataSource` (memmapped pytrees).
- `introduction/{balanced_datasets,windowed_datasets,dict_batches,argbind_guide,multiprocessing}.rst`
  — deeper topics.
- `audiotree_api/` — autodoc API reference generated from the docstrings.

**Examples are tested in two places; keep both green.** `src/` docstring examples
(`>>> ...`) run under `uv run pytest` (`--doctest-modules`), but the `docs/source/*.rst`
`.. testcode::` / `.. testoutput::` blocks run **only** under the Sphinx doctest builder
(`uv run make -C docs doctest`) — `pytest` does not collect them. CI enforces both (the
`build_docs` job runs `make -C docs doctest`), so a broken `.rst` example fails CI even
though it passes `pytest` locally; run the doctest builder yourself before pushing. When
you change public behavior, update the relevant guide and its example and run both
surfaces so a stale example can't slip through. Prefer executable `testcode`/`testoutput`
over inert `code-block` for anything that can run, and keep examples device-agnostic (no
GPU required — pass `backend="cpu"` where a device is implied).

## Changelog

User-facing changes (public API, behavior, or dependencies) are tracked in `CHANGELOG.md`. Add a bullet under its `## [Unreleased]` section, following the category conventions documented at the top of that file.
