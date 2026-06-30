# Change log

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
(categories: Added, Changed, Deprecated, Removed, Fixed, Security), and AudioTree
follows [Effort-based Versioning](https://jacobtomlinson.dev/effver/).

## Unreleased

### Added

* **`peak_norm()` transform**: Peak-normalizes audio so its largest absolute value is `1.0`, dividing by the per-item peak (across channels and samples, clamped to a small epsilon). Unlike `rescale_audio()`, which only scales down audio exceeding `[-1.0, 1.0]`, this always normalizes to a peak of `1.0`. Available in both the NumPy (`audiotree.transforms`) and JAX (`audiotree.transforms.jax`) backends.
* **`TreeWriter` and `TreeDataSource`**: A pytree-native writer and reader for memory-mapped datasets. Instead of manually extracting and reconstructing AudioTree fields, uses `jax.tree_util` to decompose any pytree into leaves—each leaf becomes a separate memmap file. A JSON structure descriptor in the manifest enables exact reconstruction. Accepts AudioTree objects, dicts of arrays, dicts of AudioTrees, or nested combinations, all through a single `write()` method. The reader takes a directory path and implements Grain's `RandomAccessDataSource` with a minimal interface (`__len__`, `__getitem__`). Drops `FieldSpec`, `AudioTreeFieldExtractor`, `audio_dtype`, string fields, and `transform_fn` in favor of a cleaner, more composable design.
* **String leaf support in `TreeWriter` / `TreeDataSource`**: `write()` now accepts `str` or `List[str]` leaves alongside arrays and AudioTrees. Strings are stored in [Bagz](https://github.com/google/bagz) files (one per string leaf) with no length limits or truncation. Single-sample reads return a bare `str`, and `AudioTree.batch` naturally collects them into `List[str]`. Backed by the `bagz` dependency, which is now a standard requirement (it builds on both Linux and macOS), so string leaves work out of the box.
* **Selective field loading via `exclude_prefixes`**: `TreeDataSource` accepts an `exclude_prefixes` parameter to skip loading specific leaves by dot-separated name prefix. For example, `exclude_prefixes=["wet.waveform"]` skips the audio memmap, and `exclude_prefixes=["dry"]` skips all leaves under the `dry` subtree. Excluded leaves are omitted from the reconstructed pytree (AudioTree fields default to `None`, metadata keys are absent). Prefix matching uses exact-or-dot semantics (`"dry"` matches `"dry"` and `"dry.waveform"` but not `"dryness"`).
* **`load_into_memory` for zero-copy worker access**: `TreeDataSource` accepts `load_into_memory=True` to load all non-excluded array and string leaves into RAM at init time. With fork-based multiprocessing (default on Linux), workers inherit the parent's data via copy-on-write, eliminating disk I/O entirely. String leaves (bagz) are read into a `List[str]`.
* **`cache_memmaps` toggle for page cache control**: `TreeDataSource` accepts `cache_memmaps=False` to reopen memmap files on every `__getitem__` call instead of caching them. This lets the OS reclaim pages between accesses, preventing the page cache from growing unboundedly when randomly accessing large files. Trades a small CPU overhead for controlled memory. Ignored when `load_into_memory=True`. See [nanoGPT](https://github.com/karpathy/nanoGPT/blob/3adf61e/train.py#L117-L118) for the same pattern.
* **Graceful under/overshoot handling in `TreeWriter`**: `expected_samples` is an allocation hint rather than an exact requirement. If a batch would exceed the allocated size, it is silently trimmed to fit. On `close()`, if fewer samples were written than allocated, memmap files are truncated to the actual sample count via `os.truncate()`. This avoids errors when the exact dataset size isn't known ahead of time (e.g., split-dependent counts) and eliminates wasted disk space.
* **`AudioTree.write()`**: Saves a single-item tree (`batch_size == 1`) to an audio file via `soundfile`, the inverse of `from_file()`. Takes a `filepath` (format inferred from its extension) plus optional `subtype`/`format`/`endian` passthroughs; it does not take a sample rate (uses `self.sample_rate` — call `resample()` first to change it). Asserts `batch_size == 1`, so index or iterate the batch first (e.g. `for item in tree: item.write(...)`).
* **`AudioTree.__iter__`**: AudioTree is now a proper `collections.abc.Iterable`. Iterating yields one batch-of-1 AudioTree per item along the leading batch axis (e.g. a batch-16 tree yields 16 trees of `batch_size == 1`). `for` loops worked before via the sequence protocol, but `isinstance(tree, Iterable)` now returns `True`.
* **`AudioTree.samples` property**: Returns the number of samples in the waveform (`waveform.shape[-1]`).
* **`AudioTree.source` property**: Returns the source group name for each item in the batch. When using `create_balanced_audio_dataset()` with `sources={"music": [...], "speech": [...]}`, each loaded AudioTree tracks which source group it came from.
* **`AudioTree.normalize_loudness(target_lufs)`**: Normalizes audio to a target LUFS level. Computes loudness if not already set, scales audio, and updates the loudness field.
* **`AudioTree.create()`**: New classmethod with automatic audio dimensionality handling (1D→3D, 2D→3D) and `filepaths` parameter support.
* **`AudioTree.filter()`**: Takes a function `filter_fn(AudioTree) -> bool`.
* **`AudioTree.split()`, `AudioTree.reshape_mini_batches()`, `AudioTree.flatten_mini_batches()`**: Split an AudioTree into a list, and add or remove a mini-batch axis (useful for `nnx.scan`).
* **`filepaths` parameter**: Now a kwarg to `AudioTree.create()` and `AudioTree.from_file()`.
* **`pad_mode` parameter**: Added to `AudioTree.from_file()`. Options include `"constant"` (zero padding), `None` (don't pad), and `"wrap"` (loop audio).
* **JAX transforms module**: `audiotree.transforms.jax` provides JAX-native transforms for GPU/JIT training pipelines. Uses `jax.random.key` and JAX operations throughout. Includes: `volume_norm`, `volume_change`, `invert_phase`, `swap_stereo`, `corrupt_phase`, `shift_phase`, `roll`, `trim`, `rescale_audio`, `peak_norm`, `resample`, `identity`.
* **`roll()` and `trim()` transforms**: For rolling audio in time and trimming/padding to fixed length with configurable padding modes.
* **`resample()` transform**: Resamples to a new sample rate in both the NumPy (`audiotree.transforms`) and JAX (`audiotree.transforms.jax`) backends, wrapping `AudioTree.resample` (so NumPy waveforms use librosa and JAX waveforms use the Julius port).
* **Transform decorators**: New `@random_transform` and `@map_transform` in `audiotree.transforms.decorators` for creating custom transforms from simple functions. Handles all boilerplate for `prob`, `scope`, `output_key`, and `split_seed` parameters.
* **`create_audio_dataset()`**: For creating simple audio datasets without balancing. Loads all files from one or more directories and applies grain's `random_map` for proper RNG seeding.
* **`find_audio_files()`**: New public helper in `audiotree.sources` for recursively listing audio files under one or more directories (skipping hidden files/dirs, filtering by extension). The result is now **sorted**, so the file order — and therefore seeded shuffling — is deterministic across machines and filesystems (previously raw `os.walk` order, which varied by filesystem).
* **`create_audio_dataset(filepaths=...)`**: `create_audio_dataset` now accepts an explicit `filepaths` list as an alternative to `sources` (exactly one of the two is required). This reuses the function's clean API and parameters for custom splits — e.g. a train/val split over a single directory — without reorganizing files on disk.
* **`load_audio_with_saliency()`**: For loading audio files with optional saliency-based excerpt selection. Designed to work with grain's `random_map`. Exported in `audiotree.sources`.
* **`AudioWriter`**: New class for writing AudioTree batches to disk with manifest generation (JSON/CSV/NPZ formats).
* **`ManifestDataSource`**: New class for reading AudioWriter outputs as a grain RandomAccessDataSource with metadata preservation and filtering capabilities.
* **New documentation guides**: Balanced Datasets, ArgBind Guide, Transform Chaining, Dict Batches, and Multiprocessing.
* **New tests**: Balanced datasets (12 tests covering hierarchical directory structures, unbalanced group sizes, statistical accuracy of weight-based sampling, and edge cases), multiprocessing/multithreading (6), transform chaining (8), and `Dict[str, AudioTree]` batch processing with scope (9).
* **`librosa` dependency**: For NumPy-based STFT operations in transforms (alongside `librosax` for JAX).
* **`bagz` dependency**: Now a standard install requirement (previously optional) so `TreeWriter` / `TreeDataSource` string leaves work without extra setup. Builds on Linux and macOS.
* **Pre-commit + Ruff**: Added a `.pre-commit-config.yaml` running Ruff (lint with `--fix` and Black-compatible formatting) plus basic file-hygiene hooks. `pre-commit` is included in the `dev` extra; enable with `pre-commit install`.

### Changed

* **Breaking — Renamed `AudioTree.audio_data` to `AudioTree.waveform`**: The primary audio field is now called `waveform` (matching torchaudio's convention and avoiding ambiguity with sample counts). This affects all keyword arguments (`AudioTree(waveform=...)`, `tree.replace(waveform=...)`) and attribute access. It also changes the on-disk leaf names produced by `TreeWriter`: new datasets are written with `waveform.bin` (or e.g. `dry.waveform.bin` when nested) and matching `manifest.json` keys. To migrate an existing dataset, rename each `*audio_data.bin` file to the corresponding `*waveform.bin` and replace every occurrence of `audio_data` with `waveform` in its `manifest.json`.
* **Breaking — Renamed `AudioTree.batch_fn` to `AudioTree.batch`**: The static collation method (used with Grain's `IterDataset.batch()`) is now `AudioTree.batch`. Update call sites to `ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)` — Grain's `batch_fn=` keyword is unchanged; only the function it points to was renamed.
* **Breaking — Separate NumPy and JAX transform backends**: Transforms now have two implementations:
  - `audiotree.transforms` (NumPy): For CPU-based grain data pipelines. Uses `np.random.Generator`.
  - `audiotree.transforms.jax` (new): For GPU/JIT training pipelines. Uses `jax.random.key`.

  The base transforms in `audiotree.transforms` now expect `np.random.Generator` instead of `jax.Array` for the RNG parameter. Use `np.random.default_rng(seed)` instead of `jax.random.key(seed)`. For JAX-based transforms in jitted training loops, import from `audiotree.transforms.jax`.

* **Breaking — Transform API redesign**: Completely redesigned from class-based to function-based pattern. All transforms are now functions decorated with `@random_transform` or `@map_transform`, enabling direct parameter binding with argbind. Parameters are now flat (e.g., `volume_norm.min_db: -20`) instead of nested in a `config` dict. This allows CLI parameter binding (e.g., `--volume_norm.min_db=-20`) which was not possible with the old API. All transform names changed from PascalCase to snake_case (e.g., `VolumeNorm` → `volume_norm`, `Trim` → `trim`). Old class-based transforms removed with no backward compatibility. Migrated transforms: `identity()`, `mono()`, `stereo()`, `volume_change()`, `volume_norm()`, `rescale_audio()`, `invert_phase()`, `swap_stereo()`, `corrupt_phase()`, `shift_phase()`, `roll()`, `trim()`, `choose()`, `encode_with_codec()`, `encode_latents()`.
* **Breaking — Renamed module `audiotree.datasources` to `audiotree.sources`**: To match `grain.sources`.
* **Breaking — Refactored `create_balanced_audio_dataset()`**: Now uses grain's `random_map` pattern for saliency-based audio loading (see Fixed for the excerpt-diversity bug this resolves). Now supports mixing pre-constructed grain MapDatasets via the `datasets` parameter. The `sources` parameter is now optional (at least one of `sources` or `datasets` must be provided). Pre-constructed datasets passed via `datasets` must already be repeated (call `.repeat()` before passing); if a finite dataset is passed, `grain.MapDataset.mix` will truncate output to the shortest dataset length. File-based sources are automatically repeated internally.
* **Breaking — Renamed `seed` to `shuffle_seed` and added `excerpt_seed`**: In `create_audio_dataset()` and `create_balanced_audio_dataset()`, the `seed` parameter has been replaced with `shuffle_seed` (controls file order) and `excerpt_seed` (controls random excerpt selection). This allows creating datasets that visit files in the same order but load different random excerpts. If `excerpt_seed` is `None`, it defaults to `shuffle_seed`. In `create_balanced_audio_dataset()`, seeds for each group are now derived from `np.random.default_rng()` for better statistical independence.
* **Breaking — `SaliencyParams.enabled` now defaults to `True`**: Previously defaulted to `False`, which caused files to always load from offset=0, which was unintuitive.
* **All `AudioTree` methods work on mini-batched trees**: After `reshape_mini_batches()`, the waveform has shape `(num_mini_batches, mini_batch_size, C, T)`. `replace_loudness()`, `normalize_loudness()`, `to_mono()`, `to_stereo()`, and `resample()` now handle any number of leading batch axes by flattening them around the underlying 3-D kernels and restoring them afterwards (`loudness` comes back shaped like the leading axes, e.g. `(num_mini_batches, mini_batch_size)`).
* **`AudioTree` array fields typed as `ArrayLike`**: The `waveform`, `loudness`, `pitch`, `velocity`, `note_duration`, `codes`, and `latents` fields (and the matching `create()` / `from_file()` parameters) are now annotated `ArrayLike = np.ndarray | jax.Array` instead of `np.ndarray`, so type checkers accept both NumPy and JAX arrays. Optional fields are explicitly `ArrayLike | None`, and `AudioTree.create()` now accepts `waveform=None` (the dimensionality-normalizing reshape is skipped) for token-only trees built from `codes` / `latents`. `core.py` uses `from __future__ import annotations` and keeps `import jax` under `TYPE_CHECKING`, so reading these annotations no longer forces a runtime JAX import (runtime behavior is otherwise unchanged).
* **Enhanced loudness computation**: `AudioTree.replace_loudness()` now uses the [loudness](https://github.com/iver56/loudness/) library for numpy arrays and [jaxloudnorm](https://github.com/DBraun/jaxloudnorm) for JAX arrays.
* **`AudioTree.resample()` CPU backend**: NumPy-backed waveforms now resample on CPU via [librosa](https://librosa.org/) (soxr), and the result **stays a NumPy array** — previously a NumPy waveform was silently returned as a JAX array. JAX-backed waveforms still use the JAX/Julius port. The NumPy path's output length is pinned to match the JAX backend (soxr can otherwise differ by a sample). `zeros`, `rolloff`, and `full` apply to the JAX backend only.
* **`jaxloudnorm` dependency**: Now installed from PyPI.
* **Loosened the `grain` pin to `>=0.2.15`** (was `==0.2.15`): `transforms/core.py` now imports `BatchOperation` and `SharedMemoryArray` from grain's public `grain.python` namespace and batches via `jax.tree_util.tree_map`, removing all `grain._src` private-API usage. Verified against grain 0.2.15 through 0.2.18. Note: grain 0.2.17+ reads absl flags inside its multiprocessing prefetch, so scripts doing multiprocessing data loading outside an `absl.app.run` entry point must call `flags.FLAGS.mark_as_parsed()` once at startup (the test suite does this in `tests/conftest.py`).
* **Filepath/source metadata strings raise instead of truncating**: Encoding a filepath or source string longer than `audiotree.core._str_max_length` now raises `ValueError` instead of silently truncating it (which corrupted the round-tripped path). The default limit was also raised from 256 to 1024 characters.
* **Updated examples**: argbind_augmentations examples demonstrate new function-based transform API with simpler YAML configs and working CLI parameter binding.
* **Consolidated tests**: Merged `test_roll.py` into `test_core.py` for better organization.

### Removed

* **`AudioDataSimpleSource`, `AudioDataBalancedSource`, `AudioDataBalancedDataset`, and `AudioDataSourceMixin`**: These deprecated classes are removed. Use `create_balanced_audio_dataset()` instead.
* **`AudioTree.from_array()`**: Use `AudioTree.create()` instead.
* **`audiotree.transforms.ReduceBatchTransform`**: Replaced with `audiotree.transforms.Batch`.
* **`num_records` parameter**: Removed from `create_audio_dataset()` and `create_balanced_audio_dataset()`. Use `.slice(slice(0, N))` on the returned dataset instead. This simplifies the API and follows the principle of separation of concerns.
* **Post-mix shuffle in `create_balanced_audio_dataset()`**: The automatic shuffle after mixing datasets has been removed. The `shuffle` parameter now only controls whether files within each group are shuffled. If you need the mixed output shuffled, call `.shuffle(seed=N)` on the result. This change simplifies the API and gives users explicit control.

### Fixed

* **`roll()` invalidates stale `metadata["offset"]`**: `AudioTree.from_file` records `metadata["offset"]` as the source-file time of sample 0. Since `roll` shifts the waveform along the time axis (in both `"wrap"` and `"constant"` modes), that offset no longer points at sample 0 and is now set to `None`, mirroring how `roll` already invalidates cached `loudness`. Trees without an `offset` key are unaffected. Applies to both the NumPy and JAX backends.
* **Stale loudness after length/channel changes**: `trim()`, `roll(mode="constant")`, and `AudioTree.to_stereo()` (mono→stereo) now invalidate the cached `loudness` field, since changing the audio length, zero-padding, or duplicating a channel all change the integrated loudness. `roll(mode="wrap")`, `invert_phase()`, and `swap_stereo()` continue to preserve `loudness` because they leave it unchanged.
* **Stale loudness after phase transforms**: `corrupt_phase()` and `shift_phase()` now invalidate the cached `loudness` field by default. Their phase changes leave the magnitude spectrum (and thus energy) intact, so loudness is approximately unchanged, but the cached value is dropped to be safe. Pass `keep_loudness=True` to retain it.
* **Excerpt diversity in `create_balanced_audio_dataset()`**: `record_key` was used as both an array index and an RNG seed, severely limiting excerpt diversity when datasets were repeated. Fixed by the `random_map` refactor (see Changed).

## audiotree 0.2.0 (Feb 17, 2025)

* `jit` has been removed in most places. We encourage users to jit as late as possible.
* New class: `AudioDataBalancedDataset`, which is a grain Dataset, **not a Data Source**.
* `AudioTree` has a `.latents` property.
* New transform: `NeuralLatentEncodeTransform`.
* Class `NeuralAudioCodecEncodeTransform` has been adjusted. The arg is now `encoder_fn` and it takes an `AudioTree` instead of an audio data array.
* In an `AudioTree`'s metadata, the offset and duration will now be 1D arrays instead of 0D arrays.
* `cpu` has a kwarg has been removed in most places. You should think of AudioTrees as existing on CPU by default. If you pass them to a jitted function then they will be put on device.
* In, `AudioDataSimpleSource` and `AudioDataBalancedSource`, `num_steps` arg is now `num_records`. Also `._filepaths` property is now `.filepaths`.

## audiotree 0.1.0 (Aug 22, 2024)

### **Breaking changes:**
* `SaliencyParams` has moved from `audiotree.datasources.SaliencyParams` to `audiotree.SaliencyParams`

### Updates:
The code has been tested with device parallel sharding.
See the recent updates to [DAC-JAX](https://github.com/DBraun/DAC-JAX/blob/main/scripts/input_pipeline.py).
SaliencyParams has a new `search_function` parameter.
The two valid strings are `SaliencyParams.search_uniform` and `SaliencyParams.search_early_bias`.
You can also plug in your own Callable function.

## audiotree 0.0.5 (Aug 8, 2024)

First release.
