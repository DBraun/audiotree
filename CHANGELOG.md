# Change log

AudioTree follows [Effort-based Versioning](https://jacobtomlinson.dev/effver/).

## Unreleased

### Breaking Changes

* **Separate NumPy and JAX transform backends**: Transforms now have two implementations:
  - `audiotree.transforms` (NumPy): For CPU-based grain data pipelines. Uses `np.random.Generator`.
  - `audiotree.transforms.jax` (new): For GPU/JIT training pipelines. Uses `jax.random.key`.

  The base transforms in `audiotree.transforms` now expect `np.random.Generator` instead of `jax.Array` for the RNG parameter. Use `np.random.default_rng(seed)` instead of `jax.random.key(seed)`. For JAX-based transforms in jitted training loops, import from `audiotree.transforms.jax`.

* **Transform API Redesign**: Completely redesigned from class-based to function-based pattern. All transforms are now functions decorated with `@random_transform` or `@map_transform`, enabling direct parameter binding with argbind. Parameters are now flat (e.g., `volume_norm.min_db: -20`) instead of nested in a `config` dict. This allows CLI parameter binding (e.g., `--volume_norm.min_db=-20`) which was not possible with the old API. All transform names changed from PascalCase to snake_case (e.g., `VolumeNorm` → `volume_norm`, `Trim` → `trim`). Old class-based transforms removed with no backward compatibility.
* **Renamed module**: `audiotree.datasources` → `audiotree.sources` to match `grain.sources`.
* **Removed deprecated classes**: `AudioDataSimpleSource`, `AudioDataBalancedSource`, `AudioDataBalancedDataset`, and `AudioDataSourceMixin`. Use `create_balanced_audio_dataset()` instead.
* **Removed method**: `AudioTree.from_array()`. Use `AudioTree.create()` instead.
* **Removed class**: `audiotree.transforms.ReduceBatchTransform`. Replaced with `audiotree.transforms.Batch`.
* **Refactored `create_balanced_audio_dataset()`**: Now uses grain's `random_map` pattern for saliency-based audio loading. This fixes a critical bug where `record_key` was used as both an array index and RNG seed, severely limiting excerpt diversity when datasets were repeated. Now supports mixing pre-constructed grain MapDatasets via the `datasets` parameter. The `sources` parameter is now optional (at least one of `sources` or `datasets` must be provided).
* **Removed `num_records` parameter**: The `num_records` parameter has been removed from `create_audio_dataset()` and `create_balanced_audio_dataset()`. Use `.slice(slice(0, N))` on the returned dataset instead. This simplifies the API and follows the principle of separation of concerns.
* **Renamed `seed` to `shuffle_seed` and added `excerpt_seed`**: In `create_audio_dataset()` and `create_balanced_audio_dataset()`, the `seed` parameter has been replaced with `shuffle_seed` (controls file order) and `excerpt_seed` (controls random excerpt selection). This allows creating datasets that visit files in the same order but load different random excerpts. If `excerpt_seed` is `None`, it defaults to `shuffle_seed`. In `create_balanced_audio_dataset()`, seeds for each group are now derived from `np.random.default_rng()` for better statistical independence.
* **Removed post-mix shuffle from `create_balanced_audio_dataset()`**: The automatic shuffle after mixing datasets has been removed. The `shuffle` parameter now only controls whether files within each group are shuffled. If you need the mixed output shuffled, call `.shuffle(seed=N)` on the result. This change simplifies the API and gives users explicit control.
* **`SaliencyParams.enabled` now defaults to `True`**: Previously defaulted to `False`, which caused files to always load from offset=0, which was unintuitive.
* **Clarified `datasets` parameter requirement**: Pre-constructed datasets passed to `create_balanced_audio_dataset()` via the `datasets` parameter must already be repeated (call `.repeat()` before passing). If a finite dataset is passed, `grain.MapDataset.mix` will truncate output to the shortest dataset length. File-based sources are automatically repeated internally.

### New Features - Transforms

* **New JAX transforms module**: `audiotree.transforms.jax` provides JAX-native transforms for GPU/JIT training pipelines. Uses `jax.random.key` and JAX operations throughout. Includes: `volume_norm`, `volume_change`, `invert_phase`, `swap_stereo`, `corrupt_phase`, `shift_phase`, `roll`, `trim`, `rescale_audio`, `identity`.
* **New transform functions**: `roll()` and `trim()` for rolling audio in time and trimming/padding to fixed length with configurable padding modes.
* **Transform decorators**: New `@random_transform` and `@map_transform` in `audiotree.transforms.decorators` for creating custom transforms from simple functions. Handles all boilerplate for `prob`, `scope`, `output_key`, and `split_seed` parameters.
* **All transforms migrated**: `identity()`, `mono()`, `stereo()`, `volume_change()`, `volume_norm()`, `rescale_audio()`, `invert_phase()`, `swap_stereo()`, `corrupt_phase()`, `shift_phase()`, `roll()`, `trim()`, `choose()`, `encode_with_codec()`, `encode_latents()`.

### New Features - Data Sources

* **New function**: `create_audio_dataset()` for creating simple audio datasets without balancing. Loads all files from one or more directories and applies grain's `random_map` for proper RNG seeding.
* **New function**: `load_audio_with_saliency()` for loading audio files with optional saliency-based excerpt selection. Designed to work with grain's `random_map`. Exported in `audiotree.sources`.
* **New property**: `AudioTree.source` returns the source group name for each item in the batch. When using `create_balanced_audio_dataset()` with `sources={"music": [...], "speech": [...]}`, each loaded AudioTree tracks which source group it came from.
* **New class**: `AudioWriter` for writing AudioTree batches to disk with manifest generation (JSON/CSV/NPZ formats).
* **New class**: `ManifestDataSource` for reading AudioWriter outputs as a grain RandomAccessDataSource with metadata preservation and filtering capabilities.

### New Features - AudioTree

* **New method**: `AudioTree.normalize_loudness(target_lufs)` to normalize audio to a target LUFS level. Computes loudness if not already set, scales audio, and updates the loudness field.
* **New method**: `AudioTree.create()` classmethod with automatic audio dimensionality handling (1D→3D, 2D→3D) and `filepaths` parameter support.
* **New method**: `AudioTree.filter()` which takes a function `filter_fn(AudioTree) -> bool`.
* **New methods**: `AudioTree.split()` to split an AudioTree into a list, `AudioTree.reshape_mini_batches()` and `AudioTree.flatten_mini_batches()` to add or remove mini-batch axis (useful for `nnx.scan`).
* **New parameter**: `filepaths` is now a kwarg to `AudioTree.create()` and `AudioTree.from_file()`.
* **New parameter**: `pad_mode` kwarg added to `AudioTree.from_file()`. Options include `"constant"` (zero padding), `None` (don't pad), and `"wrap"` (loop audio).

### Enhancements

* **Enhanced `MemmapWriter`**: Now supports AudioTree objects with automatic schema inference. Set `infer_schema=True` (default) to automatically detect and decompose AudioTree objects without manual `FieldSpec` creation. Metadata PyTrees (nested dicts) are preserved and can be reconstructed on read. Accepts a single `AudioTree` directly in `write_batch()` and `write_sample()` (not just `Dict[str, AudioTree]`). Fields are stored without a prefix and the manifest includes `"single_audiotree": true` so that `MemmapDataSource` can return an `AudioTree` directly.
* **Enhanced `MemmapDataSource`**: Now supports automatic AudioTree reconstruction. Set `reconstruct_audiotree=True` (default) to automatically rebuild AudioTree objects from flattened fields based on manifest metadata. When a dataset was written with a single `AudioTree`, `__getitem__()` and `get_slice()` return an `AudioTree` directly instead of a dict. Supports train/val/test splits via `split`, `split_ratios`, and `split_seed` parameters. Use `load_into_memory=True` to load the entire dataset into RAM for faster access.
* **Enhanced loudness computation**: `AudioTree.replace_loudness()` now uses the [loudness](https://github.com/iver56/loudness/) library for numpy arrays and [jaxloudnorm](https://github.com/DBraun/jaxloudnorm) for JAX arrays.
* **New module**: `audiotree_utils.py` with `AudioTreeFieldExtractor` class for decomposing and reconstructing AudioTree objects from flat arrays.

### Testing & Documentation

* **Comprehensive test suite for balanced datasets**: 12 new tests validating hierarchical directory structures, unbalanced group sizes, statistical accuracy of weight-based sampling, and edge cases.
* **Multiprocessing/multithreading tests**: 6 new tests for `mp_prefetch` and `ReadOptions`.
* **Transform chaining tests**: 8 new tests for dataset chaining patterns.
* **`Dict[str, AudioTree]` tests**: 9 new tests for batch processing with scope.
* **New documentation guides**: Balanced Datasets, ArgBind Guide, Transform Chaining, Dict Batches, and Multiprocessing.
* **Updated examples**: argbind_augmentations examples demonstrate new function-based transform API with simpler YAML configs and working CLI parameter binding.
* **Consolidated tests**: Merged `test_roll.py` into `test_core.py` for better organization.

### Dependencies

* Now uses `jaxloudnorm` from PyPI.
* Added `librosa` for NumPy-based STFT operations in transforms (alongside `librosax` for JAX). 

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
