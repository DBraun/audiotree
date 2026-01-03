# Change log

## Unreleased

* New `AudioTree.normalize_loudness(target_lufs)` method to normalize audio to a target LUFS level. Computes loudness if not already set, scales audio, and updates the loudness field.
* New `batch_audiotrees(audio_trees, backend=None)` function to batch a list of AudioTrees into a single AudioTree. Uses `jax.tree_util.tree_map` to properly concatenate all fields including loudness, metadata, etc. The `backend` kwarg allows explicit selection of `np` or `jnp` for concatenation; if `None`, it is inferred from the first AudioTree's `audio_data` type.
* New `AudioTree.source` property that returns the source group name for each item in the batch. When using `AudioDataSimpleSource` or `AudioDataBalancedSource` with `sources={"music": [...], "speech": [...]}`, each loaded AudioTree now tracks which source group it came from.
* New `create_balanced_audio_dataset()` function for weighted mixing of multiple audio groups using grain's public `MapDataset.mix()` API. Supports custom weights, optional shuffling, and returns a `MapDataset` with random access.
* **Enhanced `MemmapWriter`**: Now supports AudioTree objects with automatic schema inference. Set `infer_schema=True` (default) to automatically detect and decompose AudioTree objects without manual `FieldSpec` creation. AudioTree fields and metadata are automatically flattened to memmap-compatible arrays. Metadata PyTrees (nested dicts) are preserved and can be reconstructed on read.
* **Enhanced `MemmapDataSource`**: Now supports automatic AudioTree reconstruction. Set `reconstruct_audiotree=True` (default) to automatically rebuild AudioTree objects from flattened fields based on manifest metadata. Fields are grouped by prefix (e.g., `dry_audio`, `dry_mel` → `AudioTree` with mel in metadata). Supports train/val/test splits via `split`, `split_ratios`, and `split_seed` parameters. Use `load_into_memory=True` to load the entire dataset into RAM for faster access.
* New `audiotree_utils.py` module with `AudioTreeFieldExtractor` class for decomposing and reconstructing AudioTree objects from flat arrays. Includes helper methods for field extraction, metadata flattening, and FieldSpec inference.
* **Deprecated**: `AudioDataBalancedSource` and `AudioDataBalancedDataset`. Use `create_balanced_audio_dataset()` instead.
* **Breaking change**: `InvertPhase` and `SwapStereo` now inherit from `BaseRandomTransform` instead of `BaseMapTransform`. Use `.random_map(audio_tree, rng)` instead of `.map(audio_tree)`. These transforms now support the `prob` parameter for probabilistic application.
* **Breaking change**: Removed `AudioTree.from_array()` method. Use `AudioTree.create()` instead for enhanced functionality.
* `AudioTree.replace_loudness()` now uses the `loudness` library for numpy arrays and `jaxloudnorm` for JAX arrays, improving performance for numpy-based workflows.
* Added `AudioTree.create()` classmethod with automatic audio dimensionality handling (1D→3D, 2D→3D) and `filepaths` parameter support.
* `filepaths` is now a kwarg to `AudioTree.create()` and `AudioTree.from_file()`.
* The kwarg `pad_mode="constant"` has been added to `AudioTree.from_file(...)`. Other useful choices include `None` (don't pad), and `wrap` (loop audio).
* New `audiotree.transforms.Roll` Transform that rolls audio forwards or backwards in time with a padding mode.
* New `audiotree.transforms.Trim` Transform that trims or pads audio to a constant length with a padding mode.
* New `AudioTree.mini_batch_list` to split an AudioTree into a list of AudioTree.
* New `AudioTree.mini_batch` and `AudioTree.unbatch` to add or removing a mini-batch axis (useful for `nnx.scan`, etc.)
* Remove `audiotree.transforms.ReduceBatchTransform` and replace with `audiotree.transforms.Batch`, which should be used in place of `grain.transforms.Batch`
* Use `jaxloudnorm` from PyPI.
* New `AudioWriter` class for writing AudioTree batches to disk with manifest generation (JSON/CSV formats).
* New `ManifestDataSource` for reading AudioWriter outputs as a grain RandomAccessDataSource with metadata preservation and filtering capabilities.
* New `AudioTree.filter` which takes a function `filter_fn(AudioTree) -> bool` 

## audiotree 0.2.0 (Feb 17, 2025)

* `jit` has been removed in most places. We encourage users to jit as late as possible, and DAC-JAX demonstrates this.
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
