# Change log

## Unreleased

* **Breaking change**: Removed `AudioTree.from_array()` method. Use `AudioTree.create()` instead for enhanced functionality.
* Added `AudioTree.create()` classmethod with automatic audio dimensionality handling (1D→3D, 2D→3D) and `filepaths` parameter support.
* `filepaths` is now a kwarg to `AudioTree.create()` and `AudioTree.from_file()`.
* The kwarg `pad_mode="constant"` has been added to `AudioTree.from_file(...)`. Other useful choices include `None` (don't pad), and `wrap` (loop audio).
* New `audiotree.transforms.Roll` Transform that rolls audio forwards or backwards in time with a padding mode.
* New `audiotree.transforms.Trim` Transform that trims or pads audio to a constant length with a padding mode.
* New `AudioTree.mini_batch_list` to split an AudioTree into a list of AudioTree.
* New `AudioTree.mini_batch` and `AudioTree.unbatch` to add or removing a mini-batch axis (useful for `nnx.scan`, etc.)
* Remove `audiotree.transforms.ReduceBatchTransform` and replace with `audiotree.transforms.Batch`, which should be used in place of `grain.transforms.Batch`
* Use `jaxloudnorm` from PyPI.

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
