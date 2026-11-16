# AudioTree

[![PyPI](https://img.shields.io/pypi/v/audiotree.svg)](https://pypi.org/project/audiotree/)
[![Docs](https://img.shields.io/badge/docs-dirt.design%2Faudiotree-blue)](https://dirt.design/audiotree)

AudioTree loads audio files, samples excerpts, and applies augmentations. An
`AudioTree` holds batched waveforms, a sample rate, and optional per-item data
such as loudness or labels. Use the NumPy transforms in CPU data loaders,
including PyTorch pipelines, or the JAX transforms inside `jax.jit`.

## Install

Requires Python 3.12 or newer.

```bash
pip install audiotree
```

JAX, Flax, Grain, NumPy, librosa, and soundfile come with it.
On macOS and Linux, [Bagz](https://github.com/google/bagz) is also installed for
string leaves in `TreeWriter`/`TreeDataSource` and the windowed-LUFS cache.
`audiotree[progress]` (or `audiotree[all]`) adds tqdm for
`AudioWriter(show_progress=True)`.

## Quickstart

```python
from audiotree import AudioTree
from audiotree.sources import create_audio_dataset
from audiotree.transforms import stereo, volume_norm

# A single file still has a batch axis.
audio = AudioTree.from_file("/data/audio/song.wav", sample_rate=44_100)
print(audio.waveform.shape)   # (1, channels, samples)

# Sample five-second excerpts. num_epochs=None repeats indefinitely.
ds = create_audio_dataset(
    sources=["/data/audio"],
    sample_rate=44_100,
    duration=5.0,
    shuffle=True,
    num_epochs=None,
)

# Grain derives a separate random stream for each random_map call.
ds = ds.seed(42)
ds = ds.map(stereo())
ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

# Concatenate along the existing batch axis.
it = iter(ds.to_iter_dataset().batch(8, batch_fn=AudioTree.batch))

batch: AudioTree = next(it)
print(batch.waveform.shape)   # (8, 2, 220500), ordered by batch, channel, and sample
print(batch.sample_rate)      # 44100 for the whole batch
print(batch.lufs.shape)       # (8,), one measured LUFS value per item
print(batch.filepath[0])      # Source file for the first excerpt
```

- [Mix datasets with sampling weights](https://dirt.design/audiotree/introduction/balanced_datasets.html)
- [Sample windows from long recordings](https://dirt.design/audiotree/introduction/windowed_datasets.html)
- [Write datasets to disk](https://dirt.design/audiotree/introduction/writer.html)

## Versioning

AudioTree uses [Effort-based Versioning](https://jacobtomlinson.dev/effver/).
See the [changelog](https://dirt.design/audiotree/changelog.html) for changes
that may require updates to your code.

## Citation

```bibtex
@software{Braun_AudioTree_2026,
   author = {Braun, David},
   title = {{AudioTree}},
   url = {https://github.com/DBraun/audiotree},
   version = {1.1.0},
   year = {2026}
}
```

See [`CITATION.cff`](https://github.com/DBraun/audiotree/blob/main/CITATION.cff).

## License

MIT, with third-party notices for the julius-derived resampler and the
pyloudnorm-derived loudness code under
[`LICENSES/`](https://github.com/DBraun/audiotree/tree/main/LICENSES). The audio fixtures
under `tests/assets/` in the repository are carved out of the MIT grant (the MUSDB18-HQ
excerpt is CC BY-NC-SA 4.0) and are not part of any published distribution.
