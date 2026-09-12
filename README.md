# AudioTree

[![PyPI](https://img.shields.io/pypi/v/audiotree.svg)](https://pypi.org/project/audiotree/)
[![Docs](https://img.shields.io/badge/docs-dirt.design%2Faudiotree-blue)](https://dirt.design/audiotree)

AudioTree is an audio data loading and augmentation library supporting PyTorch
and with extra features for [JAX](https://jax.readthedocs.io/en/latest/). Its
central type, `AudioTree`, holds a batch of audio as a
[pytree](https://jax.readthedocs.io/en/latest/pytrees.html): waveform, sample
rate, loudness, and more, including your own per-item arrays. Augmentations come
in matched NumPy and JAX backends (NumPy for CPU data-loader workers, JAX for
jitted training steps).

The [documentation](https://dirt.design/audiotree) carries the full guides,
tested examples, and the API reference; start there.

## Install

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

# One file in, one AudioTree out. Even a single file is a batch (of 1).
audio = AudioTree.from_file("/data/audio/song.wav", sample_rate=44_100)
print(audio.waveform.shape)   # (1, channels, samples)

# The same idea for a whole directory: a shuffled, infinite stream of 5-second
# excerpts. num_epochs=None repeats forever; an integer gives that many passes
# over the files.
ds = create_audio_dataset(
    sources=["/data/audio"],
    sample_rate=44_100,
    duration=5.0,
    shuffle=True,
    num_epochs=None,
)

# Augment. One .seed() call: each random_map derives its own stream from it.
ds = ds.seed(42)
ds = ds.map(stereo())
ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

# Batch. AudioTree.batch concatenates along the leading axis items already have,
# rather than stacking a new one.
it = iter(ds.to_iter_dataset().batch(8, batch_fn=AudioTree.batch))

batch: AudioTree = next(it)
print(batch.waveform.shape)   # (8, 2, 220500) == (batch, channels, samples)
print(batch.sample_rate)      # 44100, one scalar for the whole batch
print(batch.lufs.shape)       # (8,): volume_norm leaves the achieved loudness behind
print(batch.filepath[0])      # the source file item 0 was drawn from
```

The same transforms exist in two backends: `audiotree.transforms` (NumPy, for CPU
Grain workers) and `audiotree.transforms.jax` (JAX, for jitted training steps),
bindable from YAML or the command line with DBraun's
[ArgBind fork](https://github.com/DBraun/argbind/). Beyond the quickstart: balanced
sampling across source groups, length-aware *windowed* sampling, loudness-gated
excerpt search, per-dataset read-error policies, two on-disk dataset writers, and
neural-codec protocols. The [guides](https://dirt.design/audiotree) cover all of
it, including how AudioTree compares to audiotools and torchaudio.

## Versioning

AudioTree follows [Effort-based Versioning](https://jacobtomlinson.dev/effver/):
the version communicates the effort a change is likely to cost you, not a
syntactic classification. Breaking changes are documented in the
[changelog](https://dirt.design/audiotree/changelog.html).

## Citation

```bibtex
@software{Braun_AudioTree_2026,
   author = {Braun, David},
   title = {{AudioTree}},
   url = {https://github.com/DBraun/audiotree},
   version = {1.0.0},
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
