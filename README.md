# AudioTree

[![PyPI](https://img.shields.io/pypi/v/audiotree.svg)](https://pypi.org/project/audiotree/)
[![Docs](https://img.shields.io/badge/docs-dirt.design%2Faudiotree-blue)](https://dirt.design/audiotree)

Audio as a JAX pytree: a batched `AudioTree` container, [Grain](https://github.com/google/grain)
data sources, and dual NumPy/JAX augmentations.

`AudioTree` is a [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass)
that holds a *batch* of audio under one shape convention — every array carries the
batch as its leading axis, so items stay aligned as you index, slice, batch, and
transform them:

| Field | Shape | What it is |
| --- | --- | --- |
| `waveform` | `(B, C, T)` | time-domain audio |
| `sample_rate` | scalar `int` | the one field that is *not* batched |
| `lufs` / `lufs_windows` | `(B,)` / `(B, W)` | BS.1770 integrated loudness, and a per-window curve |
| `pitch`, `velocity`, `note_duration` | `(B,)` | optional per-item labels |
| `codes` / `latents` | `(B, ...)` | neural-codec tokens or latents |
| `metadata` | `(B, ...)` | your own arrays; source paths land in `metadata["filepath"]` |

## Install

```bash
pip install audiotree
```

JAX, Flax, Grain, NumPy, librosa, and soundfile come with it.

```bash
pip install "audiotree[bagz]"
```

The `bagz` extra adds [Bagz](https://github.com/google/bagz) record files, which back
two optional features: **string leaves** in `TreeWriter` / `TreeDataSource`, and the
**windowed-LUFS cache** (`build_window_lufs_cache()`). It is an extra rather than a
dependency because bagz publishes manylinux x86-64 wheels only — no macOS, no Linux
aarch64, nothing for Python 3.14 — and a hard dependency made `pip install audiotree`
unsatisfiable on those platforms. Everything else works without it.

```bash
pip install "audiotree[progress]"
```

The `progress` extra adds [tqdm](https://github.com/tqdm/tqdm), which `AudioWriter`
requires for `show_progress=True`. `audiotree[all]` is `progress` plus `bagz` wherever
bagz has a wheel.

## Quickstart

```python
from audiotree import AudioTree
from audiotree.sources import create_audio_dataset
from audiotree.transforms import stereo, volume_norm

# A shuffled, endless stream of 5-second excerpts from a directory of audio.
# num_epochs=None never runs dry; pass an int for that many passes over the corpus.
ds = create_audio_dataset(
    sources="/data/audio",
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
print(batch.lufs.shape)       # (8,) — volume_norm leaves the achieved loudness behind
print(batch.filepath[0])      # the source file item 0 was drawn from
```

The same transforms exist in two backends: `audiotree.transforms` (NumPy, for CPU Grain
workers) and `audiotree.transforms.jax` (JAX, for jitted training steps). They compose
over any pytree of `AudioTree`s — a single tree, a list, or a `{"dry": ..., "wet": ...}`
dict — and are bindable from YAML or the command line with
[ArgBind](https://github.com/DBraun/argbind/).

Beyond the basics: balanced sampling across source groups, length-aware *windowed*
sampling, loudness-gated excerpt search, a per-dataset `on_read_error` policy for
corpora that contain unreadable files, two writers (`AudioWriter` for WAVs plus a
manifest, `TreeWriter` for memory-mapped pre-rendered pytrees), and two protocols
(`AudioCodec`, `LatentAudioCodec`) for tokenizing a corpus with a neural codec you
supply. See the [guides](https://dirt.design/audiotree).

## How it compares

AudioTree is modeled on Descript's [audiotools](https://github.com/descriptinc/audiotools)
— `AudioTree` plays the role of `AudioSignal`, and the augmentation vocabulary is
recognizably the same — but it is a pytree rather than a mutable object: transforms
return new trees, everything is batched-first, and the JAX backend traces cleanly under
`jit` and `vmap`. Against **torchaudio**, the difference is scope in both directions:
torchaudio ships feature extraction, model zoos, and codecs, none of which are here
(AudioTree defines the protocol a codec must satisfy and stores what it returns, but
ships no weights); AudioTree instead ships the data-loading and augmentation layer (Grain sources, balanced
and windowed samplers, on-disk dataset writers) that torchaudio leaves to
`torch.utils.data`. There is no torch interoperability path.

## Scope and non-goals

- **Not ML-framework-agnostic.** JAX and Flax are hard requirements: `import audiotree`
  executes `from flax import struct`, and `AudioTree` *is* a `flax.struct.dataclass`.
  There is no torch path.
- **The NumPy transform backend is for CPU Grain workers**, not a JAX-free mode. It
  exists so augmentation can happen in worker processes without a device round-trip.
- **Not a feature-extraction library.** Spectrograms, mel filterbanks, and MFCCs belong
  in [librosax](https://pypi.org/project/librosax/); AudioTree carries waveforms,
  loudness, and whatever features *you* attach.
- **No in-graph time-stretch or pitch-shift.** Resampling is provided; phase-vocoder
  style transforms are not.
- **No object storage, streaming, or sharded remote formats.** Datasets are local files
  and local memory maps.
- **No training loop and no model zoo.**
- **Loudness is BS.1770 integrated plus per-window, and that is all at 1.0.** True-peak,
  short-term, and LRA are 1.x material.

## Versioning

AudioTree follows [Effort-based Versioning](https://jacobtomlinson.dev/effver/) — the
version communicates the effort a change is likely to cost you, not a syntactic
classification. What is covered by the 1.0 contract, what is internal, the deprecation
policy, and the on-disk format guarantees are written down in
[API stability](https://dirt.design/audiotree/api_stability.html).

Upgrading from 0.2.x is a breaking change; see the
[1.0 migration guide](https://dirt.design/audiotree/migration_1_0.html).

## Documentation

<https://dirt.design/audiotree>

## Citation

```bibtex
@software{Braun_AudioTree_2026,
   author = {Braun, David},
   title = {{AudioTree}},
   url = {https://github.com/DBraun/audiotree},
   version = {1.0.0rc1},
   year = {2026}
}
```

See [`CITATION.cff`](https://github.com/DBraun/audiotree/blob/main/CITATION.cff).

## License

MIT, with third-party notices for the julius-derived resampler and the
pyloudnorm-derived loudness code under
[`LICENSES/`](https://github.com/DBraun/audiotree/tree/main/LICENSES). The audio fixtures
under `tests/assets/` in the repository are carved out of the MIT grant — the MUSDB18-HQ
excerpt is CC BY-NC-SA 4.0 — and are not part of any published distribution.
