# AudioTree contributor guide

AudioTree loads audio into Grain datasets and provides NumPy and JAX
augmentations. The package lives in `src/audiotree/`. Use the guides under
`docs/source/introduction/` for examples of how the public APIs fit together.

## Development

Use Python 3.12 or newer and `uv`. The `dev` dependency group supplies the
test, docs, and lint tools. The `all` extra enables progress reporting so
tests can exercise that path. The local `uv.lock` is ignored by Git.

```bash
uv sync --extra all
uv run pytest
uv run pytest tests/test_resample.py
uv run make -C docs doctest
uv run make -C docs html
uvx pre-commit run --all-files
uv build
```

- `uv run pytest` collects unit tests, package docstrings, and examples.
  `tests/test_docs_snippets.py` also checks guide snippets and references.
  Run the Sphinx doctest builder for the RST `testcode` and `testoutput`
  blocks. When public behavior changes, update its guide and example.
- To reproduce the minimum-dependency CI job, use a separate Python 3.12
  environment. Resolve with `uv sync --extra all --resolution lowest-direct`
  and test with `uv run --no-sync pytest` so the resolver keeps the same versions.
- Grain multiprocessing tests need process spawning and OS shared memory.
  If a sandbox denies `shm_open`, rerun in an environment that permits it.
  Error propagation differs across supported Grain versions.
- Read and write text files with an explicit encoding. The CI test matrix uses
  `PYTHONWARNDEFAULTENCODING=1` to catch platform-dependent text I/O.

## Code and tests by area

### AudioTree and transforms

- `core.py` defines `AudioTree`, a Flax pytree that holds NumPy or JAX arrays.
  Waveforms have shape `(batch, channels, samples)`. Token-only trees can
  carry `codes` or `latents` without a waveform. `reshape_mini_batches()`
  adds a leading group axis and records grouping in static pytree metadata.
  Codec feature ranks vary, so they cannot identify grouping by themselves.
- Use `AudioTree.batch` as Grain's batch function for AudioTrees. It
  concatenates along the existing leading axis. Keep labels, extras, and
  provenance aligned when slicing, grouping, or combining items.
- Waveform changes can invalidate `lufs`, `lufs_windows`, `codes`, and
  `latents`. Use `_invalidate_derived()` when implementing such changes.
  `replace_lufs()` computes gated integrated BS.1770 loudness and ungated
  window loudness. `lufs_hop_sec` controls the window stride. Gain can move
  blocks across the absolute gate, so remeasure integrated LUFS when keeping
  that cache.
- NumPy transforms in `transforms/` use `np.random.Generator` for CPU
  pipelines. JAX transforms in `transforms/jax/` use JAX keys and support
  `jax.jit`. When changing shared behavior, check both backends with
  `tests/transforms/test_backend_parity.py`. Use CPU-compatible examples.

### Data sources and storage

- `sources/core.py` builds simple and balanced Grain datasets.
  `ExcerptConfig` controls excerpt selection. `sources/windowed.py` builds
  datasets indexed by window slots and supports optional LUFS caches.
  When changing seed derivation, ordering, or worker behavior, check
  `tests/test_seed_stability.py` and `tests/sources/test_multiprocessing.py`.
- `AudioWriter` writes audio files and an NPZ manifest with one row per item.
  `write_audio=False` writes only the manifest. `AudioDataSource` reads
  individual rows, and `AudioTree.from_manifest()` reads selected rows into
  one batch. `_manifest.py` owns the NPZ codec.
- `TreeWriter` stores numeric leaves in memmapped files and strings in Bagz.
  `TreeDataSource` reconstructs samples and returns copies of array rows.
  Tree format 1.1 records numeric mini-batch grouping, and 1.2 adds separate
  Bagz limits files and grouped string leaves. Audio manifests with
  `bfloat16` columns use their own format version 1.1.
- `AudioTree.extras` holds user-supplied per-item data. `AudioTree._metadata`
  holds only `filepath`, `source`, and `offset` provenance and is serialized
  as `metadata` within an AudioTree node. `TreeWriter(metadata=...)` stores
  dataset-level metadata at the manifest root, read with
  `TreeDataSource.get_metadata()`.
- For storage changes, check the writer and reader together. Keep a published
  manifest consistent with the data its readers can open. Preserve existing
  format compatibility and use `tests/test_golden_formats.py` to verify
  reads of committed fixtures under `tests/assets/golden/`.

## Documentation and releases

- Open the guide for the API you are changing. Start with
  `docs/source/introduction/introduction.rst` for the container,
  `sources.rst` for loaders, `transforms.rst` for augmentation, and
  `writer.rst` for storage. The pages in `docs/source/audiotree_api/` come
  from docstrings. Put public API changes under `CHANGELOG.md`'s
  `[Unreleased]` section and check `tests/test_public_api.py`.
- `.github/workflows/all.yml` defines the test matrix, minimum-dependency
  job, installed-wheel tests, lint, and docs build. Docs deploy after
  successful builds on pushes to `main`.
- When preparing a release, keep `src/audiotree/__init__.py`, `CITATION.cff`,
  and the README's BibTeX version in sync. Move the release notes to a dated
  changelog section. Publish a GitHub Release with the matching `v` tag to
  start the release workflow. The PyPI upload runs after the checks pass.
