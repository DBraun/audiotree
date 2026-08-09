.. role:: python(code)
     :language: python
     :class: highlight

.. _api_stability:

API Stability
=============

1.0 draws a line around a surface and promises to keep it. This page says exactly
where the line is, what a version bump means, how anything inside the line gets
removed, and what the on-disk formats guarantee across 1.x.

What is public
--------------

**The names exported from the four public namespaces**, and the documented
attributes and methods of the classes those names resolve to.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Namespace
     - ``__all__``
   * - ``audiotree``
     - ``AudioTree``, ``ExcerptConfig``, ``AudioWriter``, ``TreeWriter``,
       ``sources``, ``transforms``
   * - ``audiotree.sources``
     - ``ExcerptConfig``, ``AudioReadError``, ``OnReadError``,
       ``READ_ERROR_KEY``, ``create_audio_dataset``,
       ``create_balanced_audio_dataset``, ``create_windowed_audio_dataset``,
       ``find_audio_files``, ``build_window_lufs_cache``, ``load_window_lufs``,
       ``save_window_lufs``, ``precompute_window_lufs``, ``scan_durations``,
       ``WindowLufsCache``, ``WindowConfig``, ``AudioDataSource``,
       ``TreeDataSource``
   * - ``audiotree.transforms``
     - ``AudioCodec``, ``LatentAudioCodec``, ``identity``, ``mono``, ``stereo``,
       ``resample``, ``volume_change``, ``volume_norm``, ``rescale_audio``,
       ``peak_norm``, ``invert_phase``, ``swap_stereo``, ``corrupt_phase``,
       ``shift_phase``, ``roll``, ``choose``, ``encode_with_codec``,
       ``encode_latents``, ``trim``, ``map_transform``, ``random_transform``
   * - ``audiotree.transforms.jax``
     - the same list, minus ``choose``

Four of the ``audiotree.sources`` exports are types and constants rather than
dataset builders:

* ``ExcerptConfig`` — the one object that says how an excerpt is chosen,
  accepted as ``excerpt=`` by the dataset builders and by
  ``AudioTree.excerpt()`` / ``AudioTree.loudest_excerpt()``; also exported from
  ``audiotree``.
* ``AudioReadError`` — the ``OSError`` subclass every read failure surfaces as,
  carrying the offending path as ``.file_path`` and the original exception as
  ``__cause__``.
* ``OnReadError`` — the ``Literal["raise", "skip", "warn"]`` type of the
  dataset builders' ``on_read_error`` parameter.
* ``READ_ERROR_KEY`` — the ``extras`` key (``"read_error"``), ``True`` on a
  silence stand-in substituted for an unreadable file and ``False`` on every
  item that really came off disk.

The exported name is the contract, not the module it happens to live in. Import
from these namespaces:

.. code-block:: python

    from audiotree import AudioTree, TreeWriter        # public
    from audiotree.core import AudioTree               # internal path — may move

The API reference renders classes under their defining module
(``audiotree.core.AudioTree``, ``audiotree.tree_writer.TreeWriter``) because that
is where autodoc finds them. That is a rendering detail.

Also public:

* **The transform calling convention.** A transform built by
  ``@map_transform`` or ``@random_transform`` takes its own parameters
  positionally-or-by-keyword, then keyword-only ``scope`` and ``output_key``;
  a random transform additionally takes keyword-only ``prob`` and
  ``split_seed``. The result exposes ``.map(...)`` (map transforms) or
  ``.random_map(...)`` (random transforms) for Grain.

  That includes the two codec transforms: ``encode_with_codec(codec, *,
  scope=None, output_key=None)`` and ``encode_latents(codec, *, scope=None,
  output_key=None)``. Their *AudioTree field* is fixed (``codes`` and
  ``latents``), but ``output_key`` renames the **dict leaf** the result is
  stored under, so ``encode_latents(codec, scope=["dry"],
  output_key="dry_latents")`` yields ``{"dry": ..., "dry_latents": ...}``.

  ``choose`` is the one export that does **not** follow the convention:

  .. list-table::
     :header-rows: 1
     :widths: 34 66

     * - Export
       - Signature
     * - ``choose``
       - ``choose(*transforms, c=1, weights=None, prob=1.0)``. A
         ``grain.transforms.RandomMap`` subclass written by hand, not a
         decorated transform: it takes ``prob`` but no ``split_seed``,
         ``scope`` or ``output_key``, and passing any of those raises
         ``TypeError``. Scope the transforms you hand it instead. Every
         positional argument must itself be a ``grain.transforms.Map`` or
         ``RandomMap`` — a bare callable raises ``TypeError`` — and ``c``,
         ``weights`` and ``prob`` are range-checked with ``ValueError``.

  ``AudioCodec`` and ``LatentAudioCodec`` are protocols, not transforms, and are
  not called this way at all.
* **The provenance properties, not their encoding.** Per-item provenance is
  written with ``create(filepath=..., source=...)`` and read with the decoded
  ``AudioTree.filepath`` / ``AudioTree.source`` properties — that pair is the
  contract. ``AudioTree.metadata``, the library-managed container that carries
  their jit-safe fixed-width integer encodings, has a closed schema (exactly
  ``filepath`` and ``source``; anything else is rejected by name at read) and
  is not meant to be indexed directly; user payload belongs in ``extras``.
* **The three on-disk formats** — a ``TreeWriter`` directory, an ``AudioWriter``
  NPZ manifest, and a windowed-LUFS cache. See `On-disk formats`_.
* **Two codec protocols**, both ``@runtime_checkable`` and each declaring exactly
  one method: ``AudioCodec`` — ``encode(AudioTree) -> (codes, scale)``, consumed
  by ``encode_with_codec``; and ``LatentAudioCodec`` —
  ``encode_to_latent(AudioTree) -> latents``, consumed by ``encode_latents``. A
  codec that does both satisfies both.
* **Documented transform semantics**, including the edge cases: ``swap_stereo``
  is a no-op on mono, swaps the two channels of stereo, and raises ``ValueError``
  on three or more channels rather than reversing the channel order.

What is internal
----------------

Everything else, including every module below. These may be renamed, moved,
merged, or deleted in any 1.x release without a deprecation cycle:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Module
     - Note
   * - ``audiotree.core``
     - Defines ``AudioTree`` and ``ExcerptConfig``; import them from
       ``audiotree``.
   * - ``audiotree.writer``, ``audiotree.tree_writer``
     - Define ``AudioWriter`` / ``TreeWriter``; import them from ``audiotree``.
   * - ``audiotree.sources.core``, ``.sources.audio``, ``.sources.tree``,
       ``.sources.windowed``
     - Define the dataset builders, the source classes, and the windowed
       helpers; import them from ``audiotree.sources``.
   * - ``audiotree.transforms.base``, ``.functional``, ``.helpers``, ``.codec``
     - Transform machinery and implementations; import the transforms from
       ``audiotree.transforms`` or ``audiotree.transforms.jax``.
   * - ``audiotree.transforms.decorators``
     - Defines ``@map_transform`` / ``@random_transform``, which are **public**
       — import them from ``audiotree.transforms`` or
       ``audiotree.transforms.jax``, not from here.
   * - ``audiotree.loudness``, ``audiotree.resample``
     - Loudness and resampling kernels; reach them through ``AudioTree``.
   * - ``audiotree._fs``, ``audiotree._format``, ``audiotree._manifest``,
       ``audiotree._bagz``
     - Path confinement, on-disk header, the NPZ manifest codec shared by
       ``AudioWriter``/``AudioDataSource``/``AudioTree.from_manifest``, and the
       lazy ``bagz`` import. The manifest *format* is public (see `On-disk
       formats`_); the module that reads and writes it is not.

.. note::
   ``@map_transform`` and ``@random_transform`` — the documented way to write your
   own transform — are exported from both ``audiotree.transforms`` and
   ``audiotree.transforms.jax`` and are fully covered by the deprecation policy
   below. Only the module they happen to be defined in
   (``audiotree.transforms.decorators``) is internal.

Anything not listed as public may also *appear* to work — a private helper is still
importable. The distinction here is about what changes without warning, not about
what Python lets you reach.

Mechanically enforced
~~~~~~~~~~~~~~~~~~~~~

``tests/test_public_api.py`` pins several of these invariants so they cannot drift
silently:

* Every name in each ``__all__`` resolves on its module.
* The NumPy and JAX transform namespaces expose the same names, with ``choose`` as
  the single sanctioned exception (it branches in Python, so it cannot be traced).
* Each listed public callable exposes at most one or two positional parameters;
  everything past the budget must be keyword-only, so adding, reordering or
  renaming a parameter cannot silently change what a positional call means.
* Every public method of ``AudioTree``, ``AudioWriter``, ``TreeWriter``,
  ``AudioDataSource`` and ``TreeDataSource`` carries a docstring.
* A misspelled transform parameter raises ``TypeError`` rather than being ignored.

Versioning
----------

AudioTree follows `Effort-based Versioning <https://jacobtomlinson.dev/effver/>`_.
The number communicates **how much effort an upgrade is likely to cost you**, not a
syntactic classification of the diff. A change can be technically backwards
compatible and still be a meso bump if adopting it is real work.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Bump
     - What to expect
   * - **Macro** — ``1.4.2`` → ``2.0.0``
     - A large, intentional break. Read a migration guide; budget real time.
       Public names may be removed here (after the deprecation cycle below), and
       an on-disk ``format_version`` major may change.
   * - **Meso** — ``1.0.3`` → ``1.1.0``
     - New features, possibly a small break or a behavior change. Skim the
       changelog; expect to adjust something, but not to rewrite a pipeline.
       Deprecations are announced here.
   * - **Micro** — ``1.0.0`` → ``1.0.1``
     - Fixes and internal changes. Upgrade without reading. If a micro release
       costs you effort, that is a bug in the release, not in your code.

The version lives in one place — ``audiotree.__version__`` — which
``pyproject.toml`` and the docs both read. Two copies exist for humans rather than
for code (``CITATION.cff`` and the BibTeX block in ``README.md``, which ships in
the wheel metadata and renders on PyPI); CI's ``version`` job fails the build if
any of them drift apart, or if a release tag disagrees with all three.

Deprecation policy
------------------

Nothing public disappears without warning. Removing a public name, parameter, or
documented behavior takes all four of these:

#. **A** ``DeprecationWarning`` **at the point of use**, whose message names the
   replacement. Not a docstring note — a runtime warning your test suite can turn
   into an error.
#. **A** ``### Deprecated`` **entry in the changelog** for the release that
   introduces the warning, saying what to use instead.
#. **At least one macro release of overlap.** A name deprecated during 1.x is
   removable no earlier than 2.0.
#. **At least six months** between the deprecating release and the removing one,
   whichever of the two constraints is longer.

Behavior changes that cannot be expressed as a rename — where the same call keeps
working but computes something different — are announced in the changelog and
carry a meso bump, since the effort is in re-validating your results rather than in
editing code.

Internal API (everything under `What is internal`_) is exempt. So are bug fixes
whose previous behavior was plainly wrong; those are described in the changelog's
``### Fixed`` section with what to re-check.

On-disk formats
---------------

Three artifacts are a compatibility contract, because people keep them for years:

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Artifact
     - ``format``
     - Written by
   * - ``TreeWriter`` directory
     - ``audiotree-tree``
     - :class:`~audiotree.tree_writer.TreeWriter`, read by ``TreeDataSource``
   * - NPZ manifest
     - ``audiotree-manifest``
     - :class:`~audiotree.writer.AudioWriter`, read by ``AudioDataSource``
       and ``AudioTree.from_manifest``
   * - windowed-LUFS cache
     - ``audiotree-lufs-windows``
     - ``build_window_lufs_cache()`` / ``save_window_lufs()``

Each carries a header — ``format``, ``format_version``, ``min_reader_version``,
``producer`` — and every reader validates it. (The NPZ manifest adds one more
header key, ``num_entries``, so a reader knows the row count without inferring it
from a column, and can reject a manifest whose columns disagree with it.) The
rules, all implemented in one place (``audiotree/_format.py``):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Situation
     - What happens
   * - No header at all
     - Refused with a "re-render the dataset" error. Only pre-1.0 artifacts are
       headerless; none of those layouts ever shipped in a release.
   * - ``format`` names a different artifact
     - Refused by name — pointing ``TreeDataSource`` at an ``AudioWriter``
       directory says so, instead of failing later as a ``KeyError``.
   * - ``format_version`` **major** differs from the reader's
     - Refused, in both directions. Major means an incompatible layout.
   * - ``format_version`` **minor** differs
     - Accepted. Minors are **additive** — new fields a reader may ignore — so an
       older 1.x reader keeps working on a newer 1.x artifact, and vice versa.
   * - ``min_reader_version`` is newer than the reader
     - Refused. This is what a writer raises instead of bumping major when it adds
       a field readers *must* honor: old readers refuse rather than silently
       ignoring it.

So across 1.x: **a dataset written by any 1.x audiotree is readable by any other
1.x audiotree, unless its** ``min_reader_version`` **says otherwise.** The format
version is tracked separately from the library version (``_format.CURRENT_VERSION``,
currently ``1.0``) — a format major bump is not planned within audiotree 1.x, and
would be a macro-level event for anyone holding data.

``tests/assets/golden/`` holds fixtures written once and committed — a ``TreeWriter``
directory and an ``AudioWriter`` manifest, both at format version 1.0 — read back in
``tests/test_golden_formats.py`` against hardcoded expected values. Every other
reader test writes its input with the code under test, so writer and reader change
together and a repacking that breaks every dataset on disk would keep the suite
green; the golden fixtures are what makes that go red. They are deliberately tiny
and ``bagz``-free so they are portable to every platform audiotree supports — which
also means the windowed-LUFS cache (a bagz format) has no golden fixture, only its
header check.

Reproducibility
---------------

This is what 1.0 *intends*, stated as intent because it has not been verified
end-to-end across environments.

**Intended**

* For a fixed corpus, a fixed seed, and pinned dependency versions, ``ds[i]``
  returns the same audio every time. Grain derives a per-index seed rather than
  advancing global state, and :func:`~audiotree.sources.find_audio_files` returns a
  sorted, de-duplicated list, so the file order that seeding is applied to does not
  depend on the filesystem or the machine.
* ``shuffle_seed`` and ``excerpt_seed`` are separable: two datasets with the same
  ``shuffle_seed`` and different ``excerpt_seed`` visit files in the same order and
  draw different excerpts.
* Dataset iterators are checkpointable and resumable via Grain's
  ``grain.checkpoint.CheckpointSave`` / ``CheckpointRestore``, so a restarted run
  resumes at the exact read position rather than at the top of the epoch. See
  :ref:`sources`.

**Not guaranteed**

* **Identical bytes across versions.** An audiotree, ``grain``, ``librosa``/``soxr``,
  ``jaxloudnorm`` or ``scipy`` upgrade may change decoded audio, resampled output, or
  loudness in the last bits. Nothing in the test suite pins cross-version bit
  equality, and the resamplers and loudness meters are third-party. If you need
  byte-stability, pin your environment; if you need it *checked*, hash your own
  pre-rendered dataset.
* **NumPy/JAX engine equality.** The two loudness engines selectable with
  ``replace_lufs(engine=...)`` are documented as not bit-identical (exact IIR
  K-weighting on CPU versus ``jaxloudnorm``'s FIR approximation), and the same
  caution applies to the two transform backends generally.
* **Batch composition under multiprocessing.** Grain shards the pipeline *upstream*
  of ``mp_prefetch`` across worker processes, so if ``.batch()`` sits before
  ``.mp_prefetch()`` each worker forms batches from its own shard and which items
  share a batch depends on ``num_workers``. Batching **after** ``mp_prefetch``
  keeps composition independent of the worker count. (Several examples in
  :ref:`multiprocessing` batch upstream, which is fine for throughput — just do not
  expect a run with 4 workers to produce the same batches as one with 8.) The set
  of items seen per epoch is unaffected either way.
* **Cross-platform bit equality.** Different CPUs, XLA backends, and BLAS builds
  are not expected to agree bit-for-bit.

The on-disk formats are the durable artifact here: if a result must be reproducible
years later, pre-render it with ``TreeWriter`` and keep the bytes, rather than
relying on re-running the pipeline.
