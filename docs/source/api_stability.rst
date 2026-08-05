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
     - ``create_audio_dataset``, ``create_balanced_audio_dataset``,
       ``create_windowed_audio_dataset``, ``find_audio_files``,
       ``build_window_lufs_cache``, ``load_window_lufs``, ``save_window_lufs``,
       ``precompute_window_lufs``, ``scan_durations``, ``WindowLufsCache``,
       ``WindowParams``, ``AudioDataSource``, ``TreeDataSource``
   * - ``audiotree.transforms``
     - ``AudioCodec``, ``identity``, ``mono``, ``stereo``, ``resample``,
       ``volume_change``, ``volume_norm``, ``rescale_audio``, ``peak_norm``,
       ``invert_phase``, ``swap_stereo``, ``corrupt_phase``, ``shift_phase``,
       ``roll``, ``choose``, ``encode_with_codec``, ``encode_latents``, ``trim``
   * - ``audiotree.transforms.jax``
     - the same list, minus ``choose``

The exported name is the contract, not the module it happens to live in. Import
from these namespaces:

.. code-block:: python

    from audiotree import AudioTree, TreeWriter        # public
    from audiotree.core import AudioTree               # internal path — may move

The API reference renders classes under their defining module
(``audiotree.core.AudioTree``, ``audiotree.tree_writer.TreeWriter``) because that
is where autodoc finds them. That is a rendering detail.

Also public:

* **The transform calling convention.** Every transform constructor takes its own
  parameters plus the keyword-only ``scope`` and ``output_key``; random transforms
  additionally take ``prob`` and ``split_seed``. The resulting object exposes
  ``.map(...)`` or ``.random_map(...)`` for Grain.
* **The three on-disk formats** — a ``TreeWriter`` directory, an ``AudioWriter``
  NPZ manifest, and a windowed-LUFS cache. See `On-disk formats`_.
* **The** ``AudioCodec`` **protocol** — ``encode(AudioTree) -> (codes, scale)`` and
  ``encode_to_latent(AudioTree) -> latents``.

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
     - ``@random_transform`` / ``@map_transform``. See the note below.
   * - ``audiotree.loudness``, ``audiotree.resample``
     - Loudness and resampling kernels; reach them through ``AudioTree``.
   * - ``audiotree._fs``, ``audiotree._format``, ``audiotree._bagz``
     - Path confinement, on-disk header, and the lazy ``bagz`` import.

.. note::
   ``@random_transform`` and ``@map_transform`` are the documented way to write
   your own transform, but they are reachable only through
   ``audiotree.transforms.decorators`` and appear in no ``__all__``. Treat them as
   **provisional** in 1.0: usable, and unlikely to change shape, but not yet covered
   by the deprecation policy below. If you depend on them, pin a minor version.

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
``pyproject.toml`` and the docs both read.

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
``producer`` — and every reader validates it. The rules, all implemented in one
place (``audiotree/_format.py``):

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
* **NumPy/JAX backend equality.** The two backends of ``replace_lufs()`` are
  documented as not bit-identical (exact IIR K-weighting on CPU versus
  ``jaxloudnorm``'s FIR approximation), and the same caution applies to the two
  transform backends generally.
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
