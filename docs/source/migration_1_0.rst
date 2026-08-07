.. role:: python(code)
     :language: python
     :class: highlight

.. _migration_1_0:

Migrating to 1.0
================

1.0 is a breaking release. Nothing from 0.2.x is kept as a shim: the loudness
vocabulary, the transform API, the data-source API, and the on-disk layouts all
changed at once, deliberately, so that the names and conventions the 1.x stability
contract freezes are the ones worth freezing. From 1.0 onward removals go through
the deprecation policy in :ref:`api_stability`.

This guide is derived from the ``Unreleased`` section of the :doc:`changelog`,
which remains the exhaustive list.

.. note::
   Some renames below are relative to the **unreleased development tree** rather
   than to 0.2.1 — ``AudioTree.batch_fn``, ``normalize_loudness()``,
   ``filter(filter_fn=...)``, ``window_duration_sec``, the ``Batch`` transform and
   the windowed-LUFS cache never appeared in a released version. They are listed
   for anyone who tracked ``git``.

At a glance
-----------

The five that break the most code, in rough order:

#. ``audio_data`` is ``waveform``, and every ``loudness`` spelling is ``lufs``.
#. Transforms are snake_case **functions**, not PascalCase classes, and take flat
   keyword parameters instead of a ``config`` dict.
#. ``audiotree.datasources`` is ``audiotree.sources``, and its data-source classes
   are gone in favor of ``create_*_audio_dataset()`` functions.
#. ``prob`` is now drawn per batch item, so your augmentation distributions change
   even where the code still runs unmodified.
#. Any dataset written by a pre-1.0 audiotree is refused at read time. Re-render it.

Renames
-------

AudioTree
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Pre-1.0
     - 1.0
   * - ``AudioTree.audio_data``
     - ``AudioTree.waveform``
   * - ``AudioTree.loudness``
     - ``AudioTree.lufs``
   * - ``AudioTree.replace_loudness()``
     - ``AudioTree.replace_lufs()``
   * - ``AudioTree.normalize_loudness()``
     - ``AudioTree.normalize_lufs()``
   * - ``replace_lufs(window_duration_sec=...)``
     - ``replace_lufs(lufs_window_sec=...)``
   * - ``replace_lufs(hop_duration_sec=...)``
     - ``replace_lufs(lufs_hop_sec=...)``
   * - ``AudioTree.batch_fn``
     - ``AudioTree.batch``
   * - ``AudioTree.filter(filter_fn=...)``
     - ``AudioTree.filter(predicate=...)``
   * - ``AudioTree.create(filepaths=...)``
     - ``AudioTree.create(filepath=...)`` — singular, and it still accepts either
       one path or a list of them (one per batch item)

The two field renames also change **keyword arguments** everywhere the fields are
constructible — ``AudioTree(waveform=..., lufs=...)``, ``create()``,
``from_file()``, ``tree.replace()`` — and the on-disk leaf/column names (see
`On-disk data`_).

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot run.

.. code-block:: python

    # Before
    tree = AudioTree.from_array(x, 44_100)
    tree = tree.replace_loudness()
    print(tree.audio_data.shape, tree.loudness)

    # After
    tree = AudioTree.create(x, 44_100)
    tree = tree.replace_lufs()
    print(tree.waveform.shape, tree.lufs)

One spelling per loudness concept
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library used to spell the analysis window two ways and the keep-threshold two
ways, in signatures that sit side by side in one pipeline. Everything is now
``lufs``:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Pre-1.0
     - 1.0
   * - ``ExcerptConfig.loudness_cutoff``
     - ``ExcerptConfig.lufs_cutoff``
   * - ``AudioDataSource.filter_by_loudness()``
     - ``AudioDataSource.filter_by_lufs()``
   * - ``corrupt_phase(keep_loudness=...)``, ``shift_phase(keep_loudness=...)``
     - ``keep_lufs=...``
   * - ``build_window_loudness_cache()`` and friends
     - ``build_window_lufs_cache()``, ``precompute_window_lufs()``,
       ``save_window_lufs()``, ``load_window_lufs()``, ``WindowLufsCache``
       (with a ``.lufs`` attribute)
   * - ``loudness_cache=`` / ``loudness_cutoff=`` on
       ``create_windowed_audio_dataset()`` and ``WindowConfig``
     - ``lufs_cache=`` / ``lufs_cutoff=``
   * - ``window_duration_sec=`` on ``build_window_lufs_cache()``,
       ``precompute_window_lufs()``, ``save_window_lufs()`` and
       ``WindowLufsCache``
     - ``lufs_window_sec=``
   * - cache file ``loudness.bagz``
     - ``lufs.bagz``

Transforms
~~~~~~~~~~

Every transform is now a snake_case function that *returns* a transform object.
Parameters are flat keyword arguments rather than a nested ``config`` dict, which
is what makes them bindable from YAML and the command line (see
:ref:`argbind_guide`).

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Pre-1.0
     - 1.0
   * - ``Identity``
     - ``identity()``
   * - ``VolumeChange``
     - ``volume_change()``
   * - ``VolumeNorm``
     - ``volume_norm()``
   * - ``RescaleAudio``
     - ``rescale_audio()``
   * - ``InvertPhase``
     - ``invert_phase()``
   * - ``SwapStereo``
     - ``swap_stereo()``
   * - ``CorruptPhase``
     - ``corrupt_phase()``
   * - ``ShiftPhase``
     - ``shift_phase()``
   * - ``Choose``
     - ``choose()``
   * - ``NeuralAudioCodecEncodeTransform``
     - ``encode_with_codec()``
   * - ``NeuralLatentEncodeTransform``
     - ``encode_latents()``
   * - ``ReduceBatchTransform``, then ``Batch``
     - removed — see `Removed API`_

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot import.

.. code-block:: python

    # Before
    from audiotree.transforms import VolumeNorm
    ds = ds.random_map(VolumeNorm(config={"min_db": -20, "max_db": -15}, prob=0.5))

    # After
    from audiotree.transforms import volume_norm
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15, prob=0.5))

``mono()``, ``stereo()``, ``trim()``, ``roll()``, ``resample()`` and ``peak_norm()``
are new in 1.0; they have no 0.2.x equivalent.

Sources
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Pre-1.0
     - 1.0
   * - ``audiotree.datasources``
     - ``audiotree.sources`` (matching ``grain.sources``)
   * - ``AudioDataSimpleSource``
     - ``create_audio_dataset()``
   * - ``AudioDataBalancedSource``, ``AudioDataBalancedDataset``,
       ``AudioDataSourceMixin``
     - ``create_balanced_audio_dataset()``
   * - ``seed=``
     - ``shuffle_seed=`` (file order) and ``excerpt_seed=`` (excerpt selection);
       ``excerpt_seed`` defaults to ``shuffle_seed``
   * - ``num_records=N``
     - ``.slice(slice(0, N))`` on the returned dataset
   * - ``repeat=True`` / ``repeat=False`` on ``create_audio_dataset()``,
       ``create_balanced_audio_dataset()`` and ``create_windowed_audio_dataset()``
     - ``num_epochs=None`` / ``num_epochs=1`` — see `A count of epochs replaces
       the repeat flag`_

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot import.

.. code-block:: python

    # Before
    from audiotree.datasources import AudioDataBalancedSource

    source = AudioDataBalancedSource(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        num_records=10_000,
        sample_rate=44_100,
        duration=5.0,
        excerpt=ExcerptConfig(loudness_cutoff=-40),
    )

    # After
    from audiotree.sources import create_balanced_audio_dataset

    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44_100,
        duration=5.0,
        excerpt=ExcerptConfig(strategy="loudest", lufs_cutoff=-40),
    ).slice(slice(0, 10_000))

Writers
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Pre-1.0
     - 1.0
   * - ``AudioWriter(output_dir=...)``, ``TreeWriter(output_dir=...)``
     - ``directory=`` on both, matching ``.directory`` on the instance

``directory`` is the first positional parameter of both writers, so
``AudioWriter("output/")`` is unaffected; only the keyword spelling moved.
``AudioDataSource.from_writer_output(output_dir)`` keeps its name — it names the
directory a writer produced, not a constructor parameter.

Removed API
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Removed
     - Use instead
   * - ``audiotree.transforms.Batch`` (and the earlier
       ``ReduceBatchTransform``)
     - Grain's own batching with ``AudioTree.batch`` as the ``batch_fn``:
       ``ds.to_iter_dataset().batch(n, batch_fn=AudioTree.batch)``
   * - ``AudioTree.from_array()``
     - ``AudioTree.create()``, which also normalizes 1-D and 2-D input to
       ``(B, C, T)`` and accepts ``filepath=`` / ``source=``
   * - ``AudioDataSimpleSource``, ``AudioDataBalancedSource``,
       ``AudioDataBalancedDataset``, ``AudioDataSourceMixin``
     - ``create_audio_dataset()`` / ``create_balanced_audio_dataset()``
   * - ``num_records=`` on both dataset builders
     - ``.slice(slice(0, N))``
   * - The automatic post-mix shuffle in ``create_balanced_audio_dataset()``
     - ``.shuffle(seed=N)`` on the result. ``shuffle=`` now only controls
       whether files are shuffled *within* each group.
   * - ``AudioWriter(sample_rate=...)``
     - ``AudioTree.resample(...)`` before writing. See `AudioWriter no longer
       resamples`_.

``Batch`` was a ``grain.python.BatchOperation`` for the older ``DataLoader`` API,
and the last thing in audiotree riding grain's private surface. Its removal is why
the ``grain`` requirement is now just ``>=0.2.15,<0.3``, with the ceiling there only
because grain is still 0.x.

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot import.

.. code-block:: python

    # Before — grain's DataLoader API
    from audiotree.transforms import Batch

    dataloader = grain.DataLoader(
        data_source=ds,
        sampler=sampler,
        operations=[Batch(batch_size=32)],
    )

    # After — grain's IterDataset API
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

Behavior changes that are not renames
-------------------------------------

These compile and run against your existing code but do something different.

``prob`` is drawn per batch item
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``random.bernoulli`` was called with no shape, so one scalar draw was broadcast
across the whole batch: ``prob=0.5`` on a batch of 32 augmented **all 32 or none**.
The marginal rate was right, so this never showed up in aggregate statistics while
removing exactly the within-batch diversity ``prob`` exists to provide.

**Check:** any ``prob < 1``. The distribution of your augmented data changes even
though nothing in your code does. If you were relying on all-or-nothing batches,
that behavior is gone.

Two consequences worth knowing: a transform that changes a field's *shape* cannot
be mixed item-by-item and now raises a clear error at ``prob < 1``; and ``prob < 1``
now works at all on the JAX backend, where it previously crashed on any tree
carrying ``metadata["filepath"]`` (which ``from_file`` always sets) or on any
transform that nulls ``lufs``.

Misspelled transform parameters raise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unknown constructor keys used to be accepted and silently dropped, so
``volume_change(min_dB=40)`` (capital ``B``) ran with the defaults and the
augmentation you configured simply never happened. They now raise ``TypeError``
with a spelling suggestion.

**Check:** a YAML config or call site that "worked" may now correctly fail. Read
the error rather than deleting the key — the parameter it names was never being
applied.

.. skip-snippet-test: the misspelling is the point — it must keep raising.

.. code-block:: python

    volume_change(min_dB=6.0)
    # TypeError: volume_change() got unexpected parameter(s): 'min_dB'
    #   (did you mean 'min_db'?). Valid parameters: max_db, min_db, plus prob,
    #   split_seed, scope, output_key.

``scope={"wet": True}`` raises
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scope entry is matched one level *above* where it sits, so the obvious shorthand
selected every leaf rather than only ``"wet"`` — silently transforming the target
signal in a dry/wet pipeline. That spelling now raises and points at the list form.

.. skip-snippet-exec: fragment; the surrounding prose supplies the batch.

.. code-block:: python

    # Before (silently transformed everything)
    volume_norm(min_db=-20, max_db=-15, scope={"wet": True})

    # After
    volume_norm(min_db=-20, max_db=-15, scope=["wet"])

The nested-dict form is still accepted where you need exclusions —
``scope={"d": {"scope": True, "f": {"scope": False}}}`` — and paths may be dotted
(``"input.dry"``) or tuples. See :ref:`dict_batches`.

Public callables are keyword-only past the first argument or two
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``create_audio_dataset``, ``create_balanced_audio_dataset``,
``create_windowed_audio_dataset``, ``find_audio_files``, ``AudioTree.create`` /
``from_file`` / ``from_manifest`` / ``write`` / ``resample``, ``AudioWriter``,
``TreeWriter``, ``AudioDataSource`` and ``TreeDataSource`` now take at most one
or two positional parameters; everything after the ``*`` must be passed by keyword.
``tests/test_public_api.py`` enforces the budget.

**Check:** positional calls past the first argument or two. They fail loudly at
call time, so there is no silent-corruption risk here — but ``TreeWriter(path, n)``
is still fine, and so is ``AudioTree.create(waveform, sample_rate)``.

Transform constructors are similar but not identical: positional arguments now bind
to the *function's own* parameters, as the rendered signature says. Previously
``prob`` and ``split_seed`` sat first, so ``roll(0.5, 1.0)`` set those instead of
the roll parameters. ``prob``, ``split_seed``, ``scope`` and ``output_key`` are now
keyword-only.

A count of epochs replaces the repeat flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The three dataset builders took a boolean ``repeat``, which could only say
"once" or "forever". They now take ``num_epochs``: an integer count, or ``None``
for an unbounded stream. The two old settings are still expressible, and the
defaults are unchanged in effect.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Pre-1.0
     - 1.0
   * - ``repeat=True``
     - ``num_epochs=None`` — a genuinely infinite grain dataset
       (``len(ds) == sys.maxsize``)
   * - ``repeat=False``
     - ``num_epochs=1``, or simply drop the argument
   * - no equivalent
     - ``num_epochs=3`` — three passes over the corpus, each with its own
       shuffle, and a finite ``len(ds)``

Defaults: ``num_epochs=1`` for :func:`~audiotree.sources.create_audio_dataset`
and :func:`~audiotree.sources.create_windowed_audio_dataset`, ``num_epochs=None``
for :func:`~audiotree.sources.create_balanced_audio_dataset` — the same behavior
their ``repeat`` defaults gave.

.. skip-snippet-exec: fragment; ``/data/audio`` stands in for your corpus.

.. code-block:: python

    # Before
    ds = create_audio_dataset("/data/audio", repeat=True)

    # After
    ds = create_audio_dataset("/data/audio", num_epochs=None)

**Check:** ``num_epochs=0``, a negative count, a float, or a bool now raises
``ValueError``/``TypeError`` at construction time. Pre-1.0 a nonsensical value
was a silent no-op.

Shuffle and excerpt seeds are expanded before use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``shuffle_seed`` and ``excerpt_seed`` keep their names, types and defaults, but
each integer is now expanded through ``np.random.SeedSequence(...)`` before it
reaches grain, so small nearby seeds no longer produce correlated streams.

**Check:** the same seed produces a **different** file order and different
excerpt offsets than pre-1.0. Nothing to edit — but a run you intend to
reproduce byte-for-byte has to be re-rendered, or its output kept.

Writers refuse to overwrite an existing dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TreeWriter`` and ``AudioWriter`` both open their outputs in truncating mode, and
neither checked whether the target directory already held a dataset — so aiming one
at a finished directory destroyed it silently, and two processes aiming at one
directory interleaved into the same files. Both now take a keyword-only
``exist_ok`` (default ``False``) and raise ``FileExistsError`` naming the offending
file.

.. skip-snippet-exec: fragment; needs a dataset directory and a writer loop.

.. code-block:: python

    # Re-rendering into a directory you know is stale
    with TreeWriter("dataset/", expected_samples=10_000, exist_ok=True) as w:
        ...

Silence is passed through instead of becoming NaN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An item measuring ``-inf`` LUFS has no gain that reaches a target (``target - -inf``
is ``+inf``, and ``0 * inf`` is ``NaN``). ``normalize_lufs()`` and ``volume_norm``
(both backends) scaled by that gain anyway and then stamped the target on the
result, so a silent item became an all-``NaN`` waveform advertising itself as, say,
−18 LUFS — and one such item poisons every batch it lands in. This was never limited
to digital silence: anything below the BS.1770 absolute gate reads ``-inf``, which
zero-padded short reads and ``trim(mode="constant")`` produce routinely.

Non-finite items are now passed through **unscaled** and keep their ``-inf``, so
they stay identifiable downstream.

**Check:** code that assumed every item hits ``target_lufs`` exactly. It no longer
does, by design. ``normalize_lufs(max_gain_db=...)`` is a new opt-in ceiling so a
very quiet but still measurable item is not amplified without bound; a capped item
lands at ``lufs + max_gain_db`` rather than at the target.

``ExcerptConfig.search`` is resolved by name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``loudest_excerpt`` called ``eval()`` on a field deliberately typed ``str`` so
argbind can bind it from YAML — which made a config file arbitrary code execution in
every data worker. Names are now looked up in a registry, with an ``importlib``
dotted-path fallback for your own function, and resolution happens in
``__post_init__`` so a typo raises when the config is built rather than minutes into
training.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Value
     - Meaning
   * - ``"uniform"`` (the new default)
     - draw the offset uniformly over the file
   * - ``"bias_early"``
     - search progressively earlier in the file as attempts accumulate
   * - ``"mypkg.offsets.my_search"``
     - dotted path to an importable function
   * - a callable
     - accepted directly in Python

The pre-1.0 spellings ``"ExcerptConfig.search_uniform"`` and
``"ExcerptConfig.search_bias_early"`` are **not** carried over: they are not
registered names, and because they contain a dot the resolver treats them as
import paths — there is no importable ``ExcerptConfig`` module, so a config still
using them raises ``ValueError`` the moment it is built. Update them to the short
names: ``"ExcerptConfig.search_uniform"`` → ``"uniform"``, and
``"ExcerptConfig.search_bias_early"`` → ``"bias_early"``.

``ExcerptConfig.enabled`` is gone, replaced by ``strategy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two interacting booleans (``enabled`` for the loudness search, ``offset=0``
otherwise) became one named ``strategy``, so the three real behaviors each have a
name and the parameters that only apply to one of them say so.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Pre-1.0
     - 1.0
   * - ``ExcerptConfig(enabled=True, ...)``
     - ``ExcerptConfig(strategy="loudest", ...)``
   * - ``ExcerptConfig(enabled=False)`` (the old default)
     - ``ExcerptConfig(strategy="start")``
   * - no equivalent
     - ``ExcerptConfig(strategy="random")`` — the new default: a uniformly random
       offset, with no loudness measured

**Check:** ``strategy`` defaults to ``"random"``, so an ``ExcerptConfig()`` that
used to load every file from sample 0 now draws a random offset. Pass
``strategy="start"`` where that determinism mattered — validation sets especially.
And note that dropping a bare ``enabled=True`` is *not* enough: without
``strategy="loudest"`` you get the random default, not the loudness search.

``num_tries``, ``lufs_cutoff``, ``search`` and ``on_failure`` mean nothing under
``"start"`` or ``"random"``, and passing them there raises rather than being
silently ignored.

``bagz`` is an optional extra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bagz`` was a hard dependency gated on ``sys_platform == 'linux'``, but it publishes
manylinux x86-64 wheels only and has no sdist — which made ``pip install audiotree``
unsatisfiable on Linux aarch64 at every Python version, and on Linux x86-64 at
Python 3.14. Install ``audiotree[bagz]`` if you use **string leaves** in
``TreeWriter``/``TreeDataSource`` or the **windowed-LUFS cache**; nothing else needs
it. The import is now lazy and happens *after* the ``exclude_prefixes`` filter, so a
Linux-written dataset can be opened elsewhere with its string leaves excluded.

AudioWriter no longer resamples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sample_rate=`` constructor argument is gone. ``AudioWriter`` adopts the sample
rate of the first ``AudioTree`` written and raises ``ValueError`` if a later write
differs, so one manifest never mixes rates.

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot run.

.. code-block:: python

    # Before
    with AudioWriter("output", sample_rate=16_000) as w:
        w.write(tree)

    # After
    with AudioWriter("output") as w:
        w.write(tree.resample(16_000))

Codec transforms take a codec object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``encode_with_codec(codec)`` and ``encode_latents(codec)`` take a single codec
object instead of a bare ``encoder_fn`` (and, for ``encode_with_codec``, a
``num_codebooks`` count). The two transforms ask for two different, separately
declared protocols, each ``@runtime_checkable`` and each with exactly one method:
``audiotree.transforms.AudioCodec`` declares ``encode(AudioTree) -> (codes,
scale)``, and ``audiotree.transforms.LatentAudioCodec`` declares
``encode_to_latent(AudioTree) -> latents``. Implement whichever you need; a codec
that does both simply satisfies both. The
codec owns resampling, channel handling and output shapes, so the transform no
longer repacks codes into ``(batch, codebooks*channels, frames)``: ``AudioTree.codes``
holds exactly what the codec returned, and a non-``None`` ``scale`` is kept under
``metadata["codec_scale"]``.

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot run.

.. code-block:: python

    # Before
    encode_with_codec(encoder_fn=my_encode, num_codebooks=9)

    # After
    class MyCodec:
        def encode(self, audio):
            return my_encode(audio), None

    encode_with_codec(MyCodec())

**Check:** anything that indexed ``AudioTree.codes`` assuming the old packing.

Two transform backends, two RNG types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``audiotree.transforms`` is now the **NumPy** backend, for CPU Grain workers, and
takes ``np.random.Generator``. ``audiotree.transforms.jax`` is the JAX backend, for
jitted training steps, and takes ``jax.random.key``. A pipeline that passed a JAX key
to ``audiotree.transforms`` must either switch the import or switch the RNG.

.. skip-snippet-exec: the "Before" half is pre-1.0 API and cannot import.

.. code-block:: python

    # Before — one namespace, JAX keys
    from audiotree.transforms import VolumeNorm
    out = VolumeNorm(config={"min_db": -20, "max_db": -15}).random_map(
        tree, jax.random.key(0)
    )

    # After — pick a backend
    from audiotree.transforms import volume_norm            # NumPy / grain workers
    out = volume_norm(min_db=-20, max_db=-15).random_map(tree, np.random.default_rng(0))

    from audiotree.transforms import jax as jax_transforms  # JAX / jitted step
    out = jax_transforms.volume_norm(min_db=-20, max_db=-15).random_map(
        tree, jax.random.key(0)
    )

Both namespaces expose the same names (``choose`` is the one sanctioned exception —
it decides in Python which sub-transform to run, so it cannot be traced).

Smaller behavior changes
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Change
     - What to check
   * - ``create_balanced_audio_dataset(datasets=...)`` accepts pre-constructed
       Grain ``MapDataset``\ s, which **must already be repeated**
     - Call ``.repeat()`` before passing; a finite dataset makes
       ``grain.MapDataset.mix`` truncate to the shortest input.
   * - ``AudioTree.resample()`` keeps NumPy waveforms on NumPy (librosa/soxr on
       the CPU)
     - A NumPy waveform used to come back as a JAX array. ``zeros``, ``rolloff``
       and ``full`` apply to the JAX backend only.
   * - Filepath/source strings longer than the limit raise instead of truncating
     - The limit rose from 256 to 1024 characters; over it you now get a
       ``ValueError`` rather than a silently corrupted path.
   * - ``trim()``, ``roll(mode="constant")`` and ``to_stereo()`` invalidate
       ``lufs``; ``corrupt_phase()`` and ``shift_phase()`` do too unless
       ``keep_lufs=True``
     - Code reading ``tree.lufs`` after these now sees ``None``. Call
       ``replace_lufs()`` again.
   * - ``roll()`` sets ``metadata["offset"]`` to ``None``
     - It no longer points at sample 0 once the waveform has been shifted.
   * - The JAX ``shift_phase()`` draws **one phase angle per batch item**, not
       one per ``(batch, channel)``
     - It now matches the NumPy backend and the rest of the "one global nudge"
       transforms (``roll``). Stereo comes out phase-coherent instead of
       decorrelated; if you were relying on the decorrelation, ``corrupt_phase``
       is the transform that randomizes independently. Recorded JAX
       augmentations will not reproduce bit-for-bit — the RNG draw for that key
       changed shape.
   * - ``corrupt_phase()`` and ``shift_phase()`` no longer zero the last
       ``length % hop_length`` samples
     - Output length is unchanged; the tail is now reconstructed instead of
       silent. A 44100-sample clip at the default hop lost its final 68 samples
       before.
   * - ``find_audio_files()`` returns a **sorted**, de-duplicated list and expands
       glob patterns
     - File order (and therefore seeded shuffling) is now stable across machines
       and filesystems; a seeded run will not reproduce a pre-1.0 run's order.
   * - Manifest-supplied paths are confined to the dataset directory
     - A manifest naming an absolute path, a ``..`` segment, or a symlink out of
       the directory is now rejected. Manifests written by audiotree are fine.
   * - grain 0.2.17+ reads absl flags inside multiprocessing prefetch
     - A script doing multiprocessing data loading outside an ``absl.app.run``
       entry point must call ``flags.FLAGS.mark_as_parsed()`` once at startup.

On-disk data
------------

**All three on-disk formats now carry a header, and every reader validates it.**
Artifacts written by a pre-1.0 audiotree carry no header and are refused:

.. code-block:: text

    ValueError: dataset/manifest.json: this TreeWriter dataset carries no format
    header, so it was written by a pre-1.0 audiotree. Those layouts were never
    released and are not read by 1.0 — re-render the dataset.

That applies to ``TreeWriter`` directories, ``AudioWriter`` NPZ manifests, and
windowed-LUFS caches. **The migration is to re-render.** None of these layouts ever
shipped in a release — 1.0 is the first release to contain ``TreeWriter`` at all —
so there is no released data to migrate, and starting the contract clean was worth
more than accommodating development-time artifacts.

If re-rendering a large pre-1.0 ``TreeWriter`` corpus is genuinely impractical, the
field renames are mechanical: rename each ``*audio_data.bin`` to ``*waveform.bin``
and each ``*loudness.bin`` to ``*lufs.bin``, replace those names in
``manifest.json``, and add the header. The same applies to the ``loudness`` column
in an ``AudioWriter`` NPZ manifest. This is unsupported; the reader validates the
manifest strictly (``num_samples``, each leaf's ``shape_per_sample`` and an
allow-listed ``dtype``, and every ``children`` key against the real ``AudioTree``
fields), so a hand-edit that is close but not exact will be rejected.

The header
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Meaning
   * - ``format``
     - Which artifact this is (``audiotree-tree``, ``audiotree-manifest``,
       ``audiotree-lufs-windows``), so pointing a reader at the wrong directory
       fails by name instead of as a ``KeyError``.
   * - ``format_version``
     - ``[major, minor]``. A major mismatch is refused; minors are additive.
   * - ``min_reader_version``
     - The oldest reader that can make sense of this artifact.
   * - ``producer``
     - The writing audiotree version, for debugging.

What that buys you across 1.x is spelled out in :ref:`api_stability`.

Checklist
---------

#. ``grep`` for ``audio_data`` and ``loudness`` and take the renames above.
#. Switch ``audiotree.datasources`` imports to ``audiotree.sources``, and the
   data-source classes to ``create_audio_dataset()`` /
   ``create_balanced_audio_dataset()``.
#. Rewrite transforms to the snake_case functions with flat parameters. Run once
   and read every ``TypeError``: each one names a parameter that was silently
   doing nothing.
#. Replace ``scope={"k": True}`` with ``scope=["k"]``.
#. Replace any ``Batch`` transform with
   ``.batch(n, batch_fn=AudioTree.batch)``.
#. Re-render every pre-1.0 ``TreeWriter`` dataset, ``AudioWriter`` manifest, and
   windowed-LUFS cache.
#. Re-tune ``prob`` if the previous all-or-nothing batches were load-bearing, and
   re-check anything downstream of ``normalize_lufs`` that assumed every item lands
   exactly on target.
