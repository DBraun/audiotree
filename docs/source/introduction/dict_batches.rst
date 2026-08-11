.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _dict_batches:

Working with Dict[str, AudioTree] Batches
==========================================

Many training setups carry several related signals per example: a dry/wet pair
for effect modeling, an input/target pair for supervised learning, a mixture and
its stems for source separation. The natural container is a dict of AudioTrees,
like ``{"dry": AudioTree, "wet": AudioTree}``, and every transform accepts one.
By default it touches all the AudioTree leaves, and its ``scope`` parameter
restricts it to the keys you name, so you can normalize both signals, then
augment only one, while the pair stays aligned.

.. testsetup::

    # Hidden shared setup for the executable examples on this page: two small
    # AudioTrees and a NumPy RNG. The NumPy-backend transforms in
    # ``audiotree.transforms`` take a ``np.random.Generator``.
    import numpy as np
    from audiotree import AudioTree

    _g = np.random.default_rng(0)
    audio1 = AudioTree(_g.standard_normal((2, 1, 44_100)), 44_100).replace_lufs()
    audio2 = AudioTree(_g.standard_normal((2, 1, 44_100)), 44_100).replace_lufs()
    rng = np.random.default_rng(42)

Three Common Setups
-------------------

These are pipeline stages: a transform with a ``scope`` chains onto a dataset
of dicts with ``.random_map()`` exactly as in :ref:`transform_chaining`, with
one ``ds.seed(n)`` up front. (The same transforms can also be applied to a
single dict directly, ``transform.random_map(batch, rng)``; the scope-shape
reference further down uses that form.)

**Dry/wet.** Audio effect modeling pairs the original signal with the processed
one; normalize both, then augment only the wet side. Runnable end to end on a
tiny two-item dataset:

.. testcode::

    import grain
    from audiotree.transforms import volume_norm, volume_change

    # A dataset whose items are {"dry": AudioTree, "wet": AudioTree} (see
    # "Creating Dict Batches from Datasets" below for building one from files).
    ds = grain.MapDataset.source([{"dry": audio1, "wet": audio2}])
    ds = ds.seed(42)

    # Normalize both signals to the same range...
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # ...then add variation only to the wet signal.
    ds = ds.random_map(volume_change(min_db=-6, max_db=6, scope=["wet"]))

    item = ds[0]
    print(sorted(item))
    print(bool(np.all((item["dry"].lufs >= -21) & (item["dry"].lufs <= -14))))

.. testoutput::

    ['dry', 'wet']
    True

**Input/target.** For supervised learning, perturb the input while the target
stays clean:

.. skip-snippet-exec: fragment; ``ds`` is the reader's own paired dataset.

.. code-block:: python

    # ds yields {"input": AudioTree, "target": AudioTree}
    ds = ds.seed(42)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))
    ds = ds.random_map(
        volume_change(min_db=-12, max_db=12, prob=0.9, scope=["input"])
    )

**More than two keys.** ``scope`` can name any subset. Here a reference signal
is left untouched while the other two are normalized:

.. skip-snippet-exec: fragment; ``ds`` is the reader's own three-key dataset.

.. code-block:: python

    # ds yields {"dry": ..., "wet": ..., "reference": ...}
    ds = ds.random_map(
        volume_norm(min_db=-20, max_db=-15, scope=["dry", "wet"])
    )

.. _loading_aligned_stems:

Loading Aligned Stems
---------------------

The patterns above assume the dict already exists. Building one from a corpus of
*stems* — several files per track that must be excerpted at the **same offset**,
as in source separation or effect modelling — is the step where alignment can
silently break.

Draw the offset **once per track** and pass it explicitly to every stem. Clamp it
against the *shortest* stem, so no stem can be asked for a window that runs past
its end:

.. skip-snippet-exec: needs a stem corpus on disk; the mechanism is tested in tests/sources/test_excerpt_selection.py.

.. code-block:: python

    from pathlib import Path

    import grain
    import numpy as np
    import soundfile
    from audiotree import AudioTree

    STEMS = ("bass", "drums", "other", "vocals")


    class PickStems(grain.transforms.RandomMap):
        """Load the same time window from every stem of one track."""

        def __init__(self, duration: float = 5.0, sample_rate: int = 44_100):
            self.duration = duration
            self.sample_rate = sample_rate

        def random_map(self, track_dir: Path, rng: np.random.Generator):
            paths = {stem: track_dir / f"{stem}.wav" for stem in STEMS}

            # One offset for the track, clamped to the shortest stem. `.info`
            # reads the header only (~35 us/file), so this costs nothing next to
            # decoding the audio -- and it is what makes alignment a property of
            # the code rather than of the corpus.
            shortest = min(soundfile.info(str(p)).duration for p in paths.values())
            latest_start = max(0.0, shortest - self.duration)
            offset = float(rng.uniform(0.0, latest_start))

            return {
                stem: AudioTree.from_file(
                    path,
                    offset=offset,
                    duration=self.duration,
                    sample_rate=self.sample_rate,
                )
                for stem, path in paths.items()
            }


    tracks = [p for p in sorted(Path("/data/musdb18hq/train").iterdir()) if p.is_dir()]
    ds = (
        grain.MapDataset.source(tracks)
        .shuffle(seed=0)
        .seed(0)
        .apply([PickStems(duration=5.0)])
    )
    stems = ds[0]  # {"bass": AudioTree, "drums": AudioTree, ...}

Every stem gets the same number, so they are aligned by construction — no
assumption that the stems are equally long, and no dependence on how any
randomness is threaded.

.. warning::

   **Do not align stems by re-seeding** :meth:`~audiotree.AudioTree.excerpt`.
   It is tempting, because ``excerpt`` is a pure function of the generator it is
   given, so handing each stem a generator built from one shared seed does
   produce one shared offset:

   .. skip-snippet-exec: deliberately wrong; its failure mode is pinned by tests/sources/test_excerpt_selection.py instead.

   .. code-block:: python

       seed = rng.integers(2**63)  # DON'T
       stems = {
           stem: AudioTree.excerpt(
               track_dir / f"{stem}.wav", np.random.default_rng(seed), duration=5.0
           )
           for stem in STEMS
       }

   It works only while every stem is exactly as long as the others. ``excerpt``
   clamps the offset it draws against *that file's* duration, so the moment one
   stem is shorter — a trailing silence trimmed, a different encoder, a stem
   rendered a few samples short — that stem alone lands somewhere else. Nothing
   raises. Misaligned stems train without any visible error and produce a model
   that cannot separate anything.

   The variant that shares one *generator* rather than one seed is worse still:
   each call advances it, so every stem gets a different offset even on a
   perfectly regular corpus.

Two more things worth knowing:

- **Sum the stems to get the mixture**, rather than reading a distributed mixture
  file, if the target must be exactly the sum of the inputs. A released mixture
  is mastered and will not be.
- **Batch with** ``batch_fn=AudioTree.batch``. It maps over the dict, so a list
  of stem dicts collapses into one dict of batched trees with no per-key
  handling.

If the header reads ever do show up in a profile, hoist them: durations are a
pure function of the corpus, so scan once with
:func:`~audiotree.sources.scan_durations` and pass the mapping into your
transform instead of calling ``.info`` per item.

Once loaded, ``scope`` selects which stems an augmentation touches, exactly as in
the patterns above.


Using Scope
-----------

**Transform all keys (default):**

.. skip-snippet-exec: fragment; ``batch`` comes from the section above.

.. code-block:: python

    # No scope specified - transforms all AudioTree leaves
    transform = volume_norm(min_db=-20, max_db=-15)
    batch = transform.random_map(batch, rng)

**Transform specific key:**

.. skip-snippet-exec: fragment; ``batch`` comes from the section above.

.. code-block:: python

    # Only transform 'src'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={"src": {"scope": True}},
    )
    batch = transform.random_map(batch, rng)

**Transform multiple specific keys:**

.. skip-snippet-exec: fragment; ``batch`` comes from the section above.

.. code-block:: python

    # Transform 'dry' and 'wet', not 'reference'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={
            "dry": {"scope": True},
            "wet": {"scope": True},
        },
    )
    batch = transform.random_map(batch, rng)

**Nested dictionaries:**

.. skip-snippet-exec: fragment; the nested trees are named but not built.

.. code-block:: python

    batch = {
        "input": {"dry": dry_audio, "wet": wet_audio},
        "target": target_audio,
    }

    # Transform only input.dry
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={"input": {"dry": {"scope": True}}},
    )
    batch = transform.random_map(batch, rng)

Scope with ArgBind
------------------

Configure scope via YAML:

**config.yml:**

.. code-block:: yaml

    volume_change.min_db: -6
    volume_change.max_db: 6
    volume_change.prob: 0.9
    volume_change.scope:
      input:  # Only transform 'input' key
        scope: true

**Python:**

.. skip-snippet-exec: fragment; argbind parses the command line at run time.

.. code-block:: python

    import argbind
    from audiotree.transforms import volume_change

    volume_change = argbind.bind(volume_change)

    args = argbind.parse_args()
    with argbind.scope(args):
        transform = volume_change()
        batch = transform.random_map(batch, rng)

Scope with Output Key
----------------------

Combine scope with output_key to create new keys:

.. testcode::

    from audiotree.transforms import volume_norm

    batch = {"src": audio1, "target": audio2}

    # Transform 'src' and output to 'src_modified'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={"src": {"scope": True}},
        output_key="modified",
    )
    batch = transform.random_map(batch, rng)

    # Result: {'src': original, 'target': original, 'modified': transformed}
    assert "src" in batch
    assert "target" in batch
    assert "modified" in batch  # New key with transformed src
    print(sorted(batch.keys()))

.. testoutput::

    ['modified', 'src', 'target']

Complete Training Pipeline
---------------------------

Full example using dict batches with scope:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    import jax
    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, volume_change, invert_phase, trim

    # Load audio
    ds = create_audio_dataset(
        sources="/data/audio",
        shuffle=True,
        num_epochs=None,
        sample_rate=48000,
        duration=5.0,
    )


    # Create dict batches with multiple variants
    def create_variants(audio):
        return {
            "clean": audio,
            "augmented": audio,  # Will be augmented
        }


    ds = ds.map(create_variants)

    # Chain augmentations with scope
    ds = ds.seed(42)

    # 1. Normalize both
    ds = ds.random_map(
        volume_norm(min_db=-25, max_db=-15),
    )

    # 2. Add variation only to 'augmented'
    ds = ds.random_map(
        volume_change(
            min_db=-6,
            max_db=6,
            prob=0.9,
            scope={"augmented": {"scope": True}},
        ),
    )

    # 3. Random phase inversion only on 'augmented'
    ds = ds.random_map(
        invert_phase(
            prob=0.5,
            scope={"augmented": {"scope": True}},
        ),
    )

    # 4. Trim both
    ds = ds.map(trim(length=3.0))

    # Convert to iterator with multiprocessing
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

    # Training loop
    for step, batch in enumerate(iter_ds):
        if step >= 1000:
            break

        # Extract clean and augmented
        clean = jax.numpy.array(batch["clean"].waveform)
        augmented = jax.numpy.array(batch["augmented"].waveform)

        # Train model
        loss = train_step(clean, augmented)

Creating Dict Batches from Datasets
------------------------------------

When the pairs exist as separate files on disk, load them with a custom
function:

.. code-block:: python

    import grain
    from audiotree import AudioTree


    def load_dry_wet_pair(filepath_pair):
        """Load a dry/wet pair from file paths."""
        dry_path, wet_path = filepath_pair

        dry = AudioTree.from_file(
            dry_path,
            sample_rate=44100,
            duration=3.0,
        )
        wet = AudioTree.from_file(
            wet_path,
            sample_rate=44100,
            duration=3.0,
        )

        return {"dry": dry, "wet": wet}


    # Create dataset of filepath pairs
    filepaths = [
        ("/data/dry/001.wav", "/data/wet/001.wav"),
        ("/data/dry/002.wav", "/data/wet/002.wav"),
        # ...
    ]

    ds = grain.MapDataset.source(filepaths)
    ds = ds.random_map(load_dry_wet_pair, seed=42)

    # Now chain transforms with scope
    ds = ds.random_map(
        volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"dry": {"scope": True}},
        ),
        seed=43,
    )

More often, though, both variants start from the *same* audio and diverge only
through augmentation, in which case a plain ``.map()`` that duplicates each
item into a dict is all it takes:

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, volume_change

    # Load single dataset
    ds = create_audio_dataset(sources="/data/audio", num_epochs=None)
    ds = ds.seed(42)


    # Create multiple versions of each item
    def create_variants(audio):
        return {
            "original": audio,
            "augmented": audio,  # Will be augmented differently
        }


    ds = ds.map(create_variants)

    # Normalize both
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Apply strong augmentation only to 'augmented'
    ds = ds.random_map(
        volume_change(
            min_db=-12,
            max_db=12,
            prob=0.9,
            scope={"augmented": {"scope": True}},
        ),
        seed=43,
    )

Each element comes out as ``{'original': clean_audio, 'augmented': noisy_audio}``,
the structure that self-supervised training typically uses.

Different Parameters Per Key
----------------------------

``scope`` selects *which* keys a transform touches. It does **not** carry
per-key parameters — a transform instance has exactly one set of parameters, and
they apply to every key it selects. Anything other than the ``scope`` sentinel
inside a scope dict is read as another path/boolean selection, not as an
override, so this does not do what it looks like:

.. skip-snippet-test: illustrates a mistake — the overrides below are not applied.

.. code-block:: yaml

    # WRONG: min_db/max_db here are NOT per-key overrides.
    volume_norm.min_db: -20
    volume_norm.max_db: -15
    volume_norm.scope:
      dry:
        scope: true
        min_db: -25   # ignored as a parameter
      wet:
        scope: true
        max_db: -10   # ignored as a parameter

To give each key its own parameters, use one transform instance per key, each
with its own parameters and each scoped to its key:

.. testcode::

    from audiotree.transforms import volume_norm

    batch = {"dry": audio1, "wet": audio2}

    # 'dry' is normalized into [-30, -15] LUFS ...
    batch = volume_norm(min_db=-30, max_db=-15, scope=["dry"]).random_map(batch, rng)

    # ... and 'wet' into [-20, -10] LUFS, with its own parameters.
    batch = volume_norm(min_db=-20, max_db=-10, scope=["wet"]).random_map(batch, rng)

    assert np.all(batch["dry"].lufs >= -31) and np.all(batch["dry"].lufs <= -14)
    assert np.all(batch["wet"].lufs >= -21) and np.all(batch["wet"].lufs <= -9)

Putting parameter values *inside* a scope entry, e.g.
``scope={'dry': {'scope': True, 'min_db': -30}}``, raises a ``ValueError``:
per-key parameter overrides are not supported, and older versions read such
keys as extra scope markers rather than overrides, so the pipeline silently
ran with the default parameters everywhere.

The same thing in YAML, using argbind's own pattern scoping (see
:ref:`argbind_guide`) to give the two bindings different parameters:

.. code-block:: yaml

    dry/volume_norm.min_db: -30
    dry/volume_norm.max_db: -15
    dry/volume_norm.scope: [dry]

    wet/volume_norm.min_db: -20
    wet/volume_norm.max_db: -10
    wet/volume_norm.scope: [wet]

Build one instance under each argbind pattern:

.. skip-snippet-exec: needs a parsed argbind config on the command line.

.. code-block:: python

    with argbind.scope(args, "dry"):
        dry_norm = volume_norm()
    with argbind.scope(args, "wet"):
        wet_norm = volume_norm()

ArgBind with Scoped Transforms
-------------------------------

Configure different pipelines for train vs validation:

**config.yml:**

.. code-block:: yaml

    # Training: aggressive augmentation
    train/volume_change.min_db: -12
    train/volume_change.max_db: 12
    train/volume_change.prob: 0.9
    train/volume_change.scope:
      dry:
        scope: true

    # Validation: no augmentation on dry
    val/volume_change.prob: 0.0

    # Shared: normalize both
    volume_norm.min_db: -20
    volume_norm.max_db: -15

**Python:**

.. code-block:: python

    import argbind
    from audiotree.transforms import volume_norm, volume_change

    volume_norm = argbind.bind(volume_norm, "train", "val")
    volume_change = argbind.bind(volume_change, "train", "val")

    args = argbind.parse_args()


    def augment_batch(batch, rng, scope_name):
        """Augment batch based on scope (train or val)."""
        with argbind.scope(args, scope_name):
            # Normalize both
            transform1 = volume_norm()
            batch = transform1.random_map(batch, rng)

            # Augment dry (if enabled for this scope). A NumPy Generator is
            # stateful, so reusing ``rng`` draws fresh randomness — no split needed.
            transform2 = volume_change()
            batch = transform2.random_map(batch, rng)

        return batch


    # Training
    train_batch = augment_batch(batch, rng, "train")

    # Validation
    val_batch = augment_batch(batch, rng, "val")

Testing Dict Batches
---------------------

A scope typo fails silently: the transform just applies everywhere, or
nowhere. A small test that the right keys changed is worth having:

.. testcode::

    from audiotree.transforms import volume_norm


    def test_scope_selective_transform():
        """Test that scope only transforms specified keys."""
        batch = {"src": audio1, "target": audio2}

        # Transform only 'src'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"src": {"scope": True}},
        )

        # NumPy-backend transforms take a np.random.Generator.
        result = transform.random_map(batch, np.random.default_rng(42))

        # Verify only src changed
        assert not np.array_equal(result["src"].lufs, audio1.lufs)
        assert np.array_equal(result["target"].lufs, audio2.lufs)


    test_scope_selective_transform()

Common Pitfalls
---------------

Three common mistakes. First, the default: with no ``scope`` at all, a
transform touches *every* AudioTree leaf, so if only one key should change, you
must say so. Second, nesting: the scope dict mirrors the batch's nesting, so for
``{'input': {'dry': ..., 'wet': ...}, 'target': ...}`` the path to the dry
signal is ``scope={'input': {'dry': {'scope': True}}}``. Third, exclusion: to
transform everything except one key, name the keys you *do* want.
``scope={'a': {'scope': True}, 'b': {'scope': True}}`` transforms ``a`` and
``b`` and leaves ``c`` alone.

Batching Dict Structures
-------------------------

When using Grain's ``IterDataset.batch()`` API with dict structures containing AudioTrees,
use ``AudioTree.batch`` as the batch function. It handles both direct AudioTree sequences
and nested structures like dicts.

**Important**: ``batch`` concatenates all arrays along axis 0. This means your data
should already have a batch dimension (even if it's size 1), which is how AudioTree works
by default with shape ``(batch, channels, samples)``.

**Basic batching with AudioTrees:**

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    ds = create_audio_dataset("/data/audio", duration=1.0)
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    for batch in iter_ds:
        print(batch.waveform.shape)  # (32, channels, samples)

**Batching dict structures:**

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    # Create dataset that yields {"src": AudioTree, "tgt": AudioTree}
    ds = create_audio_dataset("/data/audio", duration=1.0)


    def create_pair(audio):
        return {"src": audio, "tgt": audio}


    ds = ds.map(create_pair)

    # batch handles dict structures automatically
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    for batch in iter_ds:
        # batch is {"src": AudioTree, "tgt": AudioTree} with batched arrays
        print(batch["src"].waveform.shape)  # (32, channels, samples)
        print(batch["tgt"].waveform.shape)  # (32, channels, samples)

**Nested structures:**

``batch`` uses JAX's tree utilities, so it handles arbitrarily nested structures:

.. code-block:: python

    # Works with nested dicts
    {"input": {"clean": AudioTree, "noisy": AudioTree}, "target": AudioTree}

    # Works with mixed structures (AudioTrees and regular arrays)
    {"audio": AudioTree, "labels": np.array([...])}

For regular arrays (non-AudioTree), ``batch`` also concatenates along axis 0, so ensure
they have a leading batch dimension.

**Complete example with multiprocessing:**

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm

    # Create paired dataset
    ds = create_audio_dataset("/data/audio", duration=3.0, shuffle=True, num_epochs=None)


    def create_variants(audio):
        return {"clean": audio, "augmented": audio}


    ds = ds.map(create_variants)

    # Apply transforms with scope
    ds = ds.random_map(
        volume_norm(
            min_db=-20,
            max_db=-15,
            scope={"augmented": {"scope": True}},
        ),
        seed=42,
    )

    # Batch and add multiprocessing
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    iter_ds = (
        ds.to_iter_dataset()
        .batch(32, batch_fn=AudioTree.batch)
        .mp_prefetch(options=mp_options)
    )

    for batch in iter_ds:
        clean = batch["clean"].waveform  # (32, channels, samples)
        augmented = batch["augmented"].waveform  # (32, channels, samples)
        # Train model...

See Also
--------

- :ref:`transform_chaining` - Chaining transforms with datasets
- :ref:`codecs` - ``scope`` and ``output_key`` applied to the codec transforms
- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :func:`~audiotree.transforms.volume_norm` - Volume normalization
- :func:`~audiotree.transforms.volume_change` - Volume change
