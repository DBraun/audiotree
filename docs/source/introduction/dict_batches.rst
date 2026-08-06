.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _dict_batches:

Working with Dict[str, AudioTree] Batches
==========================================

A common training pattern involves processing multiple related audio signals together in a dictionary,
such as ``{"dry": AudioTree, "wet": AudioTree}`` or ``{"input": AudioTree, "target": AudioTree}``.

AudioTree transforms support this pattern with **scope** for selective transformation.

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

Why Use Dict Batches?
----------------------

Dictionary batches allow you to:

- **Process related signals**: Dry/wet, input/output, source/target pairs
- **Apply different augmentations**: Transform some signals but not others
- **Preserve relationships**: Keep multiple representations synchronized
- **Flexible pipelines**: Conditionally augment based on signal type

Common Patterns
---------------

**Pattern 1: Dry/Wet Processing**

Audio effect modeling often uses dry (original) and wet (processed) pairs:

.. skip-snippet-exec: fragment; the dry/wet trees are elided as ``AudioTree(...)``.

.. code-block:: python

    import numpy as np
    from audiotree import AudioTree
    from audiotree.transforms import volume_norm, volume_change

    # Create dry and wet signals
    dry_audio = AudioTree(...)
    wet_audio = AudioTree(...)

    dry_audio = dry_audio.replace_lufs()
    wet_audio = wet_audio.replace_lufs()

    batch = {'dry': dry_audio, 'wet': wet_audio}

    # Normalize both to same range
    transform1 = volume_norm(min_db=-20, max_db=-15)
    batch = transform1.random_map(batch, np.random.default_rng(42))

    # Add variation only to wet signal
    transform2 = volume_change(
        min_db=-6,
        max_db=6,
        scope={'wet': {'scope': True}},  # Only transform 'wet'
    )
    batch = transform2.random_map(batch, np.random.default_rng(43))

**Pattern 2: Input/Target for Supervised Learning**

.. skip-snippet-exec: fragment; the input/target trees are named but not built.

.. code-block:: python

    batch = {'input': input_audio, 'target': target_audio}

    # Normalize both
    transform1 = volume_norm(min_db=-20, max_db=-15)
    batch = transform1.random_map(batch, np.random.default_rng(42))

    # Add noise/augmentation only to input
    transform2 = volume_change(
        min_db=-12,
        max_db=12,
        prob=0.9,
        scope={'input': {'scope': True}},
    )
    batch = transform2.random_map(batch, np.random.default_rng(43))

**Pattern 3: Multi-Channel Processing**

.. skip-snippet-exec: fragment; the three trees are named but not built.

.. code-block:: python

    batch = {
        'dry': dry_audio,
        'wet': wet_audio,
        'reference': reference_audio,
    }

    # Normalize only dry and wet, not reference
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={
            'dry': {'scope': True},
            'wet': {'scope': True},
        },
    )
    batch = transform.random_map(batch, np.random.default_rng(42))

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
        scope={'src': {'scope': True}},
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
            'dry': {'scope': True},
            'wet': {'scope': True},
        },
    )
    batch = transform.random_map(batch, rng)

**Nested dictionaries:**

.. skip-snippet-exec: fragment; the nested trees are named but not built.

.. code-block:: python

    batch = {
        'input': {'dry': dry_audio, 'wet': wet_audio},
        'target': target_audio,
    }

    # Transform only input.dry
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={'input': {'dry': {'scope': True}}},
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

    batch = {'src': audio1, 'target': audio2}

    # Transform 'src' and output to 'src_modified'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={'src': {'scope': True}},
        output_key='modified',
    )
    batch = transform.random_map(batch, rng)

    # Result: {'src': original, 'target': original, 'modified': transformed}
    assert 'src' in batch
    assert 'target' in batch
    assert 'modified' in batch  # New key with transformed src
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
        repeat=True,
        sample_rate=48000,
        duration=5.0,
    )

    # Create dict batches with multiple variants
    def create_variants(audio_tree):
        return {
            'clean': audio_tree,
            'augmented': audio_tree,  # Will be augmented
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
            scope={'augmented': {'scope': True}},
        ),
    )

    # 3. Random phase inversion only on 'augmented'
    ds = ds.random_map(
        invert_phase(
            prob=0.5,
            scope={'augmented': {'scope': True}},
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
        clean = jax.numpy.array(batch['clean'].waveform)
        augmented = jax.numpy.array(batch['augmented'].waveform)

        # Train model
        loss = train_step(clean, augmented)

Creating Dict Batches from Datasets
------------------------------------

**Method 1: Custom loading function (for paired files)**

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

        return {'dry': dry, 'wet': wet}

    # Create dataset of filepath pairs
    filepaths = [
        ('/data/dry/001.wav', '/data/wet/001.wav'),
        ('/data/dry/002.wav', '/data/wet/002.wav'),
        # ...
    ]

    ds = grain.MapDataset.source(filepaths)
    ds = ds.random_map(load_dry_wet_pair, seed=42)

    # Now chain transforms with scope
    ds = ds.random_map(
        volume_norm(
            min_db=-20,
            max_db=-15,
            scope={'dry': {'scope': True}},
        ),
        seed=43,
    )

**Method 2: Create variants from single dataset (recommended)**

This is the simplest and most common pattern:

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, volume_change

    # Load single dataset
    ds = create_audio_dataset(sources="/data/audio", repeat=True)
    ds = ds.seed(42)

    # Create multiple versions of each item
    def create_variants(audio_tree):
        return {
            'original': audio_tree,
            'augmented': audio_tree,  # Will be augmented differently
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
            scope={'augmented': {'scope': True}},
        ),
        seed=43,
    )

This creates batches like ``{'original': clean_audio, 'augmented': noisy_audio}`` useful
for self-supervised learning or data augmentation research.

Scope with ArgBind
------------------

**YAML Configuration:**

.. code-block:: yaml

    # Transform only 'dry' signal
    volume_change.min_db: -6
    volume_change.max_db: 6
    volume_change.scope:
      dry:
        scope: true

    # Transform both 'dry' and 'wet'
    volume_norm.min_db: -20
    volume_norm.max_db: -15
    volume_norm.scope:
      dry:
        scope: true
      wet:
        scope: true

**Python:**

.. code-block:: python

    import argbind
    from audiotree.transforms import volume_norm, volume_change

    volume_norm = argbind.bind(volume_norm)
    volume_change = argbind.bind(volume_change)

    args = argbind.parse_args()
    with argbind.scope(args):
        transform1 = volume_norm()
        transform2 = volume_change()

        batch = transform1.random_map(batch, np.random.default_rng(42))
        batch = transform2.random_map(batch, np.random.default_rng(43))

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

To give each key its own parameters, use one transform instance per key:

.. code-block:: python

    import numpy as np
    from audiotree.transforms import volume_norm

    dry_norm = volume_norm(min_db=-25, max_db=-15, scope=["dry"])
    wet_norm = volume_norm(min_db=-20, max_db=-10, scope=["wet"])

    batch = dry_norm.random_map(batch, np.random.default_rng(42))
    batch = wet_norm.random_map(batch, np.random.default_rng(43))

The same thing in YAML, using argbind's own pattern scoping (see
:ref:`argbind_guide`) to give the two bindings different parameters:

.. code-block:: yaml

    dry/volume_norm.min_db: -25
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

Best Practices
--------------

1. **Use descriptive keys**: 'dry'/'wet', 'input'/'target', not 'x'/'y'
2. **Apply shared transforms first**: Normalize both, then augment selectively
3. **Be explicit with scope**: Always specify which keys to transform
4. **Test both keys**: Verify transforms apply correctly to each signal
5. **Document your structure**: Comment what each key represents

Common Use Cases
----------------

**Audio Effect Modeling:**

.. skip-snippet-exec: fragment; the dry/wet signals are named but not built.

.. code-block:: python

    batch = {'dry': dry_signal, 'wet': wet_signal}

    # Normalize both
    batch = volume_norm().random_map(batch, rng)

    # Augment only wet
    batch = volume_change(
        min_db=-6, max_db=6,
        scope={'wet': {'scope': True}},
    ).random_map(batch, rng)

**Source Separation:**

.. skip-snippet-exec: fragment; the stem trees are named but not built.

.. code-block:: python

    batch = {
        'mixture': mixture_audio,
        'vocals': vocals_audio,
        'drums': drums_audio,
        'bass': bass_audio,
    }

    # Normalize all sources
    batch = volume_norm(min_db=-20, max_db=-15).random_map(batch, rng)

    # Augment only mixture
    batch = volume_change(
        min_db=-3, max_db=3,
        scope={'mixture': {'scope': True}},
    ).random_map(batch, rng)

**Self-Supervised Learning:**

.. skip-snippet-exec: fragment; ``audio`` is elided.

.. code-block:: python

    batch = {
        'anchor': audio,
        'positive': audio,  # Same audio, will be augmented differently
        'negative': different_audio,
    }

    # Apply different augmentations to anchor and positive
    batch = volume_change(
        min_db=-12, max_db=12,
        scope={'anchor': {'scope': True}},
    ).random_map(batch, np.random.default_rng(42))

    batch = volume_change(
        min_db=-12, max_db=12,
        scope={'positive': {'scope': True}},
    ).random_map(batch, np.random.default_rng(999))  # Different seed!

Complete Example
----------------

Full training pipeline with dict batches:

.. skip-snippet-exec: fragment; the dataset is elided as ``...``.

.. code-block:: python

    import grain
    import jax
    from audiotree import AudioTree
    from audiotree.transforms import volume_norm, volume_change, invert_phase, trim

    # Assuming you have a dataset that loads dry/wet pairs
    # (see "Creating Dict Batches" section)

    # Start with dataset of {'dry': AudioTree, 'wet': AudioTree}
    ds = ...  # Your paired dataset

    # Build augmentation pipeline
    ds = ds.seed(42)

    # 1. Normalize both signals to similar loudness range
    ds = ds.random_map(
        volume_norm(min_db=-25, max_db=-15),
    )

    # 2. Add random volume change to dry only (for robustness)
    ds = ds.random_map(
        volume_change(
            min_db=-6,
            max_db=6,
            prob=0.8,
            scope={'dry': {'scope': True}},
        ),
    )

    # 3. Random phase inversion on wet only
    ds = ds.random_map(
        invert_phase(
            prob=0.5,
            scope={'wet': {'scope': True}},
        ),
    )

    # 4. Trim both to final length
    ds = ds.map(trim(length=3.0))

    # 5. Add multiprocessing
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

    # Training loop
    @jax.jit
    def train_step(dry, wet):
        # Your model here
        predictions = model(dry)
        loss = loss_fn(predictions, wet)
        return loss

    for step, batch in enumerate(iter_ds):
        if step >= 10000:
            break

        # Extract signals
        dry = jax.numpy.array(batch['dry'].waveform)
        wet = jax.numpy.array(batch['wet'].waveform)

        # Train
        loss = train_step(dry, wet)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

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

Advanced: Different Configs per Key
------------------------------------

Apply different parameter values to different keys:

**YAML:**

.. code-block:: yaml

    volume_norm.min_db: -20  # Default
    volume_norm.max_db: -15  # Default
    volume_norm.scope:
      dry:
        scope: true
        min_db: -30  # Override for 'dry'
      wet:
        scope: true
        max_db: -10  # Override for 'wet'

This creates:
- 'dry': normalized to [-30, -15] LUFS
- 'wet': normalized to [-20, -10] LUFS

Testing Dict Batches
---------------------

Always test that scope works correctly:

.. testcode::

    from audiotree.transforms import volume_norm

    def test_scope_selective_transform():
        """Test that scope only transforms specified keys."""
        batch = {'src': audio1, 'target': audio2}

        # Transform only 'src'
        transform = volume_norm(
            min_db=-20,
            max_db=-15,
            scope={'src': {'scope': True}},
        )

        # NumPy-backend transforms take a np.random.Generator.
        result = transform.random_map(batch, np.random.default_rng(42))

        # Verify only src changed
        assert not np.array_equal(result['src'].lufs, audio1.lufs)
        assert np.array_equal(result['target'].lufs, audio2.lufs)

    test_scope_selective_transform()

Common Pitfalls
---------------

**Issue:** All keys are transformed when I only want one

**Solution:** Use explicit scope with `{'scope': True}`:

.. code-block:: python

    # Wrong - transforms all keys
    transform = volume_norm(min_db=-20, max_db=-15)

    # Right - transforms only 'src'
    transform = volume_norm(
        min_db=-20,
        max_db=-15,
        scope={'src': {'scope': True}},
    )

**Issue:** Scope doesn't work with nested dicts

**Solution:** Use nested scope structure:

.. skip-snippet-exec: fragment; ``audio`` is elided.

.. code-block:: python

    batch = {'input': {'dry': audio, 'wet': audio}, 'target': audio}

    # Transform input.dry
    scope={'input': {'dry': {'scope': True}}}

**Issue:** Want to exclude one key, transform others

**Solution:** Scope is inclusion-based, not exclusion. Specify all keys to include:

.. code-block:: python

    # To transform 'a' and 'b', but not 'c':
    scope={
        'a': {'scope': True},
        'b': {'scope': True},
    }

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

    def create_pair(audio_tree):
        return {"src": audio_tree, "tgt": audio_tree}

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
    ds = create_audio_dataset("/data/audio", duration=3.0, shuffle=True, repeat=True)

    def create_variants(audio_tree):
        return {"clean": audio_tree, "augmented": audio_tree}

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
- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :func:`~audiotree.transforms.volume_norm` - Volume normalization
- :func:`~audiotree.transforms.volume_change` - Volume change
