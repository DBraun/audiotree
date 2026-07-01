.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _transform_chaining:

Chaining Transforms with Datasets
==================================

AudioTree transforms integrate seamlessly with Grain's dataset chaining API, allowing you to build augmentation pipelines declaratively.

Basic Transform Chaining
-------------------------

Apply transforms to a dataset using ``.random_map()`` or ``.map()``:

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, trim

    # Create dataset
    ds = create_audio_dataset(
        sources="/data/audio",
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )
    ds = ds.seed(42)  # apply seed for random_map later

    # Chain transforms
    transform1 = volume_norm(min_db=-20, max_db=-15)
    transform2 = trim(length=3.0)

    ds = ds.random_map(transform1)  # Apply volume_norm
    ds = ds.map(transform2)         # Apply trim

    # Load items - transforms are applied lazily
    audio_tree = ds[0]

**Key Points:**

- ``.random_map()`` for stochastic transforms (volume_norm, volume_change, etc.)
- ``.map()`` for deterministic transforms (trim, mono, etc.)
- Transforms are applied **lazily** when items are accessed
- Call ``ds.seed(n)`` **once** before your ``.random_map()`` calls; each one
  derives its own distinct, reproducible seed from it, so you don't pass a
  ``seed=`` to every map

Multiple Transform Pipeline
----------------------------

Build complex augmentation pipelines:

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import (
        volume_norm,
        volume_change,
        invert_phase,
        trim,
        mono,
    )

    # Create balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        weights={"speech": 0.7, "music": 0.3},
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )

    # Build augmentation pipeline
    ds = ds.seed(42)

    # 1. Normalize volume
    ds = ds.random_map(volume_norm(min_db=-25, max_db=-15))

    # 2. Random volume change (90% probability)
    ds = ds.random_map(volume_change(min_db=-6, max_db=6, prob=0.9))

    # 3. Random phase inversion (50% probability)
    ds = ds.random_map(invert_phase(prob=0.5))

    # 4. Trim to final length
    ds = ds.map(trim(length=3.0))

    # 5. Convert to mono
    ds = ds.map(mono())

    # Transforms applied when accessing items
    for item in ds.to_iter_dataset():
        # item has been through full augmentation pipeline
        print(item.waveform.shape)  # (1, 1, 132300) for 3s mono at 44.1kHz
        break

With ArgBind
------------

Use ArgBind to configure transform chains from YAML:

**pipeline.py:**

.. code-block:: python

    import argbind
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree import transforms

    # Bind transforms
    volume_norm = argbind.bind(transforms.volume_norm)
    volume_change = argbind.bind(transforms.volume_change)
    invert_phase = argbind.bind(transforms.invert_phase)
    trim = argbind.bind(transforms.trim)

    def create_pipeline():
        # Create dataset
        ds = create_balanced_audio_dataset(
            sources={
                "speech": ["/data/speech"],
                "music": ["/data/music"],
            },
            shuffle=True,
            repeat=True,
            sample_rate=44100,
            duration=5.0,
        )

        # Seed the dataset once; each random_map derives its own distinct seed.
        ds = ds.seed(42)

        # Chain transforms (configured via ArgBind)
        ds = ds.random_map(volume_norm())
        ds = ds.random_map(volume_change())
        ds = ds.random_map(invert_phase())
        ds = ds.map(trim())

        return ds

    if __name__ == "__main__":
        args = argbind.parse_args()
        with argbind.scope(args):
            ds = create_pipeline()

            # Iterate over augmented data
            for item in ds.to_iter_dataset():
                print(item.source, item.lufs)
                break

**config.yml:**

.. code-block:: yaml

    volume_norm.min_db: -25
    volume_norm.max_db: -15

    volume_change.min_db: -6
    volume_change.max_db: 6
    volume_change.prob: 0.9

    invert_phase.prob: 0.5

    trim.length: 3.0

**Run:**

.. code-block:: bash

    python pipeline.py --args.load=config.yml

Conditional Transforms with Scopes
-----------------------------------

Apply different transforms for training vs validation:

.. code-block:: python

    import argbind
    from audiotree.sources import create_audio_dataset
    from audiotree import transforms

    volume_norm = argbind.bind(transforms.volume_norm, "train", "val")
    volume_change = argbind.bind(transforms.volume_change, "train", "val")

    def create_train_pipeline():
        ds = create_audio_dataset(
            sources="/data/train",
            shuffle=True,
            repeat=True,
            sample_rate=44100,
            duration=3.0,
        )

        ds = ds.seed(42)

        # Aggressive augmentation for training
        with argbind.scope(args, "train"):
            ds = ds.random_map(volume_norm())
            ds = ds.random_map(volume_change())

        return ds

    def create_val_pipeline():
        ds = create_audio_dataset(
            sources="/data/val",
            shuffle=False,
            repeat=False,
            sample_rate=44100,
            duration=3.0,
        )

        ds = ds.seed(42)

        # Light augmentation for validation
        with argbind.scope(args, "val"):
            ds = ds.random_map(volume_norm())

        return ds

**config.yml:**

.. code-block:: yaml

    # Training: aggressive augmentation
    train/volume_norm.min_db: -30
    train/volume_norm.max_db: -10

    train/volume_change.min_db: -12
    train/volume_change.max_db: 12
    train/volume_change.prob: 0.9

    # Validation: deterministic normalization
    val/volume_norm.min_db: -20
    val/volume_norm.max_db: -20

Transform Order Matters
-----------------------

The order of transforms affects the result:

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, trim

    ds = create_audio_dataset(
        sources="/data/audio",
        sample_rate=44100,
        duration=5.0,
    )
    ds = ds.seed(42)  # both branches below inherit this seed

    # Option 1: Trim then normalize
    ds1 = ds.map(trim(length=3.0))
    ds1 = ds1.random_map(volume_norm(min_db=-20, max_db=-15))

    # Option 2: Normalize then trim
    ds2 = ds.random_map(volume_norm(min_db=-20, max_db=-15))
    ds2 = ds2.map(trim(length=3.0))

    # Results differ!
    # ds1: Loudness computed on 3s segment
    # ds2: Loudness computed on 5s segment, then trimmed to 3s

**Best practice**: Apply volume_norm early in the pipeline before trimming.

Batching in the Pipeline
-------------------------

Add batching as part of the transform chain:

.. code-block:: python

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"]},
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )
    ds = ds.seed(42)

    # Apply augmentations
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Convert to IterDataset and batch
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    # Iterate over batches
    for batch in iter_ds:
        print(batch.waveform.shape)  # (32, 1, 132300)
        break

Mixing Pre-augmented Datasets
------------------------------

Combine datasets with different augmentation strategies:

.. code-block:: python

    from audiotree.sources import create_audio_dataset, create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, volume_change

    # Clean speech dataset
    clean_ds = create_audio_dataset(
        sources="/data/clean_speech",
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )
    clean_ds = clean_ds.seed(42)
    clean_ds = clean_ds.random_map(
        volume_norm(min_db=-20, max_db=-20),  # No variation
    )

    # Noisy speech dataset
    noisy_ds = create_audio_dataset(
        sources="/data/noisy_speech",
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )
    noisy_ds = noisy_ds.seed(42)
    noisy_ds = noisy_ds.random_map(
        volume_norm(min_db=-30, max_db=-10),  # High variation
    )
    noisy_ds = noisy_ds.random_map(
        volume_change(min_db=-6, max_db=6, prob=0.8),
    )

    # Combine with balanced sampling
    mixed_ds = create_balanced_audio_dataset(
        datasets={"clean": clean_ds, "noisy": noisy_ds},
        weights={"clean": 0.3, "noisy": 0.7},
    )

Performance Considerations
--------------------------

**Lazy Evaluation**

Transforms are applied lazily when items are accessed:

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio")
    ds = ds.seed(42)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Nothing computed yet
    print("Dataset created")

    # Transforms applied now
    item = ds[0]
    print("Transform applied")

**Caching**

For an expensive pipeline, pre-compute the augmented data once and export it with
:class:`~audiotree.tree_writer.TreeWriter`, then read it back at training time with
:class:`~audiotree.sources.TreeDataSource`. Each read is a zero-copy memmap slice
with no re-augmentation or decoding cost:

.. code-block:: python

    from audiotree.sources import create_audio_dataset, TreeDataSource
    from audiotree.tree_writer import TreeWriter
    from audiotree.transforms import volume_norm

    # Load and augment (finite and non-repeating, so len(ds) is known up front).
    ds = create_audio_dataset(sources="/data/audio", duration=3.0)
    ds = ds.seed(42)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Pre-compute and write every item to a memory-mapped dataset. TreeWriter
    # pre-allocates ``expected_samples`` rows, so pass the item count up front.
    with TreeWriter("/data/augmented", expected_samples=len(ds)) as writer:
        for item in ds:
            writer.write(item)

    # Later, read the pre-augmented data back — Grain-compatible, no re-augmenting.
    augmented_ds = TreeDataSource("/data/augmented")

To shrink the cache further — e.g. when storing pre-computed spectrograms or other
float features — quantize them to ``int16`` before writing and dequantize in the
loader. See :ref:`quantized-features` for a complete round-trip example.

**Multiprocessing**

Chain transforms before adding multiprocessing:

.. code-block:: python

    import grain
    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, trim

    # Create dataset
    ds = create_audio_dataset(
        sources="/data/audio",
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )
    ds = ds.seed(42)

    # Chain transforms on MapDataset
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))
    ds = ds.map(trim(length=3.0))

    # Add multiprocessing at the end
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

    # Transforms are computed in parallel workers
    for item in iter_ds:
        print(item.waveform.shape)

Complete Training Example
--------------------------

Full pipeline with chained transforms, batching, and multiprocessing:

.. code-block:: python

    import grain
    import jax
    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, volume_change, trim, invert_phase

    # Create balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        weights={"speech": 0.6, "music": 0.4},
        shuffle=True,
        repeat=True,  # Infinite for training
        sample_rate=48000,
        duration=5.0,
    )

    # Chain augmentations
    ds = ds.seed(42)
    ds = ds.random_map(volume_norm(min_db=-25, max_db=-15))
    ds = ds.random_map(volume_change(min_db=-6, max_db=6, prob=0.9))
    ds = ds.random_map(invert_phase(prob=0.5))
    ds = ds.map(trim(length=3.0))

    # Convert to IterDataset, batch, and add multiprocessing
    iter_ds = ds.to_iter_dataset()
    iter_ds = iter_ds.batch(32, batch_fn=AudioTree.batch)

    mp_options = grain.MultiprocessingOptions(
        num_workers=8,
        per_worker_buffer_size=4,
    )
    iter_ds = iter_ds.mp_prefetch(options=mp_options)

    # Training loop
    for step, batch in enumerate(iter_ds):
        if step >= 1000:
            break

        # Convert to JAX and train
        waveform = jax.numpy.array(batch.waveform)
        loss = train_step(waveform)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

Common Patterns
---------------

**Pattern 1: Simple Augmentation**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio").seed(42)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

**Pattern 2: Multi-Stage Augmentation**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio").seed(42)
    ds = ds.random_map(volume_norm(min_db=-25, max_db=-15))
    ds = ds.random_map(volume_change(min_db=-6, max_db=6))
    ds = ds.map(trim(length=3.0))

**Pattern 3: Probabilistic Transforms**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio").seed(42)
    ds = ds.random_map(invert_phase(prob=0.5))  # 50% chance
    ds = ds.random_map(volume_change(min_db=-6, max_db=6, prob=0.8))  # 80% chance

**Pattern 4: Scoped Pipelines**

.. code-block:: python

    ds = ds.seed(42)

    with argbind.scope(args, "train"):
        train_ds = ds.random_map(volume_norm())

    with argbind.scope(args, "val"):
        val_ds = ds.random_map(volume_norm())

Best Practices
--------------

1. **Apply volume_norm early**: Compute loudness before trimming or other operations
2. **Seed once**: Call ``ds.seed(n)`` before your ``.random_map()`` calls instead of passing a ``seed=`` to each — every map derives its own distinct, reproducible seed from it
3. **Chain before batching**: Apply item-level transforms before batching
4. **Add multiprocessing last**: Convert to IterDataset and add mp_prefetch at the end
5. **Use ArgBind for configuration**: Keep transform parameters in YAML files
6. **Consider pre-computing**: For expensive pipelines, pre-compute the data and export it with :class:`~audiotree.tree_writer.TreeWriter`

Next
----

Once your pipeline produces the batches you want, the final Getting-started chapter,
:ref:`writer`, shows how to write them back to disk — as playable audio with
:class:`~audiotree.writer.AudioWriter` or as a fast memory-mapped dataset with
:class:`~audiotree.tree_writer.TreeWriter`.

See Also
--------

- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :ref:`balanced_datasets` - Creating balanced datasets
- `Grain Transformations`_ - Grain's transformation API

.. _Grain Transformations: https://github.com/google/grain/blob/main/docs/transformations.md
