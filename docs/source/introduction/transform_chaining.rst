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
        num_records=1000,
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )

    # Chain transforms
    transform1 = volume_norm(min_db=-20, max_db=-15)
    transform2 = trim(length=3.0)

    ds = ds.random_map(transform1, seed=42)  # Apply volume_norm
    ds = ds.map(transform2)                  # Apply trim

    # Load items - transforms are applied lazily
    audio_tree = ds[0]

**Key Points:**

- ``.random_map()`` for stochastic transforms (volume_norm, volume_change, etc.)
- ``.map()`` for deterministic transforms (trim, mono, etc.)
- Transforms are applied **lazily** when items are accessed
- Each ``.random_map()`` needs a seed for reproducibility

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
        num_records=10000,
        weights={"speech": 0.7, "music": 0.3},
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )

    # Build augmentation pipeline
    base_seed = 42

    # 1. Normalize volume
    ds = ds.random_map(
        volume_norm(min_db=-25, max_db=-15),
        seed=base_seed,
    )

    # 2. Random volume change (90% probability)
    ds = ds.random_map(
        volume_change(min_db=-6, max_db=6, prob=0.9),
        seed=base_seed + 1,
    )

    # 3. Random phase inversion (50% probability)
    ds = ds.random_map(
        invert_phase(prob=0.5),
        seed=base_seed + 2,
    )

    # 4. Trim to final length
    ds = ds.map(trim(length=3.0))

    # 5. Convert to mono
    ds = ds.map(mono())

    # Transforms applied when accessing items
    for item in ds.to_iter_dataset():
        # item has been through full augmentation pipeline
        print(item.audio_data.shape)  # (1, 1, 132300) for 3s mono at 44.1kHz
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
    VolumeNorm = argbind.bind(transforms.volume_norm)
    VolumeChange = argbind.bind(transforms.volume_change)
    InvertPhase = argbind.bind(transforms.invert_phase)
    Trim = argbind.bind(transforms.trim)

    def create_pipeline():
        # Create dataset
        ds = create_balanced_audio_dataset(
            sources={
                "speech": ["/data/speech"],
                "music": ["/data/music"],
            },
            num_records=10000,
            shuffle=True,
            repeat=True,
            sample_rate=44100,
            duration=5.0,
        )

        # Chain transforms (configured via ArgBind)
        ds = ds.random_map(VolumeNorm(), seed=42)
        ds = ds.random_map(VolumeChange(), seed=43)
        ds = ds.random_map(InvertPhase(), seed=44)
        ds = ds.map(Trim())

        return ds

    if __name__ == "__main__":
        args = argbind.parse_args()
        with argbind.scope(args):
            ds = create_pipeline()

            # Iterate over augmented data
            for item in ds.to_iter_dataset():
                print(item.source, item.loudness)
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

    VolumeNorm = argbind.bind(transforms.volume_norm, "train", "val")
    VolumeChange = argbind.bind(transforms.volume_change, "train", "val")

    def create_train_pipeline():
        ds = create_audio_dataset(
            sources="/data/train",
            shuffle=True,
            repeat=True,
            sample_rate=44100,
            duration=3.0,
        )

        # Aggressive augmentation for training
        with argbind.scope(args, "train"):
            ds = ds.random_map(VolumeNorm(), seed=42)
            ds = ds.random_map(VolumeChange(), seed=43)

        return ds

    def create_val_pipeline():
        ds = create_audio_dataset(
            sources="/data/val",
            shuffle=False,
            repeat=False,
            sample_rate=44100,
            duration=3.0,
        )

        # Light augmentation for validation
        with argbind.scope(args, "val"):
            ds = ds.random_map(VolumeNorm(), seed=42)

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
        num_records=100,
        sample_rate=44100,
        duration=5.0,
    )

    # Option 1: Trim then normalize
    ds1 = ds.map(trim(length=3.0))
    ds1 = ds1.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)

    # Option 2: Normalize then trim
    ds2 = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)
    ds2 = ds2.map(trim(length=3.0))

    # Results differ!
    # ds1: Loudness computed on 3s segment
    # ds2: Loudness computed on 5s segment, then trimmed to 3s

**Best practice**: Apply volume_norm early in the pipeline before trimming.

Batching in the Pipeline
-------------------------

Add batching as part of the transform chain:

.. code-block:: python

    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, Batch

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"]},
        num_records=10000,
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )

    # Apply augmentations
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)

    # Batch into groups of 32
    batch_op = Batch(batch_size=32)

    # For IterDataset, use .batch()
    iter_ds = ds.to_iter_dataset().batch(32)

    # Or for MapDataset with custom batching logic
    ds_batched = ds.map(lambda items: batch_op._default_batch_fn(items))

    # Iterate over batches
    for batch in iter_ds:
        print(batch.audio_data.shape)  # (32, 1, 132300)
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
    clean_ds = clean_ds.random_map(
        volume_norm(min_db=-20, max_db=-20),  # No variation
        seed=42,
    )

    # Noisy speech dataset
    noisy_ds = create_audio_dataset(
        sources="/data/noisy_speech",
        repeat=True,
        sample_rate=44100,
        duration=3.0,
    )
    noisy_ds = noisy_ds.random_map(
        volume_norm(min_db=-30, max_db=-10),  # High variation
        seed=42,
    )
    noisy_ds = noisy_ds.random_map(
        volume_change(min_db=-6, max_db=6, prob=0.8),
        seed=43,
    )

    # Combine with balanced sampling
    mixed_ds = create_balanced_audio_dataset(
        datasets={"clean": clean_ds, "noisy": noisy_ds},
        num_records=10000,
        weights={"clean": 0.3, "noisy": 0.7},
    )

Performance Considerations
--------------------------

**Lazy Evaluation**

Transforms are applied lazily when items are accessed:

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio", num_records=1000)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)

    # Nothing computed yet
    print("Dataset created")

    # Transforms applied now
    item = ds[0]
    print("Transform applied")

**Caching**

For expensive transforms, consider pre-computing and using manifest datasets:

.. code-block:: python

    from audiotree.sources import create_audio_dataset
    from audiotree.writer import AudioWriter
    from audiotree.transforms import volume_norm

    # Load and augment
    ds = create_audio_dataset(sources="/data/audio", num_records=1000)
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)

    # Pre-compute and write to disk
    writer = AudioWriter(output_dir="/data/augmented", format="npz")
    for item in ds.to_iter_dataset():
        writer.write(item)
    manifest_path = writer.finalize()

    # Later, load pre-augmented data
    from audiotree.sources import ManifestDataSource
    augmented_ds = ManifestDataSource(manifest_path)

**Multiprocessing**

Chain transforms before adding multiprocessing:

.. code-block:: python

    import grain
    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_norm, trim

    # Create dataset
    ds = create_audio_dataset(
        sources="/data/audio",
        num_records=10000,
        shuffle=True,
        repeat=True,
        sample_rate=44100,
        duration=5.0,
    )

    # Chain transforms on MapDataset
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)
    ds = ds.map(trim(length=3.0))

    # Add multiprocessing at the end
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    iter_ds = ds.to_iter_dataset().mp_prefetch(options=mp_options)

    # Transforms are computed in parallel workers
    for item in iter_ds:
        print(item.audio_data.shape)

Complete Training Example
--------------------------

Full pipeline with chained transforms, batching, and multiprocessing:

.. code-block:: python

    import grain
    import jax
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, volume_change, trim, invert_phase

    # Create balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        num_records=100000,
        weights={"speech": 0.6, "music": 0.4},
        shuffle=True,
        repeat=True,  # Infinite for training
        sample_rate=48000,
        duration=5.0,
    )

    # Chain augmentations
    base_seed = 42
    ds = ds.random_map(
        volume_norm(min_db=-25, max_db=-15),
        seed=base_seed,
    )
    ds = ds.random_map(
        volume_change(min_db=-6, max_db=6, prob=0.9),
        seed=base_seed + 1,
    )
    ds = ds.random_map(
        invert_phase(prob=0.5),
        seed=base_seed + 2,
    )
    ds = ds.map(trim(length=3.0))

    # Convert to IterDataset, batch, and add multiprocessing
    iter_ds = ds.to_iter_dataset()
    iter_ds = iter_ds.batch(32)

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
        audio_data = jax.numpy.array(batch.audio_data)
        loss = train_step(audio_data)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

Common Patterns
---------------

**Pattern 1: Simple Augmentation**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio")
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15), seed=42)

**Pattern 2: Multi-Stage Augmentation**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio")
    ds = ds.random_map(volume_norm(min_db=-25, max_db=-15), seed=42)
    ds = ds.random_map(volume_change(min_db=-6, max_db=6), seed=43)
    ds = ds.map(trim(length=3.0))

**Pattern 3: Probabilistic Transforms**

.. code-block:: python

    ds = create_audio_dataset(sources="/data/audio")
    ds = ds.random_map(invert_phase(prob=0.5), seed=42)  # 50% chance
    ds = ds.random_map(volume_change(min_db=-6, max_db=6, prob=0.8), seed=43)  # 80% chance

**Pattern 4: Scoped Pipelines**

.. code-block:: python

    with argbind.scope(args, "train"):
        train_ds = ds.random_map(volume_norm(), seed=42)

    with argbind.scope(args, "val"):
        val_ds = ds.random_map(volume_norm(), seed=42)

Best Practices
--------------

1. **Apply volume_norm early**: Compute loudness before trimming or other operations
2. **Use consistent seeds**: Each ``.random_map()`` needs a unique seed
3. **Chain before batching**: Apply item-level transforms before batching
4. **Add multiprocessing last**: Convert to IterDataset and add mp_prefetch at the end
5. **Use ArgBind for configuration**: Keep transform parameters in YAML files
6. **Consider pre-computing**: For expensive pipelines, pre-compute and save to manifest

See Also
--------

- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :ref:`balanced_datasets` - Creating balanced datasets
- `Grain Transformations`_ - Grain's transformation API

.. _Grain Transformations: https://github.com/google/grain/blob/main/docs/transformations.md
