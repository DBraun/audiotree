.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _transform_chaining:

Chaining Transforms with Datasets
==================================

**Transforms** are AudioTree augmentations — volume normalization, gain changes,
phase inversion, trimming, resampling, and more (the full catalog is in the
:mod:`audiotree.transforms` API reference). Each ships in two backends that share
the same names: the NumPy backend used here for CPU Grain pipelines, and a JAX
backend (:ref:`transforms`) for augmenting inside a jitted training step.

This chapter uses the NumPy backend. Transforms integrate seamlessly with
`Grain's dataset chaining API
<https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html>`_,
so you build augmentation pipelines declaratively with ``.map()`` and
``.random_map()``.

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

Configuring from YAML
---------------------

Rather than hard-coding transform parameters, you can set them from a YAML file or
the command line with ArgBind, which keeps experiments reproducible. That has its
own chapter: :ref:`argbind_guide` shows how to configure both single transforms and
a whole dataset pipeline like the ones here.

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

Normalizing Loudness Safely
---------------------------

Two practical points once ``volume_norm`` is in a pipeline.

**Deterministic target.** ``volume_norm`` normalizes each item to a *random* LUFS
in ``[min_db, max_db]``. Set ``min_db == max_db`` to make it deterministic —
normalize every item to exactly that loudness (useful for a validation set, or to
match the LUFS a model was trained at):

.. code-block:: python

    from audiotree.transforms import volume_norm

    # Every item ends up at exactly -18 LUFS, regardless of the RNG.
    ds = ds.random_map(volume_norm(min_db=-18, max_db=-18))

**Guard against clipping.** Normalizing a quiet excerpt *up* to a loud target
multiplies it by a large gain, which can push samples past ``[-1, 1]``. Left
unchecked that clips when written to 16-bit PCM and can turn into NaNs downstream.
Follow the normalization with ``rescale_audio()``, which peak-limits anything
outside ``[-1, 1]`` back into range:

.. testcode::

    import numpy as np
    from audiotree import AudioTree
    from audiotree.transforms import volume_norm, rescale_audio

    # A quiet signal normalized up to -6 LUFS overshoots 1.0...
    quiet = AudioTree(
        (0.05 * np.random.default_rng(0).standard_normal((4, 1, 16_000))).astype(np.float32),
        16_000,
    ).replace_lufs()
    loud = volume_norm(min_db=-6, max_db=-6).random_map(quiet, np.random.default_rng(1))
    print(bool(np.abs(loud.waveform).max() > 1.0))

    # ...rescale_audio() peak-limits it back into [-1, 1].
    safe = rescale_audio().map(loud)
    print(bool(np.abs(safe.waveform).max() <= 1.0))

.. testoutput::

    True
    True

Unlike :func:`~audiotree.transforms.peak_norm` (which always scales the peak to
exactly ``1.0``), ``rescale_audio`` only scales *down* items that exceed the range
and leaves quieter items untouched — so it corrects overshoot without otherwise
changing the loudness you just set.

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

To shrink the cache further — e.g., when storing pre-computed spectrograms or other
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

Best Practices
--------------

1. **Apply volume_norm early**: Compute lufs before trimming or other operations
2. **Seed once**: Call ``ds.seed(n)`` before your ``.random_map()`` calls instead of passing a ``seed=`` to each — every map derives its own distinct, reproducible seed from it
3. **Chain before batching**: Apply item-level transforms before batching
4. **Add multiprocessing last**: Convert to IterDataset and add mp_prefetch at the end
5. **Configure from YAML**: Keep transform parameters in config files with ArgBind (see :ref:`argbind_guide`)
6. **Consider pre-computing**: For expensive pipelines, pre-compute the data and export it with :class:`~audiotree.tree_writer.TreeWriter`

Next
----

The pipelines here run augmentations on the CPU in data-loader workers. The next
chapter, :ref:`transforms`, shows the JAX backend — the same augmentations applied
on the accelerator inside a ``@jax.jit`` training step.

See Also
--------

- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :ref:`balanced_datasets` - Creating balanced datasets
- `Grain Transformations`_ - Grain's transformation API

.. _Grain Transformations: https://github.com/google/grain/blob/main/docs/transformations.md
