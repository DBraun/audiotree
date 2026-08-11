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

This chapter uses the NumPy backend. A transform is a plain
`Grain transformation
<https://google-grain.readthedocs.io/en/latest/data_loader/transformations.html>`_,
so an augmentation pipeline is just a chain of ``.map()`` and ``.random_map()``
calls on the dataset.

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
        num_epochs=None,
        sample_rate=44100,
        duration=5.0,
    )
    ds = ds.seed(42)  # apply seed for random_map later

    # Chain transforms
    transform1 = volume_norm(min_db=-20, max_db=-15)
    transform2 = trim(length=3.0)

    ds = ds.random_map(transform1)  # Apply volume_norm
    ds = ds.map(transform2)  # Apply trim

    # Load items - transforms are applied lazily
    audio = ds[0]

Stochastic transforms (``volume_norm``, ``volume_change``, …) go through
``.random_map()``; deterministic ones (``trim``, ``mono``, …) through ``.map()``.
Nothing runs until an item is accessed; transforms are lazy. And call
``ds.seed(n)`` once, before the ``.random_map()`` calls: each one derives its own
distinct, reproducible seed from it, so you never pass a ``seed=`` per map.

A Longer Pipeline
-----------------

The same idea scales to several stages:

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
        num_epochs=None,
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

**Guard against clipping.** Loudness and peak are different dimensions: LUFS
averages K-weighted energy, so audio with a high crest factor — percussion,
speech with pauses, anything peaky over near-silence — measures quiet while
its peaks are already large. Normalizing such an item *up* to even an
ordinary target can push samples past ``[-1, 1]``; left unchecked that clips
when written to 16-bit PCM and can turn into NaNs downstream. Follow the
normalization with ``rescale_audio()``, which peak-limits anything outside
``[-1, 1]`` back into range:

.. testcode::

    import numpy as np
    from audiotree import AudioTree
    from audiotree.transforms import volume_norm, rescale_audio

    # A percussive signal: short decaying bursts over near-silence. It
    # measures -28.9 LUFS with a peak of only 0.47 (a ~22 dB crest factor).
    sr = 16_000
    t = np.zeros(sr, np.float32)
    n = np.arange(400)  # 25 ms burst
    burst = 0.6 * np.sin(2 * np.pi * 200 * n / sr) * np.exp(-n / 80)
    for start in range(0, sr, 4000):  # a hit every 250 ms
        t[start : start + 400] = burst
    drums = AudioTree(t.reshape(1, 1, sr), sr).replace_lufs()
    print(drums.lufs.round(1), np.abs(drums.waveform).max().round(2))

    # Normalizing up to an ordinary -18 LUFS applies ~11 dB of gain, and the
    # peaks overshoot 1.0...
    loud = volume_norm(min_db=-18, max_db=-18).random_map(
        drums, np.random.default_rng(1)
    )
    print(bool(np.abs(loud.waveform).max() > 1.0))

    # ...rescale_audio() peak-limits them back into [-1, 1].
    safe = rescale_audio().map(loud)
    print(bool(np.abs(safe.waveform).max() <= 1.0))

.. testoutput::

    [-28.9] 0.47
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
        num_epochs=None,
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
        num_epochs=None,
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
        num_epochs=None,
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

Complete Training Example
--------------------------

Full pipeline with chained transforms, batching, and multiprocessing:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

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
        num_epochs=None,  # Unbounded stream for training
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

``.batch()`` can sit on either side of ``.mp_prefetch()``. Before it (as
above), each worker assembles whole batches, parallelizing the concatenation
work — but grain shards the upstream pipeline across workers, so *which* items
share a batch then depends on ``num_workers``. Batching after ``mp_prefetch``
keeps batch composition independent of the worker count, at the cost of
concatenating on the main process. The items seen per epoch are identical
either way; see the :ref:`reproducibility` notes in the sources guide.

Next
----

The pipelines here run augmentations on the CPU in data-loader workers. The next
chapter, :ref:`transforms`, shows the JAX backend — the same augmentations applied
on the accelerator inside a ``@jax.jit`` training step.

See Also
--------

- :ref:`writer` - Pre-computing an expensive pipeline once with ``TreeWriter``,
  then training from it with no re-augmentation or decoding cost
- :ref:`argbind_guide` - Configuring transforms with ArgBind
- :ref:`multiprocessing` - Parallel data loading
- :ref:`balanced_datasets` - Creating balanced datasets
- :ref:`codecs` - Tokenizing audio with a neural codec
- `Grain Transformations`_ - Grain's transformation API

.. _Grain Transformations: https://github.com/google/grain/blob/main/docs/transformations.md
