.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _multiprocessing:

Multiprocessing and Multithreading
===================================

AudioTree datasets work seamlessly with Grain's multiprocessing (``mp_prefetch``) and multithreading (``ReadOptions``) for parallel data loading, which can significantly speed up training.

.. note::

   **Multiprocessing is Linux and macOS only.** Grain's workers return results
   through named shared memory, and Windows destroys a named mapping as soon as
   its last handle closes -- so the parent fails to attach with
   ``FileNotFoundError: [WinError 2] ... 'wnsm_<id>'`` once the worker exits.
   This is `google/grain#793 <https://github.com/google/grain/issues/793>`_,
   still open upstream, and not something AudioTree can work around.

   Multithreading via ``ReadOptions(num_threads=...)`` is unaffected -- threads
   share one address space and need no shared memory. On Windows, raise
   ``num_threads`` instead of adding workers, or run under WSL.

Why Parallel Loading?
----------------------

Audio data loading can be CPU-bound due to:

- **File I/O**: Reading audio files from disk
- **Decoding**: Decompressing FLAC, MP3, etc.
- **Resampling**: Converting sample rates
- **Loudness Computation**: Calculating LUFS values

Parallelization allows these operations to run concurrently while the GPU trains your model.

Synchronous reads are the right baseline
----------------------------------------

Grain's *default* ``ReadOptions`` is ``num_threads=16, prefetch_buffer_size=500``
-- sized for corpora of small records, not decoded audio. A 3-second stereo
excerpt at 44.1 kHz float32 is ~1 MB, so the default buffer alone can hold
~0.5 GB of decoded audio, and it applies **per worker process** once
``mp_prefetch`` is involved (8 workers -> ~4 GB of prefetched waveforms). The
read threads also compete with the training process for the GIL.

Start every pipeline with fully synchronous reads, and add parallelism only
where measurement says it pays:

.. skip-snippet-exec: fragment; ``ds`` is any MapDataset from this page.

.. code-block:: python

    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = ds.to_iter_dataset(read_options=read_options)

* **Training with** ``mp_prefetch``: keep reads synchronous inside each worker
  -- the worker processes *are* the parallelism, and per-worker thread pools
  mostly add memory and GIL churn.
* **Evaluation and pre-rendering scripts**: synchronous reads keep memory flat
  and behavior deterministic.
* **Genuinely I/O-bound reads** (network storage, cold disks) are the case for
  ``num_threads > 0`` -- the next section -- and even then prefer a small
  explicit ``prefetch_buffer_size`` over the default 500.

Multithreading with ReadOptions
--------------------------------

Use multithreading for I/O-bound operations:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree.sources import create_balanced_audio_dataset

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        sample_rate=44100,
        duration=3.0,
    )

    # Convert to IterDataset with multithreading
    read_options = grain.ReadOptions(
        num_threads=4,              # 4 threads reading in parallel
        prefetch_buffer_size=16,    # Buffer 16 items per thread
    )
    iter_ds = ds.to_iter_dataset(read_options=read_options)

    # Iterate with parallel loading
    for audio_tree in iter_ds:
        # Process audio_tree
        pass

**When to use**:

- High I/O wait (reading from network storage, slow disks)
- Python GIL is not a bottleneck (e.g., mostly C extensions for decoding)

Multiprocessing with mp_prefetch
---------------------------------

Use multiprocessing for CPU-bound operations:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree.sources import create_balanced_audio_dataset

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        sample_rate=44100,
        duration=3.0,
    )

    # Convert to IterDataset and add multiprocessing
    mp_options = grain.MultiprocessingOptions(
        num_workers=8,              # 8 worker processes
        per_worker_buffer_size=4,   # Each worker buffers 4 items
    )
    # Synchronous reads inside each worker: the processes are the parallelism.
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = ds.to_iter_dataset(read_options=read_options).mp_prefetch(
        options=mp_options
    )

    # Iterate with parallel loading
    for audio_tree in iter_ds:
        # Process audio_tree
        pass

**When to use**:

- CPU-intensive operations (resampling, loudness computation, augmentations)
- Python GIL is a bottleneck
- Large batch sizes that benefit from parallel processing

Combining Both
--------------

For maximum throughput, use both multithreading and multiprocessing:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree.sources import create_balanced_audio_dataset

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        sample_rate=44100,
        duration=3.0,
    )

    # Add multithreading for I/O
    read_options = grain.ReadOptions(
        num_threads=2,
        prefetch_buffer_size=4,
    )
    iter_ds = ds.to_iter_dataset(read_options=read_options)

    # Add multiprocessing for CPU-bound work
    mp_options = grain.MultiprocessingOptions(
        num_workers=4,
        per_worker_buffer_size=2,
    )
    iter_ds = iter_ds.mp_prefetch(options=mp_options)

    # Total threads: num_workers * num_threads = 4 * 2 = 8 threads
    for audio_tree in iter_ds:
        pass

**Note**: This creates ``num_workers × num_threads`` total threads (8 in this example).

With Balanced Datasets
-----------------------

Weight-based balancing is preserved with multiprocessing:

.. skip-snippet-exec: grain's multiprocess prefetch needs a real ``__main__`` to spawn workers from.

.. code-block:: python

    import grain
    from audiotree.sources import create_balanced_audio_dataset

    # Create balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/data/speech"],
            "music": ["/data/music"],
        },
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=3.0,
    )

    # Add multiprocessing
    mp_options = grain.MultiprocessingOptions(num_workers=8)
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = ds.to_iter_dataset(read_options=read_options).mp_prefetch(
        options=mp_options
    )

    # Verify proportions are maintained
    from collections import Counter
    sources = [item.source[0] for item in iter_ds]
    print(Counter(sources))
    # {'speech': ~7000, 'music': ~3000}

Performance Tuning
------------------

**Buffer Sizes**

Larger buffers = more memory usage but potentially higher throughput:

.. skip-snippet-exec: fragment; ``grain`` comes from the block above, which opts out.

.. code-block:: python

    # Conservative (low memory)
    mp_options = grain.MultiprocessingOptions(
        num_workers=4,
        per_worker_buffer_size=1,
    )

    # Aggressive (high throughput)
    mp_options = grain.MultiprocessingOptions(
        num_workers=8,
        per_worker_buffer_size=8,
    )

**Worker Count**

General guidelines:

- **CPU cores**: Start with ``num_workers = num_cpu_cores``
- **Memory**: Reduce if running out of memory
- **I/O bound**: Multithreading may be more efficient
- **CPU bound**: Multiprocessing usually better

.. skip-snippet-exec: fragment; ``grain`` comes from the block above, which opts out.

.. code-block:: python

    import os

    # Use all CPU cores
    num_workers = os.cpu_count()
    mp_options = grain.MultiprocessingOptions(num_workers=num_workers)

**Thread Count**

``0`` (synchronous) is the baseline -- see above. For I/O-bound workloads:

.. skip-snippet-exec: fragment; ``grain`` comes from the block above, which opts out.

.. code-block:: python

    # Conservative
    read_options = grain.ReadOptions(num_threads=2)

    # Aggressive (for network storage)
    read_options = grain.ReadOptions(num_threads=8)

Batching
--------

Use ``AudioTree.batch`` with ``IterDataset.batch()`` to batch AudioTree objects.
This concatenates along axis 0 (the batch dimension) rather than stacking, which would
add an extra dimension.

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    from audiotree import AudioTree
    from audiotree.sources import create_audio_dataset

    ds = create_audio_dataset("/data/audio", duration=1.0)

    # Batch with AudioTree.batch
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    for batch in iter_ds:
        print(batch.waveform.shape)  # (32, channels, samples)

``batch`` also handles dict structures containing AudioTrees:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    # If your dataset yields {"src": AudioTree, "tgt": AudioTree}
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    for batch in iter_ds:
        # batch is a dict with batched AudioTrees
        print(batch["src"].waveform.shape)  # (32, channels, samples)

See :ref:`dict_batches` for more details on working with dict structures.

Common Patterns
---------------

**Training Loop with Batching and Multiprocessing**

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    import jax
    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset

    # Create dataset
    ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/speech"], "music": ["/data/music"]},
        shuffle=True,
        num_epochs=None,  # Unbounded stream for training
        sample_rate=44100,
        duration=3.0,
    )

    # Add batching and multiprocessing
    mp_options = grain.MultiprocessingOptions(
        num_workers=8,
        per_worker_buffer_size=4,
    )
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = (
        ds.to_iter_dataset(read_options=read_options)
        .batch(32, batch_fn=AudioTree.batch)
        .mp_prefetch(options=mp_options)
    )

    # Training loop
    for step, batch in enumerate(iter_ds):
        if step >= max_steps:
            break

        # Convert to JAX array and train
        waveform = jnp.array(batch.waveform)  # (32, channels, samples)
        loss = train_step(waveform)

**Validation with Deterministic Order**

.. skip-snippet-exec: fragment; the training loop it feeds is elided.

.. code-block:: python

    # Create validation dataset (no shuffle, exactly one epoch)
    val_ds = create_balanced_audio_dataset(
        sources={"speech": ["/data/val_speech"], "music": ["/data/val_music"]},
        shuffle=False,  # Deterministic order
        num_epochs=1,
        excerpt_seed=42,  # Fixes which excerpt is drawn from each file
        sample_rate=44100,
        duration=3.0,
    )

    # Add batching and multiprocessing
    mp_options = grain.MultiprocessingOptions(num_workers=4)
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    val_iter_ds = (
        val_ds.to_iter_dataset(read_options=read_options)
        .batch(32, batch_fn=AudioTree.batch)
        .mp_prefetch(options=mp_options)
    )

    # Evaluate
    for batch in val_iter_ds:
        metrics = evaluate(batch)

Worker Initialization
---------------------

Perform setup in each worker process before loading data:

.. skip-snippet-exec: fragment; ``grain`` comes from the block above, which opts out.

.. code-block:: python

    def worker_init_fn(worker_id, worker_count):
        """Called once per worker before processing data."""
        print(f"Worker {worker_id}/{worker_count} initialized")

        # Example: Set different random seed per worker
        import numpy as np
        np.random.seed(42 + worker_id)

    mp_options = grain.MultiprocessingOptions(num_workers=4)
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = ds.to_iter_dataset(read_options=read_options).mp_prefetch(
        options=mp_options,
        worker_init_fn=worker_init_fn,
    )

Profiling
---------

Enable profiling to identify bottlenecks:

.. skip-snippet-exec: fragment; ``grain`` comes from the block above, which opts out.

.. code-block:: python

    mp_options = grain.MultiprocessingOptions(
        num_workers=8,
        per_worker_buffer_size=4,
        enable_profiling=True,  # Enable profiling
    )

Logs will show timing information for each stage of the pipeline.

Memory Considerations
---------------------

**Audio data can be large**:

- 3 seconds of 48kHz stereo audio = ~576KB uncompressed
- Batch of 32 = ~18MB
- 8 workers with buffer_size=4 = ~576MB in buffers

**Recommendations**:

1. **Monitor memory usage** during training
2. **Reduce buffer sizes** if memory-constrained
3. **Use shorter durations** for initial testing
4. **Stream from disk** rather than loading entire datasets into RAM

Debugging
---------

**Issue**: Workers hang or crash

**Solution**: Reduce ``num_workers`` and ``per_worker_buffer_size``. Check for memory issues.

**Issue**: No speedup with multiprocessing

**Solution**: Your bottleneck may be elsewhere (GPU, network I/O). Profile to identify actual bottleneck.

**Issue**: Different results with different worker counts

**Solution**: This is expected with stateful operations like ``filter``. Keep transformations before ``to_iter_dataset`` deterministic.

Best Practices
--------------

1. **Start conservative**: Begin with ``num_workers=2`` and ``per_worker_buffer_size=2``
2. **Profile first**: Measure actual bottlenecks before adding parallelization
3. **Monitor resources**: Watch CPU, memory, and GPU utilization
4. **Test with subset**: Verify correctness with small dataset before scaling up
5. **Use multiprocessing for CPU-bound**: Use multithreading for I/O-bound
6. **Combine carefully**: ``num_workers × num_threads`` can create many threads

Performance Checklist
---------------------

If data loading is slow, check:

- ☐ Using ``mp_prefetch`` with appropriate worker count?
- ☐ Buffer sizes large enough to keep GPU fed?
- ☐ Using multithreading for I/O-bound operations?
- ☐ Files on fast storage (SSD, not network drive)?
- ☐ Saliency disabled if not needed? (adds computation)
- ☐ Profiling enabled to identify actual bottleneck?

Example: Full Pipeline
----------------------

Complete example with batching and all optimizations:

.. skip-snippet-exec: iterates a repeated dataset, which is infinite by construction.

.. code-block:: python

    import grain
    import jax
    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.sources import ExcerptConfig

    # Saliency for loud sections
    excerpt = ExcerptConfig(
        strategy="loudest",
        lufs_cutoff=-40,
        num_tries=5,
    )

    # Create balanced dataset
    ds = create_balanced_audio_dataset(
        sources={
            "speech": ["/fast_storage/speech"],
            "music": ["/fast_storage/music"],
            "effects": ["/fast_storage/effects"],
        },
        weights={"speech": 0.5, "music": 0.3, "effects": 0.2},
        shuffle=True,
        num_epochs=None,
        shuffle_seed=42,  # excerpt_seed defaults to this
        sample_rate=48000,
        duration=3.0,
        excerpt=excerpt,
    )

    # Synchronous reads: the mp_prefetch workers below are the parallelism.
    # (Raise num_threads only for genuinely I/O-bound storage.)
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    iter_ds = ds.to_iter_dataset(read_options=read_options)

    # Batch with AudioTree.batch
    iter_ds = iter_ds.batch(32, batch_fn=AudioTree.batch)

    # Multiprocessing for CPU-bound work
    mp_options = grain.MultiprocessingOptions(
        num_workers=8,
        per_worker_buffer_size=4,
        enable_profiling=False,  # Disable in production
    )
    iter_ds = iter_ds.mp_prefetch(options=mp_options)

    # Training loop
    for step, batch in enumerate(iter_ds):
        if step >= 100000:
            break

        # Your training code here
        waveform = jnp.array(batch.waveform)  # (32, channels, samples)
        loss = train_step(waveform)

        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss}")

See Also
--------

- :ref:`transform_chaining` - Chaining transforms with datasets
- `Grain Documentation`_ - Complete Grain pipeline guide
- :ref:`balanced_datasets` - Creating balanced datasets
- :func:`~audiotree.sources.create_audio_dataset` - Simple dataset creation
- :func:`~audiotree.sources.create_balanced_audio_dataset` - Balanced dataset creation

.. _Grain Documentation: https://github.com/google/grain
