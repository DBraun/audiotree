.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _writer:

Writing Datasets
=================

.. testsetup::

    # Hidden shared setup: a temp working directory and a synthetic input WAV
    # used by the executable examples below. Examples run from inside this dir,
    # so their relative output directories ("output", etc.) land here.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _doc_dir = tempfile.mkdtemp()
    os.chdir(_doc_dir)
    soundfile.write("input.wav", (0.1 * np.random.randn(44_100, 1)).astype(np.float32), 44_100)

AudioTree provides two writers for different use cases:

.. list-table::
   :header-rows: 1
   :widths: 15 42 42

   * -
     - :class:`~audiotree.writer.AudioWriter`
     - :class:`~audiotree.tree_writer.TreeWriter`
   * - **Storage**
     - Individual WAV files + NPZ manifest
     - Memory-mapped binary files + JSON manifest
   * - **Read speed**
     - Decodes audio on each access
     - Zero-copy memmap slice
   * - **Human-readable**
     - Yes (playable audio files)
     - No (raw binary)
   * - **Best for**
     - Exporting audio for sharing, inspection, or external tools
     - Fast random-access datasets for ML training

**Use AudioWriter** when you need playable audio files on disk — for listening, sharing with collaborators,
or feeding into non-Python tools. It writes standard WAV files and tracks per-sample metadata (loudness, tags, etc.)
in an NPZ manifest that supports filtering.

**Use TreeWriter** when you need a fast pre-rendered dataset for training. It stores the entire pytree
(AudioTree, dicts of AudioTrees, nested structures) as memory-mapped arrays — one ``.bin`` file per leaf.
Reading is a memmap slice with no decoding overhead. Use :class:`~audiotree.sources.tree.TreeDataSource`
to read it back as a Grain ``RandomAccessDataSource``.

.. code-block:: python

    from audiotree import TreeWriter
    from audiotree.sources import TreeDataSource

    # Write a dataset
    with TreeWriter("dataset/", expected_samples=10000) as w:
        for batch in dataloader:
            w.write(batch)

    # Read it back (Grain-compatible)
    ds = TreeDataSource("dataset/")
    sample = ds[0]  # reconstructed AudioTree

----

TreeWriter
----------

:class:`~audiotree.tree_writer.TreeWriter` stores each pytree leaf as its own
memory-mapped ``.bin`` file and **preserves the leaf's dtype**. That makes it a
natural home for pre-computed features — a spectrogram, an embedding, a codec's
tokens — cached next to (or instead of) the waveform and read back later as a
zero-copy memmap slice via :class:`~audiotree.sources.TreeDataSource`.

.. _quantized-features:

Quantizing float features to int16
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because TreeWriter keeps each leaf's dtype, you can store a floating-point feature
as ``int16`` to halve its bytes on disk, then dequantize back to ``float32`` in the
data loader. The convention below maps the feature's value range to ``[-1, 1]`` and
then to the full ``int16`` range ``[-32767, 32767]``:

.. testcode::

    import numpy as np
    import jax.numpy as jnp
    import grain
    from audiotree import AudioTree, TreeWriter
    from audiotree.sources import TreeDataSource

    # Stand in for a feature extractor: a batch of magnitude spectrograms of
    # shape (batch, freq, frames) with values in [0, MAG_MAX].
    MAG_MAX = 4.0
    n_items = 6
    rng = np.random.default_rng(0)
    spec = rng.uniform(0.0, MAG_MAX, size=(n_items, 128, 44)).astype(np.float32)

    # 1. Remap the feature range [0, MAG_MAX] to [-1, 1].
    spec_unit = spec / MAG_MAX * 2.0 - 1.0

    # 2. Quantize [-1, 1] to the full int16 range for compact storage.
    spec_i16 = np.round(spec_unit * 32767.0).astype(np.int16)

    # Carry the int16 feature in the AudioTree's metadata (a pytree node), so it
    # batches and indexes alongside the waveform. TreeWriter keeps each leaf's
    # dtype, so the spectrogram is written to disk as int16.
    record = AudioTree.create(
        jnp.zeros((n_items, 1, 16_000)),
        16_000,
        metadata={"spectrogram": spec_i16},
    )
    with TreeWriter("features", expected_samples=n_items) as writer:
        writer.write(record)

    # In the data loader, dequantize metadata["spectrogram"] back to float32.
    def dequantize(audio_tree):
        spec = audio_tree.metadata["spectrogram"].astype(np.float32) / 32767.0
        return audio_tree.replace(
            metadata={**audio_tree.metadata, "spectrogram": spec}
        )

    ds = grain.MapDataset.source(TreeDataSource("features")).map(dequantize)

    item = ds[0]
    print(item.metadata["spectrogram"].dtype)
    print(item.metadata["spectrogram"].shape)   # the batch axis of 1 is added back
    print(bool(np.all(np.abs(item.metadata["spectrogram"]) <= 1.0)))

.. testoutput::

    float32
    (1, 128, 44)
    True

The round-trip is lossy only to ``int16`` precision (about ``1 / 32767``), which is
negligible for most spectrogram and embedding features while cutting storage in
half versus ``float32``.

----

AudioWriter
-----------

The :class:`~audiotree.writer.AudioWriter` class writes AudioTree objects as individual audio files with automatic manifest generation. This is useful for creating datasets, exporting processed audio, and maintaining organized collections of audio files with their associated metadata.

Basic Usage
~~~~~~~~~~~

AudioWriter sequentially writes AudioTree batches to disk, automatically handling file naming and optional manifest generation:

.. testcode::

    from audiotree import AudioTree, AudioWriter
    import numpy as np

    # Create an AudioTree with 3 samples
    audio_tree = AudioTree.create(
        np.random.randn(3, 2, 44_100),  # 3 batches, stereo, 1 second
        sample_rate=44_100
    )

    # Write to disk with automatic manifest
    with AudioWriter("output", pattern="audio_{index:04d}.wav") as writer:
        paths = writer.write(audio_tree)

    print(len(paths))        # One file per batch item
    print(paths[0].name)

.. testoutput::

    3
    audio_0000.wav

Manifest Formats
~~~~~~~~~~~~~~~~

AudioWriter generates manifests in NPZ format to track written files and their metadata:

NPZ Format
^^^^^^^^^^

NPZ is the recommended format for all use cases, especially for large datasets:

.. testcode::

    # NPZ format - best for large datasets
    with AudioWriter("output") as writer:
        writer.write(audio_tree)
    # Creates output/manifest.npz with efficient binary storage

**Advantages of NPZ:**

- **Compression**: Efficient storage with built-in compression
- **Speed**: Direct numpy array loading without parsing
- **Precision**: Maintains exact numeric types without conversion
- **Scalability**: Efficient storage for datasets with thousands of files

NPZ Compression Options
~~~~~~~~~~~~~~~~~~~~~~~

Control NPZ file compression for different trade-offs:

.. testcode::

    # Compressed NPZ (default) - smaller files, slightly slower writing
    writer = AudioWriter("output", compress_manifest=True)

    # Uncompressed NPZ - faster writing, larger files
    writer = AudioWriter("output", compress_manifest=False)

Compression is recommended for most cases as the size savings (often 5-10x) outweigh the minimal performance impact.

Metadata Tracking
~~~~~~~~~~~~~~~~~

AudioWriter automatically tracks all AudioTree metadata in the manifest:

.. testcode::

    # Create AudioTree with comprehensive metadata
    meta_tree = AudioTree.create(
        np.random.randn(2, 1, 44_100),
        sample_rate=44_100,
        lufs=np.array([-20.0, -15.0]),
        pitch=np.array([60.0, 62.0]),
        velocity=np.array([64, 80]),
        note_duration=np.array([1.0, 0.5]),
        filepaths=["original1.wav", "original2.wav"]
    )

    # Write with custom tags
    with AudioWriter("output_meta") as writer:
        writer.write(meta_tree, tags={"dataset": "train", "version": 2})

The manifest will contain:

- **Core info**: filename, sample_rate, channels, samples, duration_seconds
- **AudioTree fields**: loudness, pitch, velocity, duration, filepath
- **Custom tags**: Any additional metadata passed via the ``tags`` parameter

Timestamp Control
~~~~~~~~~~~~~~~~~

Control whether to include timestamps in manifest entries:

.. testcode::

    # Without timestamps (default) - smaller, cleaner manifests
    writer = AudioWriter("output", include_timestamp=False)

    # With timestamps - track when files were written
    writer = AudioWriter("output", include_timestamp=True)

Timestamps are useful for:

- Tracking dataset creation history
- Debugging data pipeline issues
- Audit trails for data processing

Resampling During Write
~~~~~~~~~~~~~~~~~~~~~~~

AudioWriter can automatically resample audio to a target sample rate:

.. testcode::

    # Original at 44.1 kHz
    resample_tree = AudioTree.create(np.zeros((1, 2, 44_100)), 44_100)

    # Resample to 16 kHz when writing
    with AudioWriter("output_16k", sample_rate=16_000) as writer:
        writer.write(resample_tree)
    # Written files will be at 16 kHz

This is useful when:

- Standardizing datasets to a common sample rate
- Reducing file sizes for storage-constrained applications
- Preparing data for models that require specific sample rates

Sequential Writing
~~~~~~~~~~~~~~~~~~

AudioWriter maintains state for sequential writing across multiple batches:

.. testcode::

    writer = AudioWriter("output_seq", pattern="sample_{index:05d}.wav")

    # Write first batch
    tree1 = AudioTree.create(np.random.randn(2, 1, 8000), 8000)
    paths1 = writer.write(tree1)
    print([p.name for p in paths1])

    # Write second batch - indexing continues
    tree2 = AudioTree.create(np.random.randn(3, 1, 8000), 8000)
    paths2 = writer.write(tree2)
    print([p.name for p in paths2])

    # Get statistics
    stats = writer.get_stats()
    print(stats['total_files'])
    print(stats['current_index'])

    # Save manifest when done
    manifest_path = writer.save_manifest()

.. testoutput::

    ['sample_00000.wav', 'sample_00001.wav']
    ['sample_00002.wav', 'sample_00003.wav', 'sample_00004.wav']
    5
    5

Progress Bars
^^^^^^^^^^^^^

AudioWriter supports progress tracking via tqdm integration:

**Using an External Progress Bar:**

.. code-block:: python

    from tqdm import tqdm

    # Create your own progress bar with custom settings
    pbar = tqdm(total=1000, desc="Generating dataset")

    with AudioWriter("output", pbar=pbar) as writer:
        for audio_tree in big_audio_tree.split(batch_size):
            if audio_tree.lufs > -30:  # Only write loud samples
                writer.write(audio_tree)  # Automatically updates pbar

**Using Internal Progress Bar:**

.. code-block:: python

    # Let AudioWriter create its own progress bar
    with AudioWriter("output", show_progress=True,
                     progress_desc="Writing audio") as writer:
        for audio_tree in audio_trees:
            writer.write(audio_tree)
    # Progress bar automatically closed

**Conditional Writing with Progress:**

.. code-block:: python

    from tqdm import tqdm

    # Count total samples that will be written
    loud_trees = [t for t in trees if t.lufs.mean() > -30]
    total_samples = sum(t.waveform.shape[0] for t in loud_trees)

    pbar = tqdm(total=total_samples, desc="Writing loud samples")
    with AudioWriter("output", pbar=pbar, close_pbar=True) as writer:
        for audio_tree in trees:
            if audio_tree.lufs.mean() > -30:
                writer.write(audio_tree)
    # pbar automatically closed when close_pbar=True

The progress bar is updated by the batch size of each AudioTree written, providing accurate progress tracking even with variable batch sizes.

Pattern Formatting
^^^^^^^^^^^^^^^^^^

The ``pattern`` parameter supports Python string formatting:

.. code-block:: python

    # Zero-padded indices
    pattern="audio_{index:04d}.wav"  # audio_0000.wav, audio_0001.wav, ...

    # Different padding
    pattern="sample_{index:06d}.wav"  # sample_000000.wav, sample_000001.wav, ...

    # Custom prefixes
    pattern="train_{index:05d}.wav"  # train_00000.wav, train_00001.wav, ...

Reading Written Data
~~~~~~~~~~~~~~~~~~~~

Use :class:`~audiotree.sources.manifest.ManifestDataSource` to read AudioWriter output:

.. testcode::

    from audiotree.sources import ManifestDataSource

    # Write some data (with per-item loudness so it round-trips into the manifest)
    loudness_tree = AudioTree.create(
        np.random.randn(3, 2, 44_100),
        sample_rate=44_100,
        lufs=np.array([-20.0, -15.0, -18.0]),
    )
    with AudioWriter("output_read") as writer:
        writer.write(loudness_tree, tags={"split": "train"})

    # Read it back
    source = ManifestDataSource.from_writer_output("output_read")

    # Access individual items
    loaded_tree = source[0]
    print(loaded_tree.sample_rate)
    print(loaded_tree.lufs)   # Metadata is restored

    # Filter by metadata
    loud_source = source.filter_by_lufs(min_lufs=-18.0)

    # Filter by tags
    train_source = source.filter_by_tag("split", "train")

.. testoutput::

    44100
    [-20.]

Integration with Data Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AudioWriter integrates seamlessly with data processing pipelines:

.. code-block:: python

    from audiotree.transforms import volume_norm

    # Process and write data
    with AudioWriter("processed_output") as writer:
        for batch in data_loader:
            # Apply transformations
            transform = volume_norm(min_db=-20, max_db=-20)
            normalized = transform.random_map(batch, rng=np.random.default_rng(42))

            # Write processed batch
            writer.write(normalized, tags={"processing": "normalized"})

    # Later, load the processed data
    source = ManifestDataSource.from_writer_output("processed_output")

Metadata Flow Example
~~~~~~~~~~~~~~~~~~~~~

Here's how metadata flows through AudioTree transformations and into the manifest:

.. testcode::

    from audiotree import AudioTree, AudioWriter
    import numpy as np

    # Load with rich metadata
    audio_tree = AudioTree.from_file(
        "input.wav",
        sample_rate=44_100,
        metadata={
            "instrument": "guitar",
            "style": "rock",
            "bpm": 120,
            "key": "A minor"
        }
    )

    # Metadata is preserved through transformations
    processed = audio_tree.resample(16_000)
    processed = processed.replace_lufs()

    # Check metadata is still there
    print(processed.metadata["instrument"])
    print(processed.metadata["bpm"])

    # Write with additional tags
    with AudioWriter("output_flow") as writer:
        writer.write(processed, tags={"processed": True, "version": 2})

.. testoutput::

    guitar
    120

When read back via :class:`~audiotree.sources.manifest.ManifestDataSource`, the per-item
metadata is restored as batched arrays (so a single-item read gives ``array(['guitar'])``
for a string field):

.. testcode::

    from audiotree.sources import ManifestDataSource
    source = ManifestDataSource.from_writer_output("output_flow")
    loaded = source[0]
    print(loaded.metadata["instrument"])

.. testoutput::

    ['guitar']

Best Practices
~~~~~~~~~~~~~~

1. **Use NPZ for large datasets**: The compression benefits become significant with 100+ files
2. **Include relevant metadata**: Track processing parameters, data sources, and versions
3. **Attach metadata early**: Use the `metadata` parameter in `from_file` to attach metadata at load time
4. **Use consistent patterns**: Maintain clear file naming conventions across projects
5. **Leverage context managers**: Ensures manifests are saved even if errors occur
6. **Consider sample rates**: Resample during writing to avoid repeated resampling later

Example: Creating a Training Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of creating a training dataset with AudioWriter:

.. code-block:: python

    from audiotree import AudioWriter
    from audiotree.sources import create_audio_dataset
    from audiotree.transforms import volume_change, shift_phase, choose
    from tqdm import tqdm

    def create_training_dataset(
        source_directory,
        output_dir="precomputed_data",
        augmentations_per_file: int = 3,
    ):
        """Turn one dataset into another with augmentations."""

        # First, get the number of source files
        base_ds = create_audio_dataset(
            sources=source_directory,
            sample_rate=16_000,
            duration=3.0,
            shuffle=False,
            repeat=False,
        )
        num_files = len(base_ds)
        total_records = num_files * augmentations_per_file

        # Create dataset with repeat to generate multiple augmentations per file
        ds = create_audio_dataset(
            sources=source_directory,
            sample_rate=16_000,
            duration=3.0,
            shuffle=True,
            repeat=True,
        )

        # Seed once; each random_map derives its own distinct seed so every
        # augmentation differs.
        ds = ds.seed(42)
        ds = ds.random_map(volume_change(min_db=-6, max_db=6))
        ds = ds.random_map(choose(shift_phase(), prob=0.5))

        pbar = tqdm(total=total_records, desc="Creating dataset")

        with AudioWriter(
            output_dir,
            pattern="train_{index:06d}.wav",
            sample_rate=16_000,
            compress_manifest=True,
            pbar=pbar,
            close_pbar=True
        ) as writer:

            for audio_tree in ds:
                augmented = audio_tree.replace_lufs()
                writer.write(augmented)

        print(f"Created dataset with {writer.get_stats()['total_files']} files")

This creates a fully tracked, augmented dataset ready for training machine learning models.

Next
----

That completes the Getting-started path — you can load audio, augment it, and write
prepared datasets back to disk. From here, the **Going further** guides dig into
:ref:`balanced_datasets`, :ref:`windowed_datasets`, :ref:`dict_batches`, command-line
configuration with :ref:`argbind_guide`, and :ref:`multiprocessing`.
