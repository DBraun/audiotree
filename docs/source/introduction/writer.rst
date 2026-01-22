.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _writer:

Writing Audio and Manifests
============================

The :class:`~audiotree.writer.AudioWriter` class provides a powerful way to write AudioTree objects to disk with automatic manifest generation for tracking metadata. This is particularly useful for creating datasets, exporting processed audio, and maintaining organized collections of audio files with their associated metadata.

Basic Usage
-----------

AudioWriter sequentially writes AudioTree batches to disk, automatically handling file naming and optional manifest generation:

.. code-block:: python

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

    >>> len(paths)
    3  # One file per batch item
    >>> paths[0].name
    'audio_0000.wav'

Manifest Formats
----------------

AudioWriter generates manifests in NPZ format to track written files and their metadata:

NPZ Format
~~~~~~~~~~

NPZ is the recommended format for all use cases, especially for large datasets:

.. code-block:: python

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
-----------------------

Control NPZ file compression for different trade-offs:

.. code-block:: python

    # Compressed NPZ (default) - smaller files, slightly slower writing
    writer = AudioWriter("output", compress_manifest=True)

    # Uncompressed NPZ - faster writing, larger files
    writer = AudioWriter("output", compress_manifest=False)

Compression is recommended for most cases as the size savings (often 5-10x) outweigh the minimal performance impact.

Metadata Tracking
-----------------

AudioWriter automatically tracks all AudioTree metadata in the manifest:

.. code-block:: python

    # Create AudioTree with comprehensive metadata
    audio_tree = AudioTree.create(
        np.random.randn(2, 1, 44_100),
        sample_rate=44_100,
        loudness=np.array([-20.0, -15.0]),
        pitch=np.array([60.0, 62.0]),
        velocity=np.array([64, 80]),
        duration=np.array([1.0, 0.5]),
        filepaths=["original1.wav", "original2.wav"]
    )

    # Write with custom tags
    with AudioWriter("output") as writer:
        writer.write(audio_tree, tags={"dataset": "train", "version": 2})

The manifest will contain:

- **Core info**: filename, sample_rate, channels, samples, duration_seconds
- **AudioTree fields**: loudness, pitch, velocity, duration, filepath
- **Custom tags**: Any additional metadata passed via the ``tags`` parameter

Timestamp Control
-----------------

Control whether to include timestamps in manifest entries:

.. code-block:: python

    # Without timestamps (default) - smaller, cleaner manifests
    writer = AudioWriter("output", include_timestamp=False)

    # With timestamps - track when files were written
    writer = AudioWriter("output", include_timestamp=True)

Timestamps are useful for:

- Tracking dataset creation history
- Debugging data pipeline issues
- Audit trails for data processing

Resampling During Write
-----------------------

AudioWriter can automatically resample audio to a target sample rate:

.. code-block:: python

    # Original at 44.1 kHz
    audio_tree = AudioTree.create(np.zeros((1, 2, 44_100)), 44_100)

    # Resample to 16 kHz when writing
    with AudioWriter("output", sample_rate=16_000) as writer:
        writer.write(audio_tree)
    # Written files will be at 16 kHz

This is useful when:

- Standardizing datasets to a common sample rate
- Reducing file sizes for storage-constrained applications
- Preparing data for models that require specific sample rates

Sequential Writing
------------------

AudioWriter maintains state for sequential writing across multiple batches:

.. code-block:: python

    writer = AudioWriter("output", pattern="sample_{index:05d}.wav")

    # Write first batch
    tree1 = AudioTree.create(np.random.randn(2, 1, 8000), 8000)
    paths1 = writer.write(tree1)
    >>> [p.name for p in paths1]
    ['sample_00000.wav', 'sample_00001.wav']

    # Write second batch - indexing continues
    tree2 = AudioTree.create(np.random.randn(3, 1, 8000), 8000)
    paths2 = writer.write(tree2)
    >>> [p.name for p in paths2]
    ['sample_00002.wav', 'sample_00003.wav', 'sample_00004.wav']

    # Get statistics
    stats = writer.get_stats()
    >>> stats['total_files']
    5
    >>> stats['current_index']
    5

    # Save manifest when done
    manifest_path = writer.save_manifest()

Progress Bars
~~~~~~~~~~~~~

AudioWriter supports progress tracking via tqdm integration:

**Using an External Progress Bar:**

.. code-block:: python

    from tqdm import tqdm

    # Create your own progress bar with custom settings
    pbar = tqdm(total=1000, desc="Generating dataset")

    with AudioWriter("output", pbar=pbar) as writer:
        for audio_tree in big_audio_tree.mini_batch_list(batch_size):
            if audio_tree.loudness > -30:  # Only write loud samples
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
    loud_trees = [t for t in trees if t.loudness.mean() > -30]
    total_samples = sum(t.audio_data.shape[0] for t in loud_trees)

    pbar = tqdm(total=total_samples, desc="Writing loud samples")
    with AudioWriter("output", pbar=pbar, close_pbar=True) as writer:
        for audio_tree in trees:
            if audio_tree.loudness.mean() > -30:
                writer.write(audio_tree)
    # pbar automatically closed when close_pbar=True

The progress bar is updated by the batch size of each AudioTree written, providing accurate progress tracking even with variable batch sizes.

Pattern Formatting
~~~~~~~~~~~~~~~~~~

The ``pattern`` parameter supports Python string formatting:

.. code-block:: python

    # Zero-padded indices
    pattern="audio_{index:04d}.wav"  # audio_0000.wav, audio_0001.wav, ...

    # Different padding
    pattern="sample_{index:06d}.wav"  # sample_000000.wav, sample_000001.wav, ...

    # Custom prefixes
    pattern="train_{index:05d}.wav"  # train_00000.wav, train_00001.wav, ...

Reading Written Data
--------------------

Use :class:`~audiotree.sources.manifest.ManifestDataSource` to read AudioWriter output:

.. code-block:: python

    from audiotree.sources import ManifestDataSource

    # Write some data
    with AudioWriter("output") as writer:
        writer.write(audio_tree, tags={"split": "train"})

    # Read it back
    source = ManifestDataSource.from_writer_output("output")

    # Access individual items
    loaded_tree = source[0]
    >>> loaded_tree.sample_rate
    44100
    >>> loaded_tree.loudness  # Metadata is restored
    array([-20.])

    # Filter by metadata
    loud_source = source.filter_by_loudness(min_lufs=-18.0)

    # Filter by tags
    train_source = source.filter_by_tag("split", "train")

Integration with Data Pipelines
-------------------------------

AudioWriter integrates seamlessly with data processing pipelines:

.. code-block:: python

    from audiotree.transforms import VolumeNorm

    # Process and write data
    with AudioWriter("processed_output") as writer:
        for batch in data_loader:
            # Apply transformations
            normalized = VolumeNorm(config={"target_db": -20}).random_map(batch)

            # Write processed batch
            writer.write(normalized, tags={"processing": "normalized"})

    # Later, load the processed data
    source = ManifestDataSource.from_writer_output("processed_output")

Metadata Flow Example
---------------------

Here's how metadata flows through AudioTree transformations and into the manifest:

.. code-block:: python

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
    processed = processed.replace_loudness()

    # Check metadata is still there
    >>> processed.metadata["instrument"]
    'guitar'
    >>> processed.metadata["bpm"]
    120

    # Write with additional tags
    with AudioWriter("output") as writer:
        writer.write(processed, tags={"processed": True, "version": 2})

    # When read back, all metadata is available
    from audiotree.sources import ManifestDataSource
    source = ManifestDataSource.from_writer_output("output")
    loaded = source[0]

    # Original metadata is preserved
    >>> loaded.metadata["instrument"]  # From from_file
    'guitar'
    >>> loaded.metadata["tags"]["processed"]  # From writer.write
    True

Best Practices
--------------

1. **Use NPZ for large datasets**: The compression benefits become significant with 100+ files
2. **Include relevant metadata**: Track processing parameters, data sources, and versions
3. **Attach metadata early**: Use the `metadata` parameter in `from_file` to attach metadata at load time
4. **Use consistent patterns**: Maintain clear file naming conventions across projects
5. **Leverage context managers**: Ensures manifests are saved even if errors occur
6. **Consider sample rates**: Resample during writing to avoid repeated resampling later

Example: Creating a Training Dataset
------------------------------------

Here's a complete example of creating a training dataset with AudioWriter:

.. code-block:: python

    import numpy as np
    from pathlib import Path
    from audiotree import AudioTree, AudioWriter
    from audiotree.sources import AudioDataSimpleSource
    from audiotree.transforms import VolumeChange, RandomPhaseShift, Batch
    from tqdm import tqdm

    # todo: write a jitted augmentation pipeline.

    def create_training_dataset(
        source_directory,
        output_dir = "precomputed_data",
        augmentations_per_file: int = 3,
        batch_size: int = 4,
    ):
        """Turn one dataset into another with augmentations."""

        # Calculate total files to be created
        data_source = AudioDataSimpleSource({"main": [source_directory]})
        total_files = len(data_source) * augmentations_per_file

        data_loader = grain.DataLoader(
            data_source,
            num_records=len(data_source),
            operations=[Batch(batch_size, drop_remainder=False)],
        )

        pbar = tqdm(total=total_files, desc="Creating dataset")

        rng = np.random.default_rng(0)

        with AudioWriter(
            output_dir,
            pattern="train_{index:06d}.wav",
            sample_rate=16_000,  # Standardize to 16 kHz
            compress_manifest=True,
            pbar=pbar,
            close_pbar=True
        ) as writer:

            for audio_tree in data_source:
                # Generate augmentations
                for aug_idx in range(augmentations_per_file):
                    # Apply random transformations
                    augmented = audio_tree
                    augmented = VolumeChange(config={"min_db": -6, "max_db": 6}).random_map(augmented, rng)
                    augmented = RandomPhaseShift(prob=0.5).random_map(augmented, rng)

                    # Compute loudness for the augmented audio
                    augmented = augmented.replace_loudness()
                    writer.write(augmented)

        print(f"Created dataset with {writer.get_stats()['total_files']} files")

This creates a fully tracked, augmented dataset ready for training machine learning models.