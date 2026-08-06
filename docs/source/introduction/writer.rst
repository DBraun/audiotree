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
     - Saving a trained model's audio outputs to listen to or share
     - Pre-computing a dataset to train a model on

**Use AudioWriter** to save the outputs of a trained model — the generated or
reconstructed audio you want to listen to, share with collaborators, or feed into
non-Python tools. It writes standard WAV files and tracks per-sample data (loudness,
latents, etc.) in an NPZ manifest that supports filtering.

**Use TreeWriter** to pre-compute data for training a model — render an augmented or
feature-extracted dataset once, then read it back with no per-item decoding cost. It
stores the entire pytree (AudioTree, dicts of AudioTrees, nested structures) as
memory-mapped arrays — one ``.bin`` file per leaf, read as a zero-copy slice via
:class:`~audiotree.sources.tree.TreeDataSource` (a Grain ``RandomAccessDataSource``).

.. skip-snippet-exec: fragment; the dataloader comes from the reader's own pipeline.

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

.. _feature-only-trees:

Feature-only trees and selective loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a large pre-rendered corpus the raw waveform often dwarfs the features you
actually train on. Two knobs keep such datasets cheap:

- **Drop the waveform at write time.** ``AudioTree.replace(waveform=None)`` yields
  a *feature-only* tree — ``codes``, ``latents``, or a ``metadata`` feature with no
  audio. :class:`~audiotree.tree_writer.TreeWriter` simply omits the missing leaf,
  and it reads back as ``None``.
- **Skip leaves at read time.** :class:`~audiotree.sources.TreeDataSource` accepts
  ``exclude_prefixes`` (dot-separated leaf-name prefixes) to avoid ever reading a
  memmap you don't need — e.g. loading only mels for one training run and only
  audio for another, from the same directory.

.. testcode::

    import jax.numpy as jnp
    import numpy as np
    from audiotree import AudioTree, TreeWriter
    from audiotree.sources import TreeDataSource

    n = 8
    # A {dry, wet} dataset: dry keeps its audio; wet keeps only a mel feature and
    # drops its waveform to save disk.
    dry = AudioTree.create(jnp.zeros((n, 1, 16_000)), 16_000)
    wet = AudioTree.create(
        jnp.zeros((n, 1, 16_000)),
        16_000,
        metadata={"mel": np.zeros((n, 80, 32), np.float32)},
    ).replace(waveform=None)              # feature-only: no audio is written

    with TreeWriter("prerendered", expected_samples=n) as writer:
        writer.write({"dry": dry, "wet": wet})

    # Full read: wet.waveform was never written, so it comes back None.
    item = TreeDataSource("prerendered")[0]
    print(item["wet"].waveform is None, item["wet"].metadata["mel"].shape)

    # Lean read: also skip the dry audio memmap (huge on a real corpus).
    lean = TreeDataSource("prerendered", exclude_prefixes=["dry.waveform"])[0]
    print(lean["dry"].waveform is None)

.. testoutput::

    True (1, 80, 32)
    True

.. tip::
   Pass ``load_into_memory=True`` to :class:`~audiotree.sources.TreeDataSource` to
   read every (non-excluded) leaf into RAM once at construction. With fork-based
   multiprocessing (the default on Linux) Grain workers then inherit that data via
   copy-on-write instead of each re-opening the memmaps — trading memory for zero
   per-worker I/O. Combine it with ``exclude_prefixes`` so only the leaves you
   train on are held in memory.

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

By default ``AudioWriter`` refuses to write into a directory that already holds a
manifest, so a finished dataset is never silently overwritten (and two writers
aimed at one directory are caught). The examples below reuse ``"output"`` and so
pass ``exist_ok=True`` to opt in.

NPZ is the recommended format for all use cases, especially for large datasets:

.. testcode::

    # NPZ format - best for large datasets
    with AudioWriter("output", exist_ok=True) as writer:
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
    writer = AudioWriter("output", compress_manifest=True, exist_ok=True)

    # Uncompressed NPZ - faster writing, larger files
    writer = AudioWriter("output", compress_manifest=False, exist_ok=True)

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
        filepath=["original1.wav", "original2.wav"]
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
    writer = AudioWriter("output", include_timestamp=False, exist_ok=True)

    # With timestamps - track when files were written
    writer = AudioWriter("output", include_timestamp=True, exist_ok=True)

Timestamps are useful for:

- Tracking dataset creation history
- Debugging data pipeline issues
- Audit trails for data processing

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

.. skip-snippet-exec: needs the optional ``tqdm`` dependency.

.. code-block:: python

    from tqdm import tqdm

    # Create your own progress bar with custom settings
    pbar = tqdm(total=1000, desc="Generating dataset")

    with AudioWriter("output", pbar=pbar, exist_ok=True) as writer:
        for audio_tree in big_audio_tree.split(batch_size):
            if audio_tree.lufs > -30:  # Only write loud samples
                writer.write(audio_tree)  # Automatically updates pbar

**Using Internal Progress Bar:**

.. skip-snippet-exec: fragment; ``audio_trees`` is the reader's own iterable.

.. code-block:: python

    # Let AudioWriter create its own progress bar
    with AudioWriter("output", show_progress=True,
                     progress_desc="Writing audio", exist_ok=True) as writer:
        for audio_tree in audio_trees:
            writer.write(audio_tree)
    # Progress bar automatically closed

**Conditional Writing with Progress:**

.. skip-snippet-exec: needs the optional ``tqdm`` dependency.

.. code-block:: python

    from tqdm import tqdm

    # Count total samples that will be written
    loud_trees = [t for t in trees if t.lufs.mean() > -30]
    total_samples = sum(t.waveform.shape[0] for t in loud_trees)

    pbar = tqdm(total=total_samples, desc="Writing loud samples")
    with AudioWriter("output", pbar=pbar, close_pbar=True, exist_ok=True) as writer:
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

Use :class:`~audiotree.sources.audio.AudioDataSource` to read AudioWriter output:

.. testcode::

    from audiotree.sources import AudioDataSource

    # Write some data (with per-item loudness so it round-trips into the manifest)
    loudness_tree = AudioTree.create(
        np.random.randn(3, 2, 44_100),
        sample_rate=44_100,
        lufs=np.array([-20.0, -15.0, -18.0]),
    )
    with AudioWriter("output_read") as writer:
        writer.write(loudness_tree, tags={"split": "train"})

    # Read it back
    source = AudioDataSource.from_writer_output("output_read")

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

Manifest-Only: Saving Embeddings (No Audio)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes the payload you want to save is not audio but a per-item array a model
produced — an embedding, a projection, a set of predicted parameters. Pass
``write_audio=False`` to write **only** the NPZ manifest, with your arrays carried
in ``metadata``. No WAV files are written, so a large evaluation set of embeddings
costs almost nothing on disk:

.. testcode::

    import os
    import numpy as np
    from audiotree import AudioTree, AudioWriter

    # A batch of clips, each with an embedding a model produced. Carry the
    # embeddings (and any ids you need) in metadata — an active pytree node.
    rng = np.random.default_rng(0)
    n_items = 100
    batch = AudioTree.create(
        rng.standard_normal((n_items, 1, 16_000)).astype(np.float32),
        16_000,
        metadata={
            "embedding": rng.standard_normal((n_items, 128)).astype(np.float32),
            "label": np.arange(n_items),
        },
    )

    # write_audio=False writes manifest.npz only (no per-item WAVs).
    with AudioWriter("embeddings", write_audio=False) as writer:
        writer.write(batch)

    print(sorted(os.listdir("embeddings")))

.. testoutput::

    ['manifest.npz']

To read the set back for analysis, :meth:`~audiotree.core.AudioTree.from_manifest`
loads the **entire** manifest into a single batched AudioTree — so every item's
embedding lands in one array rather than a stream. An optional ``filter_fn``
predicate (evaluated per manifest entry) selects a subset at load time:

.. testcode::

    # The whole manifest as one tree; metadata arrays round-trip exactly.
    tree = AudioTree.from_manifest("embeddings/manifest.npz")
    print(tree.metadata["embedding"].shape)

    # filter_fn sees each entry's columns as ``metadata_<key>``; keep labels < 10.
    subset = AudioTree.from_manifest(
        "embeddings/manifest.npz",
        filter_fn=lambda entry: entry["metadata_label"] < 10,
    )
    print(subset.metadata["embedding"].shape)

.. testoutput::

    (100, 128)
    (10, 128)

.. note::
   **Two readers, two shapes.** :meth:`~audiotree.core.AudioTree.from_manifest`
   returns *one* batched AudioTree with the whole manifest stacked along the batch
   axis — ideal for a one-shot analysis pass over saved embeddings.
   :class:`~audiotree.sources.AudioDataSource` (above) is instead a Grain
   ``RandomAccessDataSource`` that yields one item at a time in manifest order, for
   feeding a pipeline. ``from_manifest`` restores the ``metadata_*`` arrays, the
   label fields (``lufs``, ``pitch``, ``codes``, …), and the source ``filepath``
   column — so ``loaded.filepath`` matches the paths you wrote.

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

When read back via :class:`~audiotree.sources.audio.AudioDataSource`, the per-item
metadata is restored as batched arrays (so a single-item read gives ``array(['guitar'])``
for a string field):

.. testcode::

    from audiotree.sources import AudioDataSource
    source = AudioDataSource.from_writer_output("output_flow")
    loaded = source[0]
    print(loaded.metadata["instrument"])

.. testoutput::

    ['guitar']

Example: Creating a Training Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a complete example of creating a training dataset with TreeWriter:

.. skip-snippet-exec: needs the optional ``tqdm`` dependency.

.. code-block:: python

    from tqdm import tqdm
    from audiotree import TreeWriter
    from audiotree.sources import create_audio_dataset, TreeDataSource
    from audiotree.transforms import volume_change, shift_phase

    def precompute_training_dataset(source_directory, num_samples, directory="precomputed_data"):
        """Render an augmented dataset once so training never recomputes it."""

        # An infinite, shuffled stream of 3-second mono excerpts.
        ds = create_audio_dataset(
            sources=source_directory,
            sample_rate=16_000,
            duration=3.0,
            mono=True,
            shuffle=True,
            repeat=True,
        )

        # Seed once; each random_map derives its own distinct seed so every
        # augmentation differs.
        ds = ds.seed(42)
        ds = ds.random_map(volume_change(min_db=-6, max_db=6))
        ds = ds.random_map(shift_phase())

        # Take num_samples items from the infinite stream and write each one.
        # TreeWriter pre-allocates expected_samples rows up front.
        it = iter(ds.to_iter_dataset())
        pbar = tqdm(total=num_samples, desc="Precomputing")
        with TreeWriter(output_dir, expected_samples=num_samples, pbar=pbar, close_pbar=True) as writer:
            for _ in range(num_samples):
                writer.write(next(it))

        return output_dir

    # Build the dataset once...
    precompute_training_dataset("/data/audio", num_samples=10_000)

    # ...then train from it with no augmentation or decoding cost.
    train_ds = TreeDataSource("precomputed_data")

To pre-compute **input/target pairs** — say an augmented ``"input"`` beside the clean
``"target"`` — augment a ``{"input": ..., "target": ...}`` dict and restrict each
transform to one key with the ``scope`` parameter; see :ref:`dict_batches`.

Next
----

That completes the Getting-started path — you can load audio, augment it, and write
prepared datasets back to disk. From here, the **Going further** guides dig into
:ref:`balanced_datasets`, :ref:`windowed_datasets`, :ref:`dict_batches`, command-line
configuration with :ref:`argbind_guide`, and :ref:`multiprocessing`.
