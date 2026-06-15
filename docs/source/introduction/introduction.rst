.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _introduction:

Introduction to AudioTree
=========================

This guide covers the fundamentals of working with :class:`~audiotree.core.AudioTree` objects,
including instantiation, manipulation, batching, and integration with JAX's pytree system.

Basic Instantiation
-------------------

The :class:`~audiotree.core.AudioTree` class is the central data structure in the library.
It stores audio as JAX arrays with a consistent shape convention: ``(Batch, Channels, Samples)``.

Creating from NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create an AudioTree directly from NumPy or JAX NumPy arrays:

.. code-block:: python

    import numpy as np
    from audiotree import AudioTree

    # Create from 3D array (B, C, T)
    sample_rate = 44_100
    waveform = np.zeros((4, 2, 44_100))  # 4 batches, 2 channels, 1 second
    audio_tree = AudioTree(waveform, sample_rate)

    >>> audio_tree.waveform.shape
    (4, 2, 44100)
    >>> audio_tree.sample_rate
    44100

Automatic Dimensionality Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.create` method automatically handles arrays of different dimensions:

.. code-block:: python

    # From 1D array (just samples)
    audio_1d = np.zeros(44_100)
    audio_tree = AudioTree.create(audio_1d, 44_100)
    >>> audio_tree.waveform.shape
    (1, 1, 44100)  # Automatically adds batch and channel dims

    # From 2D array (channels × samples)
    audio_2d = np.zeros((2, 44_100))
    audio_tree = AudioTree.create(audio_2d, 44_100)
    >>> audio_tree.waveform.shape
    (1, 2, 44100)  # Automatically adds batch dimension

Loading Audio from Files
------------------------

AudioTree provides convenient methods for loading audio files:

.. code-block:: python

    from pathlib import Path

    # Load an audio file
    audio_tree = AudioTree.from_file("audio.wav", sample_rate=44_100)

    # Load with specific offset and duration
    audio_tree = AudioTree.from_file(
        "audio.wav",
        sample_rate=44_100,
        offset=1.0,      # Start at 1 second
        duration=2.5,    # Load 2.5 seconds
        mono=False       # Keep stereo
    )

    # Load with custom metadata
    audio_tree = AudioTree.from_file(
        "audio.wav",
        sample_rate=44_100,
        metadata={
            "params": np.zeros((1, 4,)),  # intentionally give batch axis of 1
        }
    )

    # The filepath is automatically stored in metadata
    >>> audio_tree.filepath
    ['audio.wav']
    >>> audio_tree.metadata["params"]
    array([[0., 0., 0., 0.]])

Manipulating AudioTree Objects
------------------------------

Accessing Properties
~~~~~~~~~~~~~~~~~~~~

AudioTree objects have several key properties:

.. code-block:: python

    audio_tree = AudioTree.create(np.ones((2, 2, 44_100)), 44_100)

    # Core properties
    >>> audio_tree.waveform.shape
    (2, 2, 44100)
    >>> audio_tree.sample_rate
    44100

    # Optional properties (can be None)
    >>> audio_tree.loudness  # Computed on demand
    None
    >>> audio_tree.metadata  # Dictionary for custom data
    {}

Creating Modified Copies
~~~~~~~~~~~~~~~~~~~~~~~~

AudioTree is immutable. Use :meth:`~audiotree.core.AudioTree.replace` to create modified copies:

.. code-block:: python

    # Original audio_tree with batch size 2
    audio_tree = AudioTree(np.ones((2, 2, 44_100)), 44_100)

    # Create a new audio_tree with modified audio data
    quieter_tree = audio_tree.replace(waveform=audio_tree.waveform * 0.5)

    >>> np.allclose(audio_tree.waveform[0, 0, 0], 1.0)  # Original unchanged
    True
    >>> np.allclose(quieter_tree.waveform[0, 0, 0], 0.5)
    True

Computing Loudness
~~~~~~~~~~~~~~~~~~

AudioTree can compute loudness in LUFS (Loudness Units Full Scale) for each item in the batch:

.. code-block:: python

    # Create audio_tree with 4 batches and compute loudness for each
    audio_tree = AudioTree(np.ones((4, 2, 44_100)) * 0.1, 44_100)
    tree_with_loudness = audio_tree.replace_loudness()

    >>> tree_with_loudness.loudness.shape
    (4,)  # One loudness value per batch item
    >>> tree_with_loudness.loudness
    Array([-23.456, -23.456, -23.456, -23.456], dtype=float32)  # Example LUFS values

.. note::
   **Channel Limitations for Loudness Computation**

   The loudness calculation supports up to 5 channels, following the ITU-R BS.1770-4 standard:

   - **Mono (1 channel)**: Single channel
   - **Stereo (2 channels)**: [Left, Right]
   - **5.0/5.1 Surround (5 channels)**: [Left, Right, Center, Left Surround, Right Surround]

   AudioTree objects with more than 5 channels will raise an error during loudness computation.

Resampling Audio
~~~~~~~~~~~~~~~~

To change the sample rate of audio, use the :meth:`~audiotree.core.AudioTree.resample` method:

.. code-block:: python

    # Original audio_tree at 44.1 kHz
    audio_tree = AudioTree(np.ones((1, 2, 44_100)), 44_100)

    # Resample to 48 kHz
    resampled_tree = audio_tree.resample(48_000)

    >>> audio_tree.sample_rate  # Original unchanged
    44100
    >>> resampled_tree.sample_rate
    48000
    >>> resampled_tree.waveform.shape
    (1, 2, 48000)  # Audio data is resampled

Converting Channels
~~~~~~~~~~~~~~~~~~~

Use :meth:`~audiotree.core.AudioTree.to_mono` and :meth:`~audiotree.core.AudioTree.to_stereo`
to change the channel layout. ``to_mono`` takes a ``strategy``:

.. code-block:: python

    audio_tree = AudioTree(np.random.randn(4, 2, 44_100), 44_100)

    # "average" (default) mixes all channels down
    mono = audio_tree.to_mono()
    >>> mono.waveform.shape
    (4, 1, 44100)

    # "left" / "right" select one channel of stereo audio
    left = audio_tree.to_mono("left")
    right = audio_tree.to_mono("right")
    >>> np.allclose(left.waveform[:, 0], audio_tree.waveform[:, 0])
    True

    # Duplicate a mono channel up to stereo
    stereo = mono.to_stereo()
    >>> stereo.waveform.shape
    (4, 2, 44100)

.. note::
   Changing the channel layout changes the integrated loudness, so ``to_mono`` and
   the mono→stereo path of ``to_stereo`` clear any cached ``loudness``. It is
   recomputed on the next :meth:`~audiotree.core.AudioTree.replace_loudness`.

Indexing and Iterating Batches
------------------------------

An AudioTree behaves like a sequence over its leading (batch) axis. Indexing,
slicing, ``len()``, and iteration keep every field — ``waveform``, ``codes``,
``latents``, and the ``metadata`` arrays — rank-aligned.

.. code-block:: python

    audio_tree = AudioTree(np.random.randn(16, 2, 44_100), 44_100)

    >>> len(audio_tree)               # number of batch items
    16

    # Integer indexing keeps the batch axis (a batch of 1)
    >>> audio_tree[0].waveform.shape
    (1, 2, 44100)

    # Slices select a sub-batch
    >>> audio_tree[4:8].waveform.shape
    (4, 2, 44100)

    # Negative indices work too
    >>> audio_tree[-1].waveform.shape
    (1, 2, 44100)

Because AudioTree implements ``__iter__`` it is a proper
``collections.abc.Iterable``. Iterating yields one batch-of-1 AudioTree per item:

.. code-block:: python

    for item in audio_tree:
        assert item.waveform.shape[0] == 1
        # process or write a single example...

Pairing ``__iter__`` with ``len()`` means progress bars work out of the box —
``tqdm`` reads ``len()`` to size the bar automatically:

.. code-block:: python

    import tqdm

    for item in tqdm.tqdm(audio_tree):   # shows a 0/16 ... 16/16 bar
        ...

To reassemble a batch from individual items, use
:meth:`~audiotree.core.AudioTree.batch_fn`:

.. code-block:: python

    items = [audio_tree[i] for i in range(len(audio_tree))]
    rebuilt = AudioTree.batch_fn(items)
    >>> rebuilt.waveform.shape
    (16, 2, 44100)

Batching Operations
-------------------

AudioTree provides several methods for working with batches of audio.

Creating Mini-Batches
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.reshape_mini_batches` method adds a mini-batch axis,
and :meth:`~audiotree.core.AudioTree.flatten_mini_batches` removes it again:

.. code-block:: python

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)
    >>> x.waveform.shape
    (12, 1, 44100)

    # Reshape into mini-batches of size 3
    x_batched = x.reshape_mini_batches(3)
    >>> x_batched.waveform.shape
    (4, 3, 1, 44100)  # 4 mini-batches, each with 3 samples

    # Flatten back to original shape
    x_unbatched = x_batched.flatten_mini_batches()
    >>> x_unbatched.waveform.shape
    (12, 1, 44100)  # Back to original

.. note::
   AudioTree methods operate on mini-batched trees directly. Methods like
   :meth:`~audiotree.core.AudioTree.replace_loudness`,
   :meth:`~audiotree.core.AudioTree.normalize_loudness`,
   :meth:`~audiotree.core.AudioTree.to_mono`, :meth:`~audiotree.core.AudioTree.to_stereo`,
   and :meth:`~audiotree.core.AudioTree.resample` treat *all* leading axes as batch
   axes, so you can call them on a ``(num_mini_batches, mini_batch_size, C, T)`` tree
   without flattening first. Per-item results follow the leading shape — e.g.
   ``loudness`` comes back shaped ``(num_mini_batches, mini_batch_size)``.

Splitting into Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.split` method splits a batch into separate AudioTree objects:

.. code-block:: python

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)

    # Split into 2 separate AudioTree objects
    split_trees = x.split(2)
    >>> len(split_trees)
    2
    >>> split_trees[0].waveform.shape
    (6, 1, 44100)  # First half
    >>> split_trees[1].waveform.shape
    (6, 1, 44100)  # Second half

Filtering Batch Items
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.filter` method allows you to selectively keep batch items based on a condition:

.. code-block:: python

    # Generate uniform noise and scale it to different levels
    np.random.seed(42)  # For reproducible results
    noise = np.random.uniform(-1, 1, (1, 44_100))

    # Create AudioTree with different loudness levels
    waveform = np.array([
        noise * 0.0,  # Silent
        noise * 0.1,  # Very quiet
        noise * 0.2,
        noise * 0.3,
        noise * 0.4,
        noise * 0.5,
        noise * 0.6,
        noise * 0.7,
        noise * 0.8,
        noise * 0.9,
        noise * 1.0   # Full scale
    ])

    audio_tree = AudioTree(waveform, 44_100).replace_loudness()

    # Filter to keep only audio louder than -20 LUFS
    def keep_loud_audio(mini_tree):
        return mini_tree.loudness[0] > -20.0

    filtered_tree = audio_tree.filter(keep_loud_audio)

    >>> filtered_tree.waveform.shape
    (9, 1, 44100)  # 9 batches remain (excluding silent and very quiet ones)
    >>> filtered_tree.loudness  # LUFS values for remaining batches
    Array([-15.69, -12.17, -9.67, -7.73, -6.15, -4.81, -3.65, -2.63, -1.71], dtype=float32)

Processing Mini-Batches with nnx.scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Flax's :func:`nnx.scan` to efficiently process mini-batches with neural networks:

.. code-block:: python

    from flax import nnx

    # Define a scan function to process mini-batches
    @nnx.scan(in_axes=0, out_axes=0)
    def process_mini_batches(mini_audio_tree):
        waveform = mini_audio_tree.waveform
        assert waveform.ndim == 3  # (batch, channels, samples)
        waveform = waveform * 0.5  # or use a neural network!
        return mini_audio_tree.replace(waveform=waveform)

    # Create AudioTree with 12 samples
    x = AudioTree(np.ones((12, 1, 44_100)), 44_100)

    # Create mini-batches of size 3
    x_batched = x.reshape_mini_batches(3)
    >>> x_batched.waveform.shape
    (4, 3, 1, 44100)  # 4 mini-batches of size 3

    # Process all mini-batches sequentially
    processed_batched = process_mini_batches(x_batched)
    >>> processed_batched.waveform.shape
    (4, 3, 1, 44100)  # Still mini-batched

    # Flatten back to original batch dimension
    full_batch = processed_batched.flatten_mini_batches()
    >>> full_batch.waveform.shape
    (12, 1, 44100)  # Back to original shape
    >>> np.allclose(full_batch.waveform, 0.5)
    True  # All values were processed

Working with JAX PyTrees
------------------------

AudioTree is a JAX pytree, which means it works seamlessly with JAX's audio_tree operations.

Concatenating Trees with jax.audio_tree.map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use :func:`jax.audio_tree.map` to combine multiple AudioTree objects:

.. code-block:: python

    import jax

    # Create a audio_tree with 4 batches
    x = AudioTree(np.zeros((4, 1, 44_100)), 44_100)

    # Create a list of three identical trees
    trees = [x, x, x]

    # Concatenate along the batch dimension
    big_tree = jax.audio_tree.map(
        lambda *xs: np.concatenate(xs, axis=0),
        *trees
    )

    >>> big_tree.waveform.shape
    (12, 1, 44100)  # 3 × 4 = 12 batches

Nested Structures
~~~~~~~~~~~~~~~~~

AudioTree objects can be organized in complex nested structures:

.. code-block:: python

    # Create different audio trees
    input_audio = AudioTree(np.ones((2, 1, 1000)), 44_100)
    target_audio = AudioTree(np.zeros((2, 1, 1000)), 44_100)

    # Organize in a nested structure
    batch = {
        "input": input_audio,
        "target": target_audio,
        "augmented": [input_audio, target_audio]
    }

    # Apply transformations to all trees in the structure
    def scale_audio(audio_tree):
        return audio_tree.replace(waveform=audio_tree.waveform * 0.5)

    scaled_batch = jax.audio_tree.map(
        scale_audio,
        batch,
        is_leaf=lambda x: isinstance(x, AudioTree)
    )

    >>> scaled_batch["input"].waveform[0, 0, 0]
    0.5  # Scaled from 1.0
    >>> scaled_batch["augmented"][1].waveform[0, 0, 0]
    0.0  # Scaled from 0.0 (remains 0)

audio_tree Flattening and Unflattening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX can flatten AudioTree objects for operations requiring flat arrays:

.. code-block:: python

    audio_tree = AudioTree(np.ones((1, 2, 1000)), 44_100)

    # Flatten the audio_tree into leaves and structure
    leaves, treedef = jax.audio_tree.flatten(audio_tree)

    # Modify leaves if needed...
    # Then reconstruct the audio_tree
    reconstructed = jax.audio_tree.unflatten(treedef, leaves)

    >>> np.array_equal(reconstructed.waveform, audio_tree.waveform)
    True

Metadata and Filepaths
-----------------------

AudioTree supports storing metadata and filepath information.

Storing Filepaths
~~~~~~~~~~~~~~~~~

When creating AudioTree objects, you can associate them with source files:

.. code-block:: python

    # Single filepath
    audio_tree = AudioTree.create(
        np.zeros((1, 44_100)),
        44_100,
        filepaths="audio.wav"
    )
    >>> audio_tree.filepath
    ['audio.wav']

    # Multiple filepaths for batched data
    audio_tree = AudioTree.create(
        np.zeros((3, 1, 44_100)),
        44_100,
        filepaths=["a.wav", "b.wav", "c.wav"]
    )
    >>> audio_tree.filepath
    ['a.wav', 'b.wav', 'c.wav']

Understanding Metadata
~~~~~~~~~~~~~~~~~~~~~~

The ``metadata`` field is special - it's a pytree node (``pytree_node=True``), meaning it participates
in JAX audio_tree operations like batching and concatenation. This is different from ``sample_rate``, which
is marked as ``pytree_node=False`` and remains constant across operations.

Metadata should contain array-like data with a batch dimension:

.. code-block:: python

    # Create trees with array metadata that can be batched
    tree1 = AudioTree.create(
        np.zeros((2, 1, 44_100)),
        44_100,
        metadata={
            "energy": np.array([0.8, 0.9]),  # Shape (2,) matching batch size
            "onset_times": np.array([[0.1, 0.2], [0.15, 0.25]])  # Shape (2, 2)
        }
    )

    tree2 = AudioTree.create(
        np.zeros((2, 1, 44_100)),
        44_100,
        metadata={
            "energy": np.array([0.7, 0.85]),
            "onset_times": np.array([[0.12, 0.22], [0.18, 0.28]])
        }
    )

    # Concatenate trees - metadata gets concatenated too
    import jax
    combined = jax.audio_tree.map(
        lambda *xs: np.concatenate(xs, axis=0),
        tree1, tree2
    )

    >>> combined.waveform.shape
    (4, 1, 44100)  # Batched from 2+2
    >>> combined.metadata["energy"].shape
    (4,)  # Metadata was concatenated
    >>> combined.metadata["onset_times"].shape
    (4, 2)  # 2D metadata also concatenated along batch dimension
    >>> combined.sample_rate
    44100  # Sample rate stays the same (not a pytree node)

Writing Audio to Disk
---------------------

Writing a Single File
~~~~~~~~~~~~~~~~~~~~~~

:meth:`~audiotree.core.AudioTree.write` saves one item to an audio file via
`soundfile <https://python-soundfile.readthedocs.io/>`_ — the inverse of
:meth:`~audiotree.core.AudioTree.from_file`. The tree must contain exactly one item
(``batch_size == 1``), so index or iterate a batch first. There is no sample-rate
argument: it uses ``self.sample_rate``, so call
:meth:`~audiotree.core.AudioTree.resample` beforehand to change it.

.. code-block:: python

    audio_tree = AudioTree.from_file("input.wav", sample_rate=44_100)

    # The file format is inferred from the extension.
    audio_tree.write("output.wav")

    # Control the encoding with soundfile passthroughs.
    audio_tree.write("output_24bit.wav", subtype="PCM_24")
    audio_tree.write("output.flac")  # FLAC inferred from the ".flac" extension

    # write() returns the Path it wrote.
    >>> audio_tree.write("output.wav")
    PosixPath('output.wav')

Indexing or iterating makes it easy to write every item of a batch (each item is a
batch of 1, exactly what ``write`` requires):

.. code-block:: python

    batch = AudioTree(np.random.randn(8, 2, 44_100), 44_100)
    for i, item in enumerate(batch):
        item.write(f"item_{i}.wav")

.. note::
   ``write`` operates on a single item by design. Calling it on a multi-item batch
   raises an ``AssertionError`` — use :class:`~audiotree.writer.AudioWriter` below to
   write a whole batch (with a manifest) in one call.

Writing Batches with Manifests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~audiotree.writer.AudioWriter` class provides a convenient way to write AudioTree objects to disk with automatic manifest generation for tracking metadata.

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    from audiotree import AudioTree, AudioWriter
    import numpy as np

    # Create an AudioTree with 3 samples
    audio_tree = AudioTree.create(
        np.random.randn(3, 2, 44_100),  # 3 batches, stereo, 1 second
        sample_rate=44_100,
        loudness=np.array([-20.0, -15.0, -18.0])
    )

    # Write to disk with automatic manifest
    with AudioWriter("output") as writer:
        paths = writer.write(audio_tree, tags={"dataset": "train"})

    >>> len(paths)
    3  # One file per batch item

    # Read the data back
    from audiotree.sources import ManifestDataSource
    source = ManifestDataSource.from_writer_output("output")
    >>> source[0].loudness  # Metadata is preserved
    array([-20.])

Next Steps
----------

Now that you understand the basics of AudioTree, explore:

- :ref:`writer` - Learn how to write AudioTree objects to disk with manifests
- :ref:`sources` - Learn how to create data loaders for ML pipelines
- :ref:`transforms` - Discover audio augmentations and transformations
- :class:`~audiotree.core.AudioTree` API reference for detailed documentation