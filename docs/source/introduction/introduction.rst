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
    audio_data = np.zeros((4, 2, 44_100))  # 4 batches, 2 channels, 1 second
    audio_tree = AudioTree(audio_data, sample_rate)

    >>> audio_tree.audio_data.shape
    (4, 2, 44100)
    >>> audio_tree.sample_rate
    44100

Automatic Dimensionality Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.create` method automatically handles arrays of different dimensions:

.. code-block:: python

    # From 1D array (just samples)
    audio_1d = np.zeros(44_100)
    audio_tree = AudioTree.create(audio_1d, 44_100)
    >>> audio_tree.audio_data.shape
    (1, 1, 44100)  # Automatically adds batch and channel dims

    # From 2D array (channels × samples)
    audio_2d = np.zeros((2, 44_100))
    audio_tree = AudioTree.create(audio_2d, 44_100)
    >>> audio_tree.audio_data.shape
    (1, 2, 44100)  # Automatically adds batch dimension

Loading Audio from Files
-------------------------

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
-------------------------------

Accessing Properties
~~~~~~~~~~~~~~~~~~~~

AudioTree objects have several key properties:

.. code-block:: python

    audio_tree = AudioTree.create(np.ones((2, 2, 44_100)), 44_100)

    # Core properties
    >>> audio_tree.audio_data.shape
    (2, 2, 44100)
    >>> audio_tree.sample_rate
    44100

    # Optional properties (can be None)
    >>> audio_tree.loudness  # Computed on demand
    None
    >>> audio_tree.metadata  # Dictionary for custom data
    {}

Creating Modified Copies
~~~~~~~~~~~~~~~~~~~~~~~~~

AudioTree is immutable. Use :meth:`~audiotree.core.AudioTree.replace` to create modified copies:

.. code-block:: python

    # Original audio_tree with batch size 2
    audio_tree = AudioTree(np.ones((2, 2, 44_100)), 44_100)

    # Create a new audio_tree with modified audio data
    quieter_tree = audio_tree.replace(audio_data=audio_tree.audio_data * 0.5)

    >>> np.allclose(audio_tree.audio_data[0, 0, 0], 1.0)  # Original unchanged
    True
    >>> np.allclose(quieter_tree.audio_data[0, 0, 0], 0.5)
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
    >>> resampled_tree.audio_data.shape
    (1, 2, 48000)  # Audio data is resampled

Batching Operations
-------------------

AudioTree provides several methods for working with batches of audio.

Creating Mini-Batches
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.mini_batch` method reshapes the batch dimension:

.. code-block:: python

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)
    >>> x.audio_data.shape
    (12, 1, 44100)

    # Reshape into mini-batches of size 3
    x_batched = x.mini_batch(3)
    >>> x_batched.audio_data.shape
    (4, 3, 1, 44100)  # 4 mini-batches, each with 3 samples

    # Flatten back to original shape
    x_unbatched = x_batched.unbatch()
    >>> x_unbatched.audio_data.shape
    (12, 1, 44100)  # Back to original

Splitting into Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.mini_batch_list` method splits a batch into separate AudioTree objects:

.. code-block:: python

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)

    # Split into 2 separate AudioTree objects
    split_trees = x.mini_batch_list(2)
    >>> len(split_trees)
    2
    >>> split_trees[0].audio_data.shape
    (6, 1, 44100)  # First half
    >>> split_trees[1].audio_data.shape
    (6, 1, 44100)  # Second half

Filtering Batch Items
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.filter` method allows you to selectively keep batch items based on a condition:

.. code-block:: python

    # Generate uniform noise and scale it to different levels
    np.random.seed(42)  # For reproducible results
    noise = np.random.uniform(-1, 1, (1, 44_100))

    # Create AudioTree with different loudness levels
    audio_data = np.array([
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

    audio_tree = AudioTree(audio_data, 44_100).replace_loudness()

    # Filter to keep only audio louder than -20 LUFS
    def keep_loud_audio(mini_tree):
        return mini_tree.loudness[0] > -20.0

    filtered_tree = audio_tree.filter(keep_loud_audio)

    >>> filtered_tree.audio_data.shape
    (9, 1, 44100)  # 9 batches remain (excluding silent and very quiet ones)
    >>> filtered_tree.loudness  # LUFS values for remaining batches
    Array([-15.69, -12.17, -9.67, -7.73, -6.15, -4.81, -3.65, -2.63, -1.71], dtype=float32)

Processing Mini-Batches with nnx.scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Flax's :func:`nnx.scan` to efficiently process mini-batches with neural networks:

.. code-block:: python

    from flax import nnx

    # Define a scan function to process mini-batches
    @nnx.scan(in_axes=0, out_axes=0)
    def process_mini_batches(mini_audio_tree):
        audio_data = mini_audio_tree.audio_data
        assert audio_data.ndim == 3  # (batch, channels, samples)
        audio_data = audio_data * 0.5  # or use a neural network!
        return mini_audio_tree.replace(audio_data=audio_data)

    # Create AudioTree with 12 samples
    x = AudioTree(np.ones((12, 1, 44_100)), 44_100)

    # Create mini-batches of size 3
    x_batched = x.mini_batch(3)
    >>> x_batched.audio_data.shape
    (4, 3, 1, 44100)  # 4 mini-batches of size 3

    # Process all mini-batches sequentially
    processed_batched = process_mini_batches(x_batched)
    >>> processed_batched.audio_data.shape
    (4, 3, 1, 44100)  # Still mini-batched

    # Flatten back to original batch dimension
    full_batch = processed_batched.unbatch()
    >>> full_batch.audio_data.shape
    (12, 1, 44100)  # Back to original shape
    >>> np.allclose(full_batch.audio_data, 0.5)
    True  # All values were processed

Working with JAX PyTrees
-------------------------

AudioTree is a JAX pytree, which means it works seamlessly with JAX's audio_tree operations.

Concatenating Trees with jax.audio_tree.map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    >>> big_tree.audio_data.shape
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
        return audio_tree.replace(audio_data=audio_tree.audio_data * 0.5)

    scaled_batch = jax.audio_tree.map(
        scale_audio,
        batch,
        is_leaf=lambda x: isinstance(x, AudioTree)
    )

    >>> scaled_batch["input"].audio_data[0, 0, 0]
    0.5  # Scaled from 1.0
    >>> scaled_batch["augmented"][1].audio_data[0, 0, 0]
    0.0  # Scaled from 0.0 (remains 0)

audio_tree Flattening and Unflattening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX can flatten AudioTree objects for operations requiring flat arrays:

.. code-block:: python

    audio_tree = AudioTree(np.ones((1, 2, 1000)), 44_100)

    # Flatten the audio_tree into leaves and structure
    leaves, treedef = jax.audio_tree.flatten(audio_tree)

    # Modify leaves if needed...
    # Then reconstruct the audio_tree
    reconstructed = jax.audio_tree.unflatten(treedef, leaves)

    >>> np.array_equal(reconstructed.audio_data, audio_tree.audio_data)
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

    >>> combined.audio_data.shape
    (4, 1, 44100)  # Batched from 2+2
    >>> combined.metadata["energy"].shape
    (4,)  # Metadata was concatenated
    >>> combined.metadata["onset_times"].shape
    (4, 2)  # 2D metadata also concatenated along batch dimension
    >>> combined.sample_rate
    44100  # Sample rate stays the same (not a pytree node)

Writing Audio to Disk
---------------------

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
    with AudioWriter("output", manifest_format="npz") as writer:
        paths = writer.write(audio_tree, tags={"dataset": "train"})

    >>> len(paths)
    3  # One file per batch item

    # Read the data back
    from audiotree.datasources import ManifestDataSource
    source = ManifestDataSource.from_writer_output("output")
    >>> source[0].loudness  # Metadata is preserved
    array([-20.])

Next Steps
----------

Now that you understand the basics of AudioTree, explore:

- :ref:`writer` - Learn how to write AudioTree objects to disk with manifests
- :ref:`datasources` - Learn how to create data loaders for ML pipelines
- :ref:`transforms` - Discover audio augmentations and transformations
- :class:`~audiotree.core.AudioTree` API reference for detailed documentation