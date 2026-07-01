.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _introduction:

Introduction to AudioTree
=========================

This guide covers the fundamentals of working with :class:`~audiotree.core.AudioTree` objects,
including instantiation, manipulation, batching, and integration with JAX's Pytree system.

.. testsetup::

    # Hidden shared setup for this page's executable examples. It writes a small
    # synthetic stereo WAV that the file-loading examples below read back.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _doc_dir = tempfile.mkdtemp()
    _audio_path = os.path.join(_doc_dir, "audio.wav")
    _input_path = os.path.join(_doc_dir, "input.wav")
    soundfile.write(_audio_path, np.zeros((44_100, 2), dtype=np.float32), 44_100)
    soundfile.write(_input_path, np.zeros((44_100, 2), dtype=np.float32), 44_100)

    # Run all examples from inside the temp dir so relative output paths land there.
    os.chdir(_doc_dir)

Basic Instantiation
-------------------

The :class:`~audiotree.core.AudioTree` class is the central data structure in the library.
It stores audio as arrays with a consistent shape convention: ``(Batch, Channels, Samples)``.
This format is familiar to PyTorch and librosa users.
Note that JAX and NNX follow a different convention where data is commonly in
``(Batch, Samples, Channels)`` format.

Creating from NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create an AudioTree directly from NumPy or JAX NumPy arrays:

.. testcode::

    import numpy as np
    from audiotree import AudioTree

    # Create from 3D array (B, C, T)
    sample_rate = 44_100
    waveform = np.zeros((4, 2, 88_200))  # 4 batches, 2 channels, 2 seconds
    audio_tree = AudioTree(waveform, sample_rate)

    print(audio_tree.waveform.shape)
    print(audio_tree.sample_rate)

.. testoutput::

    (4, 2, 88200)
    44100

Automatic Dimensionality Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.create` method automatically handles arrays of different dimensions:

.. testcode::

    # From 1D array (just samples)
    audio_1d = np.zeros(44_100)
    audio_tree = AudioTree.create(audio_1d, 44_100)
    print(audio_tree.waveform.shape)  # Automatically adds batch and channel dims

    # From 2D array (channels × samples)
    audio_2d = np.zeros((2, 44_100))
    audio_tree = AudioTree.create(audio_2d, 44_100)
    print(audio_tree.waveform.shape)  # Automatically adds batch dimension

.. testoutput::

    (1, 1, 44100)
    (1, 2, 44100)

Loading Audio from Files
------------------------

AudioTree provides convenient methods for loading audio files:

.. testcode::

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
            "features_4d": np.zeros((1, 4,)),  # intentionally give batch axis of 1
        }
    )

    # The filepath is automatically stored in metadata
    print(audio_tree.filepath)
    print(audio_tree.metadata["features_4d"])

.. testoutput::

    ['audio.wav']
    [[0. 0. 0. 0.]]

Manipulating AudioTree Objects
------------------------------

Accessing Properties
~~~~~~~~~~~~~~~~~~~~

AudioTree objects have several key properties:

.. testcode::

    audio_tree = AudioTree.create(np.ones((2, 2, 44_100)), 44_100)

    # Core properties
    print(audio_tree.waveform.shape)
    print(audio_tree.sample_rate)

    # The metadata dict holds custom per-item data (empty by default)
    print(audio_tree.metadata)

.. testoutput::

    (2, 2, 44100)
    44100
    {}

Creating Modified Copies
~~~~~~~~~~~~~~~~~~~~~~~~

AudioTree is immutable. Use :meth:`~audiotree.core.AudioTree.replace` to create modified copies:

.. testcode::

    # Original audio_tree with batch size 2
    audio_tree = AudioTree(np.ones((2, 2, 44_100)), 44_100)

    # Create a new audio_tree with modified audio data
    quieter_tree = audio_tree.replace(waveform=audio_tree.waveform * 0.5)

    print(np.allclose(audio_tree.waveform[0, 0, 0], 1.0))   # Original unchanged
    print(np.allclose(quieter_tree.waveform[0, 0, 0], 0.5))

.. testoutput::

    True
    True

Computing Loudness
~~~~~~~~~~~~~~~~~~

AudioTree can compute loudness in LUFS (Loudness Units Full Scale) for each item in the batch:

.. testcode::

    # Create audio_tree with 4 batches and compute loudness for each
    audio_tree = AudioTree(np.ones((4, 2, 44_100)) * 0.1, 44_100)
    tree_with_loudness = audio_tree.replace_lufs()

    print(tree_with_loudness.lufs.shape)   # One loudness value per batch item
    print(tree_with_loudness.lufs)         # LUFS values (constant 0.1 signal)

.. testoutput::

    (4,)
    [-43.25 -43.25 -43.25 -43.25]

.. note::
   **Channel Limitations for Loudness Computation**

   The loudness calculation supports up to 5 channels, following the ITU-R BS.1770-4 standard:

   - **Mono (1 channel)**: Single channel
   - **Stereo (2 channels)**: [Left, Right]
   - **5.0/5.1 Surround (5 channels)**: [Left, Right, Center, Left Surround, Right Surround]

   AudioTree objects with more than 5 channels will raise an error during loudness computation.

Choosing a compute backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a NumPy waveform, the default loudness path measures one batch item at a time
on the CPU. When the batch is large and an accelerator is available, pass
``backend="gpu"`` (or ``"tpu"``, or ``"cpu"``) to run the vmapped ``jaxloudnorm``
kernel across the whole batch at once, which is much faster:

.. testcode::

    big_batch = AudioTree(np.ones((8, 2, 44_100)) * 0.1, 44_100)

    # backend forces the JAX loudness kernel onto that XLA device. Swap in
    # "gpu" or "tpu" when you have one; "cpu" always works and is used here so
    # the example runs anywhere.
    loud = big_batch.replace_lufs(backend="cpu")

    # The returned lufs always matches the waveform's array library (NumPy here),
    # so you never need a manual jax.device_put / jax.device_get round-trip.
    print(type(loud.lufs).__module__)
    print(loud.lufs.shape)

.. testoutput::

    numpy
    (8,)

:meth:`~audiotree.core.AudioTree.normalize_lufs` computes loudness internally, so
it accepts the same ``backend`` argument.

Resampling Audio
~~~~~~~~~~~~~~~~

To change the sample rate of audio, use the :meth:`~audiotree.core.AudioTree.resample` method:

.. testcode::

    # Original audio_tree at 44.1 kHz
    audio_tree = AudioTree(np.ones((1, 2, 44_100)), 44_100)

    # Resample to 48 kHz
    resampled_tree = audio_tree.resample(48_000)

    print(audio_tree.sample_rate)      # Original unchanged
    print(resampled_tree.sample_rate)
    print(resampled_tree.waveform.shape)   # Audio data is resampled

.. testoutput::

    44100
    48000
    (1, 2, 48000)

Converting Channels
~~~~~~~~~~~~~~~~~~~

Use :meth:`~audiotree.core.AudioTree.to_mono` and :meth:`~audiotree.core.AudioTree.to_stereo`
to change the channel layout. ``to_mono`` takes a ``strategy``:

.. testcode::

    audio_tree = AudioTree(np.random.randn(4, 2, 44_100), 44_100)

    # "average" (default) mixes all channels down
    mono = audio_tree.to_mono()
    print(mono.waveform.shape)

    # "left" / "right" select one channel of stereo audio
    left = audio_tree.to_mono("left")
    right = audio_tree.to_mono("right")
    print(np.allclose(left.waveform[:, 0], audio_tree.waveform[:, 0]))

    # Duplicate a mono channel up to stereo
    stereo = mono.to_stereo()
    print(stereo.waveform.shape)

.. testoutput::

    (4, 1, 44100)
    True
    (4, 2, 44100)

.. note::
   Changing the channel layout changes the integrated loudness, so ``to_mono`` and
   the mono→stereo path of ``to_stereo`` clear any cached ``lufs``. It is
   recomputed on the next :meth:`~audiotree.core.AudioTree.replace_lufs`.

Indexing and Iterating Batches
------------------------------

An AudioTree behaves like a sequence over its leading (batch) axis. Indexing,
slicing, ``len()``, and iteration keep every field — ``waveform``, ``codes``,
``latents``, and the ``metadata`` arrays — rank-aligned.

.. testcode::

    audio_tree = AudioTree(np.random.randn(16, 2, 44_100), 44_100)

    print(len(audio_tree))                  # number of batch items

    # Integer indexing keeps the batch axis (a batch of 1)
    print(audio_tree[0].waveform.shape)

    # Slices select a sub-batch
    print(audio_tree[4:8].waveform.shape)

    # Negative indices work too
    print(audio_tree[-1].waveform.shape)

.. testoutput::

    16
    (1, 2, 44100)
    (4, 2, 44100)
    (1, 2, 44100)

Because AudioTree implements ``__iter__`` it is a proper
``collections.abc.Iterable``. Iterating yields one batch-of-1 AudioTree per item:

.. testcode::

    for item in audio_tree:
        assert item.waveform.shape[0] == 1
        # process or write a single example...

Pairing ``__iter__`` with ``len()`` means progress bars work out of the box —
``tqdm`` reads ``len()`` to size the bar automatically (``tqdm`` is an optional
dependency, so this snippet is illustrative rather than executed):

.. code-block:: python

    import tqdm

    for item in tqdm.tqdm(audio_tree):   # shows a 0/16 ... 16/16 bar
        ...

To reassemble a batch from individual items, use
:meth:`~audiotree.core.AudioTree.batch`:

.. testcode::

    items = [audio_tree[i] for i in range(len(audio_tree))]
    rebuilt = AudioTree.batch(items)
    print(rebuilt.waveform.shape)

.. testoutput::

    (16, 2, 44100)

Batching Operations
-------------------

AudioTree provides several methods for working with batches of audio.

Creating Mini-Batches
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.reshape_mini_batches` method adds a mini-batch axis,
and :meth:`~audiotree.core.AudioTree.flatten_mini_batches` removes it again:

.. testcode::

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)
    print(x.waveform.shape)

    # Reshape into mini-batches of size 3
    x_batched = x.reshape_mini_batches(3)
    print(x_batched.waveform.shape)   # 4 mini-batches, each with 3 samples

    # Flatten back to original shape
    x_unbatched = x_batched.flatten_mini_batches()
    print(x_unbatched.waveform.shape)   # Back to original

.. testoutput::

    (12, 1, 44100)
    (4, 3, 1, 44100)
    (12, 1, 44100)

.. note::
   AudioTree methods operate on mini-batched trees directly. Methods like
   :meth:`~audiotree.core.AudioTree.replace_lufs`,
   :meth:`~audiotree.core.AudioTree.normalize_lufs`,
   :meth:`~audiotree.core.AudioTree.to_mono`, :meth:`~audiotree.core.AudioTree.to_stereo`,
   and :meth:`~audiotree.core.AudioTree.resample` treat *all* leading axes as batch
   axes, so you can call them on a ``(num_mini_batches, mini_batch_size, C, T)`` tree
   without flattening first. Per-item results follow the leading shape — e.g.,
   ``lufs`` comes back shaped ``(num_mini_batches, mini_batch_size)``.

Splitting into Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.split` method splits a batch into separate AudioTree objects:

.. testcode::

    # Start with 12 audio samples
    x = AudioTree(np.zeros((12, 1, 44_100)), 44_100)

    # Split into 2 separate AudioTree objects
    split_trees = x.split(2)
    print(len(split_trees))
    print(split_trees[0].waveform.shape)   # First half
    print(split_trees[1].waveform.shape)   # Second half

.. testoutput::

    2
    (6, 1, 44100)
    (6, 1, 44100)

Filtering Batch Items
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.core.AudioTree.filter` method allows you to selectively keep batch items based on a condition:

.. testcode::

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

    audio_tree = AudioTree(waveform, 44_100).replace_lufs()

    # Filter to keep only audio louder than -20 LUFS
    def keep_loud_audio(mini_tree):
        return mini_tree.lufs[0] > -20.0

    filtered_tree = audio_tree.filter(keep_loud_audio)

    # 9 batches remain (excluding silent and very quiet ones)
    print(filtered_tree.waveform.shape)
    # LUFS values for remaining batches (rounded for display)
    print(np.round(filtered_tree.lufs, 2))

.. testoutput::

    (9, 1, 44100)
    [-15.65 -12.15  -9.65  -7.69  -6.11  -4.75  -3.61  -2.56  -1.65]

Processing Mini-Batches with nnx.scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Flax's :func:`nnx.scan` to efficiently process mini-batches with neural networks.
If a batch is too large to process in memory at once, using mini-batches can meet
the memory requirement.

.. testcode::

    from flax import nnx

    # Define a scan function to process mini-batches
    @nnx.scan(in_axes=0, out_axes=0)
    def process_mini_batches(audio_tree: AudioTree):
        waveform = audio_tree.waveform
        assert waveform.ndim == 3  # (batch, channels, samples)
        waveform = waveform * 0.5  # or use a neural network!
        return audio_tree.replace(waveform=waveform)

    # Create AudioTree with 12 samples
    x = AudioTree(np.ones((12, 1, 44_100)), 44_100)

    # Create mini-batches of size 3
    x_batched = x.reshape_mini_batches(3)
    print(x_batched.waveform.shape)   # 4 mini-batches of size 3

    # Process all mini-batches sequentially
    processed_batched = process_mini_batches(x_batched)
    print(processed_batched.waveform.shape)   # Still mini-batched

    # Flatten back to original batch dimension
    full_batch = processed_batched.flatten_mini_batches()
    print(full_batch.waveform.shape)         # Back to original shape
    print(np.allclose(full_batch.waveform, 0.5))   # All values were processed

.. testoutput::

    (4, 3, 1, 44100)
    (4, 3, 1, 44100)
    (12, 1, 44100)
    True

Working with JAX PyTrees
------------------------

AudioTree is a JAX Pytree, which means it works seamlessly with JAX's
`tree <https://docs.jax.dev/en/latest/jax.tree.html>`_ operations.

Concatenating Trees with jax.tree.map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use :func:`jax.tree.map` to combine multiple AudioTree objects:

.. testcode::

    import jax

    # Create a tree with 4 batches
    x = AudioTree(np.zeros((4, 1, 44_100)), 44_100)

    # Create a list of three identical trees
    trees = [x, x, x]

    # Concatenate along the batch dimension
    big_tree = jax.tree.map(
        lambda *xs: np.concatenate(xs, axis=0),
        *trees
    )

    print(big_tree.waveform.shape)   # 3 × 4 = 12 batches

.. testoutput::

    (12, 1, 44100)

This concatenating ``tree.map`` is exactly what
:meth:`~audiotree.core.AudioTree.batch` does for you: it maps
``np.concatenate(..., axis=0)`` over every leaf (treating each AudioTree as one
leaf), so a list of trees collapses into the identical batched tree. It is the
same function you pass to Grain as ``batch_fn=AudioTree.batch`` when building a
data loader.

.. testcode::

    same_tree = AudioTree.batch(trees)

    print(same_tree.waveform.shape)
    print(np.array_equal(same_tree.waveform, big_tree.waveform))

.. testoutput::

    (12, 1, 44100)
    True

Nested Structures
~~~~~~~~~~~~~~~~~

AudioTree objects can be organized in complex nested structures:

.. testcode::

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
    def scale_audio(audio_tree: AudioTree):
        return audio_tree.replace(waveform=audio_tree.waveform * 0.5)

    scaled_batch = jax.tree.map(
        scale_audio,
        batch,
        is_leaf=lambda x: isinstance(x, AudioTree)
    )

    print(scaled_batch["input"].waveform[0, 0, 0])         # Scaled from 1.0
    print(scaled_batch["augmented"][1].waveform[0, 0, 0])  # Scaled from 0.0 (remains 0)

.. testoutput::

    0.5
    0.0

Tree Flattening and Unflattening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX can flatten AudioTree objects for operations requiring flat arrays:

.. testcode::

    audio_tree = AudioTree(np.ones((1, 2, 1000)), 44_100)

    # Flatten the audio_tree into leaves and structure
    leaves, treedef = jax.tree.flatten(audio_tree)

    # Modify leaves if needed...
    # Then reconstruct the audio_tree
    reconstructed = jax.tree.unflatten(treedef, leaves)

    print(np.array_equal(reconstructed.waveform, audio_tree.waveform))

.. testoutput::

    True

Moving a Tree Between Devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because an AudioTree is a pytree, :func:`jax.device_put` and :func:`jax.device_get`
move *every* array leaf at once — you never touch ``waveform``, ``lufs``, and each
``metadata`` array one by one. The static ``sample_rate`` and the tree structure
are left untouched.

.. testcode::

    import jax

    audio_tree = AudioTree(np.zeros((4, 2, 44_100), dtype=np.float32), 44_100)

    # Move the whole tree onto the default JAX device (a GPU or TPU when present).
    device_tree = jax.device_put(audio_tree)
    print(isinstance(device_tree.waveform, jax.Array))

    # Pull the whole tree back to host NumPy in one call.
    host_tree = jax.device_get(device_tree)
    print(isinstance(host_tree.waveform, np.ndarray))
    print(host_tree.sample_rate)   # the static field is unchanged

.. testoutput::

    True
    True
    44100

When you're feeding a Grain data loader rather than moving a single tree, keep the
transfers off the training thread with :func:`grain.experimental.device_put`, which
prefetches whole batches onto the accelerator as you iterate — see
:ref:`streaming-device-put`.

.. tip::
   You rarely need to ``device_put`` a tree just to compute loudness on an
   accelerator: :meth:`~audiotree.core.AudioTree.replace_lufs` and
   :meth:`~audiotree.core.AudioTree.normalize_lufs` take a ``backend=`` argument
   (see `Choosing a compute backend`_) that runs the kernel on the chosen device
   and returns loudness in the waveform's own array library.

Metadata and Filepaths
-----------------------

AudioTree supports storing metadata and filepath information.

Storing Filepaths
~~~~~~~~~~~~~~~~~

When creating AudioTree objects, you can associate them with source files:

.. testcode::

    # Single filepath
    audio_tree = AudioTree.create(
        np.zeros((1, 44_100)),
        44_100,
        filepaths="audio.wav"
    )
    print(audio_tree.filepath)

    # Multiple filepaths for batched data
    audio_tree = AudioTree.create(
        np.zeros((3, 1, 44_100)),
        44_100,
        filepaths=["a.wav", "b.wav", "c.wav"]
    )
    print(audio_tree.filepath)

.. testoutput::

    ['audio.wav']
    ['a.wav', 'b.wav', 'c.wav']

Understanding Metadata
~~~~~~~~~~~~~~~~~~~~~~

The ``metadata`` field is special - it's a pytree node (``pytree_node=True``), meaning it participates
in JAX tree operations like batching and concatenation. This is different from ``sample_rate``, which
is marked as ``pytree_node=False`` and remains constant across operations.

Metadata should contain array-like data with a batch dimension:

.. testcode::

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

    # Batch the two trees together - metadata is concatenated too
    combined = AudioTree.batch([tree1, tree2])

    print(combined.waveform.shape)                  # Batched from 2+2
    print(combined.metadata["energy"].shape)        # Metadata was concatenated
    print(combined.metadata["onset_times"].shape)   # 2D metadata concatenated along batch dim
    print(combined.sample_rate)                     # Sample rate stays the same (not a pytree node)

.. testoutput::

    (4, 1, 44100)
    (4,)
    (4, 2)
    44100

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

.. testcode::

    audio_tree = AudioTree.from_file("input.wav", sample_rate=44_100)

    # The file format is inferred from the extension.
    audio_tree.write("output.wav")

    # Control the encoding with soundfile passthroughs.
    audio_tree.write("output_24bit.wav", subtype="PCM_24")
    audio_tree.write("output.flac")  # FLAC inferred from the ".flac" extension

    # write() returns the Path it wrote.
    print(audio_tree.write("output.wav"))

.. testoutput::

    output.wav

Indexing or iterating makes it easy to write every item of a batch (each item is a
batch of 1, exactly what ``write`` requires):

.. testcode::

    batch = AudioTree(np.random.randn(8, 2, 44_100), 44_100)
    for i, item in enumerate(batch):
        item.write(f"item_{i}.wav")

.. note::
   ``write`` operates on a single item by design. Calling it on a multi-item batch
   raises an ``AssertionError``. To write a whole batch in one call — with a manifest
   of per-item metadata — reach for :class:`~audiotree.writer.AudioWriter`, covered in
   the :ref:`writer` chapter.

Next Steps
----------

With the AudioTree object in hand, the next chapter builds data loaders that stream
AudioTrees straight from your audio files:

- :ref:`sources` - Load audio from directories into Grain data pipelines
- :ref:`transform_chaining` - Chain augmentations onto a data pipeline
- :ref:`writer` - Write prepared AudioTrees back to disk
- :class:`~audiotree.core.AudioTree` - Full API reference
