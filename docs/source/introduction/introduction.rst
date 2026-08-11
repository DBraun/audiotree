.. role:: python(code)
     :language: python
     :class: highlight

.. :tocdepth: 2

.. _introduction:

Introduction to AudioTree
=========================

This guide is about the :class:`~audiotree.AudioTree` object itself.

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
    # audio.wav is 4 seconds so the offset/duration example below reads real
    # audio; from_file raises on an offset at or past the end of the file.
    soundfile.write(_audio_path, np.zeros((4 * 44_100, 2), dtype=np.float32), 44_100)
    soundfile.write(_input_path, np.zeros((44_100, 2), dtype=np.float32), 44_100)

    # Run all examples from inside the temp dir so relative output paths land there.
    os.chdir(_doc_dir)

Basic Instantiation
-------------------

The :class:`~audiotree.AudioTree` class is the central data structure in the library.
It stores audio as arrays with a consistent shape convention: ``(Batch, Channels, Samples)``.
This "channels-first" format is familiar to PyTorch and librosa users.
Note that JAX and NNX are usually "channels-last": ``(Batch, Samples, Channels)``.

Creating from NumPy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create an AudioTree directly from NumPy or
:mod:`JAX NumPy <jax.numpy>` arrays:

.. testcode::

    import numpy as np
    from audiotree import AudioTree

    # Create from 3D array (B, C, T)
    sample_rate = 44_100
    waveform = np.zeros((4, 2, 88_200))  # 4 batches, 2 channels, 2 seconds
    audio = AudioTree(waveform, sample_rate)

    print(audio.waveform.shape)
    print(audio.sample_rate)

.. testoutput::

    (4, 2, 88200)
    44100

Automatic Dimensionality Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.AudioTree.create` method automatically handles arrays of different dimensions:

.. testcode::

    # From 1D array (just samples)
    audio_1d = np.zeros(44_100)
    audio = AudioTree.create(audio_1d, 44_100)
    print(audio.waveform.shape)  # Automatically adds batch and channel dims

    # From 2D array (channels × samples)
    audio_2d = np.zeros((2, 44_100))
    audio = AudioTree.create(audio_2d, 44_100)
    print(audio.waveform.shape)  # Automatically adds batch dimension

.. testoutput::

    (1, 1, 44100)
    (1, 2, 44100)

Rank 3 is where the normalization stops. A waveform with four or more axes is
rejected with a ``ValueError`` rather than accepted: ``create`` broadcasts
``filepath`` / ``source`` over the leading axis, and at rank 4 there is no single
right answer for what that axis means. Build the tree at rank 3 and add a
mini-batch axis afterwards with
:meth:`~audiotree.AudioTree.reshape_mini_batches`.

Loading Audio from Files
------------------------

:meth:`~audiotree.AudioTree.from_file` reads a file (or an excerpt of one)
into a batch-of-1 tree:

.. testcode::

    # Load an audio file
    audio = AudioTree.from_file("audio.wav", sample_rate=44_100)

    # Load with specific offset and duration
    audio = AudioTree.from_file(
        "audio.wav",
        sample_rate=44_100,
        offset=1.0,  # Start at 1 second
        duration=2.5,  # Load 2.5 seconds
        mono=False,  # Keep stereo
    )

    # Load with custom extras
    audio = AudioTree.from_file(
        "audio.wav",
        sample_rate=44_100,
        extras={
            "features_4d": np.zeros((1, 4)),  # intentionally give batch axis of 1
        },
    )

    # The filepath is recorded as per-item provenance (see "Extras and
    # Provenance" below) and read back through the property.
    print(audio.filepath)
    print(audio.extras["features_4d"])

.. testoutput::

    ['audio.wav']
    [[0. 0. 0. 0.]]

Manipulating AudioTree Objects
------------------------------

Accessing Properties
~~~~~~~~~~~~~~~~~~~~

.. testcode::

    audio = AudioTree.create(np.ones((2, 2, 44_100)), 44_100)

    # Core properties
    print(audio.waveform.shape)
    print(audio.sample_rate)

    # The extras dict holds custom per-item data (empty by default)
    print(audio.extras)

.. testoutput::

    (2, 2, 44100)
    44100
    {}

Creating Modified Copies
~~~~~~~~~~~~~~~~~~~~~~~~

AudioTree is immutable. Use :meth:`~audiotree.AudioTree.replace` to create modified copies:

.. testcode::

    # Original audio with batch size 2
    audio = AudioTree(np.ones((2, 2, 44_100)), 44_100)

    # Create a new AudioTree with modified audio data
    quieter_audio = audio.replace(waveform=audio.waveform * 0.5)

    print(np.allclose(audio.waveform[0, 0, 0], 1.0))  # Original unchanged
    print(np.allclose(quieter_audio.waveform[0, 0, 0], 0.5))

.. testoutput::

    True
    True

Computing Loudness
~~~~~~~~~~~~~~~~~~

AudioTree can compute loudness in LUFS (Loudness Units Full Scale) for each item in the batch:

.. testcode::

    # Create audio with 4 batches and compute loudness for each
    audio = AudioTree(np.full((4, 2, 44_100), 0.1), 44_100)
    audio_with_lufs = audio.replace_lufs()

    print(audio_with_lufs.lufs.shape)  # One loudness value per batch item
    print(audio_with_lufs.lufs)  # LUFS values for the constant 0.1 signal

.. testoutput::

    (4,)
    [-43.25 -43.25 -43.25 -43.25]

.. note::
   The loudness meter follows ITU-R BS.1770-4, which defines channel weights for
   mono, stereo, and 5.0/5.1 surround (left, right, center, left surround, right
   surround). A tree with more than 5 channels raises an error during loudness
   computation.

Choosing a device and an engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Where* the measurement runs and *which* kernel runs it are separate choices, so
:meth:`~audiotree.AudioTree.replace_lufs` takes two separate keyword-only
arguments:

* ``device`` — an XLA platform name (``"cpu"``, ``"gpu"``, ``"tpu"``, mirroring
  ``jax.jit``'s ``backend``) or a :class:`jax.Device`. ``None`` (the default)
  leaves the waveform where it is.
* ``engine`` — ``"numpy"`` for the exact ITU-R BS.1770 IIR meter (CPU-only), or
  ``"jax"`` for the vmapped ``jaxloudnorm`` kernel with FIR-approximated
  K-weighting. ``None`` (the default) follows the waveform's own array library,
  except that a non-CPU ``device`` implies ``"jax"``, the only engine that can
  run there.

For a NumPy waveform the default path measures one batch item at a time on the
CPU. When the batch is large, ``engine="jax"`` runs the whole batch through one
vmapped kernel instead, and ``device="gpu"`` (or ``"tpu"``) additionally moves
that work onto an accelerator:

.. testcode::

    big_batch = AudioTree(np.full((8, 2, 44_100), 0.1), 44_100)

    # engine="jax" picks the vmapped kernel; device says where to run it. Swap in
    # "gpu" or "tpu" when you have one; "cpu" always works and is used here so
    # the example runs anywhere. Note that device="cpu" on its own would *not*
    # switch engines — it only pins the device.
    loud = big_batch.replace_lufs(device="cpu", engine="jax")

    # The returned lufs always matches the waveform's array library (NumPy here),
    # so you never need a manual jax.device_put / jax.device_get round-trip.
    print(type(loud.lufs).__module__)
    print(loud.lufs.shape)

.. testoutput::

    numpy
    (8,)

:meth:`~audiotree.AudioTree.normalize_lufs` computes loudness internally, so
it accepts the same ``device`` and ``engine`` arguments.

Keeping cached loudness in sync
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once ``lufs`` (and ``lufs_windows``) are filled, they are cached on the tree. A
bare :meth:`~audiotree.AudioTree.replace` that swaps in a new ``waveform``
leaves the *old* loudness in place, making it no longer match the audio. Follow
such an edit with :meth:`~audiotree.AudioTree.clear_lufs` (shorthand for
``replace(lufs=None, lufs_windows=None)``) so the next
:meth:`~audiotree.AudioTree.replace_lufs` recomputes it:

.. testcode::

    audio = AudioTree(np.full((2, 1, 16_000), 0.1, np.float32), 16_000).replace_lufs()

    # WRONG: lufs still describes the original signal after halving the audio.
    stale = audio.replace(waveform=audio.waveform * 0.5)
    print(np.allclose(stale.lufs, audio.lufs))  # True -> stale

    # RIGHT: clear the cache, then recompute for the new, quieter audio.
    fixed = audio.replace(waveform=audio.waveform * 0.5).clear_lufs()
    fixed = fixed.replace_lufs()
    print(bool(fixed.lufs[0] < audio.lufs[0] - 5.0))  # ~6 dB quieter

.. testoutput::

    True
    True

The built-in transforms handle this for you:
:func:`~audiotree.transforms.volume_norm` /
:func:`~audiotree.transforms.volume_change` shift the cached loudness by the gain
they apply, phase transforms preserve it under ``keep_lufs=True``, and length- or
channel-changing transforms invalidate it.

Resampling Audio
~~~~~~~~~~~~~~~~

To change the sample rate of audio, use the :meth:`~audiotree.AudioTree.resample` method:

.. testcode::

    # Original audio at 44.1 kHz
    audio = AudioTree(np.ones((1, 2, 44_100)), 44_100)

    # Resample to 48 kHz
    resampled_audio = audio.resample(48_000)

    print(audio.sample_rate)  # Original unchanged
    print(resampled_audio.sample_rate)
    print(resampled_audio.waveform.shape)  # Audio data is resampled

.. testoutput::

    44100
    48000
    (1, 2, 48000)

Converting Channels
~~~~~~~~~~~~~~~~~~~

Use :meth:`~audiotree.AudioTree.to_mono` and :meth:`~audiotree.AudioTree.to_stereo`
to change the channel layout. ``to_mono`` takes a ``strategy``:

.. testcode::

    audio = AudioTree(np.random.randn(4, 2, 44_100), 44_100)

    # "average" (default) mixes all channels down
    mono = audio.to_mono()
    print(mono.waveform.shape)

    # "left" / "right" select one channel of stereo audio
    left = audio.to_mono("left")
    right = audio.to_mono("right")
    print(np.allclose(left.waveform[:, 0], audio.waveform[:, 0]))

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
   recomputed on the next :meth:`~audiotree.AudioTree.replace_lufs`.

Indexing and Iterating Batches
------------------------------

An AudioTree behaves like a sequence over its leading (batch) axis. Indexing,
slicing, ``len()``, and iteration keep every field — ``waveform``, ``codes``,
``latents``, the ``extras`` arrays, and the encoded provenance — rank-aligned.

.. testcode::

    audio = AudioTree(np.random.randn(16, 2, 44_100), 44_100)

    print(len(audio))  # number of batch items

    # Integer indexing keeps the batch axis (a batch of 1)
    print(audio[0].waveform.shape)

    # Slices select a sub-batch
    print(audio[4:8].waveform.shape)

    # Negative indices work too
    print(audio[-1].waveform.shape)

.. testoutput::

    16
    (1, 2, 44100)
    (4, 2, 44100)
    (1, 2, 44100)

Because AudioTree implements ``__iter__`` it is a proper
``collections.abc.Iterable``. Iterating yields one batch-of-1 AudioTree per item:

.. testcode::

    for item in audio:
        assert item.waveform.shape[0] == 1
        # process or write a single example...

Pairing ``__iter__`` with ``len()`` means progress bars work out of the box —
``tqdm`` reads ``len()`` to size the bar automatically (``tqdm`` is an optional
dependency, so this snippet is illustrative rather than executed):

.. skip-snippet-exec: needs the optional ``tqdm`` dependency.

.. code-block:: python

    import tqdm

    for item in tqdm.tqdm(audio):  # shows a 0/16 ... 16/16 bar
        ...

To reassemble a batch from individual items, use
:meth:`~audiotree.AudioTree.batch`:

.. testcode::

    items = [audio[i] for i in range(len(audio))]
    rebuilt = AudioTree.batch(items)
    print(rebuilt.waveform.shape)

.. testoutput::

    (16, 2, 44100)

Batching Operations
-------------------

AudioTree provides several methods for working with batches of audio.

Creating Mini-Batches
~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.AudioTree.reshape_mini_batches` method adds a mini-batch axis,
and :meth:`~audiotree.AudioTree.flatten_mini_batches` removes it again:

.. testcode::

    # Start with 12 audio samples
    audio = AudioTree(np.zeros((12, 1, 44_100)), 44_100)
    print(audio.waveform.shape)

    # Reshape into mini-batches of size 3
    audio_batched = audio.reshape_mini_batches(3)
    print(audio_batched.waveform.shape)  # 4 mini-batches, each with 3 samples

    # Flatten back to original shape
    audio_unbatched = audio_batched.flatten_mini_batches()
    print(audio_unbatched.waveform.shape)  # Back to original

.. testoutput::

    (12, 1, 44100)
    (4, 3, 1, 44100)
    (12, 1, 44100)

.. note::
   AudioTree methods operate on mini-batched trees directly. Methods like
   :meth:`~audiotree.AudioTree.replace_lufs`,
   :meth:`~audiotree.AudioTree.normalize_lufs`,
   :meth:`~audiotree.AudioTree.to_mono`, :meth:`~audiotree.AudioTree.to_stereo`,
   and :meth:`~audiotree.AudioTree.resample` treat *all* leading axes as batch
   axes, so you can call them on a ``(num_mini_batches, mini_batch_size, C, T)`` tree
   without flattening first. Per-item results follow the leading shape — e.g.,
   ``lufs`` comes back shaped ``(num_mini_batches, mini_batch_size)``.

   The methods whose contract is *per item* refuse a mini-batched tree instead
   of quietly reinterpreting the leading axis. :meth:`~audiotree.AudioTree.filter`,
   :meth:`~audiotree.AudioTree.write`, and
   :meth:`~audiotree.AudioTree.reshape_mini_batches` itself all raise
   ``ValueError`` at rank 4; call
   :meth:`~audiotree.AudioTree.flatten_mini_batches` first. The provenance
   properties do keep working: at rank 4,
   :attr:`~audiotree.AudioTree.filepath` and
   :attr:`~audiotree.AudioTree.source` return one list per mini-batch —
   ``[["0.wav", "1.wav", "2.wav"], ["3.wav", …], …]`` — rather than one flat
   list.

Splitting into Multiple Trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~audiotree.AudioTree.split` method splits a batch into separate AudioTree objects:

.. testcode::

    # Start with 12 audio samples
    audio = AudioTree(np.zeros((12, 1, 44_100)), 44_100)

    # Split into 2 separate AudioTree objects
    halves = audio.split(2)
    print(len(halves))
    print(halves[0].waveform.shape)  # First half
    print(halves[1].waveform.shape)  # Second half

.. testoutput::

    2
    (6, 1, 44100)
    (6, 1, 44100)

Filtering Batch Items
~~~~~~~~~~~~~~~~~~~~~

:meth:`~audiotree.AudioTree.filter` keeps the batch items that satisfy a
predicate:

.. testcode::

    # Generate uniform noise and scale it to different levels
    np.random.seed(42)  # For reproducible results
    noise = np.random.uniform(-1, 1, (1, 44_100))

    # Create AudioTree with different loudness levels
    waveform = np.array(
        [
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
            noise * 1.0,  # Full scale
        ]
    )

    audio = AudioTree(waveform, 44_100).replace_lufs()


    # Filter to keep only audio louder than -20 LUFS
    def keep_loud_audio(audio):
        return audio.lufs[0] > -20.0


    filtered_audio = audio.filter(keep_loud_audio)

    # 9 batches remain (excluding silent and very quiet ones)
    print(filtered_audio.waveform.shape)
    # LUFS values for remaining batches (rounded for display)
    print(np.round(filtered_audio.lufs, 2))

.. testoutput::

    (9, 1, 44100)
    [-15.65 -12.15  -9.65  -7.69  -6.11  -4.75  -3.61  -2.56  -1.65]

The predicate is called with one batch item at a time, so ``filter`` requires a
rank-3 tree; on a mini-batched one it raises ``ValueError`` rather than filtering
whole mini-batches. Call
:meth:`~audiotree.AudioTree.flatten_mini_batches` first.

Processing Mini-Batches with nnx.scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a batch is too large to run through a network in one pass, reshape it into
mini-batches and run an :class:`nnx.Module <flax.nnx.Module>` over them
sequentially with Flax's :func:`nnx.scan <flax.nnx.scan>`.
Pass the module as an argument and broadcast it with
``in_axes=(None, 0)``: the model is the same for every mini-batch, while the
audio is scanned over its leading (mini-batch) axis:

.. testcode::

    import jax.numpy as jnp
    from flax import nnx


    class LearnableGain(nnx.Module):
        """A stand-in for a real network: one trainable gain."""

        def __init__(self):
            self.gain = nnx.Param(jnp.ones((1,)))

        def __call__(self, audio: AudioTree) -> AudioTree:
            waveform = self.gain[...][:, None, None] * audio.waveform
            # The gain changes loudness, so drop any cached lufs.
            return audio.replace(waveform=waveform).clear_lufs()


    model = LearnableGain()
    model.gain[...] = 0.5 * model.gain[...]  # pretend training halved the gain


    # Scan the module over the mini-batch axis
    @nnx.scan(in_axes=(None, 0), out_axes=0)
    def process_mini_batches(model: LearnableGain, audio: AudioTree) -> AudioTree:
        assert audio.waveform.ndim == 3  # each step sees (batch, channels, samples)
        return model(audio)


    # Create AudioTree with 12 samples
    audio = AudioTree(np.ones((12, 1, 44_100)), 44_100)

    # Create mini-batches of size 3
    audio_batched = audio.reshape_mini_batches(3)
    print(audio_batched.waveform.shape)  # 4 mini-batches of size 3

    # Process all mini-batches sequentially
    processed_batched = process_mini_batches(model, audio_batched)
    print(processed_batched.waveform.shape)  # Still mini-batched

    # Flatten back to original batch dimension
    full_batch = processed_batched.flatten_mini_batches()
    print(full_batch.waveform.shape)  # Back to original shape
    print(np.allclose(full_batch.waveform, 0.5))  # The gain was applied

.. testoutput::

    (4, 3, 1, 44100)
    (4, 3, 1, 44100)
    (12, 1, 44100)
    True

Working with JAX PyTrees
------------------------

AudioTree is a JAX Pytree, so JAX's
`tree <https://docs.jax.dev/en/latest/jax.tree.html>`_ operations apply to it
directly.

Concatenating Trees with jax.tree.map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use :func:`jax.tree.map` to combine multiple AudioTree objects:

.. testcode::

    import jax

    # Create an AudioTree with 4 batch items
    audio = AudioTree(np.zeros((4, 1, 44_100)), 44_100)

    # Create a list of three identical trees
    trees = [audio, audio, audio]

    # Concatenate along the batch dimension
    big_audio = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *trees)

    print(big_audio.waveform.shape)  # 3 × 4 = 12 batches

.. testoutput::

    (12, 1, 44100)

This concatenating ``tree.map`` is exactly what
:meth:`~audiotree.AudioTree.batch` does for you: it concatenates every leaf
along axis 0 (treating each AudioTree as one leaf), so a list of trees collapses
into the identical batched tree. The array library is preserved — NumPy leaves in,
NumPy leaves out; JAX in, JAX out — so batching never forces a device round-trip.
It is the same function you pass to Grain as ``batch_fn=AudioTree.batch`` when
building a data loader. An empty sequence raises ``ValueError``.

.. testcode::

    same_audio = AudioTree.batch(trees)

    print(same_audio.waveform.shape)
    print(np.array_equal(same_audio.waveform, big_audio.waveform))

.. testoutput::

    (12, 1, 44100)
    True

Nested Structures
~~~~~~~~~~~~~~~~~

AudioTrees can sit anywhere inside a larger pytree (a dict, a list, or both):

.. testcode::

    # Create different audio trees
    input_audio = AudioTree(np.ones((2, 1, 1000)), 44_100)
    target_audio = AudioTree(np.zeros((2, 1, 1000)), 44_100)

    # Organize in a nested structure
    batch = {
        "input": input_audio,
        "target": target_audio,
        "augmented": [input_audio, target_audio],
    }


    # Apply transformations to all trees in the structure
    def scale_audio(audio: AudioTree):
        return audio.replace(waveform=audio.waveform * 0.5)


    scaled_batch = jax.tree.map(
        scale_audio, batch, is_leaf=lambda x: isinstance(x, AudioTree)
    )

    print(scaled_batch["input"].waveform[0, 0, 0])  # Scaled from 1.0
    print(scaled_batch["augmented"][1].waveform[0, 0, 0])  # Scaled from 0.0 (remains 0)

.. testoutput::

    0.5
    0.0

Tree Flattening and Unflattening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX can flatten AudioTree objects for operations requiring flat arrays:

.. testcode::

    audio = AudioTree(np.ones((1, 2, 1000)), 44_100)

    # Flatten the audio into leaves and structure
    leaves, treedef = jax.tree.flatten(audio)

    # Modify leaves if needed...
    # Then reconstruct the audio
    reconstructed = jax.tree.unflatten(treedef, leaves)

    print(np.array_equal(reconstructed.waveform, audio.waveform))

.. testoutput::

    True

Moving a Tree Between Devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because an AudioTree is a pytree, :func:`jax.device_put` and :func:`jax.device_get`
move *every* array leaf at once — you never touch ``waveform``, ``lufs``, and each
``extras`` array one by one. The static ``sample_rate`` and the tree structure
are left untouched.

.. testcode::

    import jax

    audio = AudioTree(np.zeros((4, 2, 44_100), dtype=np.float32), 44_100)

    # Move the whole tree onto the default JAX device (a GPU or TPU when present).
    device_audio = jax.device_put(audio)
    print(isinstance(device_audio.waveform, jax.Array))

    # Pull the whole tree back to host NumPy in one call.
    host_audio = jax.device_get(device_audio)
    print(isinstance(host_audio.waveform, np.ndarray))
    print(host_audio.sample_rate)  # the static field is unchanged

.. testoutput::

    True
    True
    44100

``device_put`` and ``device_get`` move a tree; :attr:`~audiotree.AudioTree.backend`
and :attr:`~audiotree.AudioTree.device` ask where it currently *is*, without
reaching into a leaf and hoping the rest agree:

.. testcode::

    print(audio.backend, audio.device)
    print(device_audio.backend, device_audio.device is not None)

.. testoutput::

    numpy None
    jax True

That matters more than it looks, because a tree does not stay homogeneous by
itself. A transform from the NumPy namespace applied to a JAX tree converts the
fields it touches, so ``audiotree.transforms.trim`` on a JAX tree hands back a
NumPy ``waveform`` — which is then re-uploaded on every ``jax.jit`` call, quietly
costing a host round trip per step. ``backend`` reports ``"mixed"`` when the
leaves disagree, so it is safe to log; ``device`` raises instead, because there is
no honest single answer.

When you're feeding a Grain data loader rather than moving a single tree, keep the
transfers off the training thread with :func:`grain.experimental.device_put`, which
prefetches whole batches onto the accelerator as you iterate — see
:ref:`streaming-device-put`.

.. tip::
   You rarely need to ``device_put`` a tree just to compute loudness on an
   accelerator: :meth:`~audiotree.AudioTree.replace_lufs` and
   :meth:`~audiotree.AudioTree.normalize_lufs` take ``device=`` and
   ``engine=`` arguments (see `Choosing a device and an engine`_) that run the
   kernel where you ask and return loudness in the waveform's own array library.

Extras and Provenance
---------------------

AudioTree keeps two per-item containers deliberately apart: ``extras``, the
dict of *your* payload arrays, and ``_metadata``, a private library-managed
container recording where each item came from.

Storing Filepaths
~~~~~~~~~~~~~~~~~

When creating AudioTree objects, you can associate them with source files:

.. testcode::

    # Single filepath
    audio = AudioTree.create(np.zeros((1, 44_100)), 44_100, filepath="audio.wav")
    print(audio.filepath)

    # Multiple filepaths for batched data
    audio = AudioTree.create(
        np.zeros((3, 1, 44_100)), 44_100, filepath=["a.wav", "b.wav", "c.wav"]
    )
    print(audio.filepath)

.. testoutput::

    ['audio.wav']
    ['a.wav', 'b.wav', 'c.wav']

Provenance: ``filepath``, ``source`` and ``offset``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``filepath``, ``source`` (the group name a balanced dataset drew an item
from) and ``offset`` (where in the source file the excerpt starts, in seconds)
are *provenance* — data about the audio, not payload that trains with it. They
live in ``AudioTree._metadata``, a library-internal container (private, as the
underscore says): pass ``filepath=`` / ``source=`` / ``offset=`` to
:meth:`~audiotree.AudioTree.create` (or let ``from_file`` and the dataset
builders stamp them) and read the values back through the ``.filepath``,
``.source`` and ``.offset`` properties. Internally each string is encoded as a
fixed-width integer array, which is what lets provenance survive ``jax.jit``,
batching, and device transfers like every other leaf; ``offset`` is a plain
float array and needs no decoding.

The container's schema is closed: it holds ``filepath``, ``source`` and
``offset`` and nothing else, and it serializes under the on-disk name
``metadata``. A ``metadata`` node read off disk that contains any other key is
rejected by name — the same strict-validation stance the manifest readers take
with unknown columns. Everything user-shaped belongs in ``extras``, and
``extras`` is entirely yours: the library plants no keys of its own there.

Understanding Extras
~~~~~~~~~~~~~~~~~~~~

The ``extras`` field is special. It's a pytree node (``pytree_node=True``), meaning it participates
in JAX tree operations like batching and concatenation. This is different from ``sample_rate``, which
is marked as ``pytree_node=False`` and remains constant across operations.

Extras should contain array-like data with a batch dimension:

.. testcode::

    # Create trees with array extras that can be batched
    audio1 = AudioTree.create(
        np.zeros((2, 1, 44_100)),
        44_100,
        extras={
            "energy": np.array([0.8, 0.9]),  # Shape (2,) matching batch size
            "onset_times": np.array([[0.1, 0.2], [0.15, 0.25]]),  # Shape (2, 2)
        },
    )

    audio2 = AudioTree.create(
        np.zeros((2, 1, 44_100)),
        44_100,
        extras={
            "energy": np.array([0.7, 0.85]),
            "onset_times": np.array([[0.12, 0.22], [0.18, 0.28]]),
        },
    )

    # Batch the two trees together - extras are concatenated too
    combined = AudioTree.batch([audio1, audio2])

    print(combined.waveform.shape)  # Batched from 2+2
    print(combined.extras["energy"].shape)  # Extras were concatenated
    print(combined.extras["onset_times"].shape)  # 2D extras concatenated along batch dim
    print(combined.sample_rate)  # Sample rate stays the same (not a pytree node)

.. testoutput::

    (4, 1, 44100)
    (4,)
    (4, 2)
    44100

Adding Extras to an Existing Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AudioTree`` is immutable, so you can't assign into ``extras`` in place. Use
:meth:`~audiotree.AudioTree.replace_extras` to merge new entries — it's
syntactic sugar for ``audio.replace(extras={**audio.extras, **kwargs})``. Keys
you pass overwrite same-named existing keys, everything else is kept, and the
original tree is untouched:

.. testcode::

    audio = AudioTree.create(
        np.zeros((2, 1, 44_100)),
        44_100,
        extras={"energy": np.array([0.8, 0.9])},
    )
    tagged = audio.replace_extras(onsets=np.array([[0.1], [0.2]]))

    print(sorted(tagged.extras.keys()))
    print(sorted(audio.extras.keys()))  # the original is unchanged

.. testoutput::

    ['energy', 'onsets']
    ['energy']

Writing Audio to Disk
---------------------

Writing a Single File
~~~~~~~~~~~~~~~~~~~~~~

:meth:`~audiotree.AudioTree.write` saves one item to an audio file via
`soundfile <https://python-soundfile.readthedocs.io/>`_ — the inverse of
:meth:`~audiotree.AudioTree.from_file`. The tree must contain exactly one item
(``batch_size == 1``), so index or iterate a batch first. There is no sample-rate
argument: it uses ``self.sample_rate``, so call
:meth:`~audiotree.AudioTree.resample` beforehand to change it.

.. testcode::

    audio = AudioTree.from_file("input.wav", sample_rate=44_100)

    # The file format is inferred from the extension.
    audio.write("output.wav")

    # Control the encoding with soundfile passthroughs.
    audio.write("output_24bit.wav", subtype="PCM_24")
    audio.write("output.flac")  # FLAC inferred from the ".flac" extension

    # write() returns the Path it wrote.
    print(audio.write("output.wav"))

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
   raises a ``ValueError``, as does calling it on a mini-batched (rank-4) tree or
   on a token-only tree that carries ``codes``/``latents`` but no ``waveform`` —
   all three by name, rather than as a shape complaint from soundfile. To write a
   whole batch in one call — with a manifest of per-item columns — reach for
   :class:`~audiotree.writer.AudioWriter`, covered in the :ref:`writer` chapter.

Next Steps
----------

With the AudioTree object in hand, the next chapter builds data loaders that stream
AudioTrees straight from your audio files:

- :ref:`sources` - Load audio from directories into Grain data pipelines
- :ref:`transform_chaining` - Chain augmentations onto a data pipeline
- :ref:`writer` - Write prepared AudioTrees back to disk
- :class:`~audiotree.AudioTree` - Full API reference
