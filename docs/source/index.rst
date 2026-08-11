AudioTree documentation
=======================

**AudioTree** is an audio data loading and augmentation library supporting PyTorch
and with extra features for `JAX <https://jax.readthedocs.io/en/latest/>`_. Its
central type, :class:`~audiotree.AudioTree`, holds a batch of audio as a
`pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_: waveform, sample
rate, loudness, and more, including your own per-item arrays. Augmentations are
defined in NumPy and JAX backends (NumPy for CPU dataloader workers, JAX for jitted
on-device training steps). JAX and Flax are hard requirements, but the training
loop can be anything using NumPy. The source code is on
`GitHub <https://github.com/DBraun/audiotree>`_.

AudioTree can be installed with pip:

.. code-block:: bash

   pip install audiotree

Concretely, :class:`~audiotree.AudioTree` is a `flax.struct.dataclass`_ where
every array carries the batch as its leading axis.

- ``waveform``: the time-domain audio, ``(B, C, T)`` (batch, channels, samples)
- ``sample_rate``: a shared scalar ``int``; the one field that is *not* batched
- ``lufs`` / ``lufs_windows``: loudness, filled on demand, ``(B,)`` / ``(B, Windows)``
- ``pitch``, ``velocity``, ``note_duration``: optional per-item labels (MIDI pitch/velocity, note length), ``(B,)`` each
- ``codes`` / ``latents``: neural-codec tokens or latent embeddings, ``(B, ...)``
- ``extras``: a dict of your own ``(B, ...)`` arrays that batch with the audio
- ``_metadata``: library-internal provenance (``filepath``, ``source``, and the excerpt ``offset``); read it through the ``.filepath``, ``.source`` and ``.offset`` properties

AudioTree integrates with `Grain`_ to provide complete data pipelines. Load audio from directories,
apply balanced sampling across groups, and chain augmentations (e.g., :meth:`~audiotree.transforms.volume_norm`):

.. testsetup::

    # Hidden setup: stand in for the "/data/speech" and "/data/music" directories
    # below with two temp dirs of small synthetic WAVs, so this example runs.
    import os
    import tempfile
    import numpy as np
    import soundfile

    _g = np.random.default_rng(0)
    _speech_dir, _music_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
    for _d in (_speech_dir, _music_dir):
        for _i in range(3):
            # 30-second files, so the 10-second excerpts below have real
            # room to draw a random offset from.
            soundfile.write(
                os.path.join(_d, f"{_i}.wav"),
                (0.1 * _g.standard_normal((30 * 44_100, 2))).astype(np.float32),
                44_100,
            )

.. testcode::

    from collections import Counter

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, stereo
    import grain

    # Create dataset with balanced sampling across groups and random sections within files
    ds = create_balanced_audio_dataset(
        sources={"speech": [_speech_dir], "music": [_music_dir]},
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=10.0,
    )

    # Chain transforms using Grain's Dataset API
    ds = ds.seed(42)
    ds = ds.map(stereo())
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Convert to iterable and batch
    iter_ds = ds.to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
    iter_ds = iter_ds.batch(32, batch_fn=AudioTree.batch)

    # Access batched AudioTrees
    batch: AudioTree = next(iter(iter_ds))
    print("shape:", batch.waveform.shape)  # (32, channels, 441000)

    # The mix follows the weights: roughly 70% speech, 30% music per batch.
    print(Counter(batch.source))

    # The source file each item was drawn from.
    print("filepath:", batch.filepath[:3])

    # Each item records where its excerpt started in its source file (seconds).
    print("offset:", batch.offset[:3].round(2))

    # `volume_norm` above caused LUFS to be calculated.
    print("LUFS:", batch.lufs[:3].round(2))

    # The whole pipeline ran on host NumPy arrays; nothing touched a device.
    print("backend:", batch.backend)

.. testoutput::
    :options: +ELLIPSIS

    shape: (32, 2, 441000)
    Counter({'speech': 23, 'music': 9})
    filepath: ['...2.wav', '...0.wav', '...1.wav']
    offset: [16.91  0.52 18.89]
    LUFS: [-16.16 -16.83 -18.4 ]
    backend: numpy

For training in PyTorch, the pipeline above still applies. The loaders and the
NumPy transform backend work on plain NumPy arrays end to end, and batches
arrive as float32, C-contiguous arrays in the ``(batch, channels, samples)``
layout PyTorch audio models expect, so the hand-off to a torch model is a
zero-copy ``torch.from_numpy``:

.. skip-snippet-exec: torch is not a dependency of audiotree.

.. code-block:: python

    for batch in iter_ds:
        waveform = torch.from_numpy(batch.waveform)  # zero-copy, (B, C, T)
        loss = model(waveform)

How it compares
---------------

        AudioTree is modeled on Descript's `AudioTools`_ (:class:`~audiotree.AudioTree`
        plays the role of ``AudioSignal``), but it is a pytree rather than a mutable
        object. Transforms return new trees, everything is batched-first, and the JAX
        backend traces cleanly under `jax.jit`_ and `jax.vmap`_. Against **torchaudio**,
        the difference is scope in both directions. Torchaudio ships feature
        extraction, model zoos, and codecs, none of which are here. AudioTree instead
        ships the data-loading and augmentation layer that torchaudio leaves to
        ``torch.utils.data``: Grain sources, balanced and windowed samplers, and
        on-disk dataset writers. Rather than a torch-specific integration layer, the
        hand-off to a torch model is the plain NumPy arrays shown above.

The guides are meant to be read in order. **Getting started** walks the main
path (the :class:`~audiotree.AudioTree` object, loading audio, augmenting it,
and writing datasets back to disk), and **Going further** collects the deeper
topics (balanced and windowed sampling, dict batches, neural codecs,
command-line or YAML configuration, and multiprocessing).

Content
--------------------------
.. toctree::
   :maxdepth: 1
   :caption: Getting started

   introduction/introduction
   introduction/sources
   introduction/transform_chaining
   introduction/transforms
   introduction/writer

.. toctree::
   :maxdepth: 1
   :caption: Going further

   introduction/balanced_datasets
   introduction/windowed_datasets
   introduction/argbind_guide
   introduction/multiprocessing
   introduction/codecs
   introduction/dict_batches

.. toctree::
   :maxdepth: 1
   :caption: AudioTree API

   audiotree_api/core
   audiotree_api/sources
   audiotree_api/transforms
   audiotree_api/writer
   audiotree_api/audio_source
   audiotree_api/tree

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Acknowledgments
---------------

AudioTree is inspired by `AudioTools`_. Thank you!

Citation
---------------

.. code-block::

   @software{Braun_AudioTree_2026,
      author = {Braun, David},
      title = {{AudioTree}},
      url = {https://github.com/DBraun/audiotree},
      version = {1.0.0},
      year = {2026}
   }

.. _flax.struct.dataclass: https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass
.. _Grain: https://github.com/google/grain
.. _Pytree: https://jax.readthedocs.io/en/latest/pytrees.html
.. _jax.tree.map: https://jax.readthedocs.io/en/latest/_autosummary/jax.tree.map.html#jax.tree.map
.. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
.. _jax.vmap: https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html
.. _ArgBind: https://github.com/DBraun/argbind/
.. _AudioTools: https://github.com/descriptinc/audiotools/
.. _librosax: https://pypi.org/project/librosax/
