AudioTree documentation
=======================

**AudioTree** represents audio as a `JAX <https://jax.readthedocs.io/en/latest/>`_ `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_:
a batched ``AudioTree`` container, `Grain`_ data sources, and dual NumPy/JAX augmentations.
JAX and Flax are hard requirements; there is no torch path (see :ref:`api_stability`).
The source code is `here <https://github.com/DBraun/audiotree>`_.

AudioTree can be installed with pip:

.. code-block:: bash

   pip install audiotree

The namesake class :class:`~audiotree.AudioTree` is a `flax.struct.dataclass`_
that holds a batch of audio under one shape convention: every array carries the
batch as its leading axis, so items stay aligned as you index, slice, batch, and
transform them.

- ``waveform`` — the time-domain audio, ``(B, C, T)`` (batch, channels, samples)
- ``sample_rate`` — a shared scalar ``int``; the one field that is *not* batched
- ``lufs`` / ``lufs_windows`` — loudness, filled on demand, ``(B,)`` / ``(B, Windows)``
- ``pitch``, ``velocity``, ``note_duration`` — optional per-item labels (MIDI pitch/velocity, note length), ``(B,)`` each
- ``codes`` / ``latents`` — neural-codec tokens or latent embeddings, ``(B, ...)``
- ``metadata`` — a dict of your own ``(B, ...)`` arrays (source paths land in ``metadata["filepath"]``)

AudioTree integrates with `Grain`_ to provide complete data pipelines. Load audio from directories,
apply balanced sampling across groups, and chain augmentations:

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
            soundfile.write(
                os.path.join(_d, f"{_i}.wav"),
                (0.1 * _g.standard_normal((44_100, 2))).astype(np.float32),
                44_100,
            )

.. testcode::

    from audiotree import AudioTree
    from audiotree.sources import create_balanced_audio_dataset
    from audiotree.transforms import volume_norm, stereo

    # Create dataset with balanced sampling across groups
    ds = create_balanced_audio_dataset(
        sources={"speech": [_speech_dir], "music": [_music_dir]},
        weights={"speech": 0.7, "music": 0.3},
        sample_rate=44100,
        duration=10.0,
    )

    # Chain transforms using Grain's API
    ds = ds.seed(42)
    ds = ds.map(stereo())
    ds = ds.random_map(volume_norm(min_db=-20, max_db=-15))

    # Convert to iterable and batch
    iter_ds = ds.to_iter_dataset().batch(32, batch_fn=AudioTree.batch)

    # Access batched AudioTrees
    batch: AudioTree = next(iter(iter_ds))
    print(batch.waveform.shape)    # (32, channels, 441000)
    print(batch.source[:3])        # ["speech", "music", ...]
    # print(batch.filepath)        # ["path/to/file_abc.wav", "path/to/file_xyz.wav", ...]

.. testoutput::

    (32, 2, 441000)
    ['speech', 'speech', 'speech']

These transforms compose over any `Pytree`_ of AudioTrees — a single tree, a list,
or a dict — so they drop straight into pipelines like the one above. With `ArgBind`_
they are also configurable from the command line and YAML:

.. code-block:: bash

    python train.py --volume_norm.min_db=-25 --volume_norm.max_db=-15

The guides are meant to be read in order. **Getting started** walks the main path —
the :class:`~audiotree.AudioTree` object, loading audio, augmenting it, and
writing datasets back to disk — and **Going further** collects the deeper topics
(balanced and windowed sampling, dict batches, neural codecs, command-line
configuration, and multiprocessing).

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
   introduction/dict_batches
   introduction/codecs
   introduction/argbind_guide
   introduction/multiprocessing

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

   api_stability
   migration_1_0
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
.. _ArgBind: https://github.com/DBraun/argbind/
.. _AudioTools: https://github.com/descriptinc/audiotools/
